"""Run nine independent conditional-distribution cells for each model.

MPI distributes independent cells, not walkers. At most 18 ranks are useful
for both models. Original global chains are never read or overwritten.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from richness_relation.cell_richness_models import (
    make_cells,config,parameter_spec,log_probability,optimize,posterior_summary,pearson_stat,
)
from richness_relation.prepare_conditional_richness import load_sample,file_sha256
from richness_relation.fit_conditional_richness_mpi import write_json
from richness_relation.check_richness_convergence import classical_rhat


def fit_cell(cell,mask,model,data,audit,args):
    import emcee
    started=time.monotonic()
    task=f'{model}_{cell["cell"]}'
    out=args.output/task/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=True)
    rm=data['lambda_rm'][mask]
    cfg=config(model,audit['rm_min'],audit['rm_max'])
    signature=dict(schema_version=1,cell=cell,model=model,data_sha256=audit['sample_sha256'],
                   IDs=data['ID'][mask].tolist(),rm_min=audit['rm_min'],rm_max=audit['rm_max'],
                   seed=args.seed,walkers=args.walkers,ensembles=args.ensembles,burn=args.burn,thin=args.thin,
                   code_sha256={name:file_sha256(Path(__file__).with_name(name)) for name in
                       ('cell_richness_models.py','conditional_richness_models.py','fit_richness_cells_mpi.py')})
    manifest=out/'manifest.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())!=signature:
            raise ValueError(f'{out}: existing fit/configuration mismatch; use a new output root')
    else: write_json(manifest,signature)
    seed=args.seed+100*cell['iz']+10*cell['ix']+(1000 if model=='mixture' else 0)
    rng=np.random.default_rng(seed)
    bestpath=out/'bestfit.json'
    names=[v[0] for v in parameter_spec(model)]
    if bestpath.exists():
        result=json.loads(bestpath.read_text())
        mle=np.asarray(result['mle_theta'])
    else:
        mle=optimize(rm,cfg,rng,args.starts).x
        result=dict(task=task,cell=cell,model=model,parameter_names=names,mle_theta=mle.tolist(),
                    priors_uniform_in_sampled_coordinates=parameter_spec(model))
        write_json(bestpath,result)
    bounds=np.array([v[1:] for v in parameter_spec(model)])
    chains,draws,diagnostics=[],[],[]
    for ensemble in range(args.ensembles):
        path=out/f'chain_{ensemble}.h5'
        backend=emcee.backends.HDFBackend(str(path))
        if path.exists():
            if not args.resume: raise FileExistsError(path)
            initial=None
        else:
            backend.reset(args.walkers,len(names))
            initial=[]
            for walker in range(args.walkers):
                for attempt in range(10000):
                    trial=mle+rng.normal(size=len(names))*.01*(bounds[:,1]-bounds[:,0])
                    if np.isfinite(log_probability(trial,rm,cfg)):
                        initial.append(trial); break
                else: raise RuntimeError('Unable to initialize finite walkers')
            initial=np.asarray(initial)
        if backend.iteration>args.steps:
            raise ValueError('Requested steps shorter than existing chain; increase --steps')
        if backend.iteration==0 and initial is None:
            raise RuntimeError('Empty existing chain; choose a new output root')
        sampler=emcee.EnsembleSampler(args.walkers,len(names),log_probability,args=(rm,cfg),backend=backend)
        if initial is not None: sampler.random_state=np.random.RandomState(seed+ensemble).get_state()
        remaining=args.steps-backend.iteration
        if remaining>0:
            print(f'{task} ensemble {ensemble}: adding {remaining} steps',flush=True)
            sampler.run_mcmc(initial,remaining,progress=False)
        chain=backend.get_chain(discard=args.burn)
        sample=chain[::args.thin].reshape(-1,len(names))
        chains.append(chain); draws.append(sample)
        tau=emcee.autocorr.integrated_time(chain,tol=0)
        good=bool(np.all(np.isfinite(tau)) and np.all(tau>0) and np.all(np.var(chain,axis=0)>0))
        diagnostics.append(dict(ensemble=ensemble,steps=int(backend.iteration),
            acceptance_mean=float(np.mean(sampler.acceptance_fraction)),
            tau_steps=tau.tolist() if good else None,
            length_exceeds_50_tau=bool(good and np.all(len(chain)>50*tau))))
    samples=np.concatenate(draws)
    # Refine the MLE from the highest-probability sampled point if needed.
    lp=np.concatenate([emcee.backends.HDFBackend(str(out/f'chain_{i}.h5'),read_only=True)
                       .get_log_prob(discard=args.burn,thin=args.thin,flat=True) for i in range(args.ensembles)])
    candidate=samples[np.argmax(lp)]
    if log_probability(candidate,rm,cfg)>log_probability(mle,rm,cfg):
        refinement=optimize(rm,cfg,rng,args.starts,initial=candidate)
        if log_probability(refinement.x,rm,cfg)>log_probability(mle,rm,cfg): mle=refinement.x
    ll=float(log_probability(mle,rm,cfg))
    n,k=len(rm),len(names)
    rhat=None
    if len(chains)>=2:
        values=np.array([classical_rhat(np.stack([c[:,w,:] for c in chains])) for w in range(args.walkers)])
        if np.all(np.isfinite(values)): rhat=values.max(axis=0).tolist()
    stat,groups=pearson_stat(rm,mle,cfg)
    converged=bool(all(d['length_exceeds_50_tau'] for d in diagnostics) and rhat is not None and max(rhat)<1.1)
    result.update(mle_theta=mle.tolist(),log_likelihood_max=ll,N=n,k=k,
        AIC=2*k-2*ll,BIC=k*np.log(n)-2*ll,
        chi2=stat,chi2_groups=groups,nominal_dof=groups-1-k,
        posterior=posterior_summary(samples,model),chain_diagnostics=diagnostics,
        rhat_max_over_fixed_walker_indices=rhat,convergence_pass=converged,
        elapsed_seconds=time.monotonic()-started,
        warnings=['Finite-bin distribution, not deconvolved intrinsic scatter.',
                  'MLE and posterior priors require boundary sensitivity checks.',
                  'No analytic chi-square p-value: use optional refitted parametric bootstrap.',
                  'Convergence diagnostics are necessary checks, not proof of exploration.'])
    subset=samples[rng.choice(len(samples),min(args.save_draws,len(samples)),replace=False)]
    temporary=out/'posterior.tmp.npz'
    np.savez_compressed(temporary,samples=subset)
    temporary.replace(out/'posterior.npz')
    write_json(out/'summary.json',result)
    print(f'{task}: COMPLETE, elapsed={result["elapsed_seconds"]:.1f}s, convergence_pass={converged}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'catalogs/conditional_richness/overnight_sample.npz')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=['both','lognormal','mixture'],default='both')
    p.add_argument('--richness-edges',type=float,nargs='+',help='Common edges; default: three full-sample tercile bins')
    p.add_argument('--walkers',type=int,default=48)
    p.add_argument('--ensembles',type=int,default=2)
    p.add_argument('--steps',type=int,default=5000)
    p.add_argument('--burn',type=int,default=1000)
    p.add_argument('--thin',type=int,default=1)
    p.add_argument('--starts',type=int,default=8)
    p.add_argument('--save-draws',type=int,default=10000)
    p.add_argument('--min-count',type=int,default=50)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--mpi',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.walkers<8 or a.walkers%2 or a.steps<=a.burn or a.burn<0 or min(a.ensembles,a.thin,a.starts,a.save_draws,a.min_count)<1:
        p.error('Invalid sampling configuration')
    comm=None; rank,size=0,1; lock=None
    if a.mpi:
        from mpi4py import MPI
        comm=MPI.COMM_WORLD; rank,size=comm.rank,comm.size
    elif int(os.environ.get('SLURM_NTASKS','1'))>1: p.error('Multiple tasks require --mpi')
    try:
        data,audit=load_sample(a.data)
        edges,cells=make_cells(data,a.richness_edges)
        models=('lognormal','mixture') if a.model=='both' else (a.model,)
        tasks=[(c,m,model) for c,m in cells for model in models if c['N']>=a.min_count]
        if rank==0:
            print('Richness edges:',edges,flush=True)
            for c,_ in cells: print(c, 'SKIPPED' if c['N']<a.min_count else '',flush=True)
            print(f'{len(tasks)} fits; {size} ranks',flush=True)
        if a.dry_run: return
        if rank==0:
            a.output.mkdir(parents=True,exist_ok=True)
            candidate=a.output/'RUNNING.lock'
            fd=os.open(candidate,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            os.close(fd); lock=candidate
            layout=dict(sample_sha256=audit['sample_sha256'],richness_edges=edges.tolist(),
                        cells=[c for c,_ in cells],rm_min=audit['rm_min'],rm_max=audit['rm_max'],min_count=a.min_count)
            path=a.output/'cells.json'
            if path.exists() and (not a.resume or json.loads(path.read_text())!=layout):
                raise ValueError('Existing cell layout differs or --resume missing')
            write_json(path,layout)
        if comm is not None: comm.Barrier()
        for index in range(rank,len(tasks),size):
            fit_cell(*tasks[index],data,audit,a)
        if comm is not None: comm.Barrier()
        if rank==0: print('All requested eligible cell fits finalized. Check convergence before interpretation.',flush=True)
    except Exception:
        traceback.print_exc()
        if comm is not None: comm.Abort(1)
        raise
    finally:
        if lock is not None: lock.unlink(missing_ok=True)


if __name__=='__main__': main()
