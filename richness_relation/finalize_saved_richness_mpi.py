"""Finalize ONE saved global fit without optimization or MCMC.

Run only after the original sampler has stopped. Original chains and metadata
are read-only; a new fit root contains derived products for existing plot tools.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from richness_relation.conditional_richness_models import (
    ModelConfig, parameter_spec, selected_logpdf, selected_rvs,
)
from richness_relation.prepare_conditional_richness import file_sha256, load_sample
from richness_relation.fit_conditional_richness_mpi import (
    discrepancies, posterior_summary, write_json,
)


def load_saved(directory, sample_path, minimum_steps):
    import emcee
    import h5py
    manifest = json.loads((directory/'manifest.json').read_text())
    summary = json.loads((directory/'bestfit.json').read_text())
    data, audit = load_sample(sample_path)
    if manifest['data_sha256'] != audit['sample_sha256']:
        raise ValueError('Sample checksum differs from the fitted sample')
    if not np.array_equal(np.asarray(manifest['IDs']).astype(str), np.asarray(data['ID']).astype(str)):
        raise ValueError('This finalizer requires a global fit with identical ordered IDs')
    for name, digest in manifest['code_sha256'].items():
        if file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError(f'Original fitting code changed: {name}; use the original revision')
    encoded = dict(summary['config'])
    if encoded != manifest['config']:
        raise ValueError('Best-fit and manifest configurations disagree')
    encoded['rm_max'] = np.inf if encoded['rm_max'] is None else encoded['rm_max']
    config = ModelConfig(**encoded)
    names = [v[0] for v in parameter_spec(config)]
    if summary['parameter_names'] != names:
        raise ValueError('Unexpected parameter ordering')
    retained, diagnostics, ensemble_summaries, snapshots = [], [], [], []
    for index in range(manifest['ensembles']):
        path = directory/f'chain_{index}.h5'
        before = path.stat()
        with h5py.File(path, 'r') as f:
            g = f['mcmc']
            steps = int(g.attrs['iteration'])
            if steps < minimum_steps or steps - manifest['burn'] < 2:
                raise ValueError(f'{path}: only {steps} recorded steps')
            chain = g['chain'][manifest['burn']:steps]
            acceptance = float(np.mean(g['accepted'][:])/steps)
        if chain.shape[1:] != (manifest['walkers'], len(names)) or not np.all(np.isfinite(chain)):
            raise ValueError(f'Invalid chain dimensions or values: {path}')
        samples = chain[::manifest['thin']].reshape(-1, len(names))
        retained.append(samples)
        ensemble_summaries.append(posterior_summary(samples, config))
        tau = emcee.autocorr.integrated_time(chain, tol=0)
        valid_tau = bool(np.all(np.isfinite(tau)) and np.all(tau > 0)
                         and np.all(np.var(chain, axis=0) > 0))
        diagnostics.append(dict(ensemble=index, steps=steps, acceptance_mean=acceptance,
            tau_steps=tau.tolist() if valid_tau else None,
            length_exceeds_50_tau=bool(valid_tau and np.all(len(chain)>50*tau))))
        snapshots.append((str(path), before.st_size, before.st_mtime_ns))
    samples = np.concatenate(retained)
    summary['posterior'] = posterior_summary(samples, config)
    summary['ensemble_posteriors'] = ensemble_summaries
    summary['chain_diagnostics'] = diagnostics
    return data, config, samples, summary, manifest, snapshots


def predictive_one(theta, data, config, seed, index):
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 881]))
    replica = selected_rvs(theta, data['lambda_spec'], data['z'], config, rng)
    return (discrepancies(data['lambda_rm'],data['lambda_spec'],data['z'],theta,config),
            discrepancies(replica,data['lambda_spec'],data['z'],theta,config))


def summarize_predictive(pairs):
    observed, replicated = zip(*pairs)
    return {name: dict(observed_median=float(np.median([o[name] for o in observed])),
        replicated_p16_p50_p84=np.percentile([r[name] for r in replicated],[16,50,84]).tolist(),
        ppc_probability_rep_ge_obs=float(np.mean([r[name]>=o[name] for o,r in pairs])))
        for name in observed[0]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True,help='Exact task/seed directory')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='NEW fit root; source remains unchanged')
    p.add_argument('--mpi',action='store_true')
    p.add_argument('--min-steps',type=int,default=5000)
    p.add_argument('--ppc-draws',type=int,default=200)
    p.add_argument('--save-draws',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if min(a.min_steps,a.ppc_draws,a.save_draws)<1 or a.seed<0: p.error('Invalid counts or seed')
    comm=None
    rank,size=0,1
    if a.mpi:
        from mpi4py import MPI
        comm=MPI.COMM_WORLD
        rank,size=comm.rank,comm.size
    elif int(os.environ.get('SLURM_NTASKS','1'))>1:
        p.error('Use --mpi with multiple Slurm ranks')
    try:
        payload=None
        if rank==0:
            destination=a.output/a.fit.parent.name/a.fit.name
            destination.mkdir(parents=True,exist_ok=False)
            print('Reading saved chains; no fitting or sampling',flush=True)
            data,config,samples,summary,manifest,snapshots=load_saved(a.fit,a.data,a.min_steps)
            rng=np.random.default_rng(a.seed)
            ppc=samples[rng.integers(len(samples),size=a.ppc_draws)]
            subset=samples[rng.choice(len(samples),min(len(samples),a.save_draws),replace=False)]
            payload=(data,config,ppc,subset)
        if comm is not None: payload=comm.bcast(payload,root=0)
        data,config,ppc,subset=payload
        local_ppc=[]
        local_ll=[]
        for i in range(rank,len(ppc),size):
            local_ppc.append((i,predictive_one(ppc[i],data,config,a.seed,i)))
            print(f'rank {rank}: PPC {i+1}/{len(ppc)}',flush=True)
        for i in range(rank,len(subset),size):
            local_ll.append((i,selected_logpdf(subset[i],data['lambda_rm'],data['lambda_spec'],data['z'],config)))
        gathered_ppc=comm.gather(local_ppc,root=0) if comm is not None else [local_ppc]
        gathered_ll=comm.gather(local_ll,root=0) if comm is not None else [local_ll]
        if rank==0:
            for path,bytes_,mtime in snapshots:
                stat=Path(path).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(bytes_,mtime):
                    raise RuntimeError('Source chain changed during finalization; stop the sampler first')
            pairs=[v for i,v in sorted(v for part in gathered_ppc for v in part)]
            likelihood=np.array([v for i,v in sorted(v for part in gathered_ll for v in part)])
            summary['posterior_predictive']=summarize_predictive(pairs)
            summary['finalization']=dict(source=str(a.fit.resolve()),ranks=size,seed=a.seed,
                ppc_draws=a.ppc_draws,save_draws=len(subset),sampling_restarted=False)
            summary['convergence_note']='50-tau is a heuristic; run independent-ensemble convergence checks on original chains.'
            shutil.copy2(a.fit/'manifest.json',destination/'manifest.json')
            shutil.copy2(a.fit/'bestfit.json',destination/'bestfit.json')
            tmp=destination/'posterior.tmp.npz'
            np.savez_compressed(tmp,samples=subset,log_likelihood=likelihood,ID=data['ID'])
            tmp.replace(destination/'posterior.npz')
            write_json(destination/'summary.json',summary)
            print('Saved derived products:',destination,flush=True)
    except Exception:
        traceback.print_exc()
        if comm is not None: comm.Abort(1)
        raise


if __name__=='__main__':
    main()
