"""Fit forward or reverse lognormal relations with independent emcee ensembles."""
import argparse
import json
import hashlib
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import emcee
import numpy as np

from global_spec_richness_model import Regression, ForwardRegression, CellResidual, parameter_names, RM_PIVOT, Z_PIVOT
from prepare_conditional_richness import load_sample
from global_spec_richness_model import EnvironmentMixture, FixedMeanMixture, classical_rhat

ROOT = Path(__file__).resolve().parents[1]


def write_atomic_json(path, value):
    path=Path(path)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def environment_model(rm, meta):
    if 'fixed_global_means' in meta:
        return FixedMeanMixture(rm, meta['fixed_global_means'], meta.get('mixture_prior','legacy'))
    return EnvironmentMixture(rm)


def ensemble_moves(meta):
    if meta.get('proposal') == 'de_snooker':
        return [(emcee.moves.DEMove(), .8), (emcee.moves.DESnookerMove(), .2)]
    return emcee.moves.StretchMove()


def richness_redshift_cells(frame, pooled_quartiles=False):
    """Common global richness tertiles, fixed z bins, no environment split."""
    edges=np.quantile(frame.lambda_spec,np.linspace(0,1,5) if pooled_quartiles else [0,1/3,2/3,1])
    if np.any(np.diff(edges)<=0):
        raise ValueError('Tied richness tertiles require explicit bin definitions')
    zedges=np.array([.10,.35] if pooled_quartiles else [.10,.18,.24,.35])
    if not np.all((frame.z>=zedges[0])&(frame.z<zedges[-1])):
        raise ValueError('Unexpected redshift range')
    frame=frame.copy()
    frame['spec_bin']=np.searchsorted(edges[1:-1],frame.lambda_spec,side='right')+1
    frame['z_bin']=np.searchsorted(zedges[1:-1],frame.z,side='right')+1
    frame['cell']=['z%d_s%d'%(z,s) for z,s in zip(frame.z_bin,frame.spec_bin)]
    frame['quartile']=0
    frame['parent_mean_rm']=frame.groupby('cell').lambda_rm.transform('mean')
    if frame.cell.nunique()!=(4 if pooled_quartiles else 9) or frame.ID.duplicated().any():
        raise ValueError('Require nine nonempty disjoint cells')
    return frame,edges,zedges


def advance_environment_ensemble(task):
    directory,rm,meta,optima,steps,i=task
    reg=environment_model(rm, meta)
    ndim=len(reg.names)
    backend=emcee.backends.HDFBackend(str(Path(directory)/f'chain_{i}.h5'))
    if not backend.initialized:
        backend.reset(meta['walkers'],ndim)
    sampler=emcee.EnsembleSampler(meta['walkers'],ndim,reg,backend=backend,vectorize=True,
                                 moves=ensemble_moves(meta))
    state=None
    if backend.iteration==0:
        seed=meta['seed']+i
        state=reg.initial(np.random.default_rng(seed),meta['walkers'],optima[i%len(optima)])
        sampler.random_state=np.random.RandomState(seed).get_state()
    if backend.iteration>steps:
        raise ValueError('Requested length shorter than existing chain')
    if backend.iteration<steps:
        sampler.run_mcmc(state,steps-backend.iteration,progress=False)
    print(f'{Path(directory).name} ensemble {i+1}/{meta["ensembles"]}: {steps} steps',flush=True)


def run_environment_cell(task):
    directory, rm, meta, steps, resume = task
    started=time.monotonic()
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    reg=environment_model(rm, meta)
    ndim=len(reg.names)
    path=directory/'run.json'
    if path.exists():
        if not resume or json.loads(path.read_text())!=meta:
            raise ValueError(f'Existing configuration mismatch: {directory}')
    else:
        if list(directory.glob('chain_*.h5')):
            raise ValueError('Chains without metadata')
        path.write_text(json.dumps(meta,indent=2))
    bestpath=directory/'bestfit.json'
    if bestpath.exists():
        best=json.loads(bestpath.read_text())
        optima=np.asarray(best['optima'])
    else:
        fits=reg.optimize(meta['seed'])
        optima=np.array([r.x for r in fits])
        if meta.get('mixture_prior') == 'beta22_uniform_sigma':
            g,e=reg.component_logpdf(optima[0],reg.rm,reg.means)
            best={'optima':optima.tolist(),'map':optima[0].tolist(),
                  'log_posterior':float(reg(optima[0])),
                  'log_likelihood_at_map':float(np.logaddexp(g,e).sum())}
        else:
            best={'optima':optima.tolist(), 'mle':optima[0].tolist(), 'log_likelihood':float(reg(optima[0]))}
        bestpath.write_text(json.dumps(best,indent=2))
    ensemble_tasks=[(directory,rm,meta,optima,steps,i) for i in range(meta['ensembles'])]
    if meta.get('ensemble_workers',1)>1:
        with ProcessPoolExecutor(max_workers=meta['ensemble_workers']) as pool:
            list(pool.map(advance_environment_ensemble,ensemble_tasks))
    else:
        for t in ensemble_tasks: advance_environment_ensemble(t)
    backends=[emcee.backends.HDFBackend(str(directory/f'chain_{i}.h5'),read_only=True)
              for i in range(meta['ensembles'])]
    chains=[b.get_chain(discard=meta['burn']) for b in backends]
    taus=np.array([emcee.autocorr.integrated_time(c,tol=0) for c in chains])
    ratios=np.array([len(c) for c in chains])[:,None]/taus
    rhat=np.max([classical_rhat(np.stack([c[:,w] for c in chains]))
                 for w in range(meta['walkers'])],axis=0)
    stuck=any(np.any(np.var(c,axis=0)==0) for c in chains)
    passed=bool(not stuck and np.all(np.isfinite(taus)) and np.all(taus>0)
                and np.all(ratios>=50) and np.all(np.isfinite(rhat)) and np.all(rhat<1.1))
    samples=np.concatenate([c[::5].reshape(-1,ndim) for c in chains])
    np.savez_compressed(directory/'posterior.npz',samples=samples)
    logs=[j for j,name in enumerate(reg.names) if name.startswith('ln_')]
    physical=samples.copy(); physical[:,logs]=np.exp(physical[:,logs])
    summary=dict(parameters=[name.removeprefix('ln_') for name in reg.names],
                 posterior_quantiles=np.percentile(physical,[16,50,84],axis=0).tolist(),
                 convergence_pass=passed,tau_steps=taus.tolist(),steps_per_tau=ratios.tolist(),
                 rhat_worst=rhat.tolist(),acceptance_mean=[float(np.mean(b.accepted/b.iteration)) for b in backends],
                 steps=steps,elapsed_seconds=time.monotonic()-started,
                 rhat_method='Worst fixed walker index across independent ensembles, never across interacting walkers',
                 prior_boundary_fraction=[float(np.mean((samples[:,j]<lo+.01*(hi-lo)) |
                                                        (samples[:,j]>hi-.01*(hi-lo))))
                                          for j,(lo,hi) in enumerate(reg.bounds)],
                 warning=('Global arithmetic means fixed; uncertainty and model misspecification not propagated.'
                          if 'fixed_global_means' in meta else
                          'Shared mean is a display offset only; all four parameters are freely fitted. Check prior sensitivity.'))
    write_atomic_json(directory/'summary.json',summary)
    print(f'{directory.name} COMPLETE: converged={passed}; {summary["elapsed_seconds"]:.1f}s',flush=True)
    return dict(task=directory.name,**summary)


def adaptive_checkpoint(report, previous, tolerance=.1):
    """Stability and repeat-pass logic, independent of how chains are stored."""
    tau=np.asarray(report['tau_steps'])
    change=None
    if previous is not None:
        old=np.asarray(previous['tau_steps'])
        if old.shape==tau.shape and np.all(np.isfinite(old)) and np.all(old>0):
            value=float(np.max(np.abs(tau-old)/old))
            if np.isfinite(value): change=value
    stable=change is not None and change<tolerance
    passed=bool(report['convergence_pass'] and stable)
    consecutive=(previous.get('consecutive_passes',0)+1 if previous else 1) if passed else 0
    return dict(steps=report['steps'],tau_steps=report['tau_steps'],
                min_steps_per_tau=float(np.min(report['steps_per_tau'])),
                max_rhat=float(np.max(report['rhat_worst'])),base_checks_pass=report['convergence_pass'],
                max_fractional_tau_change=change,tau_stable=stable,
                checkpoint_pass=passed,consecutive_passes=consecutive,converged=consecutive>=2)


def extend_environment_budget(directory, new):
    """Explicitly increase execution limits without changing the fitted target."""
    directory=Path(directory)
    if (directory/'RUNNING.lock').exists():
        raise ValueError('Cannot change a budget while this fit is running')
    path=directory/'run.json'
    old=json.loads(path.read_text())
    def identity(meta):
        result=json.loads(json.dumps(meta))
        result.pop('runner_sha256',None)
        result['adaptive'].pop('max_steps')
        result['adaptive'].pop('max_seconds')
        return result
    if identity(old)!=identity(new):
        raise ValueError('Budget extension must preserve data, model checksum, priors, sampler dimensions and stopping criteria')
    keys=('max_steps','max_seconds')
    if any(new['adaptive'][k]<old['adaptive'][k] for k in keys) or not any(new['adaptive'][k]>old['adaptive'][k] for k in keys):
        raise ValueError('At least one limit must increase; neither may decrease')
    history_path=directory/'budget_revisions.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    history.append(dict(unix_time=time.time(),previous=old,updated=new,
                        reason='Explicit --resume --extend-budget; target and convergence criteria unchanged'))
    write_atomic_json(history_path,history)
    write_atomic_json(path,new)


def run_adaptive_environment(task):
    directory,rm,meta,initial_steps,resume=task
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    rules=meta['adaptive']
    lock=directory/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.close(fd)
    started=time.monotonic()
    history_path=directory/'convergence_history.json'
    history=json.loads(history_path.read_text()) if history_path.exists() else []
    try:
        existing=directory/'run.json'
        if existing.exists() and (not resume or json.loads(existing.read_text())!=meta):
            raise ValueError('Existing adaptive run differs or --resume missing')
        lengths=[]
        for i in range(meta['ensembles']):
            backend=emcee.backends.HDFBackend(str(directory/f'chain_{i}.h5'),read_only=True)
            lengths.append(backend.iteration if backend.initialized else 0)
        target=max(initial_steps,max(lengths))
        if history and max(lengths)==history[-1]['steps']:
            if history[-1]['converged']:
                return dict(task=directory.name,**json.loads((directory/'summary.json').read_text()))
            target=max(target,history[-1]['steps']+rules['block_steps'])
        target=min(target,rules['max_steps'])
        while True:
            report=run_environment_cell((directory,rm,meta,target,resume or existing.exists()))
            previous=next((h for h in reversed(history) if h['steps']<target),None)
            entry=adaptive_checkpoint(report,previous,rules['tau_stability'])
            entry['elapsed_seconds_this_invocation']=time.monotonic()-started
            history=[h for h in history if h['steps']<target]+[entry]
            write_atomic_json(history_path,history)
            reason=('converged' if entry['converged'] else 'max_steps' if target>=rules['max_steps']
                    else 'wall_time_limit' if time.monotonic()-started>=rules['max_seconds'] else 'running')
            report.update(adaptive_convergence_pass=entry['converged'],stop_reason=reason)
            write_atomic_json(directory/'summary.json',report)
            print(f'CHECKPOINT {target}: min N/tau={entry["min_steps_per_tau"]:.1f}, '
                  f'max Rhat={entry["max_rhat"]:.3f}, tau change={entry["max_fractional_tau_change"]}, '
                  f'consecutive passes={entry["consecutive_passes"]}; {reason}',flush=True)
            if reason!='running': return report
            target=min(target+rules['block_steps'],rules['max_steps'])
            resume=True
    finally:
        lock.unlink(missing_ok=True)


def fit_environment(args):
    import pandas as pd
    if args.steps<=args.burn or args.burn<0 or args.walkers<8 or args.ensembles<2 or args.workers<1:
        raise ValueError('Invalid sampling configuration')
    if args.adaptive and (args.block_steps<1 or args.max_steps<args.steps or args.max_seconds<=0):
        raise ValueError('Invalid adaptive limits')
    source=args.environment_assignments
    frame=pd.read_csv(source,dtype={'ID':str})
    if frame.ID.duplicated().any() or frame.groupby(['cell','quartile']).ngroups!=36:
        raise ValueError('Require 36 groups and unique IDs')
    for _,g in frame.groupby('cell'):
        if g.parent_mean_rm.nunique()!=1 or not np.isclose(g.lambda_rm.mean(),g.parent_mean_rm.iloc[0]):
            raise ValueError('Parent mean does not match observed cell')
    if args.richness_redshift_grid or args.pooled_spec_quartiles:
        frame,spec_edges,z_edges=richness_redshift_cells(frame,args.pooled_spec_quartiles)
    mean_cache={}
    if args.fixed_mean_cache:
        expected=hashlib.sha256((args.fixed_global_mean/'posterior.npz').read_bytes()).hexdigest()
        for p in args.fixed_mean_cache.glob('*/run.json'):
            cached=json.loads(p.read_text())
            if cached['global_posterior_sha256']!=expected or cached['assignments_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest():
                raise ValueError('Fixed-mean cache provenance mismatch')
            for key,value in zip(cached['cluster_ids'],cached['fixed_global_means']):
                if key in mean_cache: raise ValueError('Duplicate ID in fixed-mean cache')
                mean_cache[key]=value
    if args.pooled_redshift or args.highest_richness_pooled:
        frame['spec_bin']=frame.cell.str.extract(r'_s(\d+)$')[0].astype(int)
        if not np.all((frame.z>=.1)&(frame.z<.35)):
            raise ValueError('Pooled sample must lie within 0.10 <= z < 0.35')
        if args.highest_richness_pooled:
            frame=frame[frame.spec_bin==frame.spec_bin.max()].copy()
            frame['quartile']=0
            frame['cell']='pooled_s3'
            frame['parent_mean_rm']=frame.lambda_rm.mean()
        else:
            pieces=[]
            for spec_bin,g in frame.groupby('spec_bin'):
                g=g.copy(); edges=np.quantile(g.environment,[0,.25,.5,.75,1])
                if np.any(np.diff(edges)<=0): raise ValueError('Tied environment quartiles')
                g['quartile']=np.searchsorted(edges[1:-1],g.environment,side='right')+1
                g['cell']=f'pooled_s{spec_bin}'
                g['parent_mean_rm']=g.lambda_rm.mean()
                pieces.append(g)
            frame=pd.concat(pieces)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    args.output=args.output or ROOT/'catalogs/environment_mixture_4param'
    tasks=[]
    for index,((cell,q),g) in enumerate(frame.groupby(['cell','quartile'])):
        meta=dict(model='positive_tail_mixture',cell=cell,quartile=int(q),N=len(g),
                  parent_mean_rm=float(g.parent_mean_rm.iloc[0]),
                  assignments=str(source.resolve()),assignments_sha256=digest,cluster_ids=g.ID.tolist(),
                  parameters=EnvironmentMixture.names,bounds=EnvironmentMixture.bounds,
                  prior='Independent uniform priors in mu, ln_sigma_lambda, f_proj, ln_tau within recorded bounds',
                  likelihood='Unbinned normal + positive EMG, normalized on lambda_RM > 20',
                  centering='Display only: shared measured parent mean; not a likelihood constraint',
                  seed=args.seed+1000*index,walkers=args.walkers,ensembles=args.ensembles,burn=args.burn,
                  model_sha256=hashlib.sha256(Path(__file__).with_name('global_spec_richness_model.py').read_bytes()).hexdigest())
        meta=json.loads(json.dumps(meta))
        if args.proposal!='stretch':
            meta.update(proposal=args.proposal,proposal_weights=dict(DEMove=.8,DESnookerMove=.2))
        if args.richness_redshift_grid or args.pooled_spec_quartiles:
            ispec=int(g.spec_bin.iloc[0])-1; iz=int(g.z_bin.iloc[0])-1
            meta.update(pooling='All environments; common global richness tertiles x fixed redshift bins',
                        spec_bin=ispec+1,z_bin=iz+1,spec_edges=spec_edges.tolist(),z_edges=z_edges.tolist(),
                        spec_range=spec_edges[ispec:ispec+2].tolist(),z_range=z_edges[iz:iz+2].tolist(),
                        spec_median=float(g.lambda_spec.median()),ensemble_workers=1,
                        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            if args.pooled_spec_quartiles:
                meta['pooling']='All redshifts and environments; four equal-count spec bins'
        if args.fixed_global_mean:
            global_dir=args.fixed_global_mean
            global_meta=json.loads((global_dir/'run.json').read_text())
            diagnostic=json.loads((global_dir/'diagnostics.json').read_text())
            if global_meta.get('direction')!='forward' or global_meta['model']!='noz' or not diagnostic['converged']:
                raise ValueError('Require a converged forward no-evolution global fit')
            data,audit=load_sample(Path(global_meta['data_path']))
            if audit['sample_sha256']!=global_meta['sample_sha256']:
                raise ValueError('Global sample checksum mismatch')
            lookup={str(i):s for i,s in zip(data['ID'],data['lambda_spec'])}
            if not np.allclose([lookup[i] for i in g.ID],g.lambda_spec,rtol=1e-10):
                raise ValueError('Global and cell richnesses differ')
            posterior_path=global_dir/'posterior.npz'
            with np.load(posterior_path) as archive:
                posterior=archive['samples']
            # Internal A is log arithmetic mean, not mean log richness.
            global_means=[]
            if mean_cache:
                global_means=[mean_cache[key] for key in g.ID]
            for values in ([] if mean_cache else np.array_split(g.lambda_spec.to_numpy(), max(1,int(np.ceil(len(g)/32))))):
                prediction=posterior[:,0,None]+posterior[:,1,None]*np.log(values[None,:]/40.)
                global_means.extend(np.exp(np.median(prediction,axis=0)).tolist())
            meta.update(model='positive_tail_mixture_fixed_global_mean',parameters=FixedMeanMixture.names,
                        bounds=FixedMeanMixture.bounds,fixed_global_means=global_means,
                        global_fit=str(global_dir.resolve()),
                        global_posterior_sha256=hashlib.sha256(posterior_path.read_bytes()).hexdigest(),
                        prior='Uniform in ln_sigma_lambda, f_proj, ln_tau within recorded bounds',
                        likelihood='Theta(RM-20) * (normal+positive EMG) / per-object survival at 20',
                        centering='mu_core_i = fixed_global_arithmetic_mean_i - f_proj/tau',
                        global_mean_estimator='Per-object posterior median arithmetic mean; global uncertainty not propagated')
            meta=json.loads(json.dumps(meta))
            if args.mixture_prior!='legacy':
                reg=FixedMeanMixture(g.lambda_rm.to_numpy(),global_means,args.mixture_prior)
                meta.update(mixture_prior=args.mixture_prior,bounds=reg.bounds,
                            prior='Beta(2,2) f; uniform sigma in (0,30), numerical floor 1e-12; log-uniform 1/tau in (0.1,500); log-sigma Jacobian included')
                meta=json.loads(json.dumps(meta))
        if args.adaptive:
            meta['adaptive']=dict(block_steps=args.block_steps,max_steps=args.max_steps,
                                  max_seconds=args.max_seconds,tau_stability=.1,consecutive_checks=2)
        if args.highest_richness_pooled or args.pooled_redshift:
            meta.update(pooling='all environments and redshifts' if args.highest_richness_pooled else 'redshifts pooled; environment quartiles recomputed within richness bin',
                        z_range=[.1,.35],spec_range=[float(g.lambda_spec.min()),float(g.lambda_spec.max())],
                        ensemble_workers=args.workers if args.highest_richness_pooled else 1,
                        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        directory=args.output/f'{cell}_Q{q}'
        if args.extend_budget:
            extend_environment_budget(directory,meta)
        tasks.append((directory,g.lambda_rm.to_numpy(),meta,args.steps,args.resume))
    if args.test_one:
        tasks=tasks[:1]
    if args.richness_redshift_grid or args.pooled_spec_quartiles:
        args.output.mkdir(parents=True,exist_ok=True)
        membership=args.output/'cell_assignments.csv'
        if not membership.exists():
            frame.to_csv(membership,index=False)
        print('Spec tertiles:',spec_edges,flush=True)
        print(frame.groupby('cell').size().to_string(),flush=True)
    runner=run_adaptive_environment if args.adaptive else run_environment_cell
    if args.workers==1 or args.highest_richness_pooled:
        results=[runner(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results=list(pool.map(runner,tasks))
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
    print(f'Completed {len(results)} fits; {sum(r.get("adaptive_convergence_pass",r["convergence_pass"]) for r in results)} pass convergence',flush=True)


def run_cell(task):
    directory, residual, logspec, family, meta, steps, resume = task
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    reg = CellResidual(residual, logspec, family)
    meta.update(parameters=reg.names, bounds=reg.bounds, model=family,
                mle=reg.mle.tolist(), optimizer_success=reg.optimizer_success)
    meta = json.loads(json.dumps(meta))
    path = directory / "run.json"
    if path.exists():
        old = json.loads(path.read_text())
        # Optimization can differ by machine rounding, but must not change provenance.
        compare = lambda d: {k:v for k,v in d.items() if k not in ("mle","optimizer_success")}
        if not resume or compare(old) != compare(meta):
            raise ValueError(f"Existing cell configuration differs: {directory}")
    else:
        if list(directory.glob("chain_*.h5")):
            raise ValueError("Missing cell provenance")
        path.write_text(json.dumps(meta, indent=2))
    for i in range(meta["ensembles"]):
        backend = emcee.backends.HDFBackend(str(directory/f"chain_{i}.h5"))
        if not backend.initialized:
            backend.reset(meta["walkers"],len(reg.names))
        sampler = emcee.EnsembleSampler(meta["walkers"],len(reg.names),reg,backend=backend)
        state = None
        if backend.iteration == 0:
            seed = meta["seed"]+i
            state = reg.initial(np.random.default_rng(seed),meta["walkers"])
            sampler.random_state = np.random.RandomState(seed).get_state()
        if backend.iteration < steps:
            sampler.run_mcmc(state, steps-backend.iteration, progress=False)
        print(f"{directory.parent.name}/{family} ensemble {i}: {backend.iteration} steps",flush=True)
    return str(directory)


def fit_cells(args):
    if args.steps <= args.burn or args.burn < 0 or args.ensembles < 2 or args.walkers < 6:
        raise ValueError("Invalid chain dimensions")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.only_unconverged and not args.resume:
        raise ValueError("--only-unconverged requires --resume")
    data,audit = load_sample(args.data)
    global_meta = json.loads((args.global_fits/"noz/run.json").read_text())
    diagnostics = json.loads((args.global_fits/"noz/diagnostics.json").read_text())
    if global_meta["sample_sha256"] != audit["sample_sha256"] or not diagnostics["converged"]:
        raise ValueError("Require converged no-z global fit on this exact sample")
    global_reg = Regression(data,"noz")
    fixed = global_reg.mle[:-1]
    logspec = np.log(data["lambda_spec"])
    residual = logspec-global_reg.X@fixed
    output = args.output or ROOT/"catalogs/spec_residual_cells"
    output.mkdir(parents=True,exist_ok=True)
    rm_edges, z_edges = [20.,30.,50.,100.], [.1,.18,.24,.35]
    tasks = []
    for iz,(zl,zh) in enumerate(zip(z_edges[:-1],z_edges[1:])):
        for ir,(rl,rh) in enumerate(zip(rm_edges[:-1],rm_edges[1:])):
            keep=(data["z"]>=zl)&(data["z"]<zh)&(data["lambda_rm"]>=rl)&(data["lambda_rm"]<rh)
            for family in ("lognormal","mixture"):
                meta=dict(sample_sha256=audit["sample_sha256"],data_path=str(args.data.resolve()),
                          fixed_global_mle=fixed.tolist(),rm_pivot=RM_PIVOT,
                          rm_bounds=[rl,rh],z_bounds=[zl,zh],n=int(keep.sum()),
                          cluster_ids=data["ID"][keep].astype(str).tolist(),
                          walkers=args.walkers,ensembles=args.ensembles,burn=args.burn,
                          seed=args.seed+1000*(3*iz+ir),version=1,
                          location="Fixed per-cluster no-z log-location; centered negative-tail log mixture",
                          prior="Uniform in sampled coordinates within bounds; no global uncertainty propagation")
                directory=output/f"z{iz}_rm{ir}"/family
                if args.only_unconverged:
                    report=json.loads((directory/"diagnostics.json").read_text())
                    if report["converged"]:
                        continue
                tasks.append((directory,residual[keep],logspec[keep],family,meta,args.steps,args.resume))
    if args.workers == 1:
        for task in tasks:
            run_cell(task)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(run_cell,tasks))


def fit(data_path, output, model, steps, burn, walkers, ensembles, seed, resume, direction='reverse'):
    data, audit = load_sample(data_path)
    reg = (ForwardRegression(data, model, audit['rm_min'], audit['rm_max'])
           if direction == 'forward' else Regression(data, model))
    ndim = len(reg.mle)
    if steps <= burn or burn < 0 or walkers < 2 * ndim or ensembles < 2:
        raise ValueError("Require steps > burn >= 0, walkers >= 2*ndim, ensembles >= 2")
    directory = Path(output) / model
    directory.mkdir(parents=True, exist_ok=True)
    meta = dict(model=model, sample_sha256=audit["sample_sha256"],
                data_path=str(Path(data_path).resolve()), n=reg.n,
                parameters=parameter_names(model), bounds=reg.bounds,
                rm_pivot=RM_PIVOT, z_pivot=Z_PIVOT, walkers=walkers,
                ensembles=ensembles, seed=seed, burn=burn,
                likelihood="lognormal in lambda_spec; includes Jacobian",
                prior="Uniform within bounds in coefficients and ln_s_rev",
                version=1)
    meta = json.loads(json.dumps(meta))
    if direction == 'forward':
        meta.update(direction='forward', parameters=reg.names, spec_pivot=40.,
                    rm_min=audit['rm_min'], rm_max=audit['rm_max'],
                    likelihood='P(lambda_RM|lambda_spec,z): lower-truncated lognormal including Jacobian',
                    location='ln arithmetic mean = A+B ln(spec/40)+C ln((1+z)/1.24); log-location subtracts sigma^2/2',
                    prior='Uniform in A,B,C and ln_sigma_ln_RM within recorded bounds', version=2)
    path = directory / "run.json"
    if path.exists():
        if not resume or json.loads(path.read_text()) != meta:
            raise ValueError("Existing run differs or --resume missing; use a new output directory")
    elif any(directory.glob("chain_*.h5")):
        raise ValueError("Chains exist without provenance")
    else:
        path.write_text(json.dumps(meta, indent=2))
    for i in range(ensembles):
        backend = emcee.backends.HDFBackend(str(directory / f"chain_{i}.h5"))
        fresh = not backend.initialized
        if fresh:
            backend.reset(walkers, ndim)
        sampler = emcee.EnsembleSampler(walkers, ndim, reg, backend=backend)
        state = None
        if fresh or backend.iteration == 0:
            rng = np.random.default_rng(seed + i)
            state = reg.initial(rng, walkers)
            sampler.random_state = np.random.RandomState(seed + i).get_state()
        remaining = steps - backend.iteration
        if remaining > 0:
            sampler.run_mcmc(state, remaining, progress=True)
        print(f"{model} ensemble {i}: {backend.iteration} saved steps", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "catalogs/conditional_richness/overnight_sample.npz")
    p.add_argument("--output", type=Path)
    p.add_argument("--cells", action="store_true", help="Fit 3x3 fixed-global reverse-residual cells")
    p.add_argument('--environment-assignments',type=Path,help='Run 36 free-location environment mixtures from audited cluster_assignments.csv')
    p.add_argument('--test-one',action='store_true',help='Benchmark first environment cell only')
    pooling=p.add_mutually_exclusive_group()
    pooling.add_argument('--highest-richness-pooled',action='store_true',help='One highest-richness fit pooling all environments and 0.10 <= z < 0.35')
    pooling.add_argument('--pooled-redshift',action='store_true',help='12 fits, recomputing environment quartiles in each pooled richness bin')
    pooling.add_argument('--richness-redshift-grid',action='store_true',help='Nine fits: global spec tertiles x three z bins, no environment splits')
    pooling.add_argument('--pooled-spec-quartiles',action='store_true',help='Four equal-count spec bins; all redshifts and environments')
    p.add_argument('--mixture-prior',choices=['legacy','beta22_uniform_sigma'],default='legacy')
    p.add_argument('--fixed-mean-cache',type=Path,help='Previous cell fits with identical global posterior and source sample')
    p.add_argument('--adaptive',action='store_true',help='Checkpoint and extend until stable convergence or safety limit')
    p.add_argument('--block-steps',type=int,default=5000)
    p.add_argument('--max-steps',type=int,default=100000)
    p.add_argument('--max-seconds',type=float,default=3600,help='Soft wall-time limit per fit, checked between blocks')
    p.add_argument('--extend-budget',action='store_true',help='Explicitly increase limits on a stopped adaptive run, preserving target and recording metadata revisions')
    p.add_argument("--global-fits", type=Path, default=ROOT/"catalogs/global_spec_richness")
    p.add_argument("--workers", type=int, default=1, help="Independent cell processes, not MPI")
    p.add_argument("--only-unconverged",action="store_true",help="With --cells --resume, extend only flagged fits")
    p.add_argument("--model", choices=["noz", "z", "both"], default="both")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--burn", type=int, default=1000)
    p.add_argument("--walkers", type=int, default=32)
    p.add_argument("--ensembles", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument('--direction', choices=['reverse','forward'], default='reverse')
    p.add_argument('--fixed-global-mean',type=Path,help='Converged forward noz directory: constrain per-object mixture means')
    p.add_argument('--proposal',choices=['stretch','de_snooker'],default='stretch')
    a = p.parse_args()
    if a.extend_budget and not (a.adaptive and a.resume and a.environment_assignments):
        p.error('--extend-budget requires --adaptive --resume --environment-assignments')
    if (a.adaptive or a.highest_richness_pooled or a.pooled_redshift or a.richness_redshift_grid or a.pooled_spec_quartiles or a.proposal!='stretch') and not a.environment_assignments:
        p.error('Environment options require --environment-assignments')
    if a.richness_redshift_grid and not a.fixed_global_mean:
        p.error('--richness-redshift-grid requires --fixed-global-mean')
    if (a.pooled_spec_quartiles or a.fixed_mean_cache or a.mixture_prior!='legacy') and not (a.fixed_global_mean and a.environment_assignments):
        p.error('New prior, cache and quartiles require --fixed-global-mean and --environment-assignments')
    if a.environment_assignments:
        fit_environment(a)
        return
    if a.cells:
        if a.direction != 'reverse':
            p.error('Forward environmental cell fits are not implemented by --cells')
        fit_cells(a)
        return
    a.output = a.output or ROOT/('catalogs/global_rm_richness' if a.direction=='forward' else 'catalogs/global_spec_richness')
    for model in (["noz", "z"] if a.model == "both" else [a.model]):
        fit(a.data, a.output, model, a.steps, a.burn, a.walkers, a.ensembles, a.seed, a.resume, a.direction)


if __name__ == "__main__":
    main()
