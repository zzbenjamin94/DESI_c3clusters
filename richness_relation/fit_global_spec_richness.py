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
from global_spec_richness_model import EnvironmentMixture, FixedMeanMixture, FixedMeanLognormal, FractionalMeanMixture, classical_rhat

ROOT = Path(__file__).resolve().parents[1]


def truncated_reverse_ensemble(task):
    from global_spec_richness_model import TruncatedReverseRegression, CurvedReverseRegression, BrokenForwardRegression
    folder,data,meta,steps,index=task
    cls=CurvedReverseRegression if meta['model'] in ('quadratic','broken') else TruncatedReverseRegression
    reg=BrokenForwardRegression(data) if meta.get('mean_form')=='broken_forward' else cls(data,meta['model'],meta['spec_min'])
    backend=emcee.backends.HDFBackend(str(Path(folder)/f'chain_{index}.h5'))
    fresh=not backend.initialized
    if fresh: backend.reset(meta['walkers'],len(reg.mle))
    sampler=emcee.EnsembleSampler(meta['walkers'],len(reg.mle),reg,backend=backend,
                                  moves=[(emcee.moves.DEMove(),.8),(emcee.moves.DESnookerMove(),.2)])
    state=None
    if fresh:
        state=reg.initial(np.random.default_rng(meta['seed']+index),meta['walkers'])
        sampler.random_state=np.random.RandomState(meta['seed']+index).get_state()
    remaining=steps-backend.iteration
    if remaining>0: sampler.run_mcmc(state,remaining,progress=False)
    print(meta['model'], 'ensemble',index,'saved steps',backend.iteration,flush=True)


def fit_broken_forward(args):
    from global_spec_richness_model import BrokenForwardRegression
    if args.steps<=args.burn or args.ensembles<2 or args.walkers<12 or args.output is None:
        raise ValueError('Require output, steps > burn, >=2 ensembles and >=12 walkers')
    data,audit=load_sample(args.data);reg=BrokenForwardRegression(data)
    folder=args.output
    if folder.exists() and not args.resume: raise FileExistsError(folder)
    folder.mkdir(parents=True,exist_ok=True)
    meta=dict(model='z',mean_form='broken_forward',direction='forward',scatter_model='poisson',rm_min=20.,rm_max=None,
        data_path=str(args.data.resolve()),sample_sha256=audit['sample_sha256'],n=reg.n,parameters=reg.names,bounds=reg.bounds,
        walkers=args.walkers,ensembles=args.ensembles,burn=args.burn,seed=args.seed,
        likelihood='Selected lognormal RM|spec,z; mean-log continuous broken relation; s^2=sigma0^2+exp(-ell)-exp(-2ell); ell>0',
        prior='Uniform in recorded coordinates; spec break log-uniform 20 to 300; A is mean-log RM at spec=40,z=.24',
        model_sha256=hashlib.sha256(Path(__file__).with_name('global_spec_richness_model.py').read_bytes()).hexdigest())
    if (folder/'run.json').exists() and json.loads((folder/'run.json').read_text())!=meta: raise ValueError('Resume mismatch')
    write_atomic_json(folder/'run.json',meta)
    write_atomic_json(folder/'bestfit.json',dict(mle=reg.mle.tolist(),log_likelihood=reg.log_likelihood(reg.mle)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(truncated_reverse_ensemble,[(folder,data,meta,args.steps,j) for j in range(args.ensembles)]))


def fit_truncated_reverse(args):
    from global_spec_richness_model import TruncatedReverseRegression, CurvedReverseRegression
    from prepare_conditional_richness import file_sha256
    if args.steps<=args.burn or args.ensembles<2 or args.walkers<8:
        raise ValueError('Need steps > burn, >=2 ensembles and >=8 walkers')
    data,audit=load_sample(args.data)
    keep=data['lambda_spec']>=args.spec_min
    selected={k:v[keep] for k,v in data.items()}
    root=args.output or ROOT/'catalogs/reverse_spec20_20261109'
    if root.exists() and not args.resume: raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    path=root/'sample.npz'
    if not path.exists():
        np.savez_compressed(path,**selected)
        updated=dict(audit,source_sample=str(args.data.resolve()),source_sha256=audit['sample_sha256'],
                     sample_sha256=file_sha256(path),spec_min=args.spec_min,
                     N_before=len(keep),N_after=int(keep.sum()),removed=int((~keep).sum()))
        write_atomic_json(path.with_suffix('.json'),updated)
    selected,selected_audit=load_sample(path)
    if selected_audit['source_sha256']!=audit['sample_sha256'] or selected_audit['spec_min']!=args.spec_min:
        raise ValueError('Resume sample mismatch')
    tasks=[]
    for model in (('quadratic','broken') if args.reverse_curved else ('noz','z')):
        folder=root/model; folder.mkdir(exist_ok=True)
        cls=CurvedReverseRegression if args.reverse_curved else TruncatedReverseRegression
        reg=cls(selected,model,args.spec_min)
        meta=dict(model=model,direction='reverse',spec_min=args.spec_min,rm_pivot=40.,z_pivot=.24,
                  data_path=str(path.resolve()),sample_sha256=selected_audit['sample_sha256'],n=reg.n,
                  walkers=args.walkers,ensembles=args.ensembles,burn=args.burn,seed=args.seed,
                  parameters=reg.names if args.reverse_curved else parameter_names(model),bounds=reg.bounds,
                  likelihood='lognormal spec | RM,z, spec>=threshold; exact per-object survival and Jacobian',
                  prior='Independent uniform priors in recorded coordinates and bounds; broken transition log-uniform in RM from 25 to 100' if args.reverse_curved else 'Uniform in mean-log intercept A, slope B, C, and ln_sigma within bounds',
                  model_sha256=hashlib.sha256(Path(__file__).with_name('global_spec_richness_model.py').read_bytes()).hexdigest())
        if (folder/'run.json').exists() and json.loads((folder/'run.json').read_text())!=meta:
            raise ValueError('Resume metadata mismatch')
        write_atomic_json(folder/'run.json',meta)
        write_atomic_json(folder/'bestfit.json',dict(mle=reg.mle.tolist(),log_likelihood=reg.log_likelihood(reg.mle)))
        tasks.extend((folder,selected,meta,args.steps,i) for i in range(args.ensembles))
    print('Selected',len(selected['ID']),'of',len(keep),'clusters',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(truncated_reverse_ensemble,tasks))


def fit_negative_cell(task):
    from scipy.optimize import minimize
    from global_spec_richness_model import NegativeTailMixture
    directory, values, meta = task
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    write_atomic_json(directory/'run.json',meta)
    reg=NegativeTailMixture(values)
    rng=np.random.default_rng(meta['seed'])
    optima=[]
    for _ in range(12):
        start=[np.median(values)*rng.uniform(.8,1.4),
               np.log(np.std(values)*rng.uniform(.3,.9)),rng.uniform(.1,.8),np.log(rng.uniform(10,150))]
        result=minimize(lambda t:-reg(t),start,method='Nelder-Mead',
                        options={'maxiter':1800,'xatol':1e-5})
        if np.isfinite(result.fun): optima.append(result)
    optima.sort(key=lambda r:r.fun)
    if not optima: raise ValueError('No finite optimum')
    mle=minimize(lambda t:-float(reg.log_likelihood(t)[0]),optima[0].x,
                 method='Nelder-Mead',bounds=reg.bounds,options={'maxiter':4000})
    write_atomic_json(directory/'bestfit.json',dict(map=optima[0].x.tolist(),mle=mle.x.tolist(),
                      log_likelihood=float(-mle.fun),optimizer_success=bool(mle.success)))
    samplers=[]; states=[]
    for i in range(meta['ensembles']):
        backend=emcee.backends.HDFBackend(str(directory/f'chain_{i}.h5'))
        backend.reset(meta['walkers'],4)
        sampler=emcee.EnsembleSampler(meta['walkers'],4,reg,vectorize=True,backend=backend,
                     moves=[(emcee.moves.DEMove(),.8),(emcee.moves.DESnookerMove(),.2)])
        center=optima[i%len(optima)].x
        positions=[]
        while len(positions)<meta['walkers']:
            proposal=center+rng.normal(size=4)*[2.,.06,.02,.1]
            if np.isfinite(reg(proposal)): positions.append(proposal)
        sampler.random_state=np.random.RandomState(meta['seed']+i).get_state()
        samplers.append(sampler); states.append(np.array(positions))
    started=time.monotonic(); history=[]
    for target in range(2000,meta['max_steps']+1,2000):
        for i,sampler in enumerate(samplers):
            states[i]=sampler.run_mcmc(states[i],2000,progress=False)
        chains=[s.get_chain(discard=meta['burn']) for s in samplers]
        tau=np.array([emcee.autocorr.integrated_time(c,tol=0) for c in chains])
        ratio=(target-meta['burn'])/tau
        rhat=np.max([classical_rhat(np.array([c[:,w] for c in chains]))
                      for w in range(meta['walkers'])],axis=0)
        stable=bool(history and np.max(np.abs(tau/np.asarray(history[-1]['tau'])-1))<.1)
        passed=bool(np.all(np.isfinite(tau)) and np.all(tau>0) and np.all(ratio>=50)
                    and np.all(np.isfinite(rhat)) and np.all(rhat<1.1) and stable)
        report=dict(steps=target,tau=tau.tolist(),steps_per_tau=ratio.tolist(),rhat=rhat.tolist(),
                    stable=stable,converged=passed,elapsed_seconds=time.monotonic()-started)
        history.append(report)
        write_atomic_json(directory/'diagnostics.json',report)
        write_atomic_json(directory/'convergence_history.json',history)
        print(directory.name, 'steps',target,'min N/tau',ratio.min(),'max Rhat',rhat.max(),flush=True)
        if passed or time.monotonic()-started>=meta['max_seconds']: break
    samples=np.concatenate([c.reshape(-1,4) for c in chains])
    np.savez_compressed(directory/'posterior.npz',samples=samples)
    report['stop_reason']='converged' if passed else ('step_limit' if target==meta['max_steps'] else 'wall_limit')
    write_atomic_json(directory/'diagnostics.json',report)
    return report


def fit_negative_quartiles(args):
    data,audit=load_sample(args.data)
    edges=np.quantile(data['lambda_rm'],np.linspace(0,1,5))
    root=args.output or ROOT/'catalogs/reverse_negative_mixture_20261105'
    root.mkdir(parents=True,exist_ok=False)
    tasks=[]
    for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        keep=(data['lambda_rm']>=lo)&((data['lambda_rm']<=hi) if j==3 else (data['lambda_rm']<hi))
        meta=dict(cell=j+1,rm_lo=float(lo),rm_hi=float(hi),N=int(keep.sum()),
                  sample_sha256=audit['sample_sha256'],data_path=str(args.data.resolve()),
                  seed=args.seed+100*j,walkers=args.walkers,ensembles=args.ensembles,burn=args.burn,
                  max_steps=args.max_steps,max_seconds=args.max_seconds,
                  prior='mu uniform(0,1000); sigma uniform(0.1,500); f Beta(2,2); tail scale loguniform(0.1,500)',
                  likelihood='Gaussian plus negative EMG in linear spec richness, normalized over spec>0',
                  parameters=['mu_core','ln_sigma','f_low','ln_tail_scale'],
                  model_sha256=hashlib.sha256(Path(__file__).with_name('global_spec_richness_model.py').read_bytes()).hexdigest())
        tasks.append((root/f'cell_{j+1}',data['lambda_spec'][keep],meta))
    write_atomic_json(root/'manifest.json',dict(rm_edges=edges.tolist(),cells=4))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        reports=list(pool.map(fit_negative_cell,tasks))
    print('Convergence:',[r['converged'] for r in reports],flush=True)


def write_atomic_json(path, value):
    path=Path(path)
    temporary=path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def environment_model(rm, meta):
    if meta.get('model') == 'lognormal_fixed_global_mean':
        return FixedMeanLognormal(rm, meta['fixed_global_means'])
    if meta.get('mixture_prior')=='fractional_beta22':
        return FractionalMeanMixture(rm,meta['fixed_global_means'])
    if 'fixed_global_means' in meta:
        return FixedMeanMixture(rm, meta['fixed_global_means'], meta.get('mixture_prior','legacy'))
    return EnvironmentMixture(rm)


def ensemble_moves(meta):
    if meta.get('proposal') == 'de_snooker':
        return [(emcee.moves.DEMove(), .8), (emcee.moves.DESnookerMove(), .2)]
    return emcee.moves.StretchMove()


def richness_redshift_cells(frame, pooled_quartiles=False, split_redshift=False, spec_bins=4, bin_reference='global'):
    """Common global richness tertiles, fixed z bins, no environment split."""
    if spec_bins < 2:
        raise ValueError('Require at least two richness bins')
    reference = frame[(frame.z >= .10) & (frame.z < .18)] if bin_reference == 'lowz' else frame
    edges=np.quantile(reference.lambda_spec,np.linspace(0,1,spec_bins+1) if pooled_quartiles else [0,1/3,2/3,1])
    edges[[0,-1]] = [frame.lambda_spec.min(), frame.lambda_spec.max()]
    if np.any(np.diff(edges)<=0):
        raise ValueError('Tied richness tertiles require explicit bin definitions')
    zedges=np.array([.10,.35] if pooled_quartiles and not split_redshift else [.10,.18,.24,.35])
    if not np.all((frame.z>=zedges[0])&(frame.z<zedges[-1])):
        raise ValueError('Unexpected redshift range')
    frame=frame.copy()
    frame['spec_bin']=np.searchsorted(edges[1:-1],frame.lambda_spec,side='right')+1
    frame['z_bin']=np.searchsorted(zedges[1:-1],frame.z,side='right')+1
    frame['cell']=['z%d_s%d'%(z,s) for z,s in zip(frame.z_bin,frame.spec_bin)]
    frame['quartile']=0
    frame['parent_mean_rm']=frame.groupby('cell').lambda_rm.transform('mean')
    if frame.cell.nunique()!=(len(edges)-1)*(len(zedges)-1) or frame.ID.duplicated().any():
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
        if meta.get('model') == 'lognormal_fixed_global_mean':
            best={'optima':optima.tolist(), 'mle':optima[0].tolist(),
                  'log_likelihood':float(reg.log_likelihood(optima[0])[0])}
        elif meta.get('mixture_prior') in ('beta22_uniform_sigma','fractional_beta22'):
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
        target=int(max(initial_steps,max(lengths)))
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
    if args.global_mixture:
        frame['cell']='global'
        frame['quartile']=0
        frame['parent_mean_rm']=frame.lambda_rm.mean()
    if args.richness_redshift_grid or args.pooled_spec_quartiles:
        frame,spec_edges,z_edges=richness_redshift_cells(frame,args.pooled_spec_quartiles,args.split_redshift,args.spec_bins,args.bin_reference)
    if args.spec_bin is not None:
        frame=frame[frame.spec_bin==args.spec_bin].copy()
        if frame.empty:
            raise ValueError('Requested richness bin is empty')
    if args.environment_quartiles:
        if args.bgs_environment:
            from astropy.table import Table
            env=Table.read(args.bgs_environment)
            lookup={str(row['ID']):row for row in env}
            if len(lookup)!=len(env): raise ValueError('Duplicate environment IDs')
            rows=[lookup[i] for i in frame.ID]
            if not all(bool(r['bgs_box_E_local_valid']) for r in rows):
                raise ValueError('Invalid BGS environment in selected sample')
            if not np.allclose(frame.z,[r['Z_SPEC_central'] for r in rows]):
                raise ValueError('BGS environment redshift mismatch')
            frame['environment']=[float(r['bgs_box_E_local']) for r in rows]
            if not np.isfinite(frame.environment).all(): raise ValueError('Nonfinite environment')
        pieces=[]
        environment_edges={}
        for _,g in frame.groupby('cell'):
            g=g.copy()
            edges=np.quantile(g.environment,[0,.25,.5,.75,1])
            if np.any(np.diff(edges)<=0): raise ValueError('Tied environment boundaries')
            g['quartile']=np.searchsorted(edges[1:-1],g.environment,side='right')+1
            environment_edges[g.cell.iloc[0]]=edges.tolist()
            pieces.append(g)
        frame=pd.concat(pieces)
        if args.extreme_environment_only:
            frame=frame[frame.quartile.isin([1,4])].copy()
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
                meta['pooling']=f'All redshifts and environments; {args.spec_bins} equal-count spec bins'
            if args.split_redshift:
                meta['pooling']=f'{args.spec_bins} common spec bins x three redshift bins; all environments'
            meta['bin_reference']=args.bin_reference
            if args.environment_quartiles:
                parent=frame[frame.cell==cell]
                meta.update(pooling='Pooled redshift; environment quartiles within each spec quartile',
                            environment_column='bgs_box_E_local' if args.bgs_environment else 'dr9_gaussian_E_local',
                            environment_source=str(args.bgs_environment) if args.bgs_environment else str(source),
                            environment_edges=environment_edges[cell],
                            environment_median=float(g.environment.median()))
        if args.fixed_global_mean:
            global_dir=args.fixed_global_mean
            global_meta=json.loads((global_dir/'run.json').read_text())
            diagnostic=json.loads((global_dir/'diagnostics.json').read_text())
            poisson_global=global_meta.get('scatter_model')=='poisson'
            if global_meta.get('direction')!='forward' or not diagnostic['converged'] or (not poisson_global and global_meta['model']!='noz'):
                raise ValueError('Require a converged forward global fit; evolving fits require the Poisson scatter model')
            data,audit=load_sample(Path(global_meta['data_path']))
            if audit['sample_sha256']!=global_meta['sample_sha256']:
                raise ValueError('Global sample checksum mismatch')
            if args.global_mixture and set(g.ID)!=set(str(i) for i in data['ID']):
                raise ValueError('Global mixture must contain exactly the global-anchor sample')
            lookup={str(i):s for i,s in zip(data['ID'],data['lambda_spec'])}
            if not np.allclose([lookup[i] for i in g.ID],g.lambda_spec,rtol=1e-10):
                raise ValueError('Global and cell richnesses differ')
            posterior_path=global_dir/'posterior.npz'
            with np.load(posterior_path) as archive:
                posterior=archive['samples']
            # Internal A is log arithmetic mean, not mean log richness.
            global_means=[]
            if poisson_global:
                if mean_cache:
                    raise ValueError('Do not reuse legacy mean caches for Poisson global fits')
                zlookup={str(i):z for i,z in zip(data['ID'],data['z'])}
                if not np.allclose([zlookup[i] for i in g.ID],g.z,rtol=1e-10):
                    raise ValueError('Global and cell redshifts differ')
                theta=np.median(posterior,axis=0)
                ell=theta[0]+theta[1]*np.log(g.lambda_spec.to_numpy()/global_meta['spec_pivot'])
                if global_meta['model']=='z':
                    ell+=theta[2]*np.log((1+g.z.to_numpy())/(1+global_meta['z_pivot']))
                if np.any(ell<=0):
                    raise ValueError('Poisson approximation requires positive mean log richness')
                variance=np.exp(2*theta[-1])+np.exp(-ell)-np.exp(-2*ell)
                global_means=np.exp(ell+variance/2).tolist()
            if mean_cache:
                global_means=[mean_cache[key] for key in g.ID]
            for values in ([] if mean_cache or poisson_global else np.array_split(g.lambda_spec.to_numpy(), max(1,int(np.ceil(len(g)/32))))):
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
            if poisson_global:
                meta.update(global_parameter_medians=theta.tolist(),
                            global_mean_estimator='exp(ell_i+s_i^2/2) evaluated at posterior-median global parameters; pre-selection arithmetic mean; global uncertainty not propagated')
            meta=json.loads(json.dumps(meta))
            if args.mixture_prior!='legacy':
                reg=(FractionalMeanMixture(g.lambda_rm.to_numpy(),global_means) if args.mixture_prior=='fractional_beta22'
                     else FixedMeanMixture(g.lambda_rm.to_numpy(),global_means,args.mixture_prior))
                meta.update(mixture_prior=args.mixture_prior,bounds=reg.bounds,
                            prior='Beta(2,2) f; uniform sigma in (0,30), numerical floor 1e-12; log-uniform 1/tau in (0.1,500); log-sigma Jacobian included')
                meta=json.loads(json.dumps(meta))
                if args.mixture_prior=='fractional_beta22':
                    meta.update(model='fractional_positive_tail_mixture',parameters=reg.names,
                                prior='Beta(2,2) f; uniform sigma_delta in (0,1), floor 1e-12; log-uniform inverse tau_delta in (0.01,10); log-sigma Jacobian included',
                                likelihood='Fractional mixture normalized above delta_min_i=20/global_mean_i-1; RM-density Jacobian 1/global_mean_i included',
                                centering='Fractional core=-f_proj/tau_delta; unselected fractional mean zero',
                                residual_definition='delta_i=RM_i/global_mean_i-1')
        if args.cell_family == 'lognormal':
            if 'fixed_global_means' not in meta: raise ValueError('Lognormal cells require a fixed global mean')
            meta.pop('mixture_prior', None)
            meta.update(model='lognormal_fixed_global_mean', parameters=FixedMeanLognormal.names,
                        bounds=FixedMeanLognormal.bounds,
                        prior='Uniform ln(sigma_ln_lambda) in (-6,1)',
                        likelihood='Theta(RM-20) * lognormal / per-object survival at 20',
                        centering='ell_i=ln(fixed_global_arithmetic_mean_i)-sigma_ln_lambda^2/2',
                        scatter_model='One constant total natural-log scatter per cell; no separate Poisson component')
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
        if args.global_mixture:
            meta.update(pooling='All clusters; no richness, redshift or environment splits',
                        spec_range=[float(g.lambda_spec.min()),float(g.lambda_spec.max())],
                        z_range=[float(g.z.min()),float(g.z.max())],ensemble_workers=args.workers,
                        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if args.spec_bin is not None and frame.groupby(['cell','quartile']).ngroups==1:
            meta['ensemble_workers']=args.workers
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
    if args.workers==1 or args.highest_richness_pooled or len(tasks)==1:
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


def fit(data_path, output, model, steps, burn, walkers, ensembles, seed, resume, direction='reverse', scatter_model='constant', workers=1):
    data, audit = load_sample(data_path)
    reg = (ForwardRegression(data, model, audit['rm_min'], audit['rm_max'],scatter_model)
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
        if scatter_model == 'poisson':
            meta.update(scatter_model='poisson',version=3,
                        location='ell=A+B ln(spec/40)+C ln((1+z)/1.24)',
                        variance='s_i^2=sigma0^2+exp(-ell_i)-exp(-2ell_i); ell_i>0 for all fitted objects',
                        prior='Uniform in A,B,C,ln_sigma0 within recorded bounds, conditioned on ell_i>0',
                        model_sha256=hashlib.sha256(Path(__file__).with_name('global_spec_richness_model.py').read_bytes()).hexdigest())
    path = directory / "run.json"
    if path.exists():
        if not resume or json.loads(path.read_text()) != meta:
            raise ValueError("Existing run differs or --resume missing; use a new output directory")
    elif any(directory.glob("chain_*.h5")):
        raise ValueError("Chains exist without provenance")
    else:
        path.write_text(json.dumps(meta, indent=2))
    tasks = [(directory, reg, meta, steps, i) for i in range(ensembles)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(global_environment_ensemble, tasks))
    else:
        for task in tasks:
            global_environment_ensemble(task)


def global_environment_ensemble(task):
    directory, reg, meta, target, index = task
    backend = emcee.backends.HDFBackend(str(directory/f'chain_{index}.h5'))
    fresh = not backend.initialized
    if fresh: backend.reset(meta['walkers'],len(reg.names))
    sampler = emcee.EnsembleSampler(meta['walkers'],len(reg.names),reg,backend=backend)
    state = None
    if fresh or backend.iteration == 0:
        state = reg.initial(np.random.default_rng(meta['seed']+index),meta['walkers'])
        sampler.random_state = np.random.RandomState(meta['seed']+index).get_state()
    if target > backend.iteration:
        sampler.run_mcmc(state,target-backend.iteration,progress=False)
    print(directory.name,'ensemble',index,'steps',backend.iteration,flush=True)


def fit_global_environment(args):
    """Paired global forward fits on an identical DR9 Gaussian-valid sample."""
    from astropy.table import Table
    from prepare_conditional_richness import file_sha256
    data,audit=load_sample(args.data)
    env=Table.read(args.global_environment)
    ids=np.asarray(env['ID']).astype(str)
    if len(np.unique(ids))!=len(ids): raise ValueError('Duplicate environment IDs')
    lookup={v:i for i,v in enumerate(ids)}
    rows=np.array([lookup[str(v)] for v in data['ID']])
    if not np.allclose(np.asarray(env['Z_SPEC_central'])[rows],data['z'],rtol=1e-7,atol=1e-8):
        raise ValueError('Environment redshifts do not match sample')
    values=np.ma.asarray(env['dr9_gaussian_E_local'],dtype=float).filled(np.nan)[rows]
    valid=np.ma.asarray(env['dr9_gaussian_E_local_valid'],dtype=bool).filled(False)[rows] & np.isfinite(values)
    selected={k:v[valid] for k,v in data.items()}; selected['environment']=values[valid]
    root=args.output
    if root is None: raise ValueError('Specify --output for environment comparison')
    if args.steps<=args.burn or args.ensembles<2 or args.walkers<10: raise ValueError('Invalid sampler settings')
    if root.exists() and not args.resume: raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    path=root/'sample.npz'
    source_hash=file_sha256(args.global_environment)
    if not path.exists():
        np.savez_compressed(path,**selected)
        updated=dict(audit,sample_sha256=file_sha256(path),source_sha256=audit['sample_sha256'],
            environment_sha256=source_hash,environment_column='dr9_gaussian_E_local',
            environment_path=str(args.global_environment.resolve()),N_before=len(valid),N_after=int(valid.sum()))
        write_atomic_json(path.with_suffix('.json'),updated)
    selected,updated=load_sample(path)
    if updated['source_sha256']!=audit['sample_sha256'] or updated['environment_sha256']!=source_hash:
        raise ValueError('Resume input changed')
    tasks=[]
    for name,use_environment in [('baseline',False),('environment',True)]:
        reg=ForwardRegression(selected,'z',audit['rm_min'],audit['rm_max'],'poisson',environment=use_environment)
        folder=root/name;folder.mkdir(exist_ok=True)
        meta=dict(model='z',direction='forward',scatter_model='poisson',environment=use_environment,
            environment_pivot=float(np.median(selected['environment'])),data_path=str(path.resolve()),
            sample_sha256=updated['sample_sha256'],environment_sha256=source_hash,n=reg.n,
            parameters=reg.names,bounds=reg.bounds,rm_min=audit['rm_min'],rm_max=audit['rm_max'],
            walkers=args.walkers,ensembles=args.ensembles,burn=args.burn,seed=args.seed,
            prior='Uniform in A,B,C,D (when included),ln_sigma0 within bounds; ell>0 for all observations',
            location='ell=A+B ln(spec/40)+C ln((1+z)/1.24)+D(E-E_piv); D=0 for baseline',
            variance='sigma0^2+exp(-ell)-exp(-2ell)',
            likelihood='RM>20 truncated lognormal with per-object survival normalization and Jacobian')
        if (folder/'run.json').exists() and json.loads((folder/'run.json').read_text())!=meta:
            raise ValueError('Run metadata differs')
        write_atomic_json(folder/'run.json',meta)
        write_atomic_json(folder/'bestfit.json',dict(mle=reg.mle.tolist(),log_likelihood=reg.log_likelihood(reg.mle)))
        tasks.extend((folder,reg,meta,args.steps,i) for i in range(args.ensembles))
    print('Environment comparison:',len(selected['ID']),'clusters;',args.steps,'steps per ensemble',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(global_environment_ensemble,tasks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "catalogs/conditional_richness/overnight_sample.npz")
    p.add_argument('--global-environment',type=Path,help='DR9 environment FITS: fit paired evolving global models with/without Gaussian environment')
    p.add_argument("--output", type=Path)
    p.add_argument("--cells", action="store_true", help="Fit 3x3 fixed-global reverse-residual cells")
    p.add_argument('--reverse-negative-cells',action='store_true',help='Four RM-quartile negative-tail mixture fits')
    p.add_argument('--spec-min',type=float,help='Fit both reverse global models with this explicit spec selection')
    p.add_argument('--reverse-curved',action='store_true',help='Fit quadratic and continuous broken reverse mean relations with --spec-min')
    p.add_argument('--forward-broken',action='store_true',help='Evolving broken forward lognormal with Poisson-motivated scatter')
    p.add_argument('--environment-assignments',type=Path,help='Run 36 free-location environment mixtures from audited cluster_assignments.csv')
    p.add_argument('--test-one',action='store_true',help='Benchmark first environment cell only')
    pooling=p.add_mutually_exclusive_group()
    pooling.add_argument('--highest-richness-pooled',action='store_true',help='One highest-richness fit pooling all environments and 0.10 <= z < 0.35')
    pooling.add_argument('--pooled-redshift',action='store_true',help='12 fits, recomputing environment quartiles in each pooled richness bin')
    pooling.add_argument('--richness-redshift-grid',action='store_true',help='Nine fits: global spec tertiles x three z bins, no environment splits')
    pooling.add_argument('--pooled-spec-quartiles',action='store_true',help='Four equal-count spec bins; all redshifts and environments')
    p.add_argument('--spec-bins',type=int,default=4,help='Number of equal-count bins with --pooled-spec-quartiles')
    p.add_argument('--bin-reference',choices=['global','lowz'],default='global',help='Sample defining common richness quantiles')
    p.add_argument('--cell-family',choices=['mixture','lognormal'],default='mixture')
    pooling.add_argument('--global-mixture',action='store_true',help='One shared mixture across the complete global-anchor sample')
    p.add_argument('--spec-bin',type=int,help='Run only this globally defined richness bin')
    p.add_argument('--bgs-environment',type=Path,help='Use BGS cylindrical environment joined by cluster ID')
    p.add_argument('--extreme-environment-only',action='store_true',help='Fit only bottom and top within-richness quartiles')
    p.add_argument('--environment-quartiles',action='store_true',help='Split pooled spec quartiles into sixteen environment cells')
    p.add_argument('--split-redshift',action='store_true',help='Split spec quartiles into twelve richness-redshift cells')
    p.add_argument('--mixture-prior',choices=['legacy','beta22_uniform_sigma','fractional_beta22'],default='legacy')
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
    p.add_argument('--scatter-model',choices=['constant','poisson'],default='constant')
    p.add_argument('--fixed-global-mean',type=Path,help='Converged forward global fit directory: constrain pre-selection per-object mixture means')
    p.add_argument('--proposal',choices=['stretch','de_snooker'],default='stretch')
    a = p.parse_args()
    if a.forward_broken:
        fit_broken_forward(a)
        return
    if (a.bgs_environment or a.extreme_environment_only) and not a.environment_quartiles:
        p.error('BGS/extreme environment options require --environment-quartiles')
    if a.spec_bins < 2 or (a.spec_bins != 4 and not a.pooled_spec_quartiles):
        p.error('Custom --spec-bins requires pooled richness bins')
    if a.global_environment:
        fit_global_environment(a)
        return
    if a.spec_min is not None:
        fit_truncated_reverse(a)
        return
    if a.reverse_negative_cells:
        fit_negative_quartiles(a)
        return
    if a.scatter_model == 'poisson' and (a.direction != 'forward' or a.cells or a.environment_assignments):
        p.error('Poisson scatter is available only for global forward fits')
    if a.environment_quartiles and not a.pooled_spec_quartiles:
        p.error('--environment-quartiles requires --pooled-spec-quartiles')
    if a.global_mixture and not (a.environment_assignments and a.fixed_global_mean):
        p.error('--global-mixture requires assignments and a fixed global mean fit')
    if a.spec_bin is not None and not (a.pooled_spec_quartiles or a.richness_redshift_grid):
        p.error('--spec-bin requires explicit richness-cell binning')
    if a.split_redshift and (not a.pooled_spec_quartiles or a.environment_quartiles):
        p.error('--split-redshift requires --pooled-spec-quartiles without --environment-quartiles')
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
        fit(a.data, a.output, model, a.steps, a.burn, a.walkers, a.ensembles, a.seed, a.resume, a.direction,a.scatter_model,a.workers)


if __name__ == "__main__":
    main()
