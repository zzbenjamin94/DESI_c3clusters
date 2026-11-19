"""Summarize global or residual chains without modifying saved chains.

Global maximum-likelihood estimates are recomputed for information criteria.
"""
import argparse
import json
from pathlib import Path

import emcee
import numpy as np
import pandas as pd

from global_spec_richness_model import (Regression, ForwardRegression, classical_rhat, CellResidual,
                                       residual_draw)
from scipy.stats import skew, kurtosis, norm, exponnorm
from scipy.optimize import minimize
from prepare_conditional_richness import load_sample


def forward_histogram_discrepancy(data, reg, theta, cells, edges):
    """Pearson discrepancy, integrated selected probabilities and adjacent merging."""
    from scipy.special import log_ndtr
    ell,s=reg.location_scale(theta)
    log_survival=log_ndtr((ell-np.log(20))/s)
    cdf=-np.expm1(log_ndtr((ell[None,:]-np.log(edges[:,None]))/s[None,:])-log_survival[None,:])
    cdf=np.clip(cdf,0,1);cdf[0]=0;cdf[-1]=1
    result=[];records=[]
    for cell in range(6):
        mask=cells==cell
        expected=np.diff(cdf[:,mask],axis=0).sum(axis=1)
        counts=np.histogram(data['lambda_rm'][mask],edges)[0]
        if not np.isclose(expected.sum(),mask.sum()): raise ValueError('Probability normalization failed')
        groups=[];start=0;total=0
        for j,e in enumerate(expected):
            total+=e
            if total>=5:
                groups.append([start,j+1]);start=j+1;total=0
        if start<len(expected):
            if groups:groups[-1][1]=len(expected)
            else:groups=[[0,len(expected)]]
        statistic=0.
        for lo,hi in groups:
            o=int(counts[lo:hi].sum());e=float(expected[lo:hi].sum())
            statistic+=(o-e)**2/e
            records.append(dict(cell=cell+1,left=float(edges[lo]),right=float(edges[hi]),observed=o,expected=e))
        result.append(statistic)
    return np.r_[sum(result),result],records


def forward_bootstrap_one(task):
    from scipy.stats import truncnorm
    data,meta,theta,cells,edges,seed=task
    reg=ForwardRegression(data,meta['model'],meta['rm_min'],meta['rm_max'],meta['scatter_model'])
    ell,s=reg.location_scale(theta)
    mock={k:v.copy() for k,v in data.items()}
    mock['lambda_rm']=np.exp(truncnorm.rvs((np.log(20)-ell)/s,np.inf,loc=ell,scale=s,random_state=np.random.default_rng(seed)))
    fitted=ForwardRegression(mock,meta['model'],meta['rm_min'],meta['rm_max'],meta['scatter_model'])
    stats,_=forward_histogram_discrepancy(mock,fitted,fitted.mle,cells,edges)
    return stats,fitted.mle


def forward_refitted_bootstrap(folder, output, draws, workers):
    from concurrent.futures import ProcessPoolExecutor
    folder=Path(folder);meta=json.loads((folder/'run.json').read_text())
    if meta['direction']!='forward' or meta['rm_min']!=20: raise ValueError('Requires forward RM > 20 fit')
    data,audit=load_sample(meta['data_path'])
    if audit['sample_sha256']!=meta['sample_sha256']: raise ValueError('Sample mismatch')
    output.mkdir(parents=True,exist_ok=False)
    reg=ForwardRegression(data,meta['model'],meta['rm_min'],meta['rm_max'],meta['scatter_model'])
    bounds=np.quantile(data['lambda_spec'],np.linspace(0,1,7))
    cells=np.searchsorted(bounds[1:-1],data['lambda_spec'],side='right')
    edges=np.r_[np.linspace(20,200,16),np.inf]
    observed,records=forward_histogram_discrepancy(data,reg,reg.mle,cells,edges)
    pd.DataFrame(records).to_csv(output/'observed_histograms.csv',index=False)
    tasks=[(data,meta,reg.mle,cells,edges,20261112+j) for j in range(draws)]
    simulated=[];parameters=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j,(stats,theta) in enumerate(pool.map(forward_bootstrap_one,tasks),1):
            simulated.append(stats);parameters.append(theta)
            if j%25==0 or j==draws:
                np.savez_compressed(output/'bootstrap_checkpoint.npz',statistics=simulated,parameters=parameters,observed=observed)
                print(f'Refitted bootstrap {j}/{draws}',flush=True)
    simulated=np.asarray(simulated);exceed=(simulated>=observed).sum(axis=0)
    result=pd.DataFrame(dict(cell=['combined']+[str(i) for i in range(1,7)],N=[reg.n]+[int((cells==j).sum()) for j in range(6)],chi2=observed,exceedances=exceed,replicates=draws,p_value=(exceed+1)/(draws+1)))
    result.to_csv(output/'goodness_of_fit.csv',index=False)
    (output/'settings.json').write_text(json.dumps(dict(fit=str(folder.resolve()),sample_sha256=audit['sample_sha256'],spec_edges=bounds.tolist(),histogram_edges=edges[:-1].tolist(),overflow=True,mle=reg.mle.tolist(),seed=20261112,replicates=draws,method='Parametric bootstrap at global MLE; selected per-object draws; global refit each replicate; adjacent E>=5 merging repeated each replicate; p=(exceedances+1)/(B+1)',caveat='Conditional on observed covariates; independent clusters; no spatial covariance or measurement-error deconvolution'),indent=2))
    print(result.to_string(index=False))


def reverse_log_individual_discrepancy(folder, output):
    """Selected log-space moments at saved componentwise posterior medians."""
    from scipy.special import log_ndtr
    from scipy.stats import chi2
    meta = json.loads((folder / 'run.json').read_text())
    data, audit = load_sample(meta['data_path'])
    if meta['direction'] != 'reverse' or audit['sample_sha256'] != meta['sample_sha256']:
        raise ValueError('Reverse fit/sample mismatch')
    with np.load(folder / 'posterior.npz') as archive:
        theta = np.median(archive['samples'], axis=0)
    x = np.log(data['lambda_rm'] / meta['rm_pivot'])
    ell = theta[0] + theta[1] * x
    model = meta['model']
    if model == 'z':
        ell += theta[2] * np.log((1 + data['z']) / (1 + meta['z_pivot']))
    elif model == 'quadratic':
        ell += theta[2] * x**2
    elif model == 'broken':
        ell += (theta[2] - theta[1]) * (np.maximum(x - theta[3], 0) - max(-theta[3], 0))
    elif model != 'noz':
        raise ValueError(model)
    sigma = np.exp(theta[-1])
    alpha = (np.log(meta['spec_min']) - ell) / sigma
    mills = np.exp(-alpha**2 / 2 - np.log(2 * np.pi) / 2 - log_ndtr(-alpha))
    mean = ell + sigma * mills
    variance = sigma**2 * (1 + alpha * mills - mills**2)
    if not np.all(np.isfinite(variance) & (variance > 0)):
        raise ValueError('Invalid selected log variance')
    contributions = (np.log(data['lambda_spec']) - mean)**2 / variance
    n, k = len(x), len(theta)
    statistic = float(contributions.sum())
    best = json.loads((folder / 'bestfit.json').read_text())
    report = dict(model=model, N=n, k=k, chi2=statistic, nominal_dof=n-k,
                  reduced_chi2=statistic/(n-k), nominal_p=float(chi2.sf(statistic, n-k)),
                  AIC=2*k-2*best['log_likelihood'], BIC=k*np.log(n)-2*best['log_likelihood'],
                  posterior_median=theta.tolist(), sample_sha256=audit['sample_sha256'],
                  caveat='Individual natural-log residuals; selected conditional moments at posterior medians. Nominal chi-square reference only, not calibrated goodness of fit.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'individual_log_chi2.json').write_text(json.dumps(report, indent=2))
    pd.DataFrame(dict(ID=data['ID'], selected_log_mean=mean, selected_log_variance=variance,
                      chi2_contribution=contributions)).to_csv(output / 'individual_log_residuals.csv', index=False)
    print(json.dumps(report, indent=2))


def forward_individual_discrepancy(folder,output):
    from scipy.special import log_ndtr
    meta=json.loads((folder/'run.json').read_text());data,audit=load_sample(meta['data_path'])
    if meta['direction']!='forward' or audit['sample_sha256']!=meta['sample_sha256']: raise ValueError('Fit/sample mismatch')
    reg=ForwardRegression(data,meta['model'],meta['rm_min'],meta['rm_max'],meta['scatter_model'])
    ell,s=reg.location_scale(reg.mle);a=(ell-np.log(meta['rm_min']))/s
    log_m1=ell+.5*s*s+log_ndtr(a+s)-log_ndtr(a)
    log_m2=2*ell+2*s*s+log_ndtr(a+2*s)-log_ndtr(a)
    mean=np.exp(log_m1)
    variance=np.exp(log_m2)*(-np.expm1(2*log_m1-log_m2))
    if not np.all(np.isfinite(variance)&(variance>0)): raise ValueError('Invalid selected variance')
    residual=data['lambda_rm']-mean;contribution=residual**2/variance
    report=dict(N=reg.n,parameters=len(reg.mle),chi2=float(contribution.sum()),chi2_over_N=float(contribution.mean()),nominal_N_minus_k=reg.n-len(reg.mle),chi2_over_nominal_N_minus_k=float(contribution.sum()/(reg.n-len(reg.mle))),mle=reg.mle.tolist(),mean='Per-object post-truncation arithmetic mean',variance='Per-object post-truncation second raw moment minus mean squared; includes full fitted lognormal/Poisson-motivated scatter',p_value=None,caveat='Descriptive Pearson-type statistic in linear RM; not chi-square distributed under selected skewed lognormal. No nominal p-value assigned; parameter estimation and spatial covariance not included in per-object variance.')
    output.mkdir(parents=True,exist_ok=True)
    (output/'individual_chi2.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(dict(ID=data['ID'],lambda_spec=data['lambda_spec'],lambda_rm=data['lambda_rm'],z=data['z'],selected_mean=mean,selected_variance=variance,residual=residual,chi2_contribution=contribution)).to_csv(output/'individual_residuals.csv',index=False)
    print(json.dumps(report,indent=2))


def summarize_broken_forward(folder,draws=1000):
    from global_spec_richness_model import BrokenForwardRegression
    from scipy.special import log_ndtr
    from scipy.stats import chi2
    meta=json.loads((folder/'run.json').read_text());data,audit=load_sample(meta['data_path'])
    if audit['sample_sha256']!=meta['sample_sha256']: raise ValueError('Sample mismatch')
    reg=BrokenForwardRegression(data)
    chains=[emcee.backends.HDFBackend(str(folder/f'chain_{j}.h5'),read_only=True).get_chain(discard=meta['burn']) for j in range(meta['ensembles'])]
    tau=np.array([emcee.autocorr.integrated_time(c,tol=0) for c in chains]);ratio=np.array([len(c) for c in chains])[:,None]/tau
    rhat=np.max([classical_rhat(np.stack([c[:,w] for c in chains])) for w in range(meta['walkers'])],axis=0)
    diag=dict(converged=bool(np.all(np.isfinite(tau)) and np.all(tau>0) and np.all(ratio>=50) and np.all(np.isfinite(rhat)) and np.all(rhat<1.1)),min_steps_per_tau=float(ratio.min()),max_Rhat=float(rhat.max()),tau_steps=tau.tolist(),steps_per_tau=ratio.tolist(),rhat_worst=rhat.tolist())
    (folder/'diagnostics.json').write_text(json.dumps(diag,indent=2))
    samples=np.concatenate([c.reshape(-1,c.shape[-1]) for c in chains]);np.savez_compressed(folder/'posterior.npz',samples=samples)
    rows=[]
    for j,name in enumerate(meta['parameters']):
        values=samples[:,j]
        if j==3:values=40*np.exp(values);name='lambda_spec_break'
        if j==5:values=np.exp(values);name='sigma0'
        lo,med,hi=np.percentile(values,[16,50,84]);rows.append(dict(parameter=name,median=med,err_minus=med-lo,err_plus=hi-med))
    pd.DataFrame(rows).to_csv(folder/'parameters.csv',index=False)
    rng=np.random.default_rng(20261112);ranks=(np.arange(reg.n)+.5)/reg.n;obs=[];rep=[]
    for t in samples[rng.integers(len(samples),size=draws)]:
        ell,s=reg.location_scale(t)
        pit=-np.expm1(log_ndtr((ell-reg.y)/s)-log_ndtr((ell-np.log(20))/s))
        obs.append(1/(12*reg.n)+np.sum((np.sort(pit)-ranks)**2))
        rep.append(1/(12*reg.n)+np.sum((np.sort(rng.random(reg.n))-ranks)**2))
    ppc=dict(draws=draws,exceedances=int(np.sum(np.array(rep)>=obs)),PPC=float(np.mean(np.array(rep)>=obs)),observed_CvM_median=float(np.median(obs)),replicated_CvM_median=float(np.median(rep)))
    (folder/'ppc.json').write_text(json.dumps(ppc,indent=2))
    np.savez_compressed(folder/'ppc_draws.npz',observed=obs,replicated=rep)
    t=np.median(samples,axis=0);ell,s=reg.location_scale(t);a=(ell-np.log(20))/s
    expected=np.exp(ell+.5*s*s+log_ndtr(a+s)-log_ndtr(a))
    spec=data['lambda_spec'];rm=data['lambda_rm'];edges=np.geomspace(spec.min(),np.nextafter(spec.max(),np.inf),11);stats=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(spec>=lo)&(spec<hi)
        if mask.sum()<20: continue
        sem=rm[mask].std(ddof=1)/np.sqrt(mask.sum());res=rm[mask].mean()-expected[mask].mean();stats.append((res/sem)**2)
    k=len(t);ll=reg.log_likelihood(reg.mle);baseline=ForwardRegression(data,'z',20,None,'poisson');bl=baseline.log_likelihood(baseline.mle)
    result=dict(N=reg.n,k=k,log_likelihood_max=ll,AIC=2*k-2*ll,BIC=k*np.log(reg.n)-2*ll,delta_AIC=(2*k-2*ll)-(8-2*bl),delta_BIC=(k*np.log(reg.n)-2*ll)-(4*np.log(reg.n)-2*bl),binned_chi2=float(sum(stats)),bins=len(stats),nominal_dof=len(stats)-k,nominal_p=float(chi2.sf(sum(stats),len(stats)-k)),caveat='Binned arithmetic-mean residual / observed SEM; nominal K-k p-value, not refit-bootstrap calibrated; PPC tests full selected distribution')
    (folder/'comparison.json').write_text(json.dumps(result,indent=2))
    print(pd.DataFrame(rows).to_string(index=False));print(json.dumps(dict(diagnostics=diag,ppc=ppc,comparison=result),indent=2))


def summarize_truncated_reverse(root, draws=1000):
    from global_spec_richness_model import TruncatedReverseRegression, CurvedReverseRegression
    root=Path(root); parameters=[]; comparisons=[]; ppc_rows=[]
    for model in (('quadratic','broken') if (root/'quadratic').exists() else ('noz','z')):
        folder=root/model; meta=json.loads((folder/'run.json').read_text())
        data,audit=load_sample(meta['data_path'])
        if audit['sample_sha256']!=meta['sample_sha256']: raise ValueError('Sample mismatch')
        curved=model in ('quadratic','broken')
        reg=(CurvedReverseRegression if curved else TruncatedReverseRegression)(data,model,meta['spec_min'])
        chains=[emcee.backends.HDFBackend(str(folder/f'chain_{j}.h5'),read_only=True).get_chain(discard=meta['burn'])
                for j in range(meta['ensembles'])]
        n=min(map(len,chains))
        tau=np.array([emcee.autocorr.integrated_time(c,tol=0) for c in chains])
        ratios=np.array([len(c) for c in chains])[:,None]/tau
        rhat=np.max([classical_rhat(np.array([c[:n,w] for c in chains])) for w in range(meta['walkers'])],axis=0)
        passed=bool(np.all(np.isfinite(tau)) and np.all(tau>0) and np.all(ratios>=50)
                    and np.all(np.isfinite(rhat)) and np.all(rhat<1.1))
        diag=dict(converged=passed,tau_steps=tau.tolist(),steps_per_tau=ratios.tolist(),rhat_worst=rhat.tolist(),
                  retained_steps=[len(c) for c in chains],rhat_method='Worst matched walker index across independent ensembles')
        (folder/'diagnostics.json').write_text(json.dumps(diag,indent=2))
        samples=np.concatenate([c.reshape(-1,c.shape[-1]) for c in chains])
        np.savez_compressed(folder/'posterior.npz',samples=samples)
        names=meta['parameters'] if curved else ['A_rev','B_rev']+(['C_rev'] if model=='z' else [])+['sigma_ln_spec']
        for j,name in enumerate(names):
            values=np.exp(samples[:,j]) if j==samples.shape[1]-1 else samples[:,j]
            if name=='ln_RM_break_over_40':
                values=40*np.exp(samples[:,j]);name='lambda_RM_break'
            if name=='ln_sigma_ln_spec': name='sigma_ln_spec'
            q=np.percentile(values,[16,50,84]); parameters.append(dict(model=model,parameter=name,
                median=q[1],err_minus=q[1]-q[0],err_plus=q[2]-q[1]))
        ll=reg.log_likelihood(reg.mle); k=len(reg.mle)
        comparisons.append(dict(model=model,N=reg.n,k=k,log_likelihood_max=ll,AIC=2*k-2*ll,
                                BIC=k*np.log(reg.n)-2*ll,converged=passed,
                                min_steps_per_tau=float(ratios.min()),max_Rhat=float(rhat.max())))
        edges=np.geomspace(20.,data['lambda_rm'].max(),5)
        masks=[np.ones(reg.n,dtype=bool)]+[(data['lambda_rm']>=lo)&((data['lambda_rm']<=hi) if j==3 else (data['lambda_rm']<hi))
                         for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:]))]
        rng=np.random.default_rng(20261109)
        selected=samples[rng.integers(len(samples),size=draws)]
        for cell,mask in enumerate(masks):
            count=int(mask.sum()); ranks=(np.arange(count)+.5)/count; obs=[]; rep=[]
            for theta in selected:
                u=reg.pit(theta)[mask]; v=rng.random(count)
                obs.append(1/(12*count)+np.sum((np.sort(u)-ranks)**2))
                rep.append(1/(12*count)+np.sum((np.sort(v)-ranks)**2))
            exceed=int(np.sum(np.array(rep)>=obs))
            ppc_rows.append(dict(model=model,cell=cell,N=count,
                rm_lo=20. if cell==0 else edges[cell-1],rm_hi=edges[-1] if cell==0 else edges[cell],
                cvm_observed_median=float(np.median(obs)),cvm_replicated_median=float(np.median(rep)),
                cvm_exceedances=exceed,ppc_draws=draws,cvm_ppc=exceed/draws))
    comparison=pd.DataFrame(comparisons)
    for key in ('AIC','BIC'): comparison['delta_'+key]=comparison[key]-comparison.iloc[0][key]
    comparison.to_csv(root/'model_comparison.csv',index=False)
    pd.DataFrame(parameters).to_csv(root/'parameters.csv',index=False)
    pd.DataFrame(ppc_rows).to_csv(root/'ppc.csv',index=False)
    print(comparison.to_string(index=False)); print(pd.DataFrame(ppc_rows).to_string(index=False))


def reverse_global_ppc(root, draws=1000, seed=20261105):
    """Check saved reverse posteriors without altering chains or summaries."""
    if draws < 1:
        raise ValueError('PPC draws must be positive')
    for model in ('noz', 'z'):
        directory = Path(root)/model
        meta = json.loads((directory/'run.json').read_text())
        if meta.get('direction') == 'forward':
            raise ValueError('This check is for reverse fits only')
        data, audit = load_sample(meta['data_path'])
        if audit['sample_sha256'] != meta['sample_sha256']:
            raise ValueError('Input sample changed')
        reg = Regression(data, model)
        with np.load(directory/'posterior.npz', allow_pickle=False) as archive:
            samples = archive['samples']
        rng = np.random.default_rng(seed)
        ranks = (np.arange(reg.n)+.5)/reg.n
        def cvm(u):
            return float(1/(12*reg.n)+np.sum((np.sort(u)-ranks)**2))
        observed, replicated = [], []
        for theta in samples[rng.integers(len(samples), size=draws)]:
            u = norm.cdf((reg.y-reg.X@theta[:-1])/np.exp(theta[-1]))
            observed.append(cvm(u))
            # Exact PIT distribution of replicated continuous lognormal data.
            replicated.append(cvm(rng.random(reg.n)))
        observed, replicated = np.array(observed), np.array(replicated)
        successes = int(np.sum(replicated >= observed))
        report = dict(draws=draws, seed=seed, statistic='PIT_Cramer_von_Mises',
                      upper_tail_count=successes, upper_tail_probability=successes/draws,
                      observed_median=float(np.median(observed)),
                      replicated_median=float(np.median(replicated)),
                      sample_sha256=meta['sample_sha256'],
                      selection='RM threshold is on the conditioning variable; reverse lognormal CDF',
                      replication='Uniform PITs equivalent to conditional lognormal replications')
        (directory/'ppc.json').write_text(json.dumps(report, indent=2))
        np.savez_compressed(directory/'ppc_cvm_draws.npz', observed=observed, replicated=replicated)
        print(model, json.dumps(report))


def summarize(root, tau_multiple=50., rhat_limit=1.1):
    rows, parameters = [], []
    models = ('baseline','environment') if (Path(root)/'environment/run.json').exists() else ('noz','z')
    for model in models:
        directory = Path(root) / model
        meta = json.loads((directory / "run.json").read_text())
        data, audit = load_sample(meta["data_path"])
        if audit["sample_sha256"] != meta["sample_sha256"]:
            raise ValueError("Input sample changed")
        forward = meta.get('direction') == 'forward'
        reg = (ForwardRegression(data, meta['model'], meta['rm_min'], meta['rm_max'],meta.get('scatter_model','constant'),
                                 environment=meta.get('environment',False),environment_pivot=meta.get('environment_pivot'))
               if forward else Regression(data, model))
        backends = [emcee.backends.HDFBackend(str(directory / f"chain_{i}.h5"), read_only=True)
                    for i in range(meta["ensembles"])]
        chains = [b.get_chain(discard=meta["burn"]) for b in backends]
        n = min(len(c) for c in chains)
        if n < 4:
            raise ValueError("Not enough retained steps")
        taus = np.array([emcee.autocorr.integrated_time(c, tol=0) for c in chains])
        ratios = np.array([len(c) for c in chains])[:, None] / taus
        # Compare matching walker indices across INDEPENDENT ensembles.
        # Report the worst walker-index Rhat, never GR across interacting walkers.
        rhat = np.max([classical_rhat(np.array([c[:n, w] for c in chains]))
                       for w in range(meta["walkers"])], axis=0)
        samples = np.concatenate([c.reshape(-1, c.shape[-1]) for c in chains])
        np.savez_compressed(directory / "posterior.npz", samples=samples)
        logl = reg.log_likelihood(reg.mle)
        k = len(reg.mle)
        passed = bool(np.all(np.isfinite(taus)) and np.all(taus > 0)
                      and np.all(ratios >= tau_multiple)
                      and np.all(np.isfinite(rhat)) and np.all(rhat < rhat_limit))
        report = dict(model=model, converged=passed, tau_steps=taus.tolist(),
                      steps_per_tau=ratios.tolist(), rhat_worst=rhat.tolist(),
                      tau_multiple=tau_multiple, rhat_limit=rhat_limit,
                      acceptance_mean=[float(np.mean(b.accepted / b.iteration)) for b in backends],
                      retained_steps=[len(c) for c in chains],
                      rhat_method="Classical GR per fixed walker index across independent ensembles; worst index",
                      chi2_caveat="At Gaussian MLE with fitted scatter chi2=N by construction; not an independent goodness-of-fit test.")
        (directory / "diagnostics.json").write_text(json.dumps(report, indent=2))
        if forward:
            report['chi2_caveat'] = 'PIT histogram discrepancy with 20 bins; no nominal p-value after fitting. Use selection-aware PPC.'
            (directory/'diagnostics.json').write_text(json.dumps(report, indent=2))
            from scipy.special import log_ndtr
            def pit(theta):
                loc,sigma=reg.location_scale(theta)
                return -np.expm1(log_ndtr((loc-reg.y)/sigma)-log_ndtr((loc-np.log(reg.lower))/sigma))
            def discrepancy(u):
                ordered=np.sort(u)
                return np.array([np.sum((np.histogram(u,np.linspace(0,1,21))[0]-reg.n/20)**2/(reg.n/20)),
                                 1/(12*reg.n)+np.sum((ordered-(np.arange(reg.n)+.5)/reg.n)**2)])
            rng=np.random.default_rng(20261027)
            observed=[]; replicated=[]
            for theta in samples[rng.integers(len(samples),size=1000)]:
                observed.append(discrepancy(pit(theta)))
                replicated.append(discrepancy(rng.random(reg.n)))
            observed=np.array(observed); replicated=np.array(replicated)
            ppc=dict(draws=1000,seed=20261027,statistics=['PIT_histogram_chi2','PIT_Cramer_von_Mises'],
                     observed_median=np.median(observed,axis=0).tolist(),
                     replicated_median=np.median(replicated,axis=0).tolist(),
                     upper_tail_probability=np.mean(replicated>=observed,axis=0).tolist(),
                     selection='Per-cluster lower-truncated lognormal PIT; replicated PITs uniform')
            (directory/'ppc.json').write_text(json.dumps(ppc,indent=2))
            discrepancy_mle=float(discrepancy(pit(reg.mle))[0])
        for j, name in enumerate(meta["parameters"]):
            scatter = name in ('ln_s_rev','ln_sigma_ln_RM','ln_sigma0')
            vals = np.exp(samples[:, j]) if scatter else samples[:, j]
            lo, mid, hi = np.percentile(vals, [16, 50, 84])
            parameters.append(dict(model=model, parameter=name.replace("ln_s_rev", "s_rev").replace('ln_sigma_ln_RM','sigma_ln_RM').replace('ln_sigma0','sigma0'),
                                   median=mid, err_minus=mid-lo, err_plus=hi-mid,
                                   p025=float(np.percentile(vals,2.5)),p975=float(np.percentile(vals,97.5)),
                                   mle=float(np.exp(reg.mle[j]) if scatter else reg.mle[j])))
        rows.append(dict(model=model, N=reg.n, k=k, log_likelihood_max=logl,
                         chi2_mle=discrepancy_mle if forward else reg.sse / np.exp(2 * reg.mle[-1]), dof=None if forward else reg.n-k,
                         AIC=2*k-2*logl, BIC=k*np.log(reg.n)-2*logl,
                         cvm_ppc=ppc['upper_tail_probability'][1] if forward else None,
                         converged=passed, sample_sha256=meta["sample_sha256"]))
    comparison = pd.DataFrame(rows)
    if comparison.sample_sha256.nunique() != 1:
        raise ValueError("Models use different samples")
    for key in ("AIC", "BIC", "chi2_mle"):
        comparison["delta_" + key + "_vs_" + models[0]] = comparison[key] - comparison.iloc[0][key]
    params = pd.DataFrame(parameters)
    comparison.to_csv(Path(root) / "model_comparison.csv", index=False)
    params.to_csv(Path(root) / "parameters.csv", index=False)
    if forward:
        log_rows=[]
        for model in models:
            directory=Path(root)/model
            meta=json.loads((directory/'run.json').read_text())
            with np.load(directory/'posterior.npz') as archive:
                transformed=archive['samples'].copy()
            poisson=meta.get('scatter_model')=='poisson'
            if not poisson: transformed[:,0] -= .5*np.exp(2*transformed[:,-1])
            transformed[:,-1] = np.exp(transformed[:,-1])
            for j,name in enumerate(['A','B']+(['C'] if meta['model']=='z' else [])+(['D'] if meta.get('environment') else [])+['sigma0' if poisson else 'sigma_ln_RM']):
                lo,mid,hi=np.percentile(transformed[:,j],[16,50,84])
                log_rows.append(dict(model=model,parameter=name,median=mid,err_minus=mid-lo,err_plus=hi-mid))
        pd.DataFrame(log_rows).to_csv(Path(root)/'parameters_mean_log.csv',index=False)
    comparison.drop(columns="sample_sha256").to_latex(
        Path(root) / "model_comparison.tex", index=False, float_format="%.3f")
    params.to_latex(Path(root) / "parameters.tex", index=False, float_format="%.4f")
    print(comparison.to_string(index=False))
    return comparison, params


def summarize_cells(root, ppc_draws=2000):
    """Independent cell posteriors; no refitting of the fixed global location."""
    if ppc_draws < 1:
        raise ValueError("ppc_draws must be positive")
    root=Path(root)
    rows, parameters=[],[]
    for directory in sorted(root.glob("z*_rm*/*")):
        if not (directory/"run.json").exists():
            continue
        meta=json.loads((directory/"run.json").read_text())
        data,audit=load_sample(meta["data_path"])
        if audit["sample_sha256"]!=meta["sample_sha256"]:
            raise ValueError("Cell sample checksum changed")
        keep=np.isin(data["ID"].astype(str),meta["cluster_ids"])
        if keep.sum()!=meta["n"]:
            raise ValueError("Cell IDs differ")
        logspec=np.log(data["lambda_spec"][keep])
        a,b=meta["fixed_global_mle"]
        residual=logspec-a-b*np.log(data["lambda_rm"][keep]/meta["rm_pivot"])
        family=meta["model"]
        reg=CellResidual(residual,logspec,family)
        backends=[emcee.backends.HDFBackend(str(directory/f"chain_{i}.h5"),read_only=True)
                  for i in range(meta["ensembles"])]
        chains=[be.get_chain(discard=meta["burn"]) for be in backends]
        n=min(map(len,chains))
        if n<4:
            raise ValueError("Insufficient retained chain")
        tau=np.array([emcee.autocorr.integrated_time(c,tol=0) for c in chains])
        ratio=np.array([len(c) for c in chains])[:,None]/tau
        rhat=np.max([classical_rhat(np.array([c[:n,w] for c in chains]))
                     for w in range(meta["walkers"])],axis=0)
        samples=np.concatenate([c.reshape(-1,c.shape[-1]) for c in chains])
        np.savez_compressed(directory/"posterior.npz",samples=samples)
        for be,chain in zip(backends,chains):
            lp=be.get_log_prob(discard=meta["burn"])
            candidate=chain.reshape(-1,chain.shape[-1])[np.argmax(lp)]
            if reg.log_likelihood(candidate)>reg.log_likelihood(reg.mle)+1e-6:
                result=minimize(lambda t:-reg.log_likelihood(t),candidate,
                                bounds=reg.bounds,method="L-BFGS-B")
                if -result.fun>reg.log_likelihood(reg.mle):
                    reg.mle=result.x
                    reg.optimizer_success=bool(result.success)
        passed=bool(np.all(np.isfinite(tau)) and np.all(tau>0) and np.all(ratio>=50)
                    and np.all(np.isfinite(rhat)) and np.all(rhat<1.1))
        diagnostics=dict(converged=passed,tau_steps=tau.tolist(),steps_per_tau=ratio.tolist(),
                         rhat_worst=rhat.tolist(),retained_steps=[len(c) for c in chains],
                         acceptance_mean=[float(np.mean(be.accepted/be.iteration)) for be in backends],
                         method="Rhat per fixed walker across independent ensembles; worst index",
                         tau_multiple=50,rhat_limit=1.1)
        (directory/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
        for j,name in enumerate(meta["parameters"]):
            vals=np.exp(samples[:,j]) if name.startswith("ln_") else samples[:,j]
            lo,mid,hi=np.percentile(vals,[16,50,84])
            lower,upper=meta["bounds"][j]
            boundary_fraction=float(np.mean((samples[:,j]<lower+.01*(upper-lower)) |
                                            (samples[:,j]>upper-.01*(upper-lower))))
            parameters.append(dict(cell=directory.parent.name,model=family,parameter=name.removeprefix("ln_"),
                                   median=mid,err_minus=mid-lo,err_plus=hi-mid,
                                   mle=float(np.exp(reg.mle[j]) if name.startswith("ln_") else reg.mle[j]),
                                   prior_boundary_fraction=boundary_fraction))
        logl=reg.log_likelihood(reg.mle); k=len(reg.mle)
        rng=np.random.default_rng(20261020)
        statistic=lambda v:np.array([v.mean(),v.std(ddof=0),skew(v),kurtosis(v)])
        obs=statistic(residual)
        reps=np.array([statistic(residual_draw(t,family,len(residual),rng))
                       for t in samples[rng.integers(len(samples),size=ppc_draws)]])
        ppc=np.mean(reps>=obs,axis=0)
        predictive=dict(draws=ppc_draws,seed=20261020,statistics=["mean","std","skewness","excess_kurtosis"],
                        observed=obs.tolist(),replicated_median=np.median(reps,axis=0).tolist(),
                        upper_tail_probability=ppc.tolist())
        (directory/"ppc.json").write_text(json.dumps(predictive,indent=2))
        # Pearson histogram discrepancy includes underflow/overflow; merge sparse
        # expected-count bins. No nominal chi-square p-value after fitted parameters.
        edges=np.r_[-np.inf,np.linspace(-2,2,21),np.inf]
        s=np.exp(reg.mle[0])
        if family=="lognormal":
            cdf=norm.cdf(edges,scale=s)
        else:
            f,rate=reg.mle[1],np.exp(reg.mle[2]); g=edges-f/rate
            cdf=(1-f)*norm.cdf(g,scale=s)+f*exponnorm.sf(-g,1/(rate*s),scale=s)
        expected=np.maximum(np.diff(cdf)*reg.n,0)
        observed=np.histogram(residual,edges)[0]
        groups=[]; oe=ee=0.
        for o,e in zip(observed,expected):
            oe+=o; ee+=e
            if ee>=5:
                groups.append([oe,ee]);oe=ee=0.
        if ee or oe:
            if groups:groups[-1][0]+=oe;groups[-1][1]+=ee
            else:groups.append([oe,ee])
        chi=float(sum((o-e)**2/e for o,e in groups if e>0))
        row=dict(cell=directory.parent.name,model=family,N=reg.n,converged=passed,
                 min_steps_per_tau=float(ratio.min()),max_Rhat=float(rhat.max()),
                 log_likelihood_max=logl,AIC=2*k-2*logl,BIC=k*np.log(reg.n)-2*logl,
                 chi2_histogram=chi,histogram_groups=len(groups),optimizer_success=reg.optimizer_success)
        row.update({f"ppc_{key}":float(p) for key,p in zip(predictive["statistics"],ppc)})
        rows.append(row)
    comparison=pd.DataFrame(rows)
    if len(comparison)!=18:
        raise ValueError(f"Expected 18 completed cell fits, found {len(comparison)}")
    for key in ("AIC","BIC"):
        reference=comparison[comparison.model=="lognormal"].set_index("cell")[key]
        comparison["delta_"+key+"_vs_lognormal"]=comparison[key]-comparison.cell.map(reference)
    comparison.to_csv(root/"cell_comparison.csv",index=False)
    params=pd.DataFrame(parameters)
    params.to_csv(root/"cell_parameters.csv",index=False)
    params.to_latex(root/"cell_parameters.tex",index=False,float_format="%.4f")
    comparison.to_latex(root/"cell_comparison.tex",index=False,float_format="%.4f")
    print(comparison.to_string(index=False))
    return comparison,params


def summarize_environment_burn(directory, burn):
    """Separate, auditable post-burn analysis; never alter original chains/reports."""
    from fit_global_spec_richness import adaptive_checkpoint
    directory=Path(directory)
    if (directory/'RUNNING.lock').exists():
        raise ValueError('Wait for the run to stop')
    meta=json.loads((directory/'run.json').read_text())
    if burn<=meta['burn']:
        raise ValueError('Use a larger explicitly justified burn-in')
    backends=[emcee.backends.HDFBackend(str(directory/f'chain_{i}.h5'),read_only=True)
              for i in range(meta['ensembles'])]
    if len(set(b.iteration for b in backends))!=1:
        raise ValueError('Ensemble lengths differ')
    steps=backends[0].iteration
    if steps-burn<10000:
        raise ValueError('Require at least 10000 retained steps for stability checks')
    chains=[b.get_chain(discard=burn) for b in backends]
    history=[]
    for retained in sorted(set(range(5000,steps-burn+1,5000)) | {steps-burn}):
        prefixes=[c[:retained] for c in chains]
        taus=np.array([emcee.autocorr.integrated_time(c,tol=0) for c in prefixes])
        rhat=np.max([classical_rhat(np.stack([c[:,w] for c in prefixes]))
                     for w in range(meta['walkers'])],axis=0)
        ratios=retained/taus
        passed=bool(np.all(np.isfinite(taus)) and np.all(taus>0) and np.all(ratios>=50)
                    and np.all(np.isfinite(rhat)) and np.all(rhat<1.1)
                    and not any(np.any(np.var(c,axis=0)==0) for c in prefixes))
        report=dict(steps=burn+retained,tau_steps=taus.tolist(),steps_per_tau=ratios.tolist(),
                    rhat_worst=rhat.tolist(),convergence_pass=passed)
        history.append(adaptive_checkpoint(report,history[-1] if history else None))
    samples=np.concatenate([c[::5].reshape(-1,c.shape[-1]) for c in chains])
    physical=samples.copy()
    logs=[j for j,n in enumerate(meta['parameters']) if n.startswith('ln_')]
    physical[:,logs]=np.exp(physical[:,logs])
    summary=dict(report,parameters=[n.removeprefix('ln_') for n in meta['parameters']],
                 posterior_quantiles=np.percentile(physical,[16,50,84],axis=0).tolist(),
                 adaptive_convergence_pass=history[-1]['converged'],
                 stop_reason='posthoc_burn_analysis',original_burn=meta['burn'],diagnostic_burn=burn,
                 warning='Post-hoc burn analysis after observed mode transition; original failed diagnostics preserved. Global means fixed.')
    output=directory/f'diagnostics_burn_{burn}'
    output.mkdir(exist_ok=False)
    updated=dict(meta,burn=burn,chain_directory=str(directory.resolve()),analysis_only=True,
                 original_run=str((directory/'run.json').resolve()))
    for name,value in [('run',updated),('summary',summary),('convergence_history',history),
                       ('bestfit',json.loads((directory/'bestfit.json').read_text()))]:
        (output/f'{name}.json').write_text(json.dumps(value,indent=2))
    np.savez_compressed(output/'posterior.npz',samples=samples)
    print(json.dumps(summary,indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fits", required=True, type=Path)
    p.add_argument("--cells",action="store_true")
    p.add_argument('--reverse-ppc-only', action='store_true', help='Compute reverse global CvM PPC from saved posteriors only')
    p.add_argument('--truncated-reverse',action='store_true',help='Summarize reverse global fits selected above spec threshold')
    p.add_argument('--forward-refit-bootstrap',action='store_true',help='Selected forward histogram test with refitting')
    p.add_argument('--forward-broken',action='store_true',help='Summarize evolving broken forward chains')
    p.add_argument('--forward-individual-chi2',action='store_true',help='Individual linear RM residuals divided by selected-model variance')
    p.add_argument('--reverse-log-individual-chi2',action='store_true',help='Individual selected log-spec residual statistic at posterior medians')
    p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument("--ppc-draws",type=int,default=2000)
    p.add_argument('--environment-burn',type=int,help='Separate post-hoc burn analysis of a stopped environment fit')
    a = p.parse_args()
    if a.reverse_log_individual_chi2:
        if a.output is None: p.error('Require --output')
        reverse_log_individual_discrepancy(a.fits, a.output)
    elif a.forward_individual_chi2:
        if a.output is None: p.error('Require --output')
        forward_individual_discrepancy(a.fits,a.output)
    elif a.forward_broken:
        summarize_broken_forward(a.fits,a.ppc_draws)
    elif a.forward_refit_bootstrap:
        if a.output is None or a.ppc_draws<1 or a.workers<1: p.error('Require output, positive draws/workers')
        forward_refitted_bootstrap(a.fits,a.output,a.ppc_draws,a.workers)
    elif a.truncated_reverse:
        summarize_truncated_reverse(a.fits,a.ppc_draws)
    elif a.reverse_ppc_only:
        reverse_global_ppc(a.fits,a.ppc_draws)
    elif a.environment_burn is not None:
        summarize_environment_burn(a.fits,a.environment_burn)
    elif a.cells:
        summarize_cells(a.fits,a.ppc_draws)
    else:
        summarize(a.fits)
