"""Conditional goodness of fit of the saved ten-bin mixture/lognormal fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import log_ndtr, ndtri_exp
from global_spec_richness_model import FixedMeanMixture, FixedMeanLognormal

ROOT = Path(__file__).resolve().parents[1]


def survival(theta, x, means, family):
    if family == 'lognormal':
        s = np.exp(theta[0])
        ell = np.log(means)-s*s/2
        return log_ndtr((ell-np.log(x))/s)
    ls, f, lt = theta
    s, rate = np.exp(ls), np.exp(lt)
    d = np.asarray(x)-(means-f/rate)
    return np.logaddexp(log_ndtr(-d/s), np.log(f)-rate*d+(rate*s)**2/2+log_ndtr(d/s-rate*s))


def cdf(theta, x, means, family):
    return np.clip(-np.expm1(survival(theta,x,means,family)-survival(theta,20.,means,family)),0,1)


def draw(theta, means, family, rng):
    u = rng.random(len(means))
    if family == 'lognormal':
        s = np.exp(theta[0]); ell = np.log(means)-s*s/2
        return np.exp(ell-s*ndtri_exp(survival(theta,20.,means,family)+np.log(u)))
    target = survival(theta,20.,means,family)+np.log(u)
    low = np.full(len(means),20.)
    high = np.maximum(means+100,100.)
    for _ in range(80):
        mask = survival(theta,high,means,family)>target
        if not mask.any(): break
        high[mask] *= 2
    else: raise RuntimeError('Failed inverse-CDF bracket')
    for _ in range(55):
        mid = (low+high)/2
        mask = survival(theta,mid,means,family)>target
        low = np.where(mask,mid,low); high = np.where(mask,high,mid)
    return (low+high)/2


def fit(rm, means, family, prior, starts):
    reg = FixedMeanLognormal(rm,means) if family=='lognormal' else FixedMeanMixture(rm,means,prior)
    def nll(t):
        if family=='lognormal': return -float(reg.log_likelihood(t)[0])
        a,b = reg.component_logpdf(t,rm,means)
        return -float(np.logaddexp(a,b).sum())
    if family=='lognormal':
        result = minimize_scalar(lambda x:nll([x]),bounds=(-6+1e-7,1-1e-7),method='bounded')
        if not result.success: raise RuntimeError('Lognormal optimization failed')
        return np.array([result.x]),-result.fun
    bounds = [(a+1e-7,b-1e-7) for a,b in reg.bounds]
    results = [minimize(nll,np.clip(t,np.array(bounds)[:,0],np.array(bounds)[:,1]),
                        method='Nelder-Mead',bounds=bounds,
                        options={'maxiter':2000,'xatol':1e-5,'fatol':1e-6}) for t in starts]
    good = [r for r in results if r.success and np.isfinite(r.fun)]
    if not good: raise RuntimeError('Mixture optimization failed')
    best = min(good,key=lambda r:r.fun)
    return best.x,-best.fun


def discrepancy(rm, means, theta, family, edges):
    finite = edges[:-1]
    cumulative = cdf(theta,finite[:,None],means[None,:],family).sum(axis=1)
    expected = np.diff(np.r_[cumulative,len(rm)])
    observed = np.histogram(rm,edges)[0]
    if not np.isclose(expected.sum(),len(rm)): raise ValueError('Probability not conserved')
    groups=[]; start=0; total=0.
    for j,e in enumerate(expected):
        total+=e
        if total>=5:
            groups.append([start,j+1]);start=j+1;total=0.
    if start<len(expected):
        if groups:groups[-1][1]=len(expected)
        else:groups=[[0,len(expected)]]
    records=[]
    for a,b in groups:
        o=observed[a:b].sum();e=expected[a:b].sum()
        records.append(dict(lo=edges[a],hi=edges[b],observed=int(o),expected=float(e)))
    return sum((r['observed']-r['expected'])**2/r['expected'] for r in records),records


def cvm(u):
    n=len(u)
    return 1/(12*n)+np.sum((np.sort(u)-(np.arange(n)+.5)/n)**2)


def run_cell(task):
    cell,family,reps,ppc,output=task
    root=ROOT/'catalogs'/f'{family}_spec_deciles_evolving_20261112'/cell
    meta=json.loads((root/'run.json').read_text())
    source=Path(meta['assignments'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=meta['assignments_sha256']: raise ValueError('Changed inputs')
    frame=pd.read_csv(source,dtype={'ID':str}).set_index('ID').loc[meta['cluster_ids']]
    rm=frame.lambda_rm.to_numpy(); means=np.array(meta['fixed_global_means'])
    prior=meta.get('mixture_prior','legacy')
    best=json.loads((root/'bestfit.json').read_text())
    starts=best.get('optima',[best.get('mle',best.get('map'))])
    theta,ll=fit(rm,means,family,prior,starts)
    hist=pd.read_csv(ROOT/'plots/conditional_spec_deciles_comparison_logx_20261112/histogram_measurements.csv')
    hist=hist[hist.cell==cell].sort_values('lo')
    edges=np.r_[hist.lo.to_numpy(),hist.hi.iloc[-1],np.inf]
    if not np.isclose(edges[0],20): raise ValueError('Histogram must begin at selection threshold')
    observed,records=discrepancy(rm,means,theta,family,edges)
    seed=20261112+100*meta['spec_bin']+(family=='mixture')
    rng=np.random.default_rng(seed)
    # Use multiple mixture starts, retaining the generator optimum in every refit.
    refit_starts=[theta]+([np.array([np.log(10),.2,np.log(.1)]),np.array([np.log(20),.8,np.log(.02)])] if family=='mixture' else [])
    simulated=[]
    for j in range(reps):
        mock=draw(theta,means,family,rng)
        t,_=fit(mock,means,family,prior,refit_starts)
        simulated.append(discrepancy(mock,means,t,family,edges)[0])
        if (j+1)%50==0: print(cell,family,'bootstrap',j+1,flush=True)
    with np.load(root/'posterior.npz') as archive:samples=archive['samples']
    obs=[];rep=[]
    for t in samples[rng.integers(len(samples),size=ppc)]:
        obs.append(cvm(cdf(t,rm,means,family)));rep.append(cvm(rng.random(len(rm))))
    exceed=int(np.sum(np.array(simulated)>=observed)); p=(exceed+1)/(reps+1)
    ppc_exceed=int(np.sum(np.array(rep)>=obs)); q=ppc_exceed/ppc
    k=len(theta); report=dict(cell=cell,family=family,N=len(rm),spec_lo=meta['spec_range'][0],spec_hi=meta['spec_range'][1],
        chi2=observed,histogram_groups=len(records),bootstrap_p=p,bootstrap_exceedances=exceed,bootstrap_draws=reps,
        bootstrap_mcse=float(np.sqrt(p*(1-p)/(reps+1))),CvM_PPC=q,CvM_exceedances=ppc_exceed,PPC_draws=ppc,
        AIC=2*k-2*ll,BIC=k*np.log(len(rm))-2*ll,mle=theta.tolist())
    folder=Path(output)/cell/family;folder.mkdir(parents=True,exist_ok=True)
    (folder/'summary.json').write_text(json.dumps(report,indent=2))
    pd.DataFrame(records).to_csv(folder/'histogram_counts.csv',index=False)
    np.savez_compressed(folder/'calibration.npz',bootstrap=simulated,cvm_observed=obs,cvm_replicated=rep)
    print(cell,family,'done',p,q,flush=True)
    return report


def export_table_and_residuals(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import chi2
    table=pd.read_csv(output/'goodness_of_fit.csv')
    rows=[]
    fig,axes=plt.subplots(2,5,figsize=(17,7),layout='constrained')
    hist=pd.read_csv(ROOT/'plots/conditional_spec_deciles_comparison_logx_20261112/histogram_measurements.csv')
    for i,ax in enumerate(axes.flat,1):
        cell=f'z1_s{i}_Q0';t=table[table.cell==cell].set_index('family')
        l,m=t.loc['lognormal'],t.loc['mixture']
        rows.append(dict(bin=i,spec_lo=l.spec_lo,spec_hi=l.spec_hi,N=int(l.N),
            chi2_LN=l.chi2,chi2_mix=m.chi2,p_hist_LN=l.bootstrap_p,p_hist_mix=m.bootstrap_p,
            PPC_CvM_LN=l.CvM_PPC,PPC_CvM_mix=m.CvM_PPC,delta_AIC=m.AIC-l.AIC,delta_BIC=m.BIC-l.BIC))
        h=hist[hist.cell==cell].sort_values('lo');edges=np.r_[h.lo,h.hi.iloc[-1],np.inf]
        for family,color,offset in [('lognormal','#b33b3b',-.12),('mixture','#2266aa',.12)]:
            folder=ROOT/'catalogs'/f'{family}_spec_deciles_evolving_20261112'/cell
            meta=json.loads((folder/'run.json').read_text())
            source=pd.read_csv(meta['assignments'],dtype={'ID':str}).set_index('ID')
            rm=source.loc[meta['cluster_ids'],'lambda_rm'].to_numpy();means=np.array(meta['fixed_global_means'])
            r=json.loads((output/cell/family/'summary.json').read_text());theta=r['mle']
            expected=np.diff(np.r_[cdf(theta,edges[:-1,None],means[None,:],family).sum(axis=1),len(rm)])
            counts=np.histogram(rm,edges)[0]
            lo=np.where(counts>0,chi2.ppf(.15865,2*counts)/2,0)
            hi=chi2.ppf(.84135,2*(counts+1))/2
            ax.errorbar(np.arange(len(counts))+1+offset,counts-expected,yerr=[counts-lo,hi-counts],
                        fmt='o',mfc='none',ms=3,lw=.8,color=color,label=family)
        ax.axhline(0,color='0.4',lw=.7)
        ax.set_title(f'{l.spec_lo:.1f} - {l.spec_hi:.1f}',fontsize=11)
        ax.set_xticks([1,5,10,16],['1','5','10','overflow'])
        ax.set_xlabel('Histogram interval')
        if i in (1,6):ax.set_ylabel('Observed - expected counts')
        ax.tick_params(direction='in',top=True,right=True)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.savefig(output/'count_residuals.png',dpi=180);fig.savefig(output/'count_residuals.pdf');plt.close(fig)
    pd.DataFrame(rows).to_csv(output/'comparison_table.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',type=int,default=199);p.add_argument('--ppc',type=int,default=1000)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,default=ROOT/'plots/decile_goodness_20261112')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    tasks=[(f'z1_s{i}_Q0',f,a.bootstrap,a.ppc,str(a.output)) for i in range(1,11) for f in ['lognormal','mixture']]
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(run_cell,tasks):
            results.append(result);pd.DataFrame(results).to_csv(a.output/'goodness_of_fit.csv',index=False)
    (a.output/'method.json').write_text(json.dumps(dict(bootstrap_replicates=a.bootstrap,ppc_draws=a.ppc,
        method='Fixed per-object evolving global mean anchors; selected likelihood MLE refits within original parameter bounds, without prior density; same original histogram edges for both models plus infinite overflow; repeat adjacent E>=5 merging in each bootstrap; p=(exceed+1)/(B+1). CvM PPC uses posterior draws and uniform replicated PIT values.',
        caveat='Conditional on fixed global anchors and independent observed covariates. Does not propagate global-fit uncertainty, spatial covariance, or selection beyond RM>20. No multiplicity correction.'),indent=2))
    export_table_and_residuals(a.output)
