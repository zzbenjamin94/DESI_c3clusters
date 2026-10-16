"""Independent selected P(lambda_RM | richness bin, redshift bin) models.

The location is a lognormal median or Gaussian-core location, NOT generally
an arithmetic mean. Finite-bin mixing and measurement noise contribute scatter.
"""
import numpy as np
from scipy.optimize import minimize
from richness_relation.conditional_richness_models import (
    ModelConfig, selected_logpdf, selected_cdf, selected_rvs,
)

Z_EDGES = np.array([0.10, 0.18, 0.24, 0.35])


def parameter_spec(model):
    shared = [('ln_mu', np.log(1.), np.log(1000.))]
    if model == 'lognormal':
        return shared + [('ln_sigma0', np.log(.01), np.log(2.))]
    if model == 'mixture':
        return shared + [('ln_sigma_lambda', np.log(.1), np.log(300.)),
                         ('f_proj', 0., 1.), ('ln_tau', np.log(1/500), np.log(10.))]
    raise ValueError(model)


def expanded(theta):
    # B=0 removes within-cell regression; C is absent. No dummy parameters sampled.
    return np.insert(np.asarray(theta), 1, 0.)


def config(model, lower, upper):
    return ModelConfig(model=model, evolution=False, rm_min=lower,
                       rm_max=np.inf if upper is None else upper)


def logpdf(theta, rm, cfg):
    return selected_logpdf(expanded(theta), rm, 40., .24, cfg)


def cdf(theta, rm, cfg):
    return selected_cdf(expanded(theta), rm, 40., .24, cfg)


def draw(theta, n, cfg, rng):
    return selected_rvs(expanded(theta), np.full(n,40.), np.full(n,.24),cfg,rng)


def log_probability(theta, rm, cfg):
    bounds=np.array([s[1:] for s in parameter_spec(cfg.model)])
    if not np.all(np.isfinite(theta)) or np.any(theta<=bounds[:,0]) or np.any(theta>=bounds[:,1]):
        return -np.inf
    values=logpdf(theta,rm,cfg)
    return float(values.sum()) if np.all(np.isfinite(values)) else -np.inf


def optimize(rm, cfg, rng, starts=8, initial=None):
    bounds=np.array([s[1:] for s in parameter_spec(cfg.model)])
    lo,hi=bounds[:,0]+1e-7,bounds[:,1]-1e-7
    base=[np.log(np.median(rm))]
    if cfg.model=='lognormal':
        base += [np.log(max(np.std(np.log(rm)),.1))]
    else:
        base += [np.log(max(np.std(rm)*.5,1.)),.2,np.log(1/max(np.std(rm),1.))]
    base=np.clip(base,lo,hi)
    attempts=[]
    for i in range(starts):
        x = np.clip(initial,lo,hi) if i==0 and initial is not None else (
            base if i==0 else np.clip(base+rng.normal(0,.15,len(base))*(hi-lo),lo,hi))
        fit=minimize(lambda t:-float(logpdf(t,rm,cfg).sum()),x,bounds=list(zip(lo,hi)),method='L-BFGS-B')
        if fit.success and np.isfinite(fit.fun): attempts.append(fit)
    if not attempts:
        raise RuntimeError('No successful finite MLE optimization; increase starts or inspect the cell')
    return min(attempts,key=lambda f:f.fun)


def physical_samples(samples, model):
    s=np.atleast_2d(samples)
    result={'mu':np.exp(s[:,0])}
    if model=='lognormal': result['sigma0']=np.exp(s[:,1])
    else:
        result.update(sigma_lambda=np.exp(s[:,1]), f_proj=s[:,2], tau=np.exp(s[:,3]))
    return result


def posterior_summary(samples, model):
    return {name:dict(zip(('p16','median','p84'),np.percentile(v,[16,50,84]).tolist()))
            for name,v in physical_samples(samples,model).items()}


def make_cells(data, edges=None):
    spec=np.asarray(data['lambda_spec'])
    z=np.asarray(data['z'])
    if np.any((z<Z_EDGES[0]) | (z>=Z_EDGES[-1])):
        raise ValueError('Prepared sample extends outside [0.10,0.35); reprepare consistently')
    edges=np.quantile(spec,np.linspace(0,1,5)) if edges is None else np.asarray(edges,float)
    if len(edges)!=5 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges)<=0):
        raise ValueError('Require five strictly increasing richness edges; tied quartiles need manual edges')
    if edges[0]>spec.min() or edges[-1]<spec.max():
        raise ValueError('Richness edges must encompass the full prepared sample')
    cells=[]
    for iz,(zl,zh) in enumerate(zip(Z_EDGES[:-1],Z_EDGES[1:])):
        for ix,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            mask=(z>=zl)&(z<zh)&(spec>=lo)&((spec<=hi) if ix==3 else (spec<hi))
            cells.append((dict(cell=f'z{iz+1}_r{ix+1}',iz=iz,ix=ix,zlo=float(zl),zhi=float(zh),
                               spec_lo=float(lo),spec_hi=float(hi),N=int(mask.sum())),mask))
    return edges,cells


def pearson_stat(rm,theta,cfg):
    """Fixed base edges, with fitted-expected-count merging; repeat for each bootstrap."""
    start=max(cfg.rm_min,1e-5)
    end=cfg.rm_max if np.isfinite(cfg.rm_max) else max(1000.,start*10)
    edges=np.r_[cfg.rm_min,np.geomspace(start,end,13)[1:-1],cfg.rm_max]
    observed=np.histogram(rm,edges)[0]
    expected=len(rm)*np.diff(cdf(theta,edges,cfg))
    grouped=[]
    o,e=0.,0.
    for ni,ei in zip(observed,expected):
        o+=ni; e+=max(float(ei),0.)
        if e>=5:
            grouped.append([o,e]); o,e=0.,0.
    if e>0 or o>0:
        if grouped: grouped[-1][0]+=o; grouped[-1][1]+=e
        else: grouped.append([o,e])
    if any(v[1]<=0 for v in grouped): raise ValueError('Zero expected probability')
    stat=sum((o-e)**2/e for o,e in grouped)
    return float(stat),len(grouped)
