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


def summarize(root, tau_multiple=50., rhat_limit=1.1):
    rows, parameters = [], []
    for model in ("noz", "z"):
        directory = Path(root) / model
        meta = json.loads((directory / "run.json").read_text())
        data, audit = load_sample(meta["data_path"])
        if audit["sample_sha256"] != meta["sample_sha256"]:
            raise ValueError("Input sample changed")
        forward = meta.get('direction') == 'forward'
        reg = (ForwardRegression(data, model, meta['rm_min'], meta['rm_max'])
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
                sigma=np.exp(theta[-1]); loc=reg.X@theta[:-1]-.5*sigma**2
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
            scatter = name in ('ln_s_rev','ln_sigma_ln_RM')
            vals = np.exp(samples[:, j]) if scatter else samples[:, j]
            lo, mid, hi = np.percentile(vals, [16, 50, 84])
            parameters.append(dict(model=model, parameter=name.replace("ln_s_rev", "s_rev").replace('ln_sigma_ln_RM','sigma_ln_RM'),
                                   median=mid, err_minus=mid-lo, err_plus=hi-mid,
                                   mle=float(np.exp(reg.mle[j]) if scatter else reg.mle[j])))
        rows.append(dict(model=model, N=reg.n, k=k, log_likelihood_max=logl,
                         chi2_mle=discrepancy_mle if forward else reg.sse / np.exp(2 * reg.mle[-1]), dof=None if forward else reg.n-k,
                         AIC=2*k-2*logl, BIC=k*np.log(reg.n)-2*logl,
                         converged=passed, sample_sha256=meta["sample_sha256"]))
    comparison = pd.DataFrame(rows)
    if comparison.sample_sha256.nunique() != 1:
        raise ValueError("Models use different samples")
    for key in ("AIC", "BIC", "chi2_mle"):
        comparison["delta_" + key + "_vs_noz"] = comparison[key] - comparison.iloc[0][key]
    params = pd.DataFrame(parameters)
    comparison.to_csv(Path(root) / "model_comparison.csv", index=False)
    params.to_csv(Path(root) / "parameters.csv", index=False)
    if forward:
        log_rows=[]
        for model in ('noz','z'):
            directory=Path(root)/model
            meta=json.loads((directory/'run.json').read_text())
            with np.load(directory/'posterior.npz') as archive:
                transformed=archive['samples'].copy()
            transformed[:,0] -= .5*np.exp(2*transformed[:,-1])
            transformed[:,-1] = np.exp(transformed[:,-1])
            for j,name in enumerate(['A','B']+(['C'] if model=='z' else [])+['sigma_ln_RM']):
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
    p.add_argument("--ppc-draws",type=int,default=2000)
    p.add_argument('--environment-burn',type=int,help='Separate post-hoc burn analysis of a stopped environment fit')
    a = p.parse_args()
    if a.environment_burn is not None:
        summarize_environment_burn(a.fits,a.environment_burn)
    elif a.cells:
        summarize_cells(a.fits,a.ppc_draws)
    else:
        summarize(a.fits)
