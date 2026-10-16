"""Compare saved global fits, then require explicit choices for conditional plots.

No fitting, chain modification, or automatic model selection is performed.
"""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from richness_relation.plot_conditional_richness import load_results, save, style
from richness_relation.prepare_conditional_richness import load_sample
from richness_relation.conditional_richness_models import selected_cdf, selected_logpdf


def pit_chi2(pit, bins=10):
    """Descriptive Pearson discrepancy, not a calibrated fitted-model test."""
    counts = np.histogram(pit, np.linspace(0, 1, bins + 1))[0]
    expected = len(pit) / bins
    return float(np.sum((counts - expected) ** 2 / expected))


def interval_text(q):
    return rf"${q['median']:.3g}_{{-{q['median']-q['p16']:.2g}}}^{{+{q['p84']-q['median']:.2g}}}$"


def report(data, results, output):
    rows, params = [], []
    for model in ("lognormal", "mixture"):
        pair = [results[f"{model}_combined_{mode}"] for mode in ("noz", "z")]
        for mode, r in zip(("noz", "z"), pair):
            pit = selected_cdf(r['mle_theta'], data['lambda_rm'], data['lambda_spec'], data['z'], r['model_config'])
            rows.append(dict(model=model, evolution=mode, N=len(pit),
                             chi2_PIT=pit_chi2(pit), AIC=r['AIC'], BIC=r['BIC'],
                             delta_AIC_z_minus_noz=pair[1]['AIC']-pair[0]['AIC'],
                             delta_BIC_z_minus_noz=pair[1]['BIC']-pair[0]['BIC'],
                             all_50tau_pass=all(d['length_exceeds_50_tau'] for d in r['chain_diagnostics'])))
            for name, q in r['posterior'].items():
                params.append(dict(model=model, evolution=mode, parameter=name,
                                   mle=r['mle_parameters'][name], **q))
        r = pair[1]
        c = r['samples'][:, r['parameter_names'].index('C')]
        print(model, 'C percentiles [2.5,16,50,84,97.5]:', np.percentile(c, [2.5,16,50,84,97.5]))
        print('Posterior P(C>0):', np.mean(c > 0))
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'model_comparison.csv', index=False)
    pd.DataFrame(params).to_csv(output / 'parameters.csv', index=False)
    lines = [r'\begin{table*}', r'\centering\small',
             r'\caption{Global conditional richness fits. Parameters are posterior medians with 16th--84th percentile intervals. $\Delta$ denotes redshift-dependent minus independent AIC or BIC; negative values favor evolution. $\chi^2_{\rm PIT}$ is a ten-bin probability-integral-transform discrepancy, not a mean-relation chi-square or a calibrated goodness-of-fit test.}',
             r'\label{tab:richness-redshift-comparison}',
             r'\begin{tabular}{llrrrr}', r'\hline',
             r'Model & Evolution & $\chi^2_{\rm PIT}$ & AIC & $\Delta$AIC & $\Delta$BIC \\', r'\hline']
    for row in rows:
        lines.append(f"{row['model']} & {row['evolution']} & {row['chi2_PIT']:.2f} & {row['AIC']:.2f} & {row['delta_AIC_z_minus_noz']:.2f} & {row['delta_BIC_z_minus_noz']:.2f}" + r' \\')
    lines += [r'\hline\end{tabular}', r'\par\medskip', r'\begin{tabular}{lll}', r'\hline', r'Model & Parameter & Posterior \\', r'\hline']
    for model in ('lognormal', 'mixture'):
        for mode in ('noz', 'z'):
            r = results[f'{model}_combined_{mode}']
            for name, q in r['posterior'].items():
                if name in ('sigma_log10_lambda', 'tau_inverse', 'mean_projection_boost'):
                    continue
                label = name.replace('_', r'\_')
                lines.append(f'{model} ({mode}) & {label} & {interval_text(q)}' + r' \\')
    lines += [r'\hline\end{tabular}', r'\end{table*}']
    (output / 'redshift_comparison.tex').write_text('\n'.join(lines))
    print(frame.to_string(index=False))
    print('DECISION REQUIRED: inspect convergence (including independent-ensemble R-hat), C posterior and AIC/BIC. No automatic choice made.')
    print('No classical p-value is reported: fitting the same data invalidates a naive PIT chi-square reference distribution.')


def plots(data, audit, results, choices, output):
    edges = np.unique(np.quantile(data['lambda_spec'], np.linspace(0, 1, 5)))
    if len(edges) != 5:
        raise ValueError('Tied quartiles: choose explicit richness boundaries before plotting.')
    zedges = audit['z_bin_edges']
    if len(zedges) != 4:
        raise ValueError('Expected three redshift bins.')
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 11})
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), sharex=True, sharey=True)
    rf, ra = plt.subplots(3, 4, figsize=(13, 9), sharex=True, sharey=True)
    bounds = np.geomspace(max(audit['rm_min'], 1e-3), data['lambda_rm'].max()*1.05, 16)
    grid = np.geomspace(bounds[0], bounds[-1], 250)
    colors = ['#c58100', '#d96300', '#dd3300', '#b62424']
    cell_counts = []
    for iz, (zl, zh) in enumerate(zip(zedges[:-1], zedges[1:])):
        for ix, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (data['z'] >= zl) & (data['z'] < zh) & (data['lambda_spec'] >= lo) & ((data['lambda_spec'] <= hi) if ix == 3 else (data['lambda_spec'] < hi))
            rm, spec, z = [data[k][mask] for k in ('lambda_rm', 'lambda_spec', 'z')]
            ax, rx = axes[iz, ix], ra[iz, ix]
            cell_counts.append(dict(zlo=zl,zhi=zh,spec_lo=lo,spec_hi=hi,N=len(rm)))
            label = rf'${lo:.1f}\leq\lambda_{{\rm spec}}\leq {hi:.1f}$'+'\n'+rf'${zl:.2f}\leq z<{zh:.2f}$'
            for a in (ax, rx):
                style(a)
                a.minorticks_on()
                a.tick_params(which='both', direction='in', top=True, right=True)
                a.text(.04, .96, label, transform=a.transAxes, va='top', fontsize=9)
            if len(rm) < 20:
                ax.text(.5,.5,'Insufficient sample',transform=ax.transAxes,ha='center')
                continue
            n = np.histogram(rm, bounds)[0]
            scale = len(rm)*np.diff(bounds)
            low = np.where(n > 0, chi2.ppf(.15865, 2*n)/2, 0)
            high = chi2.ppf(.84135, 2*(n+1))/2
            ax.errorbar(np.sqrt(bounds[:-1]*bounds[1:]), n/scale, yerr=np.array([n-low,high-n])/scale, fmt='o', mfc='none', color=colors[ix], ms=5, lw=.8, label='Data')
            for model, color, ls in [('lognormal','#245c91','--'),('mixture','#444444','-')]:
                r = results[f'{model}_combined_{choices[model]}']
                curve = np.mean(np.exp(selected_logpdf(r['mle_theta'], grid[:,None], spec[None,:], z[None,:], r['model_config'])),axis=1)
                ax.plot(grid,curve,color=color,ls=ls,lw=2.4,label=model)
                pit = selected_cdf(r['mle_theta'],rm,spec,z,r['model_config'])
                score = norm.ppf(np.clip(pit,1e-10,1-1e-10))
                rb = np.linspace(-4,4,17)
                rn = np.histogram(score,rb)[0]
                rx.stairs(rn/(len(rm)*np.diff(rb)),rb,color=color,ls=ls,lw=1.8,label=model)
            rx.plot(np.linspace(-4,4,100),norm.pdf(np.linspace(-4,4,100)),color='0.7',lw=2,label='Standard normal')
            ax.set_xscale('log')
            ax.set_xlim(bounds[0], bounds[-1])
            rx.set_xlim(-4, 4)
    for panels, f, xlabel, ylabel, name in [(axes,fig,r'$\lambda_{\rm RM}$',r'$p(\lambda_{\rm RM}\mid\mathrm{bin})$','conditional_distributions'),(ra,rf,r'$\Phi^{-1}(F)$','Probability density','binned_quantile_residuals')]:
        for a in panels[-1]: a.set_xlabel(xlabel)
        for a in panels[:,0]: a.set_ylabel(ylabel)
        handles, labels = [], []
        for a in panels.flat:
            handles, labels = a.get_legend_handles_labels()
            if handles: break
        f.legend(handles, labels,loc='lower center',ncol=3,frameon=False)
        f.tight_layout(rect=(0,.05,1,1))
        save(f,output,name)
    pd.DataFrame(cell_counts).to_csv(output/'bin_counts.csv',index=False)
    (output/'model_choices.json').write_text(json.dumps(dict(choices=choices,richness_edges=edges.tolist(),z_edges=list(zedges)),indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--fits',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lognormal-choice',choices=['noz','z'])
    p.add_argument('--mixture-choice',choices=['noz','z'])
    a=p.parse_args()
    data,audit=load_sample(a.data)
    results={}
    for root in a.fits:
        found=load_results(root,a.seed,audit)
        for task in results.keys() & found.keys():
            old, new = results[task], found[task]
            if (old['samples'] is not None or new['samples'] is None
                    or new.get('finalization', {}).get('source') != str(old['directory'].resolve())
                    or old['config'] != new['config']):
                raise ValueError(f'Duplicate task {task}; only a verified finalized replacement of incomplete outputs is allowed')
        results.update(found)
    for model in ('lognormal','mixture'):
        for mode in ('noz','z'):
            r=results[f'{model}_combined_{mode}']
            if r['samples'] is None:
                raise ValueError(f"Completed posterior outputs required: {r['directory']}")
    a.output.mkdir(parents=True,exist_ok=False)
    report(data,results,a.output)
    if a.lognormal_choice and a.mixture_choice:
        plots(data,audit,results,dict(lognormal=a.lognormal_choice,mixture=a.mixture_choice),a.output)
    elif a.lognormal_choice or a.mixture_choice:
        raise ValueError('Provide BOTH model choices, or neither for report-only mode')


if __name__ == '__main__':
    main()
