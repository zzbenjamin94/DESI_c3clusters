"""Plot saved DR9 environments versus richness; no measurement recomputation.

Cluster-pair bootstrap intervals are exploratory: they ignore spatial dependence.
SEM bars describe binned means, not the scatter of individual clusters.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from astropy.table import Table
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
Z_EDGES = (0.10, 0.18, 0.24, 0.35)
COLORS = ('#176b93', '#b33b3b', '#27835e')


def floats(table, name):
    return np.ma.asarray(table[name], dtype=float).filled(np.nan)


def correlation(x, y, draws, rng):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    rho, p = spearmanr(x, y)
    boot = []
    for _ in range(draws):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) > 0 and np.ptp(y[idx]) > 0:
            boot.append(spearmanr(x[idx], y[idx]).statistic)
    lo, hi = np.percentile(boot, [16, 84]) if boot else (np.nan, np.nan)
    return rho, p, lo, hi, len(boot)


def binned_means(x, y, max_bins, min_count):
    nbin = min(max_bins, len(x) // min_count)
    if nbin == 0:
        return []
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbin + 1)))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        keep = (x >= lo) & ((x <= hi) if i == len(edges)-2 else (x < hi))
        n = int(keep.sum())
        if n >= min_count:
            rows.append(dict(bin_lo=lo, bin_hi=hi, N=n, x_mean=x[keep].mean(),
                             y_mean=y[keep].mean(), sem=y[keep].std(ddof=1)/np.sqrt(n)))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=ROOT/'local_overdensity/dr9_outputs/section4/dr9/environment.fits')
    p.add_argument('--output', type=Path, default=ROOT/'plots/environment_richness')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-bins', type=int, default=7)
    p.add_argument('--min-count', type=int, default=20)
    p.add_argument('--coverage-min', type=float, default=None,
                   help='Optional threshold for BOTH signal and background; default applies no new coverage cut')
    a = p.parse_args()
    if a.bootstrap < 1 or a.max_bins < 1 or a.min_count < 2:
        p.error('Require bootstrap >= 1, max-bins >= 1 and min-count >= 2')
    t = Table.read(a.input)
    if len(np.unique(t['ID'])) != len(t):
        raise ValueError('Expected one row per cluster; duplicate IDs found')
    # This stage column is the catalog builder's source for noproj_weighted.
    spec_col = 'lambda_spec_noproj_weighted' if 'lambda_spec_noproj_weighted' in t.colnames else 'lambda_spec_geo_comp_lf'
    if spec_col not in t.colnames:
        raise KeyError('Missing fully weighted continuum-subtracted richness')
    z = floats(t, 'Z_SPEC_central')
    rng = np.random.default_rng(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'serif', 'mathtext.fontset':'stix', 'font.size':12})
    records, bin_records = [], []
    for filt in ('box', 'gaussian'):
        for metric in ('excess', 'E_local'):
            prefix = f'dr9_{filt}'
            ycol = f'{prefix}_{metric}'
            y = floats(t, ycol)
            flag = f'{prefix}_valid' if metric == 'excess' else f'{prefix}_E_local_valid'
            valid = np.ma.asarray(t[flag], dtype=bool).filled(False) & np.isfinite(z) & np.isfinite(y)
            if a.coverage_min is not None:
                valid &= (floats(t,f'{prefix}_coverage') >= a.coverage_min) & (floats(t,'dr9_background_coverage') >= a.coverage_min)
            fig, axes = plt.subplots(1,2,figsize=(13,5),layout='constrained')
            for ax, xcol, label in zip(axes, ('LAMBDA',spec_col), (r'$\lambda_{\rm RM}$',r'$\lambda_{\rm spec}$')):
                x = floats(t,xcol)
                handles = []
                for color, lo, hi in zip(COLORS,Z_EDGES[:-1],Z_EDGES[1:]):
                    mask = valid & np.isfinite(x) & (x > 0) & (z >= lo) & (z < hi)
                    xx, yy = x[mask], y[mask]
                    rho, pv, lower, upper, successful = correlation(xx,yy,a.bootstrap,rng)
                    records.append(dict(filter=filt,metric=metric,richness_col=xcol,zlo=lo,zhi=hi,N=len(xx),
                                        spearman_r=rho,p_asymptotic=pv,bootstrap_p16=lower,bootstrap_p84=upper,
                                        bootstrap_valid=successful))
                    print(f'{filt} {metric} {xcol} {lo}-{hi}: N={len(xx)}, rho={rho:.3f}, 68% interval=[{lower:.3f}, {upper:.3f}]',flush=True)
                    ax.scatter(xx,yy,s=7,color=color,alpha=.15,edgecolors='none',rasterized=True)
                    bins = binned_means(xx,yy,a.max_bins,a.min_count)
                    for row in bins:
                        bin_records.append(dict(filter=filt,metric=metric,richness_col=xcol,zlo=lo,zhi=hi,**row))
                    if bins:
                        ax.errorbar([b['x_mean'] for b in bins],[b['y_mean'] for b in bins],
                                    yerr=[b['sem'] for b in bins],fmt='o',mfc='white',mec=color,
                                    ecolor=color,ms=6,capsize=2,lw=1.2,zorder=5)
                    text = rf'${lo:.2f}\leq z<{hi:.2f}$: $r_s={rho:.2f}$ [{lower:.2f}, {upper:.2f}]'
                    handles.append(Line2D([],[],marker='o',linestyle='none',color=color,mfc='white',label=text))
                ax.set_xscale('log')
                ax.set_xlabel(label,fontsize=16)
                ax.set_ylabel(r'$\Sigma_{\rm excess}$ [$h^2\,\mathrm{Mpc}^{-2}$]' if metric=='excess' else r'$E_{\rm local}$',fontsize=16)
                ax.axhline(0,color='0.5',lw=.8)
                ax.grid(False)
                ax.minorticks_on()
                ax.tick_params(which='both',direction='in',top=True,right=True)
                ax.legend(handles=handles,loc='upper right',frameon=False,fontsize=9)
            for ext in ('png','pdf'):
                fig.savefig(a.output/f'dr9_{filt}_{metric}_richness.{ext}',dpi=200)
            plt.close(fig)
    pd.DataFrame(records).to_csv(a.output/'spearman_summary.csv',index=False)
    pd.DataFrame(bin_records).to_csv(a.output/'binned_means_sem.csv',index=False)
    (a.output/'settings.json').write_text(json.dumps(dict(input=str(a.input.resolve()),spec_column=spec_col,
        z_edges=Z_EDGES,bootstrap=a.bootstrap,seed=a.seed,min_count=a.min_count,max_bins=a.max_bins,
        coverage_min=a.coverage_min,errors='Linear-space SEM; cluster-pair bootstrap 16-84% intervals; spatial correlations ignored'),indent=2))
    print('Saved:',a.output)


if __name__ == '__main__':
    main()
