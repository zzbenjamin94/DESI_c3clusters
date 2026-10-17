"""Publication plot from the saved global log-normal posterior with C=0.

No fitting. Binned geometric means use SEM in natural-log richness.
"""
import argparse
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from richness_relation.prepare_conditional_richness import load_sample
from richness_relation.plot_conditional_richness import load_results, binned_mean_log, save, style


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'catalogs/conditional_richness/overnight_sample.npz')
    p.add_argument('--fits', type=Path, default=ROOT/'catalogs/conditional_richness/lognormal_fits')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bins', type=int, default=7)
    p.add_argument('--min-count', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.bins < 1 or args.min_count < 2:
        p.error('bins must be positive and min-count at least two')
    data, audit = load_sample(args.data)
    fit = load_results(args.fits, args.seed, audit)['lognormal_combined_noz']
    if fit['samples'] is None:
        raise ValueError('Completed lognormal_combined_noz posterior required')
    cfg = fit['model_config']
    if cfg.evolution or cfg.model != 'lognormal':
        raise ValueError('This plot requires the no-evolution log-normal fit')
    if args.output.exists():
        raise FileExistsError('Choose a fresh output directory')
    x, y = data['lambda_rm'], data['lambda_spec']
    in_x_range = (x >= 20) & (x <= 100)
    points = binned_mean_log(x[in_x_range], y[in_x_range], args.bins, args.min_count)
    grid = np.geomspace(20, 600, 250)
    rng = np.random.default_rng(args.seed)
    samples = fit['samples']
    samples = samples[rng.choice(len(samples), min(2000, len(samples)), replace=False)]
    names = fit['parameter_names']
    mu = samples[:, names.index('ln_A'), None] + samples[:, names.index('B'), None]*np.log(grid/cfg.lambda_piv)
    q = np.percentile(np.exp(mu), [16, 50, 84], axis=0)
    # Estimate density in the coordinates displayed on the logarithmic axes.
    log_points = np.vstack([np.log10(x), np.log10(y)])
    density = gaussian_kde(log_points)(log_points)
    order = np.argsort(density)
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'stix', 'font.size': 12})
    fig, ax = plt.subplots(figsize=(7, 5.6))
    sc = ax.scatter(x[order], y[order], s=3, alpha=.6, c=density[order],
                    cmap='viridis', edgecolors='none', rasterized=True)
    # Swap the original conditional relation's coordinates, not its conditioning.
    ax.fill_betweenx(grid, q[0], q[2], color='#cf573b', alpha=.35, linewidth=0,
                    label='68% posterior interval', zorder=3)
    ax.plot(q[1], grid, color='#982d20', lw=1.0, label='Log-linear model', zorder=4)
    if len(points):
        ax.errorbar(points[:,0], points[:,1], yerr=points[:,2:].T,
                    fmt='o', mfc='white', mec='black', ecolor='black',
                    ms=3, mew=.8, capsize=3, lw=1.2, label='Binned geometric mean with SEM', zorder=5)
    ax.set_xscale('log'); ax.set_yscale('log'); style(ax)
    ax.set_xlabel(r'$\lambda_{\rm RM}$', fontsize=18)
    ax.set_ylabel(r'$\lambda_{\rm spec}$', fontsize=18)
    ax.set_xlim(20, 100)
    ax.set_ylim(20, 600)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]))
        axis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
        axis.set_minor_formatter(NullFormatter())
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    cbar = fig.colorbar(sc, ax=ax, pad=.025, fraction=.045)
    cbar.set_label(r'Point density [dex$^{-2}$]')
    fig.tight_layout()
    save(fig, args.output, 'global_richness_relation_noz')
    np.savetxt(args.output/'binned_geometric_means.csv', points, delimiter=',',
               header='lambda_RM_geomean,lambda_spec_geomean,SEM_lower,SEM_upper', comments='')
    np.savetxt(args.output/'posterior_relation.csv', np.column_stack([grid,q.T]), delimiter=',',
               header='lambda_spec,core_p16,core_p50,core_p84', comments='')
    print('Saved', args.output)
    print('Bands are 68% pointwise posterior intervals, not population scatter.')
    print('This uses the existing constant-scatter global fit; no new variable-scatter fit is performed.')
    print('Saved 50-tau checks:', [d['length_exceeds_50_tau'] for d in fit['chain_diagnostics']])


if __name__ == '__main__':
    main()
