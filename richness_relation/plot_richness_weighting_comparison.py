"""Three-stage richness scatter and geometric means; no regression or MCMC."""

import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from richness_relation.prepare_richness_weighting import DEFAULT_OUTPUT, STAGES, load_stages
from richness_relation.weighting_plot_utils import (
    COLORS, MARKERS, TITLES, RM_BINS, Z_BINS, common_sample, binned_log_mean, style, legend, save,
)


def plot_relation(table, rm_bins=RM_BINS, min_count=20, xlim=(20, 100), ylim=None):
    table = common_sample(table)
    rm, z = np.asarray(table["LAMBDA"]), np.asarray(table["Z_SPEC_central"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    for ax, key, title in zip(axes, STAGES, TITLES):
        spec = np.asarray(table[key])
        for (lo, hi), color, marker in zip(Z_BINS, COLORS, MARKERS):
            use = (z >= lo) & (z < hi)
            ax.scatter(rm[use], spec[use], s=10, alpha=0.18, color=color,
                       edgecolors="none", rasterized=True)
            points = binned_log_mean(rm[use], spec[use], rm_bins, min_count)
            if len(points):
                x, mean, sem, _ = points.T
                y = 10 ** mean
                ax.errorbar(x, y, yerr=[y - 10 ** (mean - sem), 10 ** (mean + sem) - y],
                            fmt=marker, ls="none", ms=5, capsize=3, elinewidth=1.3,
                            color=color, zorder=4)
        ax.set(xscale="log", yscale="log", xlabel=r"$\lambda_{\rm RM}$", title=title, xlim=xlim)
        style(ax)
    axes[0].set_ylabel(r"$\lambda_{\rm spec}$")
    if ylim is None:
        visible = (rm >= xlim[0]) & (rm <= xlim[1])
        if np.any(visible):
            values = np.concatenate([np.asarray(table[key])[visible] for key in STAGES])
            ylim = (values.min() * 0.8, values.max() * 1.25)
    if ylim is not None:
        axes[0].set_ylim(ylim)
    legend(fig)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "plots/richness_weighting")
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()
    table, _ = load_stages(args.data)
    fig = plot_relation(table, min_count=args.min_count)
    save(fig, args.output, "richness_weighting_comparison")
    plt.close(fig)
    print(f"Saved richness-relation plots to {args.output}")


if __name__ == "__main__":
    main()
