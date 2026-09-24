"""Data-only spectroscopic offsets and conditional distributions; no fitted residuals."""

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
    COLORS, TITLES, Z_BINS, common_sample, validate_bins, style, legend, save,
)


def plot_distributions(table, rm_bins=(20, 30, 50, 100), residual=True, min_count=20, hist_bins=18):
    """Rows are weighting stages, columns are RM bins; colors distinguish redshift.

    residual=True: log10(lambda_spec/lambda_RM), relative to equality.
    residual=False: raw lambda_spec distributions, as in the archived notebook.
    Histograms in the latter mode are densities per unit LINEAR richness.
    """
    table = common_sample(table)
    edges = validate_bins(rm_bins)
    if min_count < 1 or hist_bins < 1:
        raise ValueError("min_count and hist_bins must be positive")
    rm, z = np.asarray(table["LAMBDA"]), np.asarray(table["Z_SPEC_central"])
    values = [np.log10(np.asarray(table[k])) - np.log10(rm) if residual else np.asarray(table[k])
              for k in STAGES]
    fig, axes = plt.subplots(3, len(edges) - 1, figsize=(5 * (len(edges) - 1), 9),
                             squeeze=False, sharex="col", sharey="col")
    for j, (rlo, rhi) in enumerate(zip(edges[:-1], edges[1:])):
        in_rm = (rm >= rlo) & (rm < rhi)
        combined = np.concatenate([v[in_rm] for v in values])
        bounds = np.histogram_bin_edges(combined, bins=hist_bins) if len(combined) else None
        for i, (key, title) in enumerate(zip(STAGES, TITLES)):
            ax = axes[i, j]
            shown = False
            for color, (lo, hi) in zip(COLORS, Z_BINS):
                use = in_rm & (z >= lo) & (z < hi)
                if use.sum() < min_count:
                    continue
                ax.hist(values[i][use], bins=bounds, density=True, histtype="stepfilled",
                        alpha=0.18, color=color, edgecolor=color)
                ax.hist(values[i][use], bins=bounds, density=True, histtype="step", color=color, lw=1.3)
                shown = True
            if residual:
                ax.axvline(0, color="black", lw=1, ls="--")
            if not shown:
                ax.text(0.5, 0.5, f"No redshift bin with N >= {min_count}",
                        transform=ax.transAxes, ha="center", fontsize=9)
            if i == 0:
                ax.set_title(rf"${rlo:g}\leq\lambda_{{\rm RM}}<{rhi:g}$")
            if j == 0:
                ax.set_ylabel(title + ("\nDensity [dex$^{-1}$]" if residual else "\nProbability density"))
            if i == 2:
                ax.set_xlabel(r"$\log_{10}(\lambda_{\rm spec}/\lambda_{\rm RM})$ [dex]"
                              if residual else r"$\lambda_{\rm spec}$")
            style(ax)
    legend(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "plots/richness_weighting")
    parser.add_argument("--min-count", type=int, default=20)
    args = parser.parse_args()
    table, _ = load_stages(args.data)
    for residual, name in [(True, "spectroscopic_richness_offsets"), (False, "spectroscopic_richness_distributions")]:
        fig = plot_distributions(table, residual=residual, min_count=args.min_count)
        save(fig, args.output, name)
        plt.close(fig)
    print(f"Saved offset and conditional-distribution plots to {args.output}")


if __name__ == "__main__":
    main()
