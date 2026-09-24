"""Shared selections and visual conventions for data-only weight comparisons."""

from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from richness_relation.prepare_richness_weighting import STAGES
from tools.richness_selection import BCG_Z_BIN_EDGES

COLORS = ("crimson", "darkorange", "royalblue")
MARKERS = ("o", "s", "^")
TITLES = ("No weights", "Geometry + completeness", "Geometry + completeness + LF")
RM_BINS = np.geomspace(20, 100, 9)
Z_BINS = tuple(zip(BCG_Z_BIN_EDGES[:-1], BCG_Z_BIN_EDGES[1:]))


def common_sample(table):
    rm, z = np.asarray(table["LAMBDA"]), np.asarray(table["Z_SPEC_central"])
    mask = np.isfinite(rm) & (rm >= 20) & np.isfinite(z) & (z >= 0.1) & (z < 0.35)
    for key in STAGES:
        values = np.asarray(table[key])
        mask &= np.isfinite(values) & (values > 0)
    print(f"Common positive sample: {mask.sum()}/{len(table)} clusters; "
          "nonpositive values are excluded, never floored.")
    if not np.any(mask):
        raise ValueError("No clusters pass the common positive three-stage plotting selection")
    return table[mask]


def validate_bins(edges):
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Bin edges must be finite and strictly increasing")
    return edges


def binned_log_mean(rm, spec, edges=RM_BINS, min_count=20):
    edges = validate_bins(edges)
    if min_count < 2:
        raise ValueError("SEM requires min_count >= 2")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        use = (rm >= lo) & (rm < hi)
        if use.sum() < min_count:
            continue
        log_spec = np.log10(spec[use])
        rows.append((10 ** np.log10(rm[use]).mean(), log_spec.mean(),
                     log_spec.std(ddof=1) / np.sqrt(use.sum()), use.sum()))
    return np.asarray(rows).reshape(-1, 4)


def style(ax):
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def legend(fig):
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=5,
                      label=rf"${lo:.2f}\leq z<{hi:.2f}$")
               for c, m, (lo, hi) in zip(COLORS, MARKERS, Z_BINS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)


def save(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
