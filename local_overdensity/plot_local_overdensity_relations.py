"""
Plot local-overdensity statistics against lambda_true and redMaPPer richness.

This script is plotting-only. It reads the saved FITS table produced by the
DR9 sweep workflow after lambda_true/lambda_spec has been joined, applies the
same coverage mask, and writes non-redshift-binned plus redshift-binned plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.stats import binned_statistic, pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "local_overdensity" / "dr9_one_file_outputs"
INPUT_TABLE = OUTPUT_DIR / "rm_dr9_local_overdensity_sweep_with_lambda_spec.fits"
PLOT_DIR = OUTPUT_DIR / "local_overdensity_plots"

COVERAGE_MIN = 0.8
NBINS = 8
Z_BINS = [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]
Z_COLORS = ["crimson", "darkorange", "royalblue"]


def col_float(table: Table, col: str) -> np.ndarray:
    arr = np.ma.asarray(table[col], dtype=float)
    return np.ma.filled(arr, np.nan)


def safe_divide(numerator, denominator, min_denominator=1.0e-12):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full_like(numerator, np.nan, dtype=float)
    good = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > min_denominator)
    )
    out[good] = numerator[good] / denominator[good]
    return out


def add_binned_mean(ax, x, y, nbins=NBINS, color="crimson", label="binned mean"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x = x[mask]
    y = y[mask]
    if len(x) < nbins:
        return

    bins = np.logspace(np.log10(np.nanmin(x)), np.log10(np.nanmax(x)), nbins + 1)
    centers = np.sqrt(bins[:-1] * bins[1:])
    mean, _, _ = binned_statistic(x, y, statistic="mean", bins=bins)
    std, _, _ = binned_statistic(x, y, statistic="std", bins=bins)
    count, _, _ = binned_statistic(x, y, statistic="count", bins=bins)
    good = count > 3
    sem = std / np.sqrt(np.clip(count, 1, None))

    ax.errorbar(
        centers[good],
        mean[good],
        yerr=sem[good],
        fmt="o",
        color=color,
        ecolor=color,
        capsize=3,
        label=label,
        zorder=4,
    )


def correlation_summary(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    if np.count_nonzero(mask) < 3:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    pearson_r, pearson_p = pearsonr(x[mask], y[mask])
    pearson_logx_r, pearson_logx_p = pearsonr(np.log10(x[mask]), y[mask])
    spearman_r, spearman_p = spearmanr(x[mask], y[mask])
    return (
        pearson_r,
        pearson_p,
        pearson_logx_r,
        pearson_logx_p,
        spearman_r,
        spearman_p,
        np.count_nonzero(mask),
    )


def prepare_table():
    table = Table.read(INPUT_TABLE)

    if "lambda_true" not in table.colnames and "lambda_spec" in table.colnames:
        table["lambda_true"] = table["lambda_spec"]
    if "lambda_spec" not in table.colnames and "lambda_true" in table.colnames:
        table["lambda_spec"] = table["lambda_true"]

    coverage_signal = col_float(table, "coverage_signal_sweep")
    coverage_bg = col_float(table, "coverage_bg_sweep")
    n_signal = col_float(table, "Ngal_signal_DR9_annulus")
    n_bg = col_float(table, "Ngal_bg_DR9_annulus")
    area_signal = col_float(table, "area_signal_deg2")
    covered_area_signal = col_float(table, "covered_area_signal_deg2")
    covered_area_bg = col_float(table, "covered_area_bg_deg2")

    table["Ngal_signal_DR9_annulus_covcorr"] = safe_divide(
        n_signal, coverage_signal, min_denominator=1.0e-6
    )
    table["Ngal_bg_DR9_annulus_covcorr"] = safe_divide(
        n_bg, coverage_bg, min_denominator=1.0e-6
    )

    sigma_signal = safe_divide(n_signal, covered_area_signal)
    sigma_bg = safe_divide(n_bg, covered_area_bg)
    sigma_excess = sigma_signal - sigma_bg

    table["Sigma_signal_covcorr_recomputed"] = sigma_signal
    table["Sigma_bg_covcorr_recomputed"] = sigma_bg
    table["Sigma_excess_local_recomputed"] = sigma_excess
    table["Nexcess_local_recomputed"] = sigma_excess * area_signal

    base_mask = (
        np.isfinite(col_float(table, "lambda_true"))
        & (col_float(table, "lambda_true") > 0)
        & np.isfinite(col_float(table, "LAMBDA"))
        & (col_float(table, "LAMBDA") > 0)
        & np.isfinite(coverage_signal)
        & np.isfinite(coverage_bg)
        & (coverage_signal > COVERAGE_MIN)
        & (coverage_bg > COVERAGE_MIN)
    )

    return table, base_mask


Y_COLUMNS = [
    ("Ngal_signal_DR9_annulus_covcorr", r"$N_{\rm signal}/f_{\rm cov}$"),
    ("Ngal_bg_DR9_annulus_covcorr", r"$N_{\rm bg}/f_{\rm cov}$"),
    ("Sigma_signal_covcorr_recomputed", r"$\Sigma_{\rm signal}$"),
    ("Sigma_bg_covcorr_recomputed", r"$\Sigma_{\rm bg}$"),
    ("Sigma_excess_local_recomputed", r"$\Sigma_{\rm excess,local}$"),
    ("Nexcess_local_recomputed", r"$N_{\rm excess,local}$"),
]


def make_redshift_binned_grid(table, base_mask, x_col, x_label, filename_prefix):
    x = col_float(table, x_col)
    z = col_float(table, "Z_SPEC_x")
    rows = []

    for y_col, y_label in Y_COLUMNS:
        y = col_float(table, y_col)
        fig, axes = plt.subplots(1, len(Z_BINS), figsize=(16, 4.7), sharex=True)

        for ax, (zlo, zhi), color in zip(axes, Z_BINS, Z_COLORS):
            zmask = (z >= zlo) & (z < zhi)
            mask = base_mask & zmask & np.isfinite(x) & (x > 0) & np.isfinite(y)

            ax.scatter(x[mask], y[mask], s=14, alpha=0.45, color=color, edgecolor="none")
            add_binned_mean(ax, x[mask], y[mask], nbins=NBINS, color="black")

            if "excess" in y_col.lower():
                ax.axhline(0.0, color="black", lw=1.0, alpha=0.45)

            ax.set_xscale("log")
            ax.set_xlabel(x_label)
            ax.set_title(rf"${zlo}<z<{zhi}$, $N={np.count_nonzero(mask)}$")
            ax.legend(frameon=False, fontsize=9)

            pr, pp, plr, plp, sr, sp, n = correlation_summary(x[mask], y[mask])
            ax.text(
                0.04,
                0.96,
                rf"$r_s={sr:.2f}$" + "\n" + rf"$p={sp:.1e}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            )
            rows.append((x_col, y_col, zlo, zhi, n, pr, pp, plr, plp, sr, sp))

        axes[0].set_ylabel(y_label)
        fig.suptitle(
            f"{y_label} versus {x_label} in redshift bins; coverage > {COVERAGE_MIN}",
            fontsize=15,
        )
        fig.savefig(PLOT_DIR / f"{filename_prefix}_{y_col}_z_binned.png", dpi=180)
        plt.close(fig)

    return Table(
        rows=rows,
        names=[
            "x_quantity",
            "y_quantity",
            "z_low",
            "z_high",
            "N",
            "pearson_r_x",
            "pearson_p_x",
            "pearson_r_logx",
            "pearson_p_logx",
            "spearman_r_x",
            "spearman_p_x",
        ],
    )


def make_redshift_binned_focused_plot(table, base_mask, x_col, x_label, filename_prefix):
    x = col_float(table, x_col)
    y = col_float(table, "Nexcess_local_recomputed")
    z = col_float(table, "Z_SPEC_x")

    fig, axes = plt.subplots(1, len(Z_BINS), figsize=(16, 4.7), sharex=True, sharey=True)

    for ax, (zlo, zhi), color in zip(axes, Z_BINS, Z_COLORS):
        zmask = (z >= zlo) & (z < zhi)
        mask = base_mask & zmask & np.isfinite(x) & (x > 0) & np.isfinite(y)

        ax.scatter(x[mask], y[mask], s=18, alpha=0.55, color=color, edgecolor="none")
        add_binned_mean(ax, x[mask], y[mask], nbins=NBINS, color="black")
        ax.axhline(0.0, color="black", lw=1.0, alpha=0.55)
        ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_title(rf"${zlo}<z<{zhi}$, $N={np.count_nonzero(mask)}$")

        _, _, _, _, sr, sp, _ = correlation_summary(x[mask], y[mask])
        ax.text(
            0.04,
            0.96,
            rf"$r_s={sr:.2f}$" + "\n" + rf"$p={sp:.1e}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
        )
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel(r"$N_{\rm excess,local}$")
    fig.suptitle(
        rf"$N_{{\rm excess,local}}$ versus {x_label} in redshift bins; coverage $>{COVERAGE_MIN}$",
        fontsize=15,
    )
    fig.savefig(PLOT_DIR / f"{filename_prefix}_Nexcess_local_z_binned.png", dpi=180)
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    table, base_mask = prepare_table()
    print(f"Rows: {len(table):,}")
    print(f"Rows after richness + coverage cut: {np.count_nonzero(base_mask):,}")

    corr_true = make_redshift_binned_grid(
        table, base_mask, "lambda_true", r"$\lambda_{\rm true}$", "local_overdensity_vs_lambda_true"
    )
    make_redshift_binned_focused_plot(
        table, base_mask, "lambda_true", r"$\lambda_{\rm true}$", "local_overdensity_vs_lambda_true"
    )
    corr_true.write(
        PLOT_DIR / "local_overdensity_vs_lambda_true_redshift_binned_correlations.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )

    corr_rm = make_redshift_binned_grid(
        table, base_mask, "LAMBDA", r"$\lambda_{\rm RM}$", "local_overdensity_vs_lambda_RM"
    )
    make_redshift_binned_focused_plot(
        table, base_mask, "LAMBDA", r"$\lambda_{\rm RM}$", "local_overdensity_vs_lambda_RM"
    )
    corr_rm.write(
        PLOT_DIR / "local_overdensity_vs_lambda_RM_redshift_binned_correlations.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )

    print(f"Wrote plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
