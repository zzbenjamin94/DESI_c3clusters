"""
Diagnostics for DR9 local-overdensity coverage and zero-count failures.

This script does not regenerate DR9 counts. It reads the saved cluster table
from the sweep workflow and writes summary tables/plots that answer:

1. Which clusters fail the signal/background coverage cuts?
2. Are failures localized on the sky, in redshift, or in north/south footprint?
3. Do any high-coverage clusters have zero signal/background galaxy counts?
4. Are zero counts plausible given covered area and typical measured density?

Run from anywhere inside the repo/workspace:

    python DESI_c3clusters/local_overdensity/diagnose_dr9_overdensity_coverage.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "local_overdensity" / "dr9_one_file_outputs"
INPUT_TABLE = OUTPUT_DIR / "rm_dr9_local_overdensity_sweep_with_lambda_spec.fits"
DIAG_DIR = OUTPUT_DIR / "coverage_diagnostics"

COVERAGE_MIN = 0.8


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


def describe_mask(name: str, mask: np.ndarray, total: int) -> tuple[str, int, float]:
    n = int(np.count_nonzero(mask))
    frac = n / total if total else np.nan
    print(f"{name:42s}: {n:6d} / {total:6d} = {frac:7.3f}")
    return name, n, frac


def save_sky_plot(table, masks):
    ra = col_float(table, "RA_x")
    dec = col_float(table, "DEC_x")
    coverage_signal = col_float(table, "coverage_signal_sweep")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    sc = axes[0].scatter(
        ra,
        dec,
        c=coverage_signal,
        s=8,
        alpha=0.65,
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolor="none",
    )
    axes[0].set_title("Cluster centers colored by signal coverage")
    axes[0].set_xlabel("RA [deg]")
    axes[0].set_ylabel("Dec [deg]")
    fig.colorbar(sc, ax=axes[0], label="coverage_signal_sweep")

    axes[1].scatter(ra, dec, s=6, color="lightgray", alpha=0.25, label="all")
    axes[1].scatter(
        ra[masks["coverage_fail"]],
        dec[masks["coverage_fail"]],
        s=10,
        color="crimson",
        alpha=0.75,
        label="coverage fail",
    )
    axes[1].scatter(
        ra[masks["zero_signal_after_cut"]],
        dec[masks["zero_signal_after_cut"]],
        s=18,
        facecolors="none",
        edgecolors="black",
        linewidths=0.7,
        label="zero signal after cut",
    )
    axes[1].scatter(
        ra[masks["zero_bg_after_cut"]],
        dec[masks["zero_bg_after_cut"]],
        s=18,
        facecolors="none",
        edgecolors="royalblue",
        linewidths=0.7,
        label="zero bg after cut",
    )
    axes[1].set_title("Coverage failures and zero-count clusters")
    axes[1].set_xlabel("RA [deg]")
    axes[1].set_ylabel("Dec [deg]")
    axes[1].legend(frameon=False, fontsize=9)

    fig.savefig(DIAG_DIR / "sky_coverage_and_zero_counts.png", dpi=180)
    plt.close(fig)


def save_histograms(table, masks):
    coverage_signal = col_float(table, "coverage_signal_sweep")
    coverage_bg = col_float(table, "coverage_bg_sweep")
    n_touch = col_float(table, "n_files_touching_cluster")
    z = col_float(table, "Z_SPEC_x")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].hist(coverage_signal[np.isfinite(coverage_signal)], bins=50, alpha=0.75)
    axes[0, 0].axvline(COVERAGE_MIN, color="crimson", lw=2)
    axes[0, 0].set_xlabel("coverage_signal_sweep")
    axes[0, 0].set_ylabel("clusters")

    axes[0, 1].hist(coverage_bg[np.isfinite(coverage_bg)], bins=50, alpha=0.75)
    axes[0, 1].axvline(COVERAGE_MIN, color="crimson", lw=2)
    axes[0, 1].set_xlabel("coverage_bg_sweep")

    axes[1, 0].hist(n_touch[np.isfinite(n_touch)], bins=np.arange(np.nanmax(n_touch) + 2) - 0.5)
    axes[1, 0].set_xlabel("n_files_touching_cluster")
    axes[1, 0].set_ylabel("clusters")

    axes[1, 1].hist(z[np.isfinite(z)], bins=40, histtype="step", lw=1.5, label="all")
    axes[1, 1].hist(
        z[masks["coverage_fail"] & np.isfinite(z)],
        bins=40,
        histtype="step",
        lw=1.5,
        label="coverage fail",
    )
    axes[1, 1].hist(
        z[masks["zero_signal_after_cut"] & np.isfinite(z)],
        bins=40,
        histtype="step",
        lw=1.5,
        label="zero signal after cut",
    )
    axes[1, 1].set_xlabel("Z_SPEC_x")
    axes[1, 1].legend(frameon=False)

    fig.savefig(DIAG_DIR / "coverage_histograms.png", dpi=180)
    plt.close(fig)


def save_count_area_plots(table, masks):
    n_signal = col_float(table, "Ngal_signal_DR9_annulus")
    n_bg = col_float(table, "Ngal_bg_DR9_annulus")
    area_signal = col_float(table, "covered_area_signal_deg2")
    area_bg = col_float(table, "covered_area_bg_deg2")
    coverage_signal = col_float(table, "coverage_signal_sweep")
    coverage_bg = col_float(table, "coverage_bg_sweep")

    good_signal_density = (
        masks["coverage_pass"]
        & np.isfinite(n_signal)
        & np.isfinite(area_signal)
        & (area_signal > 0)
        & (n_signal > 0)
    )
    good_bg_density = (
        masks["coverage_pass"]
        & np.isfinite(n_bg)
        & np.isfinite(area_bg)
        & (area_bg > 0)
        & (n_bg > 0)
    )

    median_signal_density = np.nanmedian(safe_divide(n_signal[good_signal_density], area_signal[good_signal_density]))
    median_bg_density = np.nanmedian(safe_divide(n_bg[good_bg_density], area_bg[good_bg_density]))

    expected_signal = median_signal_density * area_signal
    expected_bg = median_bg_density * area_bg

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    axes[0].scatter(area_signal, n_signal, c=coverage_signal, s=9, alpha=0.55, cmap="viridis")
    axes[0].scatter(
        area_signal[masks["zero_signal_after_cut"]],
        n_signal[masks["zero_signal_after_cut"]],
        s=25,
        facecolors="none",
        edgecolors="crimson",
        linewidths=0.9,
        label="zero signal after cut",
    )
    xline = np.linspace(0, np.nanpercentile(area_signal, 99), 200)
    axes[0].plot(xline, median_signal_density * xline, color="black", lw=1.5, label="median density expectation")
    axes[0].set_xlabel("covered signal area [deg$^2$]")
    axes[0].set_ylabel("Ngal_signal_DR9_annulus")
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].scatter(area_bg, n_bg, c=coverage_bg, s=9, alpha=0.55, cmap="viridis")
    axes[1].scatter(
        area_bg[masks["zero_bg_after_cut"]],
        n_bg[masks["zero_bg_after_cut"]],
        s=25,
        facecolors="none",
        edgecolors="crimson",
        linewidths=0.9,
        label="zero bg after cut",
    )
    xline = np.linspace(0, np.nanpercentile(area_bg, 99), 200)
    axes[1].plot(xline, median_bg_density * xline, color="black", lw=1.5, label="median density expectation")
    axes[1].set_xlabel("covered background area [deg$^2$]")
    axes[1].set_ylabel("Ngal_bg_DR9_annulus")
    axes[1].legend(frameon=False, fontsize=9)

    fig.savefig(DIAG_DIR / "counts_vs_covered_area.png", dpi=180)
    plt.close(fig)

    table["expected_signal_from_median_density"] = expected_signal
    table["expected_bg_from_median_density"] = expected_bg
    return table


def save_zero_count_tables(table, masks):
    keep_cols = [
        "ID",
        "RA_x",
        "DEC_x",
        "Z_SPEC_x",
        "LAMBDA",
        "lambda_spec_tot",
        "Ngal_signal_DR9_annulus",
        "Ngal_bg_DR9_annulus",
        "coverage_signal_sweep",
        "coverage_bg_sweep",
        "covered_area_signal_deg2",
        "covered_area_bg_deg2",
        "n_files_touching_cluster",
        "expected_signal_from_median_density",
        "expected_bg_from_median_density",
    ]
    keep_cols = [c for c in keep_cols if c in table.colnames]

    table[masks["coverage_fail"]][keep_cols].write(
        DIAG_DIR / "coverage_fail_clusters.ecsv", format="ascii.ecsv", overwrite=True
    )
    table[masks["zero_signal_after_cut"]][keep_cols].write(
        DIAG_DIR / "zero_signal_after_coverage_cut.ecsv", format="ascii.ecsv", overwrite=True
    )
    table[masks["zero_bg_after_cut"]][keep_cols].write(
        DIAG_DIR / "zero_bg_after_coverage_cut.ecsv", format="ascii.ecsv", overwrite=True
    )


def main():
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    table = Table.read(INPUT_TABLE)
    total = len(table)

    coverage_signal = col_float(table, "coverage_signal_sweep")
    coverage_bg = col_float(table, "coverage_bg_sweep")
    n_signal = col_float(table, "Ngal_signal_DR9_annulus")
    n_bg = col_float(table, "Ngal_bg_DR9_annulus")
    n_touch = col_float(table, "n_files_touching_cluster")
    z = col_float(table, "Z_SPEC_x")
    dec = col_float(table, "DEC_x")

    finite_geometry = (
        np.isfinite(coverage_signal)
        & np.isfinite(coverage_bg)
        & np.isfinite(z)
        & (z > 0)
    )
    coverage_pass = (
        finite_geometry
        & (coverage_signal > COVERAGE_MIN)
        & (coverage_bg > COVERAGE_MIN)
    )
    coverage_fail = finite_geometry & ~coverage_pass
    zero_signal_after_cut = coverage_pass & np.isfinite(n_signal) & (n_signal == 0)
    zero_bg_after_cut = coverage_pass & np.isfinite(n_bg) & (n_bg == 0)
    no_touching_file = finite_geometry & np.isfinite(n_touch) & (n_touch == 0)
    raw_coverage_gt_one = (
        (col_float(table, "coverage_signal_sweep_raw") > 1.05)
        | (col_float(table, "coverage_bg_sweep_raw") > 1.05)
    )

    masks = {
        "finite_geometry": finite_geometry,
        "coverage_pass": coverage_pass,
        "coverage_fail": coverage_fail,
        "zero_signal_after_cut": zero_signal_after_cut,
        "zero_bg_after_cut": zero_bg_after_cut,
        "no_touching_file": no_touching_file,
        "raw_coverage_gt_one": raw_coverage_gt_one,
    }

    print(f"Input table: {INPUT_TABLE}")
    print(f"Diagnostics dir: {DIAG_DIR}")
    summary_rows = [
        describe_mask("finite geometry", finite_geometry, total),
        describe_mask(f"coverage pass > {COVERAGE_MIN}", coverage_pass, total),
        describe_mask("coverage fail", coverage_fail, total),
        describe_mask("no touching sweep file", no_touching_file, total),
        describe_mask("zero signal after coverage cut", zero_signal_after_cut, total),
        describe_mask("zero bg after coverage cut", zero_bg_after_cut, total),
        describe_mask("raw accumulated coverage > 1.05", raw_coverage_gt_one, total),
        describe_mask("Dec >= 32 deg", np.isfinite(dec) & (dec >= 32), total),
        describe_mask("Dec < 32 deg", np.isfinite(dec) & (dec < 32), total),
    ]

    summary = Table(rows=summary_rows, names=["diagnostic", "N", "fraction"])
    summary.write(DIAG_DIR / "coverage_diagnostic_summary.ecsv", format="ascii.ecsv", overwrite=True)

    save_sky_plot(table, masks)
    save_histograms(table, masks)
    table = save_count_area_plots(table, masks)
    save_zero_count_tables(table, masks)

    print(f"Wrote diagnostics to {DIAG_DIR}")


if __name__ == "__main__":
    main()
