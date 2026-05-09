"""
Diagnostic plots for comparing individual-first and stack-first spectroscopic
richness estimates.

The central idea is to compare the two workflows after each transformation:

1. raw redshift-offset counts
2. normalized PDFs
3. continuum/background model
4. bin weights
5. per-bin richness contributions
6. final spectroscopic-redMaPPer richness relation

Example
-------
from astropy.table import Table
import numpy as np
from richness_diagnostics import run_diagnostics

table = Table.read("my_matched_catalog.fits")
bin_edges = np.linspace(-0.08, 0.08, 81)

run_diagnostics(
    table,
    bin_edges,
    original_module_path="/Users/zzbenjamin94/Downloads/projection_functions (1).py",
    redmapper_col="LAMBDA_CHISQ",
    output_dir="diagnostic_plots",
)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_original_module(path):
    """Load the original projection_functions file, even if its filename has spaces."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location("projection_functions_original", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bin_centers(bin_edges):
    bin_edges = np.asarray(bin_edges)
    return 0.5 * (bin_edges[:-1] + bin_edges[1:])


def bin_widths(bin_edges):
    bin_edges = np.asarray(bin_edges)
    return np.diff(bin_edges)


def z_diff(table, z_bgs_col="Z_BGS", z_cluster_col="Z_SPEC_x"):
    return (np.asarray(table[z_bgs_col]) - np.asarray(table[z_cluster_col])) / (
        1.0 + np.asarray(table[z_cluster_col])
    )


def histogram_components(table, bin_edges, z_bgs_col="Z_BGS", z_cluster_col="Z_SPEC_x"):
    """Return raw counts, in-range count, total count, and manually normalized PDF."""
    dz = z_diff(table, z_bgs_col=z_bgs_col, z_cluster_col=z_cluster_col)
    counts, _ = np.histogram(dz, bins=bin_edges, density=False)
    widths = bin_widths(bin_edges)
    n_in = int(np.sum(counts))
    n_total = len(dz)
    pdf = np.full_like(counts, np.nan, dtype=float)
    if n_in > 0:
        pdf = counts / (n_in * widths)
    return {
        "dz": dz,
        "counts": counts.astype(float),
        "pdf": pdf,
        "n_in": n_in,
        "n_total": n_total,
        "frac_out": 1.0 - n_in / n_total if n_total else np.nan,
    }


def group_tables(table, id_col="ID"):
    grouped = table.group_by(id_col)
    return list(grouped.groups)


def safe_original_zdiff(original_module, bin_edges, table):
    try:
        pdf, err = original_module.calc_zDiff(bin_edges, table, numCount_bool=False)
        counts, count_err = original_module.calc_zDiff(bin_edges, table, numCount_bool=True)
    except Exception:
        pdf = np.full(len(bin_edges) - 1, np.nan)
        err = np.full(len(bin_edges) - 1, np.nan)
        counts = np.full(len(bin_edges) - 1, np.nan)
        count_err = np.full(len(bin_edges) - 1, np.nan)
    return pdf, err, counts, count_err


def get_weights(original_module, bin_edges, table):
    x, weights = original_module.calc_weights_all(bin_edges, table)
    return np.asarray(x), np.asarray(weights, dtype=float)


def get_continuum(original_module, bin_edges, centers, table):
    return np.asarray(original_module.calc_Continuum(bin_edges, centers, table), dtype=float)


def compare_stack_vs_individual(table, bin_edges, original_module, id_col="ID"):
    groups = group_tables(table, id_col=id_col)
    centers = bin_centers(bin_edges)
    widths = bin_widths(bin_edges)

    stack = histogram_components(table, bin_edges)
    individual = [histogram_components(group, bin_edges) for group in groups]

    sum_ind_counts = np.sum([item["counts"] for item in individual], axis=0)
    n_in_ind = np.asarray([item["n_in"] for item in individual])
    n_total_ind = np.asarray([item["n_total"] for item in individual])
    frac_out_ind = np.asarray([item["frac_out"] for item in individual])

    pdfs = np.asarray([item["pdf"] for item in individual])
    valid_pdf = np.isfinite(pdfs).all(axis=1)

    galaxy_weighted_pdf = np.full(len(centers), np.nan)
    if np.sum(n_in_ind[valid_pdf]) > 0:
        galaxy_weighted_pdf = np.average(
            pdfs[valid_pdf], axis=0, weights=n_in_ind[valid_pdf]
        )

    cluster_equal_pdf = np.nanmean(pdfs, axis=0)

    original_stack_pdf, original_stack_err, original_stack_counts, _ = safe_original_zdiff(
        original_module, bin_edges, table
    )

    original_ind_pdf = []
    original_ind_counts = []
    original_ind_valid = []
    for group in groups:
        pdf, err, counts, count_err = safe_original_zdiff(original_module, bin_edges, group)
        original_ind_pdf.append(pdf)
        original_ind_counts.append(counts)
        original_ind_valid.append(np.isfinite(pdf).all())

    original_ind_pdf = np.asarray(original_ind_pdf)
    original_ind_counts = np.asarray(original_ind_counts)
    original_ind_valid = np.asarray(original_ind_valid)

    weights_x, weights = get_weights(original_module, bin_edges, table)
    continuum = get_continuum(original_module, bin_edges, centers, table)

    stack_bin_lambda = (original_stack_pdf - continuum) * widths * weights * len(table)
    stack_bin_lambda_per_cluster = stack_bin_lambda / len(groups)

    ind_bin_lambda = []
    for group, pdf in zip(groups, original_ind_pdf):
        ind_bin_lambda.append((pdf - continuum) * widths * weights * len(group))
    ind_bin_lambda = np.asarray(ind_bin_lambda)

    return {
        "centers": centers,
        "widths": widths,
        "groups": groups,
        "stack": stack,
        "individual": individual,
        "sum_ind_counts": sum_ind_counts,
        "n_in_ind": n_in_ind,
        "n_total_ind": n_total_ind,
        "frac_out_ind": frac_out_ind,
        "manual_galaxy_weighted_pdf": galaxy_weighted_pdf,
        "manual_cluster_equal_pdf": cluster_equal_pdf,
        "original_stack_pdf": original_stack_pdf,
        "original_stack_err": original_stack_err,
        "original_stack_counts": original_stack_counts,
        "original_ind_pdf": original_ind_pdf,
        "original_ind_counts": original_ind_counts,
        "original_ind_valid": original_ind_valid,
        "weights_x": weights_x,
        "weights": weights,
        "continuum": continuum,
        "stack_bin_lambda_per_cluster": stack_bin_lambda_per_cluster,
        "ind_bin_lambda": ind_bin_lambda,
    }


def plot_step_diagnostics(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    centers = result["centers"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, result["stack"]["counts"], label="stack-first raw counts", lw=2)
    ax.plot(
        centers,
        result["sum_ind_counts"],
        "--",
        label="sum of individual raw counts",
        lw=2,
    )
    ax.set_xlabel("Delta z")
    ax.set_ylabel("Galaxy count per bin")
    ax.set_title("Step 1: raw counts should match exactly")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_raw_counts_stack_vs_sum_individual.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, result["stack"]["pdf"], label="manual stack PDF", lw=2)
    ax.plot(
        centers,
        result["manual_galaxy_weighted_pdf"],
        "--",
        label="galaxy-weighted mean individual PDF",
        lw=2,
    )
    ax.plot(
        centers,
        result["manual_cluster_equal_pdf"],
        ":",
        label="cluster-equal mean individual PDF",
        lw=2,
    )
    ax.set_xlabel("Delta z")
    ax.set_ylabel("PDF")
    ax.set_title("Step 2: PDF normalization comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "02_pdf_normalization_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, result["original_stack_pdf"], label="original stack PDF", lw=2)
    ax.plot(centers, result["stack"]["pdf"], "--", label="manual stack PDF", lw=2)
    ax.plot(centers, result["continuum"], label="continuum model", lw=2)
    ax.axvspan(-0.02, 0.02, color="0.85", label="continuum excluded region")
    ax.set_xlabel("Delta z")
    ax.set_ylabel("PDF")
    ax.set_title("Step 3: original PDF and continuum model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_continuum_fit_check.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(centers, result["weights"], marker="o", lw=1.5)
    ax.set_xlabel("Delta z")
    ax.set_ylabel("Mean WEIGHT / geoFrac")
    ax.set_title("Step 4: bin weights used by both estimators")
    fig.tight_layout()
    fig.savefig(output_dir / "04_bin_weights.png", dpi=180)
    plt.close(fig)

    mean_ind_bin_lambda = np.nanmean(result["ind_bin_lambda"], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        centers,
        result["stack_bin_lambda_per_cluster"],
        label="stack-first contribution per cluster",
        lw=2,
    )
    ax.plot(
        centers,
        mean_ind_bin_lambda,
        "--",
        label="mean individual-first contribution",
        lw=2,
    )
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlabel("Delta z")
    ax.set_ylabel("Contribution to lambda_spec")
    ax.set_title("Step 5: per-bin richness contribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "05_richness_contribution_by_bin.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        centers,
        np.nancumsum(result["stack_bin_lambda_per_cluster"]),
        label="stack-first cumulative",
        lw=2,
    )
    ax.plot(
        centers,
        np.nancumsum(mean_ind_bin_lambda),
        "--",
        label="individual-first cumulative",
        lw=2,
    )
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlabel("Delta z")
    ax.set_ylabel("Cumulative lambda_spec")
    ax.set_title("Step 6: where the final richness difference accumulates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "06_cumulative_richness_difference.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(result["n_total_ind"], result["frac_out_ind"], s=14, alpha=0.65)
    ax.set_xlabel("Galaxies associated with cluster table")
    ax.set_ylabel("Fraction outside Delta z bin range")
    ax.set_title("Diagnostic: possible density=True normalization bias")
    fig.tight_layout()
    fig.savefig(output_dir / "07_out_of_range_fraction_by_cluster.png", dpi=180)
    plt.close(fig)


def plot_redmapper_relation(
    table,
    bin_edges,
    original_module,
    redmapper_col,
    output_dir,
    id_col="ID",
    redmapper_bins=None,
):
    """Compare individual-first and stack-first richness in redMaPPer richness bins."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if redmapper_bins is None:
        values = np.asarray(table[redmapper_col], dtype=float)
        redmapper_bins = np.nanquantile(values, np.linspace(0, 1, 6))
        redmapper_bins = np.unique(redmapper_bins)

    centers_rm = 0.5 * (redmapper_bins[:-1] + redmapper_bins[1:])
    ind_first = []
    ind_first_err = []
    stack_first = []
    stack_first_err = []
    n_clusters = []

    for lo, hi in zip(redmapper_bins[:-1], redmapper_bins[1:]):
        mask = (np.asarray(table[redmapper_col]) >= lo) & (np.asarray(table[redmapper_col]) < hi)
        sub = table[mask]
        if len(sub) == 0:
            ind_first.append(np.nan)
            stack_first.append(np.nan)
            n_clusters.append(0)
            continue

        ids = np.unique(np.asarray(sub[id_col]))
        n_clusters.append(len(ids))

        _, _, lambda_true_ind = original_module.calc_specRichness_individual(bin_edges, bin_centers(bin_edges), sub)
        _, lambda_true_stack, _ = original_module.calc_specRichness_stacked(
            bin_edges, bin_centers(bin_edges), sub
        )
        ind_first.append(np.nanmean(lambda_true_ind))
        ind_first_err.append(np.sqrt(np.nanmean(lambda_true_ind)/len(ids))) ## Poissonian errors
        stack_first.append(lambda_true_stack)
        stack_first_err.append(np.sqrt(lambda_true_stack/len(ids))) ## Poissonan errors
        

    ind_first = np.asarray(ind_first, dtype=float)
    stack_first = np.asarray(stack_first, dtype=float)

    ##Poissonian errors
    ind_first_err = np.asarray(ind_first_err, dtype=float)
    stack_first_err = np.asarray(stack_first_err, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    offset = 1.05
    ax.errorbar(centers_rm, ind_first, yerr=ind_first_err, marker="o", label="individual first, then average")
    ax.errorbar(centers_rm*offset, stack_first, yerr=stack_first_err, marker="s", label="stack first")
    ax.set_xlabel(redmapper_col)
    ax.set_ylabel("lambda_spec")
    ax.set_title("Final relation comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "08_redmapper_relation_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color="0.2", lw=0.8)
    ax.plot(centers_rm, ind_first - stack_first, marker="o")
    ax.set_xlabel(redmapper_col)
    ax.set_ylabel("individual-first minus stack-first")
    ax.set_title("Final relation residual")
    fig.tight_layout()
    fig.savefig(output_dir / "09_redmapper_relation_residual.png", dpi=180)
    plt.close(fig)

    return {
        "redmapper_bin_centers": centers_rm,
        "individual_first": ind_first,
        "stack_first": stack_first,
        "n_clusters": np.asarray(n_clusters),
    }


def run_diagnostics(
    table,
    bin_edges,
    original_module_path="/Users/zzbenjamin94/Downloads/projection_functions (1).py",
    output_dir="diagnostic_plots",
    id_col="ID",
    redmapper_col=None,
    redmapper_bins=None,
):
    """Run the full diagnostic plot suite."""
    original_module = load_original_module(original_module_path)
    output_dir = Path(output_dir)
    result = compare_stack_vs_individual(
        table, bin_edges, original_module=original_module, id_col=id_col
    )
    plot_step_diagnostics(result, output_dir)

    relation = None
    if redmapper_col is not None:
        relation = plot_redmapper_relation(
            table,
            bin_edges,
            original_module=original_module,
            redmapper_col=redmapper_col,
            output_dir=output_dir,
            id_col=id_col,
            redmapper_bins=redmapper_bins,
        )

    print(f"Wrote diagnostic plots to: {output_dir.resolve()}")
    print(f"Clusters: {len(result['groups'])}")
    print(f"Galaxies: {len(table)}")
    print(f"Galaxies inside bin range: {result['stack']['n_in']}")
    print(f"Fraction outside bin range: {result['stack']['frac_out']:.4f}")
    print(f"Clusters with finite original individual PDFs: {np.sum(result['original_ind_valid'])}")
    print(f"Clusters skipped/invalid in original individual PDFs: {np.sum(~result['original_ind_valid'])}")

    return result, relation
