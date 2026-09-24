"""
MPI sweep over cumulative local-environment outer radii.

This script counts selected DR9 photometric galaxies in cumulative projected
annuli around redMaPPer cluster centers,

    R_MIN < R < R_OUT,

for a grid of outer radii. It then ranks which R_OUT gives the strongest
absolute Spearman correlation with cluster richness. The coverage fractions are
saved as diagnostics, but no coverage cut is applied by default.

Example on Perlmutter, from the repository root:

    srun -n 8 -c 1 python local_overdensity/DR9_outer_radius_spearman_sweep_mpi.py \
        --r-mag-limit 22.0
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, unique
from scipy.stats import binned_statistic, pearsonr, rankdata, spearmanr

MPI_IMPORT_ERROR = None
try:
    from mpi4py import MPI
except Exception as exc:
    MPI = None
    MPI_IMPORT_ERROR = exc

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from DR9_localOverdensity_sweep_mpi import (  # noqa: E402
    DEFAULT_SWEEP_DIRS,
    INPUT_DIR,
    OUTPUT_DIR,
    annulus_area_for_clusters,
    clusters_overlapping_sweep_box,
    count_galaxies_in_annuli,
    read_one_dr9_sweep,
    read_sweep_bounds,
    safe_divide,
    sweep_annulus_coverage_for_clusters,
)


def reduce_sum(comm, local_array, root=0):
    if comm is None:
        return local_array
    if MPI is None:
        raise RuntimeError("MPI communicator is active, but mpi4py.MPI is unavailable.") from MPI_IMPORT_ERROR
    global_array = np.empty_like(local_array) if comm.Get_rank() == root else None
    comm.Reduce(local_array, global_array, op=MPI.SUM, root=root)
    return global_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=INPUT_DIR / "bgs_clus_RM_gal_matched.pickle",
        help="Input redMaPPer/BGS matched pickle file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "outer_radius_spearman_sweep",
        help="Directory for radius-sweep outputs.",
    )
    parser.add_argument(
        "--sweep-dir",
        action="append",
        type=Path,
        default=None,
        help="DR9 sweep directory. Can be supplied multiple times.",
    )
    parser.add_argument("--pattern", default="sweep-*.fits")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--r-mag-limit", type=float, default=22.0)
    parser.add_argument("--rmin-hmpc", type=float, default=1.5)
    parser.add_argument(
        "--rout-hmpc",
        type=float,
        nargs="+",
        default=[5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        help="Outer radii for cumulative annuli Rmin < R < Rout.",
    )
    parser.add_argument("--nside-gal", type=int, default=4096)
    parser.add_argument("--nside-sweep-coverage", type=int, default=1024)
    parser.add_argument(
        "--primary-richness-col",
        default="lambda_spec_tot",
        help="Primary richness column used to pick the best radius.",
    )
    parser.add_argument(
        "--primary-y-col",
        default="Sigma_env_covcorr",
        choices=["N_env", "Sigma_env_geom", "Sigma_env_covcorr"],
        help="Environment statistic used to pick the best radius.",
    )
    parser.add_argument("--nbins", type=int, default=8)
    return parser.parse_args()


def find_sweep_files(sweep_dirs, pattern="sweep-*.fits", max_files=None):
    files = []
    for sweep_dir in sweep_dirs:
        sweep_dir = Path(sweep_dir)
        files.extend(glob(str(sweep_dir / pattern)))
        files.extend(glob(str(sweep_dir / "*" / pattern)))
    files = sorted(set(files))
    if max_files is not None:
        files = files[:max_files]
    if len(files) == 0:
        raise FileNotFoundError(
            f"No sweep files matched {[str(Path(d) / pattern) for d in sweep_dirs]}"
        )
    return [Path(f) for f in files]


def col_float(table: Table, col: str) -> np.ndarray:
    arr = np.ma.asarray(table[col], dtype=float)
    return np.ma.filled(arr, np.nan)


def finite_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.isfinite(x) & (x > 0)


def correlation_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = finite_positive(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 4:
        return {
            "N": int(np.count_nonzero(mask)),
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pearson_r_logx": np.nan,
            "pearson_p_logx": np.nan,
        }

    spearman_r, spearman_p = spearmanr(x[mask], y[mask])
    pearson_r_logx, pearson_p_logx = pearsonr(np.log10(x[mask]), y[mask])
    return {
        "N": int(np.count_nonzero(mask)),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r_logx": float(pearson_r_logx),
        "pearson_p_logx": float(pearson_p_logx),
    }


def add_binned_mean(ax, x, y, nbins=8, color="crimson", label="binned mean"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = finite_positive(x) & np.isfinite(y)
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
        zorder=5,
    )


def add_binned_mean_linear_x(ax, x, y, nbins=8, color="crimson", label="binned mean"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < nbins:
        return

    bins = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
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
        zorder=5,
    )


def make_best_radius_plots(
    output_dir: Path,
    table: Table,
    richness_col: str,
    y_col: str,
    rmin_hmpc: float,
    rout_hmpc: float,
    nbins: int,
) -> None:
    x = col_float(table, richness_col)
    y = col_float(table, y_col)
    mask = finite_positive(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return

    sr, sp = spearmanr(x, y)
    pr, pp = pearsonr(np.log10(x), y)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    axes[0].scatter(x, y, s=12, alpha=0.35, color="0.25", edgecolor="none")
    add_binned_mean(axes[0], x, y, nbins=nbins, color="crimson")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(richness_col)
    axes[0].set_ylabel(y_col)
    axes[0].set_title(rf"${rmin_hmpc:g}<R<{rout_hmpc:g}\ h^{{-1}}\,{{\rm Mpc}}$")
    axes[0].text(
        0.04,
        0.96,
        rf"Spearman $r_s={sr:.3f}$, $p={sp:.1e}$"
        + "\n"
        + rf"Pearson $(\log x,y)$ $r={pr:.3f}$, $p={pp:.1e}$",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )
    axes[0].legend(frameon=False)

    rx = rankdata(x)
    ry = rankdata(y)
    axes[1].scatter(rx, ry, s=12, alpha=0.35, color="0.25", edgecolor="none")
    add_binned_mean_linear_x(axes[1], rx, ry, nbins=nbins, color="crimson")
    axes[1].plot(
        [np.nanmin(rx), np.nanmax(rx)],
        [np.nanmin(rx), np.nanmax(rx)],
        color="black",
        lw=1.0,
        alpha=0.35,
    )
    axes[1].set_xlabel(f"rank({richness_col})")
    axes[1].set_ylabel(f"rank({y_col})")
    axes[1].set_title("Rank-rank visualization")

    fig.tight_layout()
    fig.savefig(
        output_dir / f"best_radius_{richness_col}_{y_col}_binned_mean_rank_rank.png",
        dpi=180,
    )
    plt.close(fig)


def make_radius_summary_plot(output_dir: Path, summary: Table, richness_col: str, y_col: str):
    mask = (
        np.asarray(summary["richness_col"]) == richness_col
    ) & (np.asarray(summary["y_col"]) == y_col)
    sub = summary[mask]
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.5)
    ax.plot(
        sub["rout_hmpc"],
        sub["spearman_r"],
        marker="o",
        color="crimson",
        label="Spearman",
    )
    ax.plot(
        sub["rout_hmpc"],
        sub["pearson_r_logx"],
        marker="s",
        color="royalblue",
        label=r"Pearson $(\log x,y)$",
    )
    best_idx = int(np.nanargmax(np.abs(sub["spearman_r"])))
    ax.scatter(
        sub["rout_hmpc"][best_idx],
        sub["spearman_r"][best_idx],
        s=90,
        facecolor="none",
        edgecolor="black",
        lw=1.6,
        zorder=5,
        label="max |Spearman|",
    )
    ax.set_xlabel(r"$R_{\rm out}\ [h^{-1}{\rm Mpc}]$")
    ax.set_ylabel("correlation coefficient")
    ax.set_title(f"{y_col} versus {richness_col}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"radius_sweep_correlations_{richness_col}_{y_col}.png", dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if MPI is None:
        comm = None
        rank = 0
        size = 1
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    sweep_dirs = args.sweep_dir or DEFAULT_SWEEP_DIRS
    sweep_files = find_sweep_files(sweep_dirs, args.pattern, max_files=args.max_files)
    my_files = sweep_files[rank::size]

    rout_grid = np.asarray(args.rout_hmpc, dtype=float)
    if np.any(rout_grid <= args.rmin_hmpc):
        raise ValueError("All --rout-hmpc values must be larger than --rmin-hmpc.")
    rout_grid = np.sort(np.unique(rout_grid))
    n_r = len(rout_grid)

    if rank == 0:
        print(f"MPI size: {size}")
        print(f"Catalog: {args.catalog_path}")
        print(f"Output dir: {output_dir}")
        print(f"Found {len(sweep_files):,} sweep files")
        print(f"Radii: {args.rmin_hmpc:g} < R < {rout_grid}")
    print(f"Rank {rank:03d}: assigned {len(my_files):,} files", flush=True)

    with args.catalog_path.open("rb") as handle:
        bgs_matched = pickle.load(handle)
    rm_tab = unique(bgs_matched, keys="ID")

    n_cl = len(rm_tab)
    ra_cl = np.asarray(rm_tab["RA_x"], dtype=float)
    dec_cl = np.asarray(rm_tab["DEC_x"], dtype=float)
    z_cl = np.asarray(rm_tab["Z_SPEC_x"], dtype=float)

    area_deg2_all = np.zeros((n_r, n_cl), dtype=float)
    for ir, rout in enumerate(rout_grid):
        _, area_deg2_all[ir] = annulus_area_for_clusters(z_cl, args.rmin_hmpc, rout)

    n_env_local = np.zeros((n_r, n_cl), dtype=np.int64)
    covered_area_local = np.zeros((n_r, n_cl), dtype=float)
    n_files_touching_local = np.zeros(n_cl, dtype=np.int64)

    for i_file, sweep_file in enumerate(my_files, start=1):
        print(
            f"Rank {rank:03d}: [{i_file}/{len(my_files)}] {sweep_file.name}",
            flush=True,
        )
        sweep_bounds = read_sweep_bounds(sweep_file)
        overlap = clusters_overlapping_sweep_box(
            ra_cl, dec_cl, z_cl, sweep_bounds, rmax_hmpc=float(np.max(rout_grid))
        )
        idx = np.where(overlap)[0]
        if len(idx) == 0:
            continue

        dr9 = read_one_dr9_sweep(sweep_file, r_mag_limit=args.r_mag_limit)
        ra_gal = np.asarray(dr9["RA"], dtype=float)
        dec_gal = np.asarray(dr9["DEC"], dtype=float)

        n_files_touching_local[idx] += 1

        for ir, rout in enumerate(rout_grid):
            counts, _ = count_galaxies_in_annuli(
                ra_cl[idx],
                dec_cl[idx],
                z_cl[idx],
                ra_gal,
                dec_gal,
                rmin_hmpc=args.rmin_hmpc,
                rmax_hmpc=float(rout),
                nside=args.nside_gal,
            )
            coverage, _ = sweep_annulus_coverage_for_clusters(
                ra_cl[idx],
                dec_cl[idx],
                z_cl[idx],
                sweep_bounds,
                rmin_hmpc=args.rmin_hmpc,
                rmax_hmpc=float(rout),
                nside=args.nside_sweep_coverage,
            )
            n_env_local[ir, idx] += counts
            covered_area_local[ir, idx] += (
                np.nan_to_num(coverage, nan=0.0) * area_deg2_all[ir, idx]
            )

        del dr9, ra_gal, dec_gal
        gc.collect()

    n_env_all = reduce_sum(comm, n_env_local)
    covered_area_all = reduce_sum(comm, covered_area_local)
    n_files_touching_all = reduce_sum(comm, n_files_touching_local)

    if rank != 0:
        return 0

    richness_cols = [
        col
        for col in ["lambda_spec_tot", "lambda_spec", "lambda_true", "LAMBDA"]
        if col in rm_tab.colnames
    ]
    if args.primary_richness_col not in richness_cols:
        raise KeyError(
            f"{args.primary_richness_col!r} is not in the cluster table. "
            f"Available richness columns: {richness_cols}"
        )

    rows = []
    for ir, rout in enumerate(rout_grid):
        coverage = np.clip(safe_divide(covered_area_all[ir], area_deg2_all[ir]), 0.0, 1.0)
        sigma_geom = safe_divide(n_env_all[ir], area_deg2_all[ir])
        sigma_covcorr = safe_divide(n_env_all[ir], covered_area_all[ir])

        radius_table = rm_tab.copy()
        radius_table["Rmin_hmpc"] = np.full(n_cl, args.rmin_hmpc)
        radius_table["Rout_hmpc"] = np.full(n_cl, rout)
        radius_table["N_env"] = n_env_all[ir]
        radius_table["area_env_deg2"] = area_deg2_all[ir]
        radius_table["covered_area_env_deg2"] = covered_area_all[ir]
        radius_table["coverage_env_sweep"] = coverage
        radius_table["n_files_touching_cluster"] = n_files_touching_all
        radius_table["Sigma_env_geom"] = sigma_geom
        radius_table["Sigma_env_covcorr"] = sigma_covcorr

        suffix = f"R{args.rmin_hmpc:g}_{rout:g}".replace(".", "p")
        radius_table.write(
            output_dir / f"rm_dr9_environment_outer_radius_{suffix}.ecsv",
            format="ascii.ecsv",
            overwrite=True,
        )
        radius_table.write(
            output_dir / f"rm_dr9_environment_outer_radius_{suffix}.fits",
            overwrite=True,
        )

        y_arrays = {
            "N_env": np.asarray(n_env_all[ir], dtype=float),
            "Sigma_env_geom": sigma_geom,
            "Sigma_env_covcorr": sigma_covcorr,
        }
        for richness_col in richness_cols:
            x = col_float(radius_table, richness_col)
            for y_col, y in y_arrays.items():
                stats = correlation_summary(x, y)
                rows.append(
                    {
                        "rmin_hmpc": args.rmin_hmpc,
                        "rout_hmpc": float(rout),
                        "richness_col": richness_col,
                        "y_col": y_col,
                        **stats,
                    }
                )

    summary = Table(rows=rows)
    summary["abs_spearman_r"] = np.abs(np.asarray(summary["spearman_r"], dtype=float))
    summary.sort("abs_spearman_r")
    summary.reverse()
    summary.write(
        output_dir / "outer_radius_spearman_summary.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )

    primary = summary[
        (np.asarray(summary["richness_col"]) == args.primary_richness_col)
        & (np.asarray(summary["y_col"]) == args.primary_y_col)
    ]
    best_idx = int(np.nanargmax(np.abs(primary["spearman_r"])))
    best_rout = float(primary["rout_hmpc"][best_idx])

    for richness_col in richness_cols:
        for y_col in ["N_env", "Sigma_env_geom", "Sigma_env_covcorr"]:
            make_radius_summary_plot(output_dir, summary, richness_col, y_col)

    suffix = f"R{args.rmin_hmpc:g}_{best_rout:g}".replace(".", "p")
    best_table = Table.read(output_dir / f"rm_dr9_environment_outer_radius_{suffix}.fits")
    make_best_radius_plots(
        output_dir,
        best_table,
        args.primary_richness_col,
        args.primary_y_col,
        args.rmin_hmpc,
        best_rout,
        args.nbins,
    )

    print("\nFinished outer-radius Spearman sweep")
    print(f"Best primary radius: {args.rmin_hmpc:g} < R < {best_rout:g} h^-1 Mpc")
    print(f"Primary x: {args.primary_richness_col}")
    print(f"Primary y: {args.primary_y_col}")
    print("Top rows:")
    print(summary[:10])
    print("Saved:")
    print(" ", output_dir / "outer_radius_spearman_summary.ecsv")
    print(" ", output_dir / f"best_radius_{args.primary_richness_col}_{args.primary_y_col}_binned_mean_rank_rank.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
