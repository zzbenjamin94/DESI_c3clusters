"""
MPI DR9 local-overdensity sweep over signal/background radius definitions.

The script reads a cluster-level table with one BCG/central position per
cluster, redMaPPer richness (LAMBDA), and spectroscopic richness
(lambda_spec_tot). It then streams through DR9 sweep files once per MPI rank
and accumulates photometric-galaxy counts for many signal/background annuli.

Default radius definitions are paired:

    signal:     1.5 < R < X
    background: X   < R < X + 5

for X = 5, 6, 7, 8, 9, 10 h^-1 Mpc.

Example on Perlmutter:

    srun -n 16 -c 1 python local_overdensity/DR9_radius_grid_overdensity_mpi.py \
        --r-mag-limit 22.0

Quick test:

    srun -n 4 -c 1 python local_overdensity/DR9_radius_grid_overdensity_mpi.py \
        --max-files 8 --r-mag-limit 22.0
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import warnings
from glob import glob
from pathlib import Path

import numpy as np
from astropy.table import Table, unique, vstack
from astropy.units import UnitsWarning
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings(
    "ignore",
    message=r".*did not parse as fits unit.*",
    category=UnitsWarning,
)

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


DEFAULT_SIGNAL_ANNULI = "1.5:5,1.5:6,1.5:7,1.5:8,1.5:9,1.5:10"
DEFAULT_BACKGROUND_ANNULI = "5:10,6:11,7:12,8:13,9:14,10:15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=INPUT_DIR / "rm_clusters_with_spec_richness.pickle",
        help=(
            "Cluster-level table containing one RA_x/DEC_x center per cluster, "
            "LAMBDA, and lambda_spec_tot."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "radius_grid_overdensity",
        help="Directory for output tables.",
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
    parser.add_argument(
        "--no-r-mag-limit",
        action="store_true",
        help="Do not apply an r-band magnitude limit to the DR9 photometric galaxies.",
    )
    parser.add_argument(
        "--signal-annuli",
        default=DEFAULT_SIGNAL_ANNULI,
        help="Comma-separated signal annuli in h^-1 Mpc, e.g. '1.5:5,1.5:8'.",
    )
    parser.add_argument(
        "--background-annuli",
        default=DEFAULT_BACKGROUND_ANNULI,
        help="Comma-separated background annuli in h^-1 Mpc, e.g. '5:10,10:15'.",
    )
    parser.add_argument(
        "--pair-mode",
        choices=["paired", "all"],
        default="paired",
        help="'paired' matches signal/background annuli by index; 'all' tests all combinations.",
    )
    parser.add_argument("--nside-gal", type=int, default=4096)
    parser.add_argument("--nside-sweep-coverage", type=int, default=1024)
    parser.add_argument(
        "--coverage-min-for-summary",
        type=float,
        default=0.0,
        help="Optional coverage cut used only for correlation summary rows.",
    )
    return parser.parse_args()


def parse_annuli(text: str) -> list[tuple[float, float]]:
    annuli = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Annulus {chunk!r} must have form rmin:rmax")
        rmin_text, rmax_text = chunk.split(":", 1)
        rmin = float(rmin_text)
        rmax = float(rmax_text)
        if rmax <= rmin:
            raise ValueError(f"Annulus {chunk!r} has rmax <= rmin")
        annuli.append((rmin, rmax))
    if len(annuli) == 0:
        raise ValueError("No annuli were parsed.")
    return annuli


def build_radius_definitions(
    signal_annuli: list[tuple[float, float]],
    background_annuli: list[tuple[float, float]],
    pair_mode: str,
) -> Table:
    rows = []
    if pair_mode == "paired":
        if len(signal_annuli) != len(background_annuli):
            raise ValueError("paired mode requires the same number of signal and background annuli.")
        iterator = zip(signal_annuli, background_annuli)
    else:
        iterator = ((sig, bg) for sig in signal_annuli for bg in background_annuli)

    for i, (sig, bg) in enumerate(iterator):
        rows.append(
            {
                "radius_def_id": i,
                "r_sig_in_hmpc": sig[0],
                "r_sig_out_hmpc": sig[1],
                "r_bg_in_hmpc": bg[0],
                "r_bg_out_hmpc": bg[1],
                "radius_label": (
                    f"sig_{sig[0]:g}_{sig[1]:g}__bg_{bg[0]:g}_{bg[1]:g}"
                ).replace(".", "p"),
            }
        )
    return Table(rows=rows)


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


def read_pickle_table(path: Path) -> Table:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, Table):
        return obj
    if hasattr(obj, "to_pandas"):
        return Table.from_pandas(obj.to_pandas())
    return Table(obj)


def choose_col(table: Table, candidates: list[str]) -> str:
    for col in candidates:
        if col in table.colnames:
            return col
    raise KeyError(f"None of these columns were found: {candidates}")


def col_float(table: Table, col: str) -> np.ndarray:
    arr = np.ma.asarray(table[col], dtype=float)
    return np.ma.filled(arr, np.nan)


def finite_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.isfinite(x) & (x > 0)


def reduce_sum(comm, local_array, root=0):
    if comm is None:
        return local_array
    if MPI is None:
        raise RuntimeError("MPI communicator is active, but mpi4py.MPI is unavailable.") from MPI_IMPORT_ERROR
    global_array = np.empty_like(local_array) if comm.Get_rank() == root else None
    comm.Reduce(local_array, global_array, op=MPI.SUM, root=root)
    return global_array


def correlation_summary(x, y, mask):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool) & finite_positive(x) & np.isfinite(y)
    n = int(np.count_nonzero(mask))
    if n < 4:
        return {
            "N_corr": n,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "pearson_r_logx": np.nan,
            "pearson_p_logx": np.nan,
        }
    spearman_r, spearman_p = spearmanr(x[mask], y[mask])
    pearson_r_logx, pearson_p_logx = pearsonr(np.log10(x[mask]), y[mask])
    return {
        "N_corr": n,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "pearson_r_logx": float(pearson_r_logx),
        "pearson_p_logx": float(pearson_p_logx),
    }


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

    signal_annuli = parse_annuli(args.signal_annuli)
    background_annuli = parse_annuli(args.background_annuli)
    radius_defs = build_radius_definitions(signal_annuli, background_annuli, args.pair_mode)
    n_def = len(radius_defs)

    sweep_dirs = args.sweep_dir or DEFAULT_SWEEP_DIRS
    sweep_files = find_sweep_files(sweep_dirs, args.pattern, max_files=args.max_files)
    my_files = sweep_files[rank::size]

    if rank == 0:
        print(f"MPI size: {size}")
        print(f"Catalog: {args.catalog_path}")
        print(f"Output dir: {output_dir}")
        print(f"Found {len(sweep_files):,} sweep files")
        print(f"Radius definitions: {n_def}")
        print(
            "DR9 r-band magnitude limit: "
            + ("none" if args.no_r_mag_limit else str(args.r_mag_limit))
        )
        print(radius_defs)
    print(f"Rank {rank:03d}: assigned {len(my_files):,} files", flush=True)

    # This is a cluster-level table, not the full BGS member-galaxy table.
    # The annuli are centered on one BCG/central position per cluster.
    cluster_table = read_pickle_table(args.catalog_path)
    if "ID" in cluster_table.colnames:
        cluster_table = unique(cluster_table, keys="ID")

    # Cluster center: use the BCG/central-galaxy coordinates from the matched table.
    ra_col = choose_col(cluster_table, ["RA_x"])
    dec_col = choose_col(cluster_table, ["DEC_x"])
    z_col = choose_col(cluster_table, ["Z_SPEC_x", "Z_SPEC", "Z_LAMBDA"])
    required_cols = ["LAMBDA", "lambda_spec_tot"]
    missing = [col for col in required_cols if col not in cluster_table.colnames]
    if missing:
        raise KeyError(f"Missing required columns in cluster table: {missing}")

    n_cl = len(cluster_table)
    ra_cl = col_float(cluster_table, ra_col)
    dec_cl = col_float(cluster_table, dec_col)
    z_cl = col_float(cluster_table, z_col)

    if rank == 0:
        print(f"Using cluster centers from columns: {ra_col}, {dec_col}")
        print(f"Using cluster redshift column: {z_col}")

    area_sig_deg2 = np.zeros((n_def, n_cl), dtype=float)
    area_bg_deg2 = np.zeros((n_def, n_cl), dtype=float)
    for idef, radius_def in enumerate(radius_defs):
        _, area_sig_deg2[idef] = annulus_area_for_clusters(
            z_cl, radius_def["r_sig_in_hmpc"], radius_def["r_sig_out_hmpc"]
        )
        _, area_bg_deg2[idef] = annulus_area_for_clusters(
            z_cl, radius_def["r_bg_in_hmpc"], radius_def["r_bg_out_hmpc"]
        )

    n_sig_local = np.zeros((n_def, n_cl), dtype=np.int64)
    n_bg_local = np.zeros((n_def, n_cl), dtype=np.int64)
    covered_sig_local = np.zeros((n_def, n_cl), dtype=float)
    covered_bg_local = np.zeros((n_def, n_cl), dtype=float)
    n_files_touching_local = np.zeros(n_cl, dtype=np.int64)

    max_radius = float(
        max(
            np.max(radius_defs["r_sig_out_hmpc"]),
            np.max(radius_defs["r_bg_out_hmpc"]),
        )
    )

    for i_file, sweep_file in enumerate(my_files, start=1):
        print(
            f"Rank {rank:03d}: [{i_file}/{len(my_files)}] {sweep_file.name}",
            flush=True,
        )
        sweep_bounds = read_sweep_bounds(sweep_file)
        overlap = clusters_overlapping_sweep_box(
            ra_cl, dec_cl, z_cl, sweep_bounds, rmax_hmpc=max_radius
        )
        idx = np.where(overlap)[0]
        if len(idx) == 0:
            continue

        r_mag_limit = None if args.no_r_mag_limit else args.r_mag_limit
        dr9 = read_one_dr9_sweep(sweep_file, r_mag_limit=r_mag_limit)
        ra_gal = np.asarray(dr9["RA"], dtype=float)
        dec_gal = np.asarray(dr9["DEC"], dtype=float)
        n_files_touching_local[idx] += 1

        for idef, radius_def in enumerate(radius_defs):
            n_sig, _ = count_galaxies_in_annuli(
                ra_cl[idx],
                dec_cl[idx],
                z_cl[idx],
                ra_gal,
                dec_gal,
                rmin_hmpc=float(radius_def["r_sig_in_hmpc"]),
                rmax_hmpc=float(radius_def["r_sig_out_hmpc"]),
                nside=args.nside_gal,
            )
            n_bg, _ = count_galaxies_in_annuli(
                ra_cl[idx],
                dec_cl[idx],
                z_cl[idx],
                ra_gal,
                dec_gal,
                rmin_hmpc=float(radius_def["r_bg_in_hmpc"]),
                rmax_hmpc=float(radius_def["r_bg_out_hmpc"]),
                nside=args.nside_gal,
            )
            cov_sig, _ = sweep_annulus_coverage_for_clusters(
                ra_cl[idx],
                dec_cl[idx],
                z_cl[idx],
                sweep_bounds,
                rmin_hmpc=float(radius_def["r_sig_in_hmpc"]),
                rmax_hmpc=float(radius_def["r_sig_out_hmpc"]),
                nside=args.nside_sweep_coverage,
            )
            cov_bg, _ = sweep_annulus_coverage_for_clusters(
                ra_cl[idx],
                dec_cl[idx],
                z_cl[idx],
                sweep_bounds,
                rmin_hmpc=float(radius_def["r_bg_in_hmpc"]),
                rmax_hmpc=float(radius_def["r_bg_out_hmpc"]),
                nside=args.nside_sweep_coverage,
            )

            n_sig_local[idef, idx] += n_sig
            n_bg_local[idef, idx] += n_bg
            covered_sig_local[idef, idx] += (
                np.nan_to_num(cov_sig, nan=0.0) * area_sig_deg2[idef, idx]
            )
            covered_bg_local[idef, idx] += (
                np.nan_to_num(cov_bg, nan=0.0) * area_bg_deg2[idef, idx]
            )

        del dr9, ra_gal, dec_gal, n_sig, n_bg, cov_sig, cov_bg
        gc.collect()

    n_sig_all = reduce_sum(comm, n_sig_local)
    n_bg_all = reduce_sum(comm, n_bg_local)
    covered_sig_all = reduce_sum(comm, covered_sig_local)
    covered_bg_all = reduce_sum(comm, covered_bg_local)
    n_files_touching_all = reduce_sum(comm, n_files_touching_local)

    if rank != 0:
        return 0

    radius_defs.write(
        output_dir / "radius_definitions.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )

    keep_cols = [
        col
        for col in [
            "ID",
            ra_col,
            dec_col,
            z_col,
            "LAMBDA",
            "lambda_spec_tot",
            "lambda_spec",
            "lambda_true",
        ]
        if col in cluster_table.colnames
    ]

    base = cluster_table[keep_cols]
    long_tables = []
    summary_rows = []
    richness_cols = ["LAMBDA", "lambda_spec_tot"]

    for idef, radius_def in enumerate(radius_defs):
        coverage_sig_raw = safe_divide(covered_sig_all[idef], area_sig_deg2[idef])
        coverage_bg_raw = safe_divide(covered_bg_all[idef], area_bg_deg2[idef])
        coverage_sig = np.clip(coverage_sig_raw, 0.0, 1.0)
        coverage_bg = np.clip(coverage_bg_raw, 0.0, 1.0)

        sigma_sig_geom = safe_divide(n_sig_all[idef], area_sig_deg2[idef])
        sigma_bg_geom = safe_divide(n_bg_all[idef], area_bg_deg2[idef])
        sigma_sig_covcorr = safe_divide(n_sig_all[idef], covered_sig_all[idef])
        sigma_bg_covcorr = safe_divide(n_bg_all[idef], covered_bg_all[idef])
        sigma_excess = sigma_sig_covcorr - sigma_bg_covcorr
        n_excess = sigma_excess * area_sig_deg2[idef]

        one = base.copy()
        for col in radius_defs.colnames:
            one[col] = np.repeat(radius_def[col], n_cl)
        one["N_signal"] = n_sig_all[idef]
        one["N_background"] = n_bg_all[idef]
        one["area_signal_deg2"] = area_sig_deg2[idef]
        one["area_background_deg2"] = area_bg_deg2[idef]
        one["covered_area_signal_deg2"] = covered_sig_all[idef]
        one["covered_area_background_deg2"] = covered_bg_all[idef]
        one["coverage_signal"] = coverage_sig
        one["coverage_background"] = coverage_bg
        one["coverage_signal_raw"] = coverage_sig_raw
        one["coverage_background_raw"] = coverage_bg_raw
        one["n_files_touching_cluster"] = n_files_touching_all
        one["Sigma_signal_geom"] = sigma_sig_geom
        one["Sigma_background_geom"] = sigma_bg_geom
        one["Sigma_signal_covcorr"] = sigma_sig_covcorr
        one["Sigma_background_covcorr"] = sigma_bg_covcorr
        one["Sigma_excess"] = sigma_excess
        one["N_excess"] = n_excess
        long_tables.append(one)

        summary_mask = (
            np.isfinite(sigma_excess)
            & np.isfinite(n_excess)
            & (coverage_sig >= args.coverage_min_for_summary)
            & (coverage_bg >= args.coverage_min_for_summary)
        )
        y_arrays = {
            "Sigma_excess": sigma_excess,
            "N_excess": n_excess,
            "Sigma_signal_covcorr": sigma_sig_covcorr,
            "Sigma_background_covcorr": sigma_bg_covcorr,
        }
        for richness_col in richness_cols:
            x = col_float(cluster_table, richness_col)
            for y_col, y in y_arrays.items():
                stats = correlation_summary(x, y, summary_mask)
                summary_rows.append(
                    {
                        "radius_def_id": int(radius_def["radius_def_id"]),
                        "radius_label": str(radius_def["radius_label"]),
                        "r_sig_in_hmpc": float(radius_def["r_sig_in_hmpc"]),
                        "r_sig_out_hmpc": float(radius_def["r_sig_out_hmpc"]),
                        "r_bg_in_hmpc": float(radius_def["r_bg_in_hmpc"]),
                        "r_bg_out_hmpc": float(radius_def["r_bg_out_hmpc"]),
                        "richness_col": richness_col,
                        "y_col": y_col,
                        "coverage_min_for_summary": args.coverage_min_for_summary,
                        **stats,
                    }
                )

    output_table = vstack(long_tables, metadata_conflicts="silent")
    summary = Table(rows=summary_rows)
    summary["abs_spearman_r"] = np.abs(col_float(summary, "spearman_r"))
    summary.sort("abs_spearman_r")
    summary.reverse()

    output_ecsv = output_dir / "rm_dr9_radius_grid_overdensity.ecsv"
    output_fits = output_dir / "rm_dr9_radius_grid_overdensity.fits"
    summary_ecsv = output_dir / "rm_dr9_radius_grid_overdensity_summary.ecsv"
    summary_fits = output_dir / "rm_dr9_radius_grid_overdensity_summary.fits"

    output_table.write(output_ecsv, format="ascii.ecsv", overwrite=True)
    output_table.write(output_fits, overwrite=True)
    summary.write(summary_ecsv, format="ascii.ecsv", overwrite=True)
    summary.write(summary_fits, overwrite=True)

    print("\nFinished DR9 radius-grid local-overdensity MPI run")
    print(f"Rows written: {len(output_table):,}")
    print("Top correlation summary rows:")
    print(summary[:10])
    print("Saved:")
    print(" ", output_ecsv)
    print(" ", output_fits)
    print(" ", summary_ecsv)
    print(" ", summary_fits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
