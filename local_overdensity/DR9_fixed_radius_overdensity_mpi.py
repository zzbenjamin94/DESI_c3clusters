"""
MPI-parallel DR9 local-overdensity calculation for one fixed radius definition.

This production script uses the radius definition selected from the radius-grid
test:

    signal annulus:      1.5 < R < 7  h^-1 Mpc
    background annulus:  7   < R < 12 h^-1 Mpc

For each redMaPPer cluster, the code counts DR9 photometric galaxies in the
signal and background annuli, estimates the sweep-file footprint coverage of
each annulus, and writes coverage-corrected density and excess-density columns.
The cluster-level catalog is expected to contain one central BCG position per
cluster using RA_x and DEC_x, redMaPPer richness LAMBDA, and spectroscopic
richness lambda_spec_true.

Example quick test on Perlmutter:

    srun -n 16 -c 1 python local_overdensity/DR9_fixed_radius_overdensity_mpi.py \
        --max-files 100

Full run:

    srun -n 16 -c 1 python local_overdensity/DR9_fixed_radius_overdensity_mpi.py

Optional bright DR9 cut:

    srun -n 16 -c 1 python local_overdensity/DR9_fixed_radius_overdensity_mpi.py \
        --r-mag-limit 19.5
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
from astropy.table import Table, unique
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-path",
        type=Path,
        default=INPUT_DIR / "rm_clusters_with_spec_richness.pickle",
        help=(
            "Cluster-level pickle table. Must contain RA_x, DEC_x, LAMBDA, "
            "and lambda_spec_true."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR / "fixed_radius_overdensity_1p5_7",
        help="Directory for output tables and sweep manifest.",
    )
    parser.add_argument(
        "--sweep-dir",
        action="append",
        type=Path,
        default=None,
        help="DR9 sweep directory. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--sweep-manifest",
        type=Path,
        default=None,
        help=(
            "Optional text file with one sweep FITS path per line. Use this to "
            "force the MPI run and diagnostic notebooks to read exactly the "
            "same sweep files."
        ),
    )
    parser.add_argument("--pattern", default="sweep-*.fits")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--r-mag-limit",
        type=float,
        default=None,
        help=(
            "Optional dereddened r-band magnitude limit for DR9 photometric "
            "galaxies. Default is no magnitude limit."
        ),
    )
    parser.add_argument("--r-sig-in-hmpc", type=float, default=1.5)
    parser.add_argument("--r-sig-out-hmpc", type=float, default=7.0)
    parser.add_argument("--r-bg-in-hmpc", type=float, default=7.0)
    parser.add_argument("--r-bg-out-hmpc", type=float, default=12.0)
    parser.add_argument("--nside-gal", type=int, default=4096)
    parser.add_argument("--nside-sweep-coverage", type=int, default=1024)
    parser.add_argument(
        "--coverage-min-for-summary",
        type=float,
        default=0.0,
        help="Optional coverage cut used only for correlation summary rows.",
    )
    return parser.parse_args()


def find_sweep_files(
    sweep_dirs,
    pattern: str = "sweep-*.fits",
    max_files: int | None = None,
    manifest: Path | None = None,
) -> list[Path]:
    if manifest is not None:
        with manifest.open() as handle:
            files = [
                Path(line.strip())
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            ]
    else:
        files = []
        for sweep_dir in sweep_dirs:
            sweep_dir = Path(sweep_dir)
            files.extend(glob(str(sweep_dir / pattern)))
            files.extend(glob(str(sweep_dir / "*" / pattern)))
        files = sorted(set(Path(f) for f in files))

    if max_files is not None:
        files = files[:max_files]
    if len(files) == 0:
        if manifest is not None:
            raise FileNotFoundError(f"No sweep files listed in manifest: {manifest}")
        raise FileNotFoundError(
            f"No sweep files matched {[str(Path(d) / pattern) for d in sweep_dirs]}"
        )
    return [Path(f) for f in files]


def write_sweep_manifest(files: list[Path], output_path: Path) -> None:
    with output_path.open("w") as handle:
        for filename in files:
            handle.write(f"{filename}\n")


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


def require_cols(table: Table, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in table.colnames]
    if missing:
        lambda_cols = [col for col in table.colnames if "lambda" in col.lower()]
        raise KeyError(
            "Missing required columns in cluster table: "
            f"{missing}. Available lambda-like columns are: {lambda_cols}"
        )


def col_float(table: Table, col: str) -> np.ndarray:
    arr = np.ma.asarray(table[col], dtype=float)
    return np.ma.filled(arr, np.nan)


def finite_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.isfinite(x) & (x > 0)


def reduce_sum(comm, local_array, root: int = 0):
    if comm is None:
        return local_array
    if MPI is None:
        raise RuntimeError("MPI communicator is active, but mpi4py.MPI is unavailable.") from MPI_IMPORT_ERROR
    global_array = np.empty_like(local_array) if comm.Get_rank() == root else None
    comm.Reduce(local_array, global_array, op=MPI.SUM, root=root)
    return global_array


def correlation_summary(x, y, mask) -> dict[str, float | int]:
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if MPI is None:
        comm = None
        rank = 0
        size = 1
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    sweep_dirs = args.sweep_dir or DEFAULT_SWEEP_DIRS
    sweep_files = find_sweep_files(
        sweep_dirs,
        pattern=args.pattern,
        max_files=args.max_files,
        manifest=args.sweep_manifest,
    )
    my_files = sweep_files[rank::size]

    if rank == 0:
        write_sweep_manifest(
            sweep_files,
            args.output_dir / "sweep_files_used_by_fixed_radius_mpi.txt",
        )
        print(f"MPI size: {size}")
        print(f"Catalog: {args.catalog_path}")
        print(f"Output dir: {args.output_dir}")
        print(f"Found {len(sweep_files):,} sweep files")
        print(f"Signal annulus: {args.r_sig_in_hmpc:g} < R < {args.r_sig_out_hmpc:g} h^-1 Mpc")
        print(f"Background annulus: {args.r_bg_in_hmpc:g} < R < {args.r_bg_out_hmpc:g} h^-1 Mpc")
        print(
            "DR9 r-band magnitude limit: "
            + ("none" if args.r_mag_limit is None else str(args.r_mag_limit))
        )
    print(f"Rank {rank:03d}: assigned {len(my_files):,} files", flush=True)

    cluster_table = read_pickle_table(args.catalog_path)
    if "ID" in cluster_table.colnames:
        cluster_table = unique(cluster_table, keys="ID")

    ra_col = choose_col(cluster_table, ["RA_x"])
    dec_col = choose_col(cluster_table, ["DEC_x"])
    z_col = choose_col(cluster_table, ["Z_SPEC_x", "Z_SPEC", "Z_LAMBDA"])
    require_cols(cluster_table, ["LAMBDA", "lambda_spec_true"])

    n_cl = len(cluster_table)
    ra_cl = col_float(cluster_table, ra_col)
    dec_cl = col_float(cluster_table, dec_col)
    z_cl = col_float(cluster_table, z_col)

    _, area_sig_deg2 = annulus_area_for_clusters(
        z_cl, args.r_sig_in_hmpc, args.r_sig_out_hmpc
    )
    _, area_bg_deg2 = annulus_area_for_clusters(
        z_cl, args.r_bg_in_hmpc, args.r_bg_out_hmpc
    )

    n_sig_local = np.zeros(n_cl, dtype=np.int64)
    n_bg_local = np.zeros(n_cl, dtype=np.int64)
    covered_sig_local = np.zeros(n_cl, dtype=float)
    covered_bg_local = np.zeros(n_cl, dtype=float)
    n_files_touching_local = np.zeros(n_cl, dtype=np.int64)
    max_radius = max(args.r_sig_out_hmpc, args.r_bg_out_hmpc)

    if rank == 0:
        print(f"Using cluster centers from columns: {ra_col}, {dec_col}")
        print(f"Using cluster redshift column: {z_col}")
        print("Using spectroscopic richness column: lambda_spec_true")

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

        dr9 = read_one_dr9_sweep(sweep_file, r_mag_limit=args.r_mag_limit)
        ra_gal = np.asarray(dr9["RA"], dtype=float)
        dec_gal = np.asarray(dr9["DEC"], dtype=float)
        n_files_touching_local[idx] += 1

        n_sig, _ = count_galaxies_in_annuli(
            ra_cl[idx],
            dec_cl[idx],
            z_cl[idx],
            ra_gal,
            dec_gal,
            rmin_hmpc=args.r_sig_in_hmpc,
            rmax_hmpc=args.r_sig_out_hmpc,
            nside=args.nside_gal,
        )
        n_bg, _ = count_galaxies_in_annuli(
            ra_cl[idx],
            dec_cl[idx],
            z_cl[idx],
            ra_gal,
            dec_gal,
            rmin_hmpc=args.r_bg_in_hmpc,
            rmax_hmpc=args.r_bg_out_hmpc,
            nside=args.nside_gal,
        )
        cov_sig, _ = sweep_annulus_coverage_for_clusters(
            ra_cl[idx],
            dec_cl[idx],
            z_cl[idx],
            sweep_bounds,
            rmin_hmpc=args.r_sig_in_hmpc,
            rmax_hmpc=args.r_sig_out_hmpc,
            nside=args.nside_sweep_coverage,
        )
        cov_bg, _ = sweep_annulus_coverage_for_clusters(
            ra_cl[idx],
            dec_cl[idx],
            z_cl[idx],
            sweep_bounds,
            rmin_hmpc=args.r_bg_in_hmpc,
            rmax_hmpc=args.r_bg_out_hmpc,
            nside=args.nside_sweep_coverage,
        )

        n_sig_local[idx] += n_sig
        n_bg_local[idx] += n_bg
        covered_sig_local[idx] += np.nan_to_num(cov_sig, nan=0.0) * area_sig_deg2[idx]
        covered_bg_local[idx] += np.nan_to_num(cov_bg, nan=0.0) * area_bg_deg2[idx]

        del dr9, ra_gal, dec_gal, n_sig, n_bg, cov_sig, cov_bg
        gc.collect()

    n_sig_all = reduce_sum(comm, n_sig_local)
    n_bg_all = reduce_sum(comm, n_bg_local)
    covered_sig_all = reduce_sum(comm, covered_sig_local)
    covered_bg_all = reduce_sum(comm, covered_bg_local)
    n_files_touching_all = reduce_sum(comm, n_files_touching_local)

    if rank != 0:
        return 0

    coverage_sig_raw = safe_divide(covered_sig_all, area_sig_deg2)
    coverage_bg_raw = safe_divide(covered_bg_all, area_bg_deg2)
    coverage_sig = np.clip(coverage_sig_raw, 0.0, 1.0)
    coverage_bg = np.clip(coverage_bg_raw, 0.0, 1.0)

    sigma_sig_geom = safe_divide(n_sig_all, area_sig_deg2)
    sigma_bg_geom = safe_divide(n_bg_all, area_bg_deg2)
    sigma_sig_covcorr = safe_divide(n_sig_all, covered_sig_all)
    sigma_bg_covcorr = safe_divide(n_bg_all, covered_bg_all)
    sigma_excess = sigma_sig_covcorr - sigma_bg_covcorr
    n_excess = sigma_excess * area_sig_deg2

    keep_cols = [
        col
        for col in [
            "ID",
            ra_col,
            dec_col,
            z_col,
            "Z_LAMBDA",
            "LAMBDA",
            "lambda_spec_true",
            "lambda_spec_tot",
            "lambda_true",
            "lambda_tot",
        ]
        if col in cluster_table.colnames
    ]
    output_table = cluster_table[keep_cols].copy()
    output_table["r_sig_in_hmpc"] = np.repeat(args.r_sig_in_hmpc, n_cl)
    output_table["r_sig_out_hmpc"] = np.repeat(args.r_sig_out_hmpc, n_cl)
    output_table["r_bg_in_hmpc"] = np.repeat(args.r_bg_in_hmpc, n_cl)
    output_table["r_bg_out_hmpc"] = np.repeat(args.r_bg_out_hmpc, n_cl)
    output_table["N_signal"] = n_sig_all
    output_table["N_background"] = n_bg_all
    output_table["area_signal_deg2"] = area_sig_deg2
    output_table["area_background_deg2"] = area_bg_deg2
    output_table["covered_area_signal_deg2"] = covered_sig_all
    output_table["covered_area_background_deg2"] = covered_bg_all
    output_table["coverage_signal"] = coverage_sig
    output_table["coverage_background"] = coverage_bg
    output_table["coverage_signal_raw"] = coverage_sig_raw
    output_table["coverage_background_raw"] = coverage_bg_raw
    output_table["n_files_touching_cluster"] = n_files_touching_all
    output_table["Sigma_signal_geom"] = sigma_sig_geom
    output_table["Sigma_background_geom"] = sigma_bg_geom
    output_table["Sigma_signal_covcorr"] = sigma_sig_covcorr
    output_table["Sigma_background_covcorr"] = sigma_bg_covcorr
    output_table["Sigma_excess"] = sigma_excess
    output_table["N_excess"] = n_excess

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
    summary_rows = []
    for richness_col in ["LAMBDA", "lambda_spec_true"]:
        x = col_float(cluster_table, richness_col)
        for y_col, y in y_arrays.items():
            stats = correlation_summary(x, y, summary_mask)
            summary_rows.append(
                {
                    "richness_col": richness_col,
                    "y_col": y_col,
                    "r_sig_in_hmpc": args.r_sig_in_hmpc,
                    "r_sig_out_hmpc": args.r_sig_out_hmpc,
                    "r_bg_in_hmpc": args.r_bg_in_hmpc,
                    "r_bg_out_hmpc": args.r_bg_out_hmpc,
                    "coverage_min_for_summary": args.coverage_min_for_summary,
                    **stats,
                }
            )

    summary = Table(rows=summary_rows)
    summary["abs_spearman_r"] = np.abs(col_float(summary, "spearman_r"))
    summary.sort("abs_spearman_r")
    summary.reverse()

    output_ecsv = args.output_dir / "rm_dr9_fixed_radius_overdensity_1p5_7.ecsv"
    output_fits = args.output_dir / "rm_dr9_fixed_radius_overdensity_1p5_7.fits"
    summary_ecsv = args.output_dir / "rm_dr9_fixed_radius_overdensity_1p5_7_summary.ecsv"
    summary_fits = args.output_dir / "rm_dr9_fixed_radius_overdensity_1p5_7_summary.fits"

    output_table.write(output_ecsv, format="ascii.ecsv", overwrite=True)
    output_table.write(output_fits, overwrite=True)
    summary.write(summary_ecsv, format="ascii.ecsv", overwrite=True)
    summary.write(summary_fits, overwrite=True)

    print("\nFinished fixed-radius DR9 local-overdensity MPI run")
    print(f"Rows written: {len(output_table):,}")
    print("Correlation summary:")
    print(summary)
    print("Saved:")
    print(" ", output_ecsv)
    print(" ", output_fits)
    print(" ", summary_ecsv)
    print(" ", summary_fits)
    print(" ", args.output_dir / "sweep_files_used_by_fixed_radius_mpi.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
