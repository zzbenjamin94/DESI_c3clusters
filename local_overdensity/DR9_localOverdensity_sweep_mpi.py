"""
MPI-parallel DR9 local-overdensity sweep calculation.

Each MPI rank receives a subset of DR9 sweep files, reads one file at a time,
counts selected DR9 photometric galaxies in cluster-centered annuli, estimates
the sweep-footprint coverage for those annuli, and accumulates rank-local
arrays. Rank 0 then reduces the arrays, writes the final redMaPPer cluster table
with local-overdensity columns, and saves all-sweep density diagnostics.

Example on Perlmutter, from the repository root:

    srun -n 8 -c 1 python local_overdensity/DR9_localOverdensity_sweep_mpi.py \
        --max-files 32

Remove ``--max-files`` for the full run.
"""

from __future__ import annotations

import argparse
import gc
import pickle
import sys
import warnings
from glob import glob
from pathlib import Path

import astropy.units as u
from astropy.cosmology import Planck18
from astropy.table import Table, unique
from astropy.units import UnitsWarning
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MPI_IMPORT_ERROR = None
try:
    from mpi4py import MPI
except Exception as exc:
    MPI = None
    MPI_IMPORT_ERROR = exc


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "catalogs"
OUTPUT_DIR = REPO_ROOT / "local_overdensity" / "dr9_outputs"

DEFAULT_SWEEP_DIRS = [
    Path("/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/north/sweep/9.0"),
    Path("/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0"),
]


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
        default=OUTPUT_DIR,
        help="Directory for local-overdensity outputs.",
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
    parser.add_argument("--r-mag-limit", type=float, default=23.5)
    parser.add_argument("--rmin-hmpc", type=float, default=1.5)
    parser.add_argument("--rmax-hmpc", type=float, default=8.0)
    parser.add_argument("--bg-rmin-hmpc", type=float, default=12.0)
    parser.add_argument("--bg-rmax-hmpc", type=float, default=20.0)
    parser.add_argument("--coverage-min", type=float, default=0.8)
    parser.add_argument("--nside-gal", type=int, default=4096)
    parser.add_argument("--nside-sweep-coverage", type=int, default=1024)
    parser.add_argument("--nside-density", type=int, default=256)
    parser.add_argument(
        "--skip-density-diagnostic",
        action="store_true",
        help="Skip the all-sweep HEALPix galaxy-density diagnostic map.",
    )
    return parser.parse_args()


def rank_print(rank: int, *args, **kwargs) -> None:
    if rank == 0:
        print(*args, **kwargs)


def nanomaggies_to_mag(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    mag = np.full(flux.shape, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    return mag


def dereddened_mag(table: Table, band: str) -> np.ndarray:
    band = band.upper()
    flux = np.asarray(table[f"FLUX_{band}"], dtype=float)
    transmission = np.asarray(table[f"MW_TRANSMISSION_{band}"], dtype=float)
    return nanomaggies_to_mag(flux / transmission)


def dr9_galaxy_mask(table: Table, r_mag_limit: float | None = 23.5) -> np.ndarray:
    mask = np.isfinite(table["RA"]) & np.isfinite(table["DEC"])
    if "TYPE" in table.colnames:
        mask &= np.asarray(table["TYPE"]) != "PSF"
    for band in ("G", "R", "Z"):
        col = f"FLUX_IVAR_{band}"
        if col in table.colnames:
            mask &= np.asarray(table[col], dtype=float) > 0
    if r_mag_limit is not None:
        r_dered = dereddened_mag(table, "R")
        mask &= np.isfinite(r_dered) & (r_dered < r_mag_limit)
    return np.asarray(mask, dtype=bool)


def read_table_silent_units(filename: str | Path) -> Table:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*did not parse as fits unit.*",
            category=UnitsWarning,
        )
        return Table.read(filename, hdu=1, memmap=True)


def read_one_dr9_sweep(filename: str | Path, r_mag_limit: float = 23.5) -> Table:
    table = read_table_silent_units(filename)
    mask = dr9_galaxy_mask(table, r_mag_limit=r_mag_limit)
    out = table[mask][["RA", "DEC", "TYPE"]].copy()
    return out


def read_sweep_bounds(filename: str | Path) -> dict[str, float]:
    table = read_table_silent_units(filename)
    ra = np.asarray(table["RA"], dtype=float)
    dec = np.asarray(table["DEC"], dtype=float)
    good = np.isfinite(ra) & np.isfinite(dec)
    return {
        "ra_min": float(np.nanmin(ra[good])),
        "ra_max": float(np.nanmax(ra[good])),
        "dec_min": float(np.nanmin(dec[good])),
        "dec_max": float(np.nanmax(dec[good])),
    }


def inside_ra_dec_box(ra, dec, bounds):
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    return (
        (ra >= bounds["ra_min"])
        & (ra <= bounds["ra_max"])
        & (dec >= bounds["dec_min"])
        & (dec <= bounds["dec_max"])
    )


def radec_to_unitvec(ra_deg, dec_deg):
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cosd = np.cos(dec)
    return np.column_stack((cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec)))


def annulus_theta_radians(
    z, Rmin_hMpc=1.5, Rmax_hMpc=10.0, cosmo=Planck18
) -> tuple[float, float]:
    h = cosmo.h
    rmin_mpc = (Rmin_hMpc / h) * u.Mpc
    rmax_mpc = (Rmax_hMpc / h) * u.Mpc
    dm = cosmo.comoving_transverse_distance(z)
    th_min = (rmin_mpc / dm).decompose().value
    th_max = (rmax_mpc / dm).decompose().value
    return th_min, th_max


def spherical_annulus_area_sr(theta_min, theta_max):
    return 2.0 * np.pi * (np.cos(theta_min) - np.cos(theta_max))


def annulus_area_for_clusters(z_cl, rmin_hmpc, rmax_hmpc):
    z_cl = np.asarray(z_cl, dtype=float)
    area_sr = np.full(len(z_cl), np.nan, dtype=float)
    for i, z in enumerate(z_cl):
        if not np.isfinite(z) or z <= 0:
            continue
        th_min, th_max = annulus_theta_radians(
            z, Rmin_hMpc=rmin_hmpc, Rmax_hMpc=rmax_hmpc
        )
        area_sr[i] = spherical_annulus_area_sr(th_min, th_max)
    return area_sr, area_sr * (180.0 / np.pi) ** 2


def build_galaxy_healpix_index(ra_gal, dec_gal, nside=4096, nest=False):
    theta = np.deg2rad(90.0 - np.asarray(dec_gal, dtype=float))
    phi = np.deg2rad(np.asarray(ra_gal, dtype=float))
    pix = hp.ang2pix(nside, theta, phi, nest=nest)
    order = np.argsort(pix)
    pix_sorted = pix[order]
    uniq_pix, start = np.unique(pix_sorted, return_index=True)
    end = np.r_[start[1:], len(pix_sorted)]
    return {
        "nside": nside,
        "nest": nest,
        "order": order,
        "uniq_pix": uniq_pix,
        "start": start,
        "end": end,
    }


def gather_candidates(index, pix_list):
    uniq_pix = index["uniq_pix"]
    pix_list = np.asarray(pix_list, dtype=uniq_pix.dtype)
    pos = np.searchsorted(uniq_pix, pix_list)
    inside = pos < len(uniq_pix)
    pos_valid = pos[inside]
    pix_valid = pix_list[inside]
    good = uniq_pix[pos_valid] == pix_valid
    pos = pos_valid[good]
    if len(pos) == 0:
        return np.empty(0, dtype=np.int64)
    order = index["order"]
    start = index["start"]
    end = index["end"]
    return np.concatenate([order[start[p] : end[p]] for p in pos])


def count_galaxies_in_annuli(
    ra_cl,
    dec_cl,
    z_cl,
    ra_gal,
    dec_gal,
    rmin_hmpc=1.5,
    rmax_hmpc=10.0,
    nside=4096,
    nest=False,
):
    ra_cl = np.asarray(ra_cl, dtype=float)
    dec_cl = np.asarray(dec_cl, dtype=float)
    z_cl = np.asarray(z_cl, dtype=float)
    gal_vecs = radec_to_unitvec(ra_gal, dec_gal)
    cl_vecs = radec_to_unitvec(ra_cl, dec_cl)
    index = build_galaxy_healpix_index(ra_gal, dec_gal, nside=nside, nest=nest)

    counts = np.zeros(len(ra_cl), dtype=np.int64)
    area_sr = np.full(len(ra_cl), np.nan, dtype=float)
    for i, (vec, z) in enumerate(zip(cl_vecs, z_cl)):
        if not np.isfinite(z) or z <= 0:
            continue
        th_min, th_max = annulus_theta_radians(
            z, Rmin_hMpc=rmin_hmpc, Rmax_hMpc=rmax_hmpc
        )
        area_sr[i] = spherical_annulus_area_sr(th_min, th_max)
        pix_outer = hp.query_disc(nside, vec, th_max, inclusive=True, nest=nest)
        cand = gather_candidates(index, pix_outer)
        if len(cand) == 0:
            continue
        dot = np.clip(gal_vecs[cand] @ vec, -1.0, 1.0)
        sep = np.arccos(dot)
        counts[i] = np.count_nonzero((sep >= th_min) & (sep < th_max))
    return counts, area_sr


def sweep_annulus_coverage_for_clusters(
    ra_cl,
    dec_cl,
    z_cl,
    sweep_bounds,
    rmin_hmpc=1.5,
    rmax_hmpc=8.0,
    nside=1024,
    nest=False,
):
    ra_cl = np.asarray(ra_cl, dtype=float)
    dec_cl = np.asarray(dec_cl, dtype=float)
    z_cl = np.asarray(z_cl, dtype=float)
    cl_vecs = radec_to_unitvec(ra_cl, dec_cl)
    coverage = np.full(len(ra_cl), np.nan, dtype=float)
    n_pix_annulus = np.zeros(len(ra_cl), dtype=int)
    for i, (vec, z) in enumerate(zip(cl_vecs, z_cl)):
        if not np.isfinite(z) or z <= 0:
            continue
        th_min, th_max = annulus_theta_radians(
            z, Rmin_hMpc=rmin_hmpc, Rmax_hMpc=rmax_hmpc
        )
        pix_out = hp.query_disc(nside, vec, th_max, inclusive=True, nest=nest)
        pix_in = hp.query_disc(nside, vec, th_min, inclusive=True, nest=nest)
        pix_ann = np.setdiff1d(pix_out, pix_in, assume_unique=False)
        n_pix_annulus[i] = len(pix_ann)
        if len(pix_ann) == 0:
            continue
        theta_pix, phi_pix = hp.pix2ang(nside, pix_ann, nest=nest)
        ra_pix = np.rad2deg(phi_pix)
        dec_pix = 90.0 - np.rad2deg(theta_pix)
        inside = inside_ra_dec_box(ra_pix, dec_pix, sweep_bounds)
        coverage[i] = np.count_nonzero(inside) / len(pix_ann)
    return coverage, n_pix_annulus


def clusters_overlapping_sweep_box(ra_cl, dec_cl, z_cl, sweep_bounds, rmax_hmpc):
    ra_cl = np.asarray(ra_cl, dtype=float)
    dec_cl = np.asarray(dec_cl, dtype=float)
    z_cl = np.asarray(z_cl, dtype=float)
    theta_max_deg = np.full(len(z_cl), np.nan, dtype=float)
    finite_z = np.isfinite(z_cl) & (z_cl > 0)
    theta_max_deg[finite_z] = [
        np.rad2deg(annulus_theta_radians(z, Rmax_hMpc=rmax_hmpc)[1])
        for z in z_cl[finite_z]
    ]
    cos_dec = np.cos(np.deg2rad(dec_cl))
    ra_pad = theta_max_deg / np.clip(cos_dec, 0.2, None)
    dec_pad = theta_max_deg
    return (
        finite_z
        & (ra_cl >= sweep_bounds["ra_min"] - ra_pad)
        & (ra_cl <= sweep_bounds["ra_max"] + ra_pad)
        & (dec_cl >= sweep_bounds["dec_min"] - dec_pad)
        & (dec_cl <= sweep_bounds["dec_max"] + dec_pad)
    )


def safe_divide(numerator, denominator, min_denominator=1.0e-12):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    num = np.broadcast_to(numerator, out.shape)
    den = np.broadcast_to(denominator, out.shape)
    good = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > min_denominator)
    out[good] = num[good] / den[good]
    return out


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


def reduce_sum(comm, local_array, root=0):
    if comm is None:
        return local_array
    if MPI is None:
        raise RuntimeError("MPI communicator is active, but mpi4py.MPI is unavailable.") from MPI_IMPORT_ERROR
    global_array = np.empty_like(local_array) if comm.Get_rank() == root else None
    comm.Reduce(local_array, global_array, op=MPI.SUM, root=root)
    return global_array


def write_density_diagnostics(
    output_dir,
    galaxy_counts_map,
    file_density_table,
    cluster_ra,
    cluster_dec,
    nside_density,
):
    pix_area_deg2 = hp.nside2pixarea(nside_density, degrees=True)
    density_map_deg2 = galaxy_counts_map.astype(float) / pix_area_deg2
    density_plot = density_map_deg2.copy()
    density_plot[galaxy_counts_map == 0] = hp.UNSEEN

    finite_density = density_map_deg2[galaxy_counts_map > 0]
    vmin = vmax = None
    if len(finite_density) > 0:
        vmin, vmax = np.nanpercentile(finite_density, [2, 98])

    fig = plt.figure(figsize=(11, 6.5))
    hp.mollview(
        density_plot,
        fig=fig.number,
        title="DR9 selected galaxy surface density, all north+south sweeps",
        unit=r"galaxies deg$^{-2}$",
        min=vmin,
        max=vmax,
        cmap="viridis",
    )
    hp.projscatter(
        cluster_ra,
        cluster_dec,
        lonlat=True,
        s=2,
        alpha=0.35,
        color="crimson",
    )
    hp.graticule(color="white", alpha=0.25)
    fig.savefig(
        output_dir / "dr9_all_sweeps_mollview_density_clusters.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    good_density = np.isfinite(file_density_table["selected_density_deg2"]) & (
        file_density_table["selected_density_deg2"] > 0
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(
        np.asarray(file_density_table["selected_density_deg2"][good_density], dtype=float),
        bins=50,
        histtype="stepfilled",
        alpha=0.75,
    )
    ax.set_xlabel(r"per-file selected galaxy density [deg$^{-2}$]")
    ax.set_ylabel("number of sweep files")
    ax.set_title("DR9 sweep-to-sweep density variation")
    fig.savefig(output_dir / "dr9_all_sweeps_file_density_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.scatter(
        file_density_table["occupied_area_deg2"],
        file_density_table["n_selected"],
        s=12,
        alpha=0.6,
    )
    ax.set_xlabel(r"occupied HEALPix area [deg$^2$]")
    ax.set_ylabel("selected DR9 galaxies")
    ax.set_title("Selected galaxies per sweep file")
    fig.savefig(output_dir / "dr9_all_sweeps_file_counts_vs_area.png", dpi=180)
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

    rank_print(rank, f"MPI size: {size}")
    rank_print(rank, f"Catalog: {args.catalog_path}")
    rank_print(rank, f"Output dir: {output_dir}")
    rank_print(rank, f"Found {len(sweep_files):,} sweep files")
    for sweep_dir in sweep_dirs:
        rank_print(rank, f"  sweep dir: {sweep_dir}")
    print(f"Rank {rank:03d}: assigned {len(my_files):,} files", flush=True)

    with args.catalog_path.open("rb") as handle:
        bgs_matched = pickle.load(handle)
    rm_tab = unique(bgs_matched, keys="ID")

    n_cl = len(rm_tab)
    ra_cl = np.asarray(rm_tab["RA_x"], dtype=float)
    dec_cl = np.asarray(rm_tab["DEC_x"], dtype=float)
    z_cl = np.asarray(rm_tab["Z_SPEC_x"], dtype=float)

    _, area_signal_deg2_all = annulus_area_for_clusters(
        z_cl, args.rmin_hmpc, args.rmax_hmpc
    )
    _, area_bg_deg2_all = annulus_area_for_clusters(
        z_cl, args.bg_rmin_hmpc, args.bg_rmax_hmpc
    )

    n_signal_local = np.zeros(n_cl, dtype=np.int64)
    n_bg_local = np.zeros(n_cl, dtype=np.int64)
    covered_signal_local = np.zeros(n_cl, dtype=float)
    covered_bg_local = np.zeros(n_cl, dtype=float)
    n_files_touching_local = np.zeros(n_cl, dtype=np.int64)

    npix_density = hp.nside2npix(args.nside_density)
    density_counts_local = np.zeros(npix_density, dtype=np.int64)
    pix_area_deg2 = hp.nside2pixarea(args.nside_density, degrees=True)
    file_rows = []

    for i_file, sweep_file in enumerate(my_files, start=1):
        print(
            f"Rank {rank:03d}: [{i_file}/{len(my_files)}] {sweep_file.name}",
            flush=True,
        )
        sweep_bounds = read_sweep_bounds(sweep_file)
        overlap = clusters_overlapping_sweep_box(
            ra_cl, dec_cl, z_cl, sweep_bounds, rmax_hmpc=args.bg_rmax_hmpc
        )
        idx = np.where(overlap)[0]
        dr9 = None
        if len(idx) > 0 or not args.skip_density_diagnostic:
            dr9 = read_one_dr9_sweep(sweep_file, r_mag_limit=args.r_mag_limit)
            ra_gal = np.asarray(dr9["RA"], dtype=float)
            dec_gal = np.asarray(dr9["DEC"], dtype=float)
            finite = np.isfinite(ra_gal) & np.isfinite(dec_gal)
        else:
            ra_gal = np.empty(0, dtype=float)
            dec_gal = np.empty(0, dtype=float)
            finite = np.zeros(0, dtype=bool)

        if not args.skip_density_diagnostic and np.any(finite):
            pix = hp.ang2pix(
                args.nside_density,
                np.deg2rad(90.0 - dec_gal[finite]),
                np.deg2rad(ra_gal[finite]),
            )
            np.add.at(density_counts_local, pix, 1)
            occupied_pix = np.unique(pix)
            occupied_area_deg2 = len(occupied_pix) * pix_area_deg2
            selected_density = np.count_nonzero(finite) / occupied_area_deg2
            ra_min = float(np.nanmin(ra_gal[finite]))
            ra_max = float(np.nanmax(ra_gal[finite]))
            dec_min = float(np.nanmin(dec_gal[finite]))
            dec_max = float(np.nanmax(dec_gal[finite]))
        else:
            occupied_area_deg2 = 0.0
            selected_density = np.nan
            ra_min = ra_max = dec_min = dec_max = np.nan

        file_rows.append(
            {
                "filename": str(sweep_file),
                "rank": rank,
                "n_selected": int(np.count_nonzero(finite)),
                "n_overlap_clusters": int(len(idx)),
                "occupied_area_deg2": float(occupied_area_deg2),
                "selected_density_deg2": float(selected_density),
                "ra_min": ra_min,
                "ra_max": ra_max,
                "dec_min": dec_min,
                "dec_max": dec_max,
            }
        )

        if len(idx) == 0:
            del dr9
            gc.collect()
            continue

        if dr9 is not None and len(dr9) > 0:
            n_sig, _ = count_galaxies_in_annuli(
                ra_cl[idx],
                dec_cl[idx],
                z_cl[idx],
                ra_gal,
                dec_gal,
                rmin_hmpc=args.rmin_hmpc,
                rmax_hmpc=args.rmax_hmpc,
                nside=args.nside_gal,
            )
            n_bg, _ = count_galaxies_in_annuli(
                ra_cl[idx],
                dec_cl[idx],
                z_cl[idx],
                ra_gal,
                dec_gal,
                rmin_hmpc=args.bg_rmin_hmpc,
                rmax_hmpc=args.bg_rmax_hmpc,
                nside=args.nside_gal,
            )
        else:
            n_sig = np.zeros(len(idx), dtype=np.int64)
            n_bg = np.zeros(len(idx), dtype=np.int64)

        cov_sig, _ = sweep_annulus_coverage_for_clusters(
            ra_cl[idx],
            dec_cl[idx],
            z_cl[idx],
            sweep_bounds,
            rmin_hmpc=args.rmin_hmpc,
            rmax_hmpc=args.rmax_hmpc,
            nside=args.nside_sweep_coverage,
        )
        cov_bg, _ = sweep_annulus_coverage_for_clusters(
            ra_cl[idx],
            dec_cl[idx],
            z_cl[idx],
            sweep_bounds,
            rmin_hmpc=args.bg_rmin_hmpc,
            rmax_hmpc=args.bg_rmax_hmpc,
            nside=args.nside_sweep_coverage,
        )

        n_signal_local[idx] += n_sig
        n_bg_local[idx] += n_bg
        covered_signal_local[idx] += np.nan_to_num(cov_sig, nan=0.0) * area_signal_deg2_all[idx]
        covered_bg_local[idx] += np.nan_to_num(cov_bg, nan=0.0) * area_bg_deg2_all[idx]
        n_files_touching_local[idx] += 1

        del dr9, n_sig, n_bg, cov_sig, cov_bg
        gc.collect()

    n_signal_all = reduce_sum(comm, n_signal_local)
    n_bg_all = reduce_sum(comm, n_bg_local)
    covered_signal_all = reduce_sum(comm, covered_signal_local)
    covered_bg_all = reduce_sum(comm, covered_bg_local)
    n_files_touching_all = reduce_sum(comm, n_files_touching_local)
    density_counts_all = reduce_sum(comm, density_counts_local)

    if comm is not None:
        gathered_rows = comm.gather(file_rows, root=0)
    else:
        gathered_rows = [file_rows]

    if rank != 0:
        return 0

    flat_rows = [row for rows in gathered_rows for row in rows]
    file_density_table = Table(rows=flat_rows)
    file_density_table.write(
        output_dir / "dr9_all_sweeps_file_density_diagnostic.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )

    coverage_signal_raw = safe_divide(covered_signal_all, area_signal_deg2_all)
    coverage_bg_raw = safe_divide(covered_bg_all, area_bg_deg2_all)
    coverage_signal = np.clip(coverage_signal_raw, 0.0, 1.0)
    coverage_bg = np.clip(coverage_bg_raw, 0.0, 1.0)

    sigma_signal = safe_divide(n_signal_all, covered_signal_all)
    sigma_bg = safe_divide(n_bg_all, covered_bg_all)
    sigma_excess_local = sigma_signal - sigma_bg
    n_excess_local = sigma_excess_local * area_signal_deg2_all

    rm_tab_dr9 = rm_tab.copy()
    rm_tab_dr9["Ngal_signal_DR9_annulus"] = n_signal_all
    rm_tab_dr9["Ngal_bg_DR9_annulus"] = n_bg_all
    rm_tab_dr9["area_signal_deg2"] = area_signal_deg2_all
    rm_tab_dr9["area_bg_deg2"] = area_bg_deg2_all
    rm_tab_dr9["covered_area_signal_deg2"] = covered_signal_all
    rm_tab_dr9["covered_area_bg_deg2"] = covered_bg_all
    rm_tab_dr9["coverage_signal_sweep"] = coverage_signal
    rm_tab_dr9["coverage_bg_sweep"] = coverage_bg
    rm_tab_dr9["coverage_signal_sweep_raw"] = coverage_signal_raw
    rm_tab_dr9["coverage_bg_sweep_raw"] = coverage_bg_raw
    rm_tab_dr9["n_files_touching_cluster"] = n_files_touching_all
    rm_tab_dr9["Sigma_signal_covcorr"] = sigma_signal
    rm_tab_dr9["Sigma_bg_covcorr"] = sigma_bg
    rm_tab_dr9["Sigma_excess_local_DR9_annulus"] = sigma_excess_local
    rm_tab_dr9["Nexcess_local_DR9_annulus"] = n_excess_local

    full_mask = (
        (np.asarray(rm_tab_dr9["coverage_signal_sweep"], dtype=float) > args.coverage_min)
        & (np.asarray(rm_tab_dr9["coverage_bg_sweep"], dtype=float) > args.coverage_min)
        & np.isfinite(rm_tab_dr9["Nexcess_local_DR9_annulus"])
        & np.isfinite(rm_tab_dr9["Sigma_excess_local_DR9_annulus"])
    )
    rm_tab_dr9_filt = rm_tab_dr9[full_mask]

    rm_tab_dr9.write(
        output_dir / "rm_dr9_local_overdensity_sweep_mpi.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    rm_tab_dr9.write(output_dir / "rm_dr9_local_overdensity_sweep_mpi.fits", overwrite=True)
    rm_tab_dr9_filt.write(
        output_dir / "rm_dr9_local_overdensity_sweep_mpi_coverage_cut.ecsv",
        format="ascii.ecsv",
        overwrite=True,
    )
    rm_tab_dr9_filt.write(
        output_dir / "rm_dr9_local_overdensity_sweep_mpi_coverage_cut.fits",
        overwrite=True,
    )

    if not args.skip_density_diagnostic:
        write_density_diagnostics(
            output_dir,
            density_counts_all,
            file_density_table,
            ra_cl,
            dec_cl,
            args.nside_density,
        )

    print("\nFinished MPI DR9 local-overdensity sweep")
    print(f"Kept {len(rm_tab_dr9_filt):,}/{len(rm_tab_dr9):,} clusters after coverage cut")
    print(f"Clusters with signal count > 0: {np.count_nonzero(n_signal_all > 0):,}")
    print(f"Clusters touching >=1 sweep file: {np.count_nonzero(n_files_touching_all > 0):,}")
    print("Saved:")
    print(" ", output_dir / "rm_dr9_local_overdensity_sweep_mpi.ecsv")
    print(" ", output_dir / "rm_dr9_local_overdensity_sweep_mpi.fits")
    print(" ", output_dir / "rm_dr9_local_overdensity_sweep_mpi_coverage_cut.ecsv")
    print(" ", output_dir / "rm_dr9_local_overdensity_sweep_mpi_coverage_cut.fits")
    print(" ", output_dir / "dr9_all_sweeps_file_density_diagnostic.ecsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
