"""
Compute redMaPPer geometric coverage fractions from DESI random catalogs.

This is the expensive part of the projected matching workflow, so it is split
from ``projection_match_catalogs.py`` and parallelized with MPI.  Each MPI
rank reads a subset of random catalogs, counts random points inside the angular
aperture around every redMaPPer cluster, and rank 0 combines the partial counts.

Run on NERSC, for example:

    srun -n 16 -c 1 python make_catalogs/compute_geo_fraction_mpi.py

The output is a cluster-level table keyed by ``ID``.  Use
``patch_geo_fraction_to_matched_catalog.py`` afterward to join it onto the
galaxy-level matched catalog.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.spatial import KDTree

try:
    from mpi4py import MPI
except ImportError as exc:
    raise ImportError(
        "This script requires mpi4py. On NERSC, source your MPI setup script "
        "before running with srun."
    ) from exc

from projection_match_catalogs import (
    CATALOG_DIR,
    PROJECTED_APERTURE_HMPC,
    RM_PICKLE,
    angular_radius_deg_from_hmpc,
    load_redmapper_catalog,
)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

RANDOM_DIR = Path("/global/cfs/cdirs/desi/survey/catalogs/dr1/LSS/iron/LSScats/v1.5pip")
RANDOM_PATTERN = "BGS_ANY_{}_full_HPmapcut.ran.fits"
N_RANDOM_FILES = 18

RANDOM_DENSITY_PER_DEG2 = 2500.0
N_KDTREE_WORKERS = 1

OUTPUT_DIR = CATALOG_DIR
OUTPUT_PICKLE = OUTPUT_DIR / "rm_cluster_geo_fraction_1p5hmpc.pickle"
OUTPUT_FITS = OUTPUT_DIR / "rm_cluster_geo_fraction_1p5hmpc.fits"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def spherical_to_cartesian(ra_deg, dec_deg):
    """Convert RA/Dec in degrees to unit-sphere Cartesian coordinates."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack(
        (cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec))
    )


def assigned_random_indices(rank: int, size: int):
    """Return the random-catalog indices assigned to this MPI rank."""
    return list(range(N_RANDOM_FILES))[rank::size]


def count_randoms_for_rank(cluster_xyz, theta_rad, rank: int, size: int):
    """Count random points around every cluster for the files assigned to rank."""
    local_counts = np.zeros(len(cluster_xyz), dtype=np.float64)
    local_files_read = 0

    for random_index in assigned_random_indices(rank, size):
        random_path = RANDOM_DIR / RANDOM_PATTERN.format(random_index)
        if not random_path.exists():
            print(f"[rank {rank}] missing random catalog: {random_path}", flush=True)
            continue

        t0 = time.time()
        ran = Table.read(random_path)
        random_xyz = spherical_to_cartesian(ran["RA"], ran["DEC"])
        tree = KDTree(random_xyz)
        counts = tree.query_ball_point(
            cluster_xyz,
            theta_rad,
            workers=N_KDTREE_WORKERS,
            return_length=True,
        )
        local_counts += np.asarray(counts, dtype=np.float64)
        local_files_read += 1
        dt = time.time() - t0
        print(
            f"[rank {rank}] read {random_path.name}: "
            f"{len(ran):,} randoms in {dt:.1f} s",
            flush=True,
        )

        del ran, random_xyz, tree, counts

    return local_counts, local_files_read


def build_geo_table(rm_clus, total_counts, n_files_read):
    """Build the cluster-level geometric-fraction table on rank 0."""
    theta_deg = angular_radius_deg_from_hmpc(
        PROJECTED_APERTURE_HMPC,
        rm_clus["Z_SPEC_x"],
    )
    area_deg2 = np.pi * theta_deg**2
    expected_per_file = area_deg2 * RANDOM_DENSITY_PER_DEG2
    expected_total = expected_per_file * n_files_read

    with np.errstate(divide="ignore", invalid="ignore"):
        geo_frac = total_counts / expected_total
    geo_frac[~np.isfinite(geo_frac)] = np.nan

    out = Table()
    out["ID"] = rm_clus["ID"]
    out["RA_x"] = rm_clus["RA_x"]
    out["DEC_x"] = rm_clus["DEC_x"]
    out["Z_SPEC_x"] = rm_clus["Z_SPEC_x"]
    out["angRad_deg"] = theta_deg
    out["sq_deg"] = area_deg2
    out[f"Nr_{PROJECTED_APERTURE_HMPC:g}hmpc_expected_per_file"] = expected_per_file
    out["N_random_total"] = total_counts
    out["N_random_files_geoFrac"] = n_files_read
    out["geoFrac"] = geo_frac
    return out


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    t_start = time.time()

    if rank == 0:
        print(f"Running with {size} MPI ranks", flush=True)
        print(f"Loading redMaPPer catalog: {RM_PICKLE}", flush=True)
        rm_clus, _ = load_redmapper_catalog(RM_PICKLE)
        cluster_xyz = spherical_to_cartesian(rm_clus["RA_x"], rm_clus["DEC_x"])
        theta_deg = angular_radius_deg_from_hmpc(
            PROJECTED_APERTURE_HMPC,
            rm_clus["Z_SPEC_x"],
        )
        theta_rad = np.deg2rad(theta_deg)
    else:
        rm_clus = None
        cluster_xyz = None
        theta_rad = None

    rm_clus = comm.bcast(rm_clus, root=0)
    cluster_xyz = comm.bcast(cluster_xyz, root=0)
    theta_rad = comm.bcast(theta_rad, root=0)

    local_counts, local_files_read = count_randoms_for_rank(
        cluster_xyz,
        theta_rad,
        rank,
        size,
    )

    total_counts = np.zeros_like(local_counts)
    total_files_read = comm.reduce(local_files_read, op=MPI.SUM, root=0)
    comm.Reduce(local_counts, total_counts, op=MPI.SUM, root=0)

    if rank == 0:
        if total_files_read == 0:
            raise RuntimeError(f"No random catalogs were read from {RANDOM_DIR}")

        geo = build_geo_table(rm_clus, total_counts, total_files_read)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with OUTPUT_PICKLE.open("wb") as handle:
            pickle.dump(geo, handle, protocol=pickle.HIGHEST_PROTOCOL)
        geo.write(OUTPUT_FITS, overwrite=True)

        elapsed = time.time() - t_start
        print(f"Random files read: {total_files_read}", flush=True)
        print(f"Saved {OUTPUT_PICKLE}", flush=True)
        print(f"Saved {OUTPUT_FITS}", flush=True)
        print(f"Elapsed wall time: {elapsed:.1f} s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
