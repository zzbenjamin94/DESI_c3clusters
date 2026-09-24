"""
MPI practice script for reading DR9 sweep files one file at a time.

This is intentionally diagnostic-only.  Each MPI rank receives a subset of
sweep files, reads them one at a time, applies the same basic DR9 photometric
galaxy cuts used by the overdensity workflow, and reports per-file counts and
RA/Dec bounds.  Use it to verify that you are reading the full north+south DR9
sweep footprint before rerunning expensive cluster counts.

Example on Perlmutter:

    srun -n 8 python mpi_sweep_file_reader_practice.py --max-files 32

For a quick single-process test:

    python mpi_sweep_file_reader_practice.py --max-files 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
from astropy.units import UnitsWarning
from astropy.table import Table

try:
    from mpi4py import MPI
except ImportError:  # Allows the script to run without MPI for debugging.
    MPI = None

try:
    from .dr9_photometry import dr9_galaxy_mask
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dr9_photometry import dr9_galaxy_mask


DEFAULT_SWEEP_DIRS = [
    "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/north/sweep/9.0",
    "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        action="append",
        default=None,
        help=(
            "DR9 sweep directory. Can be supplied multiple times. "
            "Defaults to both north and south DR9 sweep directories."
        ),
    )
    parser.add_argument("--pattern", default="sweep-*.fits")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--r-mag-limit", type=float, default=23.5)
    parser.add_argument("--output", default="mpi_sweep_file_read_summary.ecsv")
    return parser.parse_args()


def find_sweep_files(sweep_dirs, pattern, max_files=None):
    files = []
    for sweep_dir in sweep_dirs:
        sweep_dir = Path(sweep_dir)
        files.extend(sorted(sweep_dir.glob(pattern)))
        files.extend(sorted(sweep_dir.glob(f"*/{pattern}")))
    files = sorted(files)
    if max_files is not None:
        files = files[:max_files]
    return files


def summarize_one_file(filename, r_mag_limit):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*'1/arcsec\^2' did not parse as fits unit.*",
            category=UnitsWarning,
        )
        table = Table.read(filename, hdu=1, memmap=True)
    ra = np.asarray(table["RA"], dtype=float)
    dec = np.asarray(table["DEC"], dtype=float)
    finite_pos = np.isfinite(ra) & np.isfinite(dec)

    selected = dr9_galaxy_mask(
        table,
        r_mag_limit=r_mag_limit,
        require_grz_ivar=True,
        remove_psf=True,
        color_gr_range=None,
    )

    return {
        "filename": str(filename),
        "n_raw": len(table),
        "n_finite_position": int(np.count_nonzero(finite_pos)),
        "n_selected": int(np.count_nonzero(selected)),
        "ra_min": float(np.nanmin(ra[finite_pos])) if np.any(finite_pos) else np.nan,
        "ra_max": float(np.nanmax(ra[finite_pos])) if np.any(finite_pos) else np.nan,
        "dec_min": float(np.nanmin(dec[finite_pos])) if np.any(finite_pos) else np.nan,
        "dec_max": float(np.nanmax(dec[finite_pos])) if np.any(finite_pos) else np.nan,
    }


def main():
    args = parse_args()

    if MPI is None:
        rank = 0
        size = 1
        comm = None
    else:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    sweep_dirs = args.sweep_dir or DEFAULT_SWEEP_DIRS
    files = find_sweep_files(sweep_dirs, args.pattern, max_files=args.max_files)

    if rank == 0:
        print(f"Found {len(files):,} sweep files")
        for sweep_dir in sweep_dirs:
            print(f"  sweep dir: {sweep_dir}")
        print(f"MPI size: {size}")

    my_files = files[rank::size]
    print(f"Rank {rank:03d}: assigned {len(my_files):,} files")

    rows = []
    for i, filename in enumerate(my_files, start=1):
        print(f"Rank {rank:03d}: reading {i}/{len(my_files)} {filename.name}")
        row = summarize_one_file(filename, args.r_mag_limit)
        row["rank"] = rank
        rows.append(row)

    if comm is not None:
        gathered = comm.gather(rows, root=0)
    else:
        gathered = [rows]

    if rank == 0:
        flat_rows = [row for chunk in gathered for row in chunk]
        out = Table(rows=flat_rows)
        out.write(args.output, format="ascii.ecsv", overwrite=True)
        print(f"Wrote {args.output}")
        if len(out) > 0:
            print(f"Total raw objects:      {int(np.sum(out['n_raw'])):,}")
            print(f"Total selected objects: {int(np.sum(out['n_selected'])):,}")


if __name__ == "__main__":
    main()
