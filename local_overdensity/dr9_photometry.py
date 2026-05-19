"""
Utilities for loading DR9 photometric galaxies from Legacy Survey sweep files.

Example
-------

from local_overdensity.dr9_photometry import load_dr9_galaxies

dr9 = load_dr9_galaxies(
    "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/north/sweep/9.0/",
    pattern="sweep-*.fits",
    r_mag_limit=23.5,
)

ra_dr9 = dr9["RA"]
dec_dr9 = dr9["DEC"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import glob

import numpy as np
from astropy.table import Table, vstack


DEFAULT_DR9_SWEEP_DIR = (
    "/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/north/sweep/9.0/"
)


def nanomaggies_to_mag(flux: np.ndarray) -> np.ndarray:
    """Convert Legacy Survey nanomaggies to AB magnitudes."""

    flux = np.asarray(flux, dtype=float)
    mag = np.full(flux.shape, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    mag[good] = 22.5 - 2.5 * np.log10(flux[good])
    return mag


def dereddened_mag(table: Table, band: str) -> np.ndarray:
    """
    Return dereddened AB magnitude for a Legacy Survey band.

    Parameters
    ----------
    table:
        DR9 sweep table.
    band:
        One of ``"G"``, ``"R"``, or ``"Z"``. Lowercase is accepted.
    """

    band = band.upper()
    flux = np.asarray(table[f"FLUX_{band}"], dtype=float)
    transmission = np.asarray(table[f"MW_TRANSMISSION_{band}"], dtype=float)
    return nanomaggies_to_mag(flux / transmission)


def dr9_galaxy_mask(
    table: Table,
    r_mag_limit: float | None = 23.5,
    require_grz_ivar: bool = True,
    remove_psf: bool = True,
    color_gr_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Build the DR9 photometric-galaxy selection used for local overdensity work.

    The baseline cuts mirror the notebook:
    finite coordinates, ``TYPE != "PSF"``, positive GRZ inverse variances, and
    optional dereddened r-band magnitude and g-r color cuts.
    """

    mask = np.isfinite(table["RA"]) & np.isfinite(table["DEC"])

    if remove_psf and "TYPE" in table.colnames:
        source_type = np.asarray(table["TYPE"])
        mask &= source_type != "PSF"

    if require_grz_ivar:
        for band in ("G", "R", "Z"):
            col = f"FLUX_IVAR_{band}"
            if col in table.colnames:
                mask &= np.asarray(table[col], dtype=float) > 0

    r_dered = None
    if r_mag_limit is not None:
        r_dered = dereddened_mag(table, "R")
        mask &= np.isfinite(r_dered) & (r_dered < r_mag_limit)

    if color_gr_range is not None:
        g_dered = dereddened_mag(table, "G")
        if r_dered is None:
            r_dered = dereddened_mag(table, "R")
        gr = g_dered - r_dered
        lo, hi = color_gr_range
        mask &= np.isfinite(gr) & (gr > lo) & (gr < hi)

    return np.asarray(mask, dtype=bool)


def find_dr9_sweep_files(
    sweep_dir: str | Path = DEFAULT_DR9_SWEEP_DIR,
    pattern: str = "sweep-*.fits",
) -> list[str]:
    """Return sorted DR9 sweep files matching a glob pattern."""

    sweep_dir = Path(sweep_dir)
    return sorted(glob.glob(str(sweep_dir / pattern)))


def read_dr9_sweep_file(
    filename: str | Path,
    r_mag_limit: float | None = 23.5,
    require_grz_ivar: bool = True,
    remove_psf: bool = True,
    color_gr_range: tuple[float, float] | None = None,
    keep_columns: Iterable[str] = ("RA", "DEC", "TYPE"),
    add_dereddened_columns: bool = True,
) -> Table:
    """Read one DR9 sweep file and return selected photometric galaxies."""

    table = Table.read(filename, hdu=1, memmap=True)
    mask = dr9_galaxy_mask(
        table,
        r_mag_limit=r_mag_limit,
        require_grz_ivar=require_grz_ivar,
        remove_psf=remove_psf,
        color_gr_range=color_gr_range,
    )

    keep = [col for col in keep_columns if col in table.colnames]
    out = table[mask][keep].copy()

    if add_dereddened_columns:
        selected = table[mask]
        out["g_dered"] = dereddened_mag(selected, "G")
        out["r_dered"] = dereddened_mag(selected, "R")
        out["z_dered"] = dereddened_mag(selected, "Z")
        out["gmr"] = out["g_dered"] - out["r_dered"]
        out["rmz"] = out["r_dered"] - out["z_dered"]

    return out


def load_dr9_galaxies(
    sweep_dir: str | Path = DEFAULT_DR9_SWEEP_DIR,
    pattern: str = "sweep-*.fits",
    max_files: int | None = None,
    r_mag_limit: float | None = 23.5,
    require_grz_ivar: bool = True,
    remove_psf: bool = True,
    color_gr_range: tuple[float, float] | None = None,
    keep_columns: Iterable[str] = ("RA", "DEC", "TYPE"),
    add_dereddened_columns: bool = True,
    verbose: bool = True,
) -> Table:
    """
    Load and concatenate selected DR9 photometric galaxies from sweep files.

    Parameters
    ----------
    max_files:
        Optional cap for quick tests.
    """

    files = find_dr9_sweep_files(sweep_dir, pattern=pattern)
    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No DR9 sweep files matched {sweep_dir}/{pattern}")

    chunks = []
    for i, filename in enumerate(files):
        if verbose:
            print(f"Reading {i + 1}/{len(files)}: {filename}")

        chunk = read_dr9_sweep_file(
            filename,
            r_mag_limit=r_mag_limit,
            require_grz_ivar=require_grz_ivar,
            remove_psf=remove_psf,
            color_gr_range=color_gr_range,
            keep_columns=keep_columns,
            add_dereddened_columns=add_dereddened_columns,
        )
        chunks.append(chunk)

        if verbose:
            print(f"  kept {len(chunk):,} galaxies")

    if len(chunks) == 1:
        return chunks[0]

    return vstack(chunks, metadata_conflicts="silent")
