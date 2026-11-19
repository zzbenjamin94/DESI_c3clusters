"""Native DR9 sidecars, standard imaging ownership, and Gaussian photo-z weights."""

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from scipy.special import ndtr


def gaussian_probability(mean, sigma, z, half_width=0.05):
    """Probability in the cluster window, conditional on nonnegative redshift."""
    lo = (max(0.0, z-half_width)-mean)/sigma
    hi = (z+half_width-mean)/sigma
    # Survival-function subtraction avoids cancellation in the upper tail.
    integral = np.where(lo > 0, ndtr(-lo)-ndtr(-hi), ndtr(hi)-ndtr(lo))
    return np.clip(integral/ndtr(mean/sigma), 0, 1)


def ownership(table, region):
    if region not in ("north", "south"):
        raise ValueError("DR9 region must be north or south")
    ra, dec = np.asarray(table["RA"], float), np.asarray(table["DEC"], float)
    finite = np.isfinite(ra) & np.isfinite(dec) & (abs(dec) <= 90)
    north = np.zeros(len(table), dtype=bool)
    north[finite] = (dec[finite] >= 32.375) & (
        SkyCoord(ra[finite], dec[finite], unit="deg").galactic.b.deg > 0)
    return finite & (north if region == "north" else ~north)


def coverage(table):
    return np.logical_and.reduce([
        np.isfinite(np.asarray(table[f"NOBS_{b}"], float)) &
        (np.asarray(table[f"NOBS_{b}"], float) >= 1) for b in "GRZ"])


def paired_chunks(job, size):
    with fits.open(job["path"], memmap=True) as data, fits.open(job["photoz"], memmap=True) as photo:
        if data[1].header["NAXIS2"] != photo[1].header["NAXIS2"]:
            raise ValueError("Sweep/photo-z row counts differ")
        for start in range(0, data[1].header["NAXIS2"], size):
            table = Table(data[1].data[start:start+size], copy=True)
            p = photo[1].data[start:start+size]
            for key in ("RELEASE", "BRICKID", "OBJID"):
                if not np.array_equal(table[key], p[key]):
                    raise ValueError(f"Sweep/photo-z {key} mismatch; refusing positional join")
            for key in ("Z_PHOT_MEAN", "Z_PHOT_STD"):
                table[key] = np.array(p[key], copy=True)
            yield table
