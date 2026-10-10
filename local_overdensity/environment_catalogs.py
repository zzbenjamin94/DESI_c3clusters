"""Chunked FITS readers and strict adapters; never infer a PDF from an error bar."""

from pathlib import Path
import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree
from local_overdensity.dr9_photometry import dereddened_mag


def discover(patterns):
    files = sorted({str(Path(p).resolve()) for pattern in patterns for p in glob.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No input files: {patterns}")
    return files


def fits_chunks(path, size):
    with fits.open(path, memmap=True) as hdus:
        for start in range(0, len(hdus[1].data), size):
            yield start, Table(hdus[1].data[start:start + size], copy=True)


def xyz(ra, dec):
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack((np.cos(dec)*np.cos(ra), np.cos(dec)*np.sin(ra), np.sin(dec)))


def sky_tree(table):
    ra, dec = np.asarray(table["RA"], float), np.asarray(table["DEC"], float)
    if np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)) or np.any(abs(dec) > 90):
        raise ValueError("Invalid tracer/random coordinates")
    vectors = xyz(ra, dec)
    return vectors, cKDTree(vectors)


def load_clusters(path):
    from richness_relation.prepare_conditional_richness import read_catalog
    table = read_catalog(Path(path))
    required = ("ID", "RA_central", "DEC_central", "Z_SPEC_central")
    for key in required:
        if key not in table.colnames:
            raise KeyError(f"Cluster catalog requires {key}")
    ids, first, inverse = np.unique(np.asarray(table["ID"]).astype(str), return_index=True, return_inverse=True)
    keep = list(required) + [k for k in ("LAMBDA", "lambda_spec_geo_comp_lf") if k in table.colnames]
    for key in keep[1:]:
        v = np.ma.asarray(table[key], dtype=float).filled(np.nan)
        if not np.all(np.isclose(v, v[first][inverse], equal_nan=True)):
            raise ValueError(f"Cluster-level {key} varies within ID")
    table = table[first][keep].copy()
    z = np.asarray(table["Z_SPEC_central"], float)
    table = table[np.isfinite(z) & (z >= 0.1) & (z < 0.35)]
    if not len(table):
        raise ValueError("No clusters in 0.1 <= z < 0.35")
    sky_tree(Table({"RA": table["RA_central"], "DEC": table["DEC_central"]}))
    return table


def curve(path, z, value_column):
    tab = Table.read(path, format="ascii.csv")
    x, y = np.asarray(tab["z"], float), np.asarray(tab[value_column], float)
    if len(x) < 2 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0) or np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError(f"Invalid {value_column} curve: {path}")
    if np.any(z < x[0]) or np.any(z > x[-1]):
        raise ValueError(f"{path} does not cover required redshifts; extrapolation is disabled")
    return np.interp(z, x, y)


def photo_selection(table, config):
    # Required fields are checked explicitly; never silently skip a missing cut.
    mask = np.ones(len(table), bool)
    kind = np.char.strip(np.asarray(table["TYPE"]).astype(str))
    mask &= kind != "PSF"
    for band in "GRZ":
        ivar = np.asarray(table[f"FLUX_IVAR_{band}"], float)
        mask &= np.isfinite(ivar) & (ivar > 0)
    transmission = np.asarray(table["MW_TRANSMISSION_R"], float)
    mask &= np.isfinite(transmission) & (transmission > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = dereddened_mag(table, "R")
    mask &= np.isfinite(mag) & (mag < config["r_mag_limit"])
    mask &= (np.asarray(table["MASKBITS"], np.int64) & int(config["maskbits_reject"])) == 0
    if config.get("gr_max") is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            g = dereddened_mag(table, "G")
        mask &= np.isfinite(g) & ((g - mag) < config["gr_max"])
    return mask


def load_photoz(path):
    """One NPZ per sweep: keyed rows, PDF grid, primary ownership and sys weights.

    PRIMARY must encode a resolved union of north/south, not merely BRICK_PRIMARY.
    The product-specific conversion is intentionally external until identified.
    """
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in ("RELEASE", "BRICKID", "OBJID", "z_grid", "pdf", "PRIMARY", "SYS_WEIGHT")}
    n = len(data["OBJID"])
    if any(data[k].shape != (n,) for k in ("RELEASE", "BRICKID", "PRIMARY", "SYS_WEIGHT")):
        raise ValueError("Photo-z sidecar column lengths differ")
    if data["pdf"].shape != (n, len(data["z_grid"])) or data["PRIMARY"].dtype.kind != "b":
        raise ValueError("Invalid PDF shape or non-boolean PRIMARY")
    keys = list(zip(data["RELEASE"], data["BRICKID"], data["OBJID"]))
    lookup = {key: i for i, key in enumerate(keys)}
    if len(lookup) != n:
        raise ValueError("Duplicate photo-z keys")
    return data, lookup


def photo_indices(table, lookup):
    keys = zip(table["RELEASE"], table["BRICKID"], table["OBJID"])
    try:
        return np.array([lookup[k] for k in keys], dtype=int)
    except KeyError as error:
        raise ValueError(f"Selected photometric object has no PDF: {error}") from error
