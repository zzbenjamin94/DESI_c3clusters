"""
redMaPPer--DESI BGS projected catalog matching workflow.

This script rewrites the matching logic from ``projection_matchCatalogs.ipynb``
in a more explicit and safer way.  The main change is that the physical
aperture cut is applied directly to the candidate redMaPPer/BGS pairs, rather
than running a second all-to-all 3D search on already-expanded matched arrays.

The intended output is a galaxy-level table with one row per unique
redMaPPer-cluster/BGS-galaxy pair.  The table contains BGS columns,
redMaPPer cluster columns, optional redMaPPer member-galaxy columns, a central
flag, projected radius, geometric coverage fraction, and explicit weight
columns.

Weight convention:

    COMP_WEIGHT      = DESI/BGS completeness weight, 1 / PROB_OBS.
                       This combines target-observation completeness and
                       redshift/template success effects represented by
                       PROB_OBS in the DR2 catalog.
    GEOMETRIC_FRACTION = cluster-level geometric coverage fraction.
    GEOMETRIC_WEIGHT = 1 / GEOMETRIC_FRACTION, constant for a given cluster.
    LF_WEIGHT        = cluster-level luminosity-function completeness weight.
    TOTAL_WEIGHT     = COMP_WEIGHT * GEOMETRIC_WEIGHT * LF_WEIGHT.

The geometric coverage fraction is computed with MPI.  Rank 0 performs the
catalog matching, all ranks read disjoint subsets of random catalogs to count
random points around the cluster apertures, and rank 0 joins the final
geometric fraction onto the matched catalog.

Edit the configuration block below before running on NERSC.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import sys
import warnings

import astropy.units as u
import astropy.cosmology.units as cu
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18
from astropy.table import Table, join, unique
from astropy.coordinates import SkyCoord
from scipy.integrate import quad
from scipy.spatial import KDTree

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path("/global/homes/z/zzhang13/DESI/Projection")
if not REPO_ROOT.exists():
    REPO_ROOT = Path(__file__).resolve().parents[1]

CATALOG_DIR = REPO_ROOT / "catalogs"
OUTPUT_DIR = CATALOG_DIR

RM_PICKLE = CATALOG_DIR / "RM_SDSS_df.pkl"
BGS_LSSCAT_DIR = Path("/global/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1")
BGS_CATALOG = BGS_LSSCAT_DIR / "BGS_BRIGHT_full_noveto.dat.fits"
RANDOM_DIR = BGS_LSSCAT_DIR

OUTPUT_PICKLE = OUTPUT_DIR / "bgs_clus_RM_gal_matched_with_weights.pickle"
OUTPUT_FITS = OUTPUT_DIR / "bgs_clus_RM_gal_matched_with_weights.fits"
LF_SUMMARY_CSV = CATALOG_DIR / "bgs_direct_lf_logL_global_vmax_schechter_fit_summary.csv"

Z_MIN = 0.0
Z_MAX = 0.4
PROJECTED_APERTURE_HMPC = 1.5
DZ_ABS_MAX = 0.2
RM_MEMBER_MATCH_MAX_SEP = 0.1 * u.arcsec
CENTRAL_RADIUS_HMPC = 0.005

RANDOM_DENSITY_PER_DEG2 = 2500.0
N_RANDOM_FILES = 18
RANDOM_PATTERN = "BGS_BRIGHT_{}_full.ran.fits"
RANDOM_GLOB_PATTERNS = [
    "BGS_BRIGHT_*_full.ran.fits",
]
BGS_DATA_GLOB_PATTERNS = [
    "BGS_BRIGHT_full_noveto.dat.fits",
]
N_KDTREE_WORKERS = 1
ADD_LF_WEIGHT_COLUMNS = True
ADD_SPEC_RICHNESS_COLUMNS = True

R_MAG_LIMIT = 19.5
REFERENCE_Z = 0.1
LOG_L_MIN_FIT = 9.0
M_SUN_R_AB = 4.64

COSMO = Planck18
H = COSMO.H0.value / 100.0


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def spherical_to_cartesian(ra_deg, dec_deg):
    """Convert RA/Dec in degrees to unit-sphere Cartesian coordinates."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack(
        (cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec))
    )


def comoving_distance_mpc(z):
    """Comoving transverse distance in Mpc for redshift array ``z``."""
    z = np.asarray(z, dtype=float)
    zq = z * cu.redshift
    return zq.to(u.Mpc, cu.redshift_distance(COSMO, kind="comoving")).value


def angular_radius_deg_from_hmpc(radius_hmpc, z):
    """
    Convert a projected comoving aperture in h^-1 Mpc to angular radius in deg.

    Astropy gives D_M in Mpc.  A comoving projected separation in h^-1 Mpc is

        R_hmpc = theta * D_M * h.

    Therefore theta = R_hmpc / (D_M * h).
    """
    dm_mpc = comoving_distance_mpc(z)
    theta_rad = np.asarray(radius_hmpc, dtype=float) / (dm_mpc * H)
    return np.rad2deg(theta_rad)


def projected_radius_hmpc(ra1, dec1, z_cluster, ra2, dec2):
    """Projected comoving separation at the cluster redshift, in h^-1 Mpc."""
    c1 = SkyCoord(ra=np.asarray(ra1) * u.deg, dec=np.asarray(dec1) * u.deg)
    c2 = SkyCoord(ra=np.asarray(ra2) * u.deg, dec=np.asarray(dec2) * u.deg)
    theta_rad = c1.separation(c2).rad
    return theta_rad * comoving_distance_mpc(z_cluster) * H


def mpi_context():
    """Return ``comm, rank, size`` with a serial fallback if mpi4py is absent."""
    if MPI is None:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def mpi_print(rank, message):
    """Print a message with rank information."""
    print(f"[rank {rank}] {message}", flush=True)


# -----------------------------------------------------------------------------
# Catalog loading and preparation
# -----------------------------------------------------------------------------

def read_pickle_table(path: Path) -> Table:
    """Read a pickle object as an Astropy Table."""
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, Table):
        return obj
    if obj.__class__.__module__.startswith("pandas"):
        return Table.from_pandas(obj)
    if hasattr(obj, "to_pandas"):
        return Table.from_pandas(obj.to_pandas())
    return Table(obj)


def table_col(table: Table, preferred: str, *fallbacks: str) -> str:
    """Return the first available column name from preferred/fallback choices."""
    for col in (preferred, *fallbacks):
        if col in table.colnames:
            return col
    choices = ", ".join(repr(col) for col in (preferred, *fallbacks))
    raise KeyError(f"Missing required column. Tried: {choices}")


def load_redmapper_catalog(path: Path = RM_PICKLE):
    """Load redMaPPer table and split it into cluster and member tables."""
    rm_data = read_pickle_table(path)

    z_central_col = table_col(rm_data, "Z_SPEC_central", "Z_SPEC_x")
    z_member_col = table_col(rm_data, "Z_SPEC_member", "Z_SPEC_y")

    good = (
        np.isfinite(np.asarray(rm_data[z_central_col], dtype=float))
        & (rm_data[z_central_col] > Z_MIN)
        & (rm_data[z_central_col] < Z_MAX)
        & (rm_data[z_member_col] != rm_data[z_central_col])
    )
    rm_data = rm_data[good]

    cluster_cols = [
        "ID",
        "LAMBDA",
        "Z_LAMBDA",
        "R_LAMBDA",
        table_col(rm_data, "Z_SPEC_central", "Z_SPEC_x"),
        table_col(rm_data, "RA_central", "RA_x"),
        table_col(rm_data, "DEC_central", "DEC_x"),
        table_col(rm_data, "MODEL_MAG_R_central", "MODEL_MAG_R_x"),
        table_col(rm_data, "MODEL_MAGERR_R_central", "MODEL_MAGERR_R_x"),
    ]
    member_cols = [
        "ID",
        table_col(rm_data, "Z_SPEC_member", "Z_SPEC_y"),
        table_col(rm_data, "RA_member", "RA_y"),
        table_col(rm_data, "DEC_member", "DEC_y"),
        table_col(rm_data, "R_member", "R"),
        table_col(rm_data, "P_member", "P"),
        table_col(rm_data, "MODEL_MAG_R_member", "MODEL_MAG_R_y"),
        table_col(rm_data, "MODEL_MAGERR_R_member", "MODEL_MAGERR_R_y"),
    ]
    cluster_cols = list(dict.fromkeys(cluster_cols))
    member_cols = list(dict.fromkeys(member_cols))

    rm_clus = unique(rm_data[cluster_cols], keys="ID")
    rm_gal = rm_data[member_cols]

    rename_map = {
        "Z_SPEC_x": "Z_SPEC_central",
        "RA_x": "RA_central",
        "DEC_x": "DEC_central",
        "MODEL_MAG_R_x": "MODEL_MAG_R_central",
        "MODEL_MAGERR_R_x": "MODEL_MAGERR_R_central",
        "Z_SPEC_y": "Z_SPEC_member",
        "RA_y": "RA_member",
        "DEC_y": "DEC_member",
        "R": "R_member",
        "P": "P_member",
        "MODEL_MAG_R_y": "MODEL_MAG_R_member",
        "MODEL_MAGERR_R_y": "MODEL_MAGERR_R_member",
    }
    for table in (rm_clus, rm_gal):
        for old, new in rename_map.items():
            if old in table.colnames and new not in table.colnames:
                table.rename_column(old, new)
    return rm_clus, rm_gal


def discover_files(path: Path, patterns: list[str]) -> list[Path]:
    """Return FITS files from a file path or directory using candidate patterns."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(path.glob(pattern)))

    # Preserve order while removing duplicates introduced by overlapping globs.
    seen = set()
    unique_files = []
    for file_path in files:
        resolved = str(file_path)
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(file_path)
    return unique_files


def standardize_column_name(table: Table, canonical: str, candidates: list[str], required: bool = True):
    """Rename the first available candidate column to a canonical name."""
    if canonical in table.colnames:
        return canonical

    for candidate in candidates:
        if candidate in table.colnames:
            table.rename_column(candidate, canonical)
            return canonical

    if required:
        preview = ", ".join(table.colnames[:80])
        raise KeyError(
            f"Could not find required column {canonical!r}. "
            f"Tried candidates: {candidates}. Available columns begin with: {preview}"
        )
    return None


def load_bgs_catalog(path: Path = BGS_CATALOG):
    """Load DESI BGS catalog(s) and standardize key column names."""
    paths = discover_files(path, BGS_DATA_GLOB_PATTERNS)
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No BGS clustering data files found in {path}. "
            f"Tried patterns: {BGS_DATA_GLOB_PATTERNS}"
        )

    if len(paths) != 1:
        raise ValueError(f"Expected one DR2 BGS Bright noveto data file, found {len(paths)}: {paths}")

    print(f"Reading BGS catalog: {paths[0]}")
    bgs = Table.read(paths[0])

    standardize_column_name(bgs, "RA_BGS", ["RA", "TARGET_RA"])
    standardize_column_name(bgs, "DEC_BGS", ["DEC", "TARGET_DEC"])
    standardize_column_name(
        bgs,
        "Z_BGS",
        ["Z", "Z_not4clus", "Z_NOT4CLUS", "Z_COSMO", "Z_RR", "Z_DESI"],
    )
    standardize_column_name(bgs, "TARGETID", ["TARGETID", "TARGET_ID"])

    if "PROB_OBS" not in bgs.colnames:
        raise KeyError("DR2 BGS catalog must contain PROB_OBS.")
    prob_obs = np.asarray(bgs["PROB_OBS"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        comp_weight = 1.0 / prob_obs
    comp_weight[~np.isfinite(comp_weight) | (comp_weight <= 0)] = 0.0
    bgs["COMP_WEIGHT"] = comp_weight

    required_keep_cols = [
        "TARGETID",
        "RA_BGS",
        "DEC_BGS",
        "Z_BGS",
        "PROB_OBS",
        "COMP_WEIGHT",
    ]
    optional_keep_cols = [
        "FLUX_G",
        "FLUX_R",
        "FLUX_Z",
        "FLUX_W1",
        "FLUX_W2",
        "flux_g_dered",
        "flux_r_dered",
        "flux_z_dered",
        "flux_w1_dered",
        "flux_w2_dered",
    ]
    keep_cols = required_keep_cols + [col for col in optional_keep_cols if col in bgs.colnames]
    return bgs[keep_cols]


def add_weight_columns(table: Table) -> Table:
    """
    Add explicit completeness, geometric, LF, and total weight columns.

    ``GEOMETRIC_FRACTION`` and ``LF_WEIGHT`` are cluster-level quantities.  This
    function assumes they have already been joined onto each galaxy row by
    ``ID``.  If no LF correction has been computed yet, ``LF_WEIGHT`` defaults
    to 1.
    """
    out = table.copy()

    if "COMP_WEIGHT" not in out.colnames:
        if "PROB_OBS" in out.colnames:
            prob_obs = np.asarray(out["PROB_OBS"], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["COMP_WEIGHT"] = 1.0 / prob_obs
        elif "IID_WEIGHT" in out.colnames:
            out["COMP_WEIGHT"] = np.asarray(out["IID_WEIGHT"], dtype=float)
        elif "WEIGHT" in out.colnames:
            out["COMP_WEIGHT"] = np.asarray(out["WEIGHT"], dtype=float)
        else:
            out["COMP_WEIGHT"] = np.ones(len(out), dtype=float)

    if "GEOMETRIC_FRACTION" not in out.colnames:
        if "geoFrac" in out.colnames:
            out["GEOMETRIC_FRACTION"] = np.asarray(out["geoFrac"], dtype=float)
        else:
            out["GEOMETRIC_FRACTION"] = np.ones(len(out), dtype=float)

    if "LF_WEIGHT" not in out.colnames:
        if "lf_weight" in out.colnames:
            out["LF_WEIGHT"] = np.asarray(out["lf_weight"], dtype=float)
        else:
            out["LF_WEIGHT"] = np.ones(len(out), dtype=float)

    comp = np.asarray(out["COMP_WEIGHT"], dtype=float)
    geo = np.asarray(out["GEOMETRIC_FRACTION"], dtype=float)
    lf = np.asarray(out["LF_WEIGHT"], dtype=float)
    comp[~np.isfinite(comp) | (comp <= 0)] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        geometric_weight = 1.0 / geo
        total_weight = comp * geometric_weight * lf

    geometric_weight[~np.isfinite(geometric_weight)] = 0.0
    total_weight[~np.isfinite(total_weight)] = 0.0

    out["GEOMETRIC_WEIGHT"] = geometric_weight
    out["TOTAL_WEIGHT"] = total_weight
    return out


def read_lf_fit_summary(path: Path) -> dict[str, float]:
    """Read the first row of the global LF fit summary CSV."""
    summary = pd.read_csv(path)
    if len(summary) == 0:
        raise ValueError(f"LF summary has no rows: {path}")
    required = ["log10_L_star", "alpha"]
    missing = [col for col in required if col not in summary.columns]
    if missing:
        raise KeyError(f"LF summary missing required columns: {missing}")

    row = summary.iloc[0]
    return {
        "log10_L_star": float(row["log10_L_star"]),
        "alpha": float(row["alpha"]),
        "M_star_minus_5logh": float(row["M_star_minus_5logh"])
        if "M_star_minus_5logh" in summary.columns
        else np.nan,
    }


def distance_modulus(z):
    """Distance modulus for redshift array z."""
    z = np.asarray(z, dtype=float)
    dist = (z * cu.redshift).to(u.pc, cu.redshift_distance(COSMO, kind="luminosity"))
    return 5.0 * np.log10(dist.value / 10.0)


def M_minus_5logh_from_apparent(mag, z):
    """Return M - 5 log10(h), matching the LF notebook convention."""
    return np.asarray(mag, dtype=float) - distance_modulus(z) - 5.0 * np.log10(H)


def logL_from_M_minus_5logh(M_h):
    """Log-luminosity variable used by the LF notebook."""
    return -0.4 * (np.asarray(M_h, dtype=float) - M_SUN_R_AB)


def magnitude_limit_to_logL(z, m_lim=R_MAG_LIMIT):
    """Log-luminosity implied by the apparent magnitude limit at redshift z."""
    M_lim_h = M_minus_5logh_from_apparent(m_lim, z)
    return logL_from_M_minus_5logh(M_lim_h)


def schechter_logL_shape(logL_values, logL_star, alpha):
    """Single-Schechter shape per dex, without normalization."""
    x = 10.0 ** (np.asarray(logL_values, dtype=float) - logL_star)
    return np.log(10.0) * x ** (alpha + 1.0) * np.exp(-x)


def integrate_schechter_logL_from(logL_min, logL_star, alpha, logL_max=14.0):
    """Integrate the Schechter shape from logL_min to logL_max."""
    value, _ = quad(
        lambda ell: schechter_logL_shape(ell, logL_star, alpha),
        float(logL_min),
        float(logL_max),
        epsabs=1e-10,
        epsrel=1e-6,
        limit=200,
    )
    return value


def lf_observed_fraction_logL(logL_lim, logL_ref, logL_star, alpha):
    """
    Fraction of the reference LF observable above a redshift-dependent limit.

    If the survey is deeper than the reference limit, the numerator is evaluated
    at the reference limit, so the observable fraction is capped at 1.
    """
    logL_lim = np.asarray(logL_lim, dtype=float)
    frac = np.full(len(logL_lim), np.nan, dtype=float)
    denom = integrate_schechter_logL_from(logL_ref, logL_star, alpha)
    if denom <= 0 or not np.isfinite(denom):
        return frac

    logL_lim_eff = np.maximum(logL_lim, logL_ref)
    for i, ell in enumerate(logL_lim_eff):
        if not np.isfinite(ell):
            continue
        frac[i] = integrate_schechter_logL_from(ell, logL_star, alpha) / denom

    return np.clip(frac, 0.0, 1.0)


def add_lf_weight_columns(table: Table, lf_summary_csv: Path = LF_SUMMARY_CSV) -> Table:
    """
    Add one luminosity-function completeness weight per cluster ID.

    The LF factor is cluster-level: all galaxies in the same redMaPPer cluster
    receive the same ``LF_WEIGHT``.  After joining the LF table, this recomputes
    ``GEOMETRIC_WEIGHT`` and ``TOTAL_WEIGHT`` so the final total weight includes
    completeness, geometry, and LF corrections.
    """
    if not ADD_LF_WEIGHT_COLUMNS:
        return add_weight_columns(table)

    if not lf_summary_csv.exists():
        raise FileNotFoundError(
            f"LF summary file does not exist: {lf_summary_csv}. "
            "Run the LF fit notebook/script first, or set ADD_LF_WEIGHT_COLUMNS=False."
        )

    out = table.copy()
    if "ID" not in out.colnames:
        raise KeyError("Input catalog must contain an 'ID' column.")
    z_central_col = table_col(out, "Z_SPEC_central", "Z_SPEC_x")

    lf_params = read_lf_fit_summary(lf_summary_csv)
    cluster_table = unique(out[["ID", z_central_col]], keys="ID")
    z_cluster = np.asarray(cluster_table[z_central_col], dtype=float)

    logL_lim = magnitude_limit_to_logL(z_cluster, R_MAG_LIMIT)
    logL_ref_raw = magnitude_limit_to_logL(REFERENCE_Z, R_MAG_LIMIT)
    logL_ref = max(float(logL_ref_raw), LOG_L_MIN_FIT)

    obs_frac = lf_observed_fraction_logL(
        logL_lim,
        logL_ref,
        lf_params["log10_L_star"],
        lf_params["alpha"],
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        lf_weight = 1.0 / obs_frac
    lf_weight[~np.isfinite(lf_weight)] = np.nan
    lf_weight = np.maximum(lf_weight, 1.0)

    lf_table = Table()
    lf_table["ID"] = np.asarray(cluster_table["ID"])
    lf_table["lf_z_cluster"] = z_cluster
    lf_table["logL_lim_lf"] = logL_lim
    lf_table["logL_ref_lf"] = np.full(len(lf_table), logL_ref)
    lf_table["lf_log10_L_star"] = np.full(len(lf_table), lf_params["log10_L_star"])
    lf_table["lf_M_star_minus_5logh"] = np.full(len(lf_table), lf_params["M_star_minus_5logh"])
    lf_table["lf_alpha"] = np.full(len(lf_table), lf_params["alpha"])
    lf_table["lf_observed_fraction"] = obs_frac
    lf_table["LF_WEIGHT"] = lf_weight

    for col in lf_table.colnames:
        if col != "ID" and col in out.colnames:
            out.remove_column(col)

    out = join(out, lf_table, keys="ID", join_type="left", metadata_conflicts="silent")
    return add_weight_columns(out)


def default_richness_bins():
    """Return the redshift-difference bins used for spectroscopic richness."""
    wide_bin_1 = np.linspace(-0.1, -0.005, 21, endpoint=False)
    small_bin = np.linspace(-0.005, 0.005, 21, endpoint=False)
    wide_bin_2 = np.linspace(0.005, 0.1, 21, endpoint=False)
    bin_boundaries = np.hstack((wide_bin_1, small_bin, wide_bin_2))
    bin_centers = np.asarray(
        [
            0.5 * (bin_boundaries[i] + bin_boundaries[i + 1])
            for i in range(len(bin_boundaries) - 1)
        ]
    )
    if len(set(bin_centers)) != len(bin_centers):
        raise ValueError("Overlapping richness bin centers")
    if len(set(bin_boundaries)) != len(bin_boundaries):
        raise ValueError("Overlapping richness bin boundaries")
    return bin_boundaries, bin_centers


def add_spectroscopic_richness_columns(table: Table) -> Table:
    """Append lambda_spec_* columns to a matched galaxy-level catalog."""
    if not ADD_SPEC_RICHNESS_COLUMNS:
        return table

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from tools.projection_functions import append_spec_richness_columns

    bin_boundaries, bin_centers = default_richness_bins()
    return append_spec_richness_columns(
        table,
        bin_boundaries,
        bin_centers,
        total_weight_col="TOTAL_WEIGHT",
    )


# -----------------------------------------------------------------------------
# Matching steps
# -----------------------------------------------------------------------------

def match_bgs_to_clusters_projected(
    rm_clus: Table,
    bgs: Table,
    aperture_hmpc: float = PROJECTED_APERTURE_HMPC,
    dz_abs_max: float = DZ_ABS_MAX,
) -> Table:
    """
    Match BGS galaxies to redMaPPer cluster centers in projected radius.

    Steps:
    1. Use one broad angular search radius large enough for the lowest-redshift
       cluster in the sample.
    2. For each candidate pair, compute the projected separation at the cluster
       redshift.
    3. Keep pairs with R_perp < aperture_hmpc.
    4. Apply a broad redshift-difference sanity cut.
    """
    rm_coord = SkyCoord(ra=rm_clus["RA_central"] * u.deg, dec=rm_clus["DEC_central"] * u.deg)
    bgs_coord = SkyCoord(ra=bgs["RA_BGS"] * u.deg, dec=bgs["DEC_BGS"] * u.deg)

    theta_deg = angular_radius_deg_from_hmpc(aperture_hmpc, rm_clus["Z_SPEC_central"])
    max_theta = np.nanmax(theta_deg) * u.deg
    print(f"Angular preselection radius: {max_theta.to(u.arcmin):.3f}")

    idx_rm, idx_bgs, d2d, _ = bgs_coord.search_around_sky(rm_coord, max_theta)
    if len(idx_rm) == 0:
        warnings.warn("No redMaPPer/BGS angular candidate pairs found.", RuntimeWarning)
        return Table()

    rm_pair = rm_clus[idx_rm]
    bgs_pair = bgs[idx_bgs]

    rproj_hmpc = projected_radius_hmpc(
        rm_pair["RA_central"],
        rm_pair["DEC_central"],
        rm_pair["Z_SPEC_central"],
        bgs_pair["RA_BGS"],
        bgs_pair["DEC_BGS"],
    )
    keep_radius = np.isfinite(rproj_hmpc) & (rproj_hmpc < aperture_hmpc)

    rm_pair = rm_pair[keep_radius]
    bgs_pair = bgs_pair[keep_radius]
    rproj_hmpc = rproj_hmpc[keep_radius]

    dz = (bgs_pair["Z_BGS"] - rm_pair["Z_SPEC_central"]) / (1.0 + rm_pair["Z_SPEC_central"])
    keep_dz = np.isfinite(dz) & (np.abs(dz) <= dz_abs_max)

    rm_pair = rm_pair[keep_dz]
    bgs_pair = bgs_pair[keep_dz]
    rproj_hmpc = rproj_hmpc[keep_dz]
    dz = dz[keep_dz]

    matched = bgs_pair.copy()
    for col in rm_pair.colnames:
        matched[col] = rm_pair[col]
    matched["R_PROJ_HMPC"] = rproj_hmpc
    matched["DZ_CLUSTER"] = dz
    matched["central_flag"] = rproj_hmpc < CENTRAL_RADIUS_HMPC

    matched = unique(matched, keys=["ID", "TARGETID"])
    return matched


def attach_redmapper_member_match(
    bgs_matched: Table,
    rm_gal: Table,
    max_sep=RM_MEMBER_MATCH_MAX_SEP,
) -> Table:
    """
    Mark BGS rows that are also redMaPPer member-galaxy rows.

    A match is required to be within ``max_sep`` on the sky and to have the same
    redMaPPer cluster ID.  If multiple candidate member rows match the same BGS
    row, the closest one is used.
    """
    out = bgs_matched.copy()
    out["RM_gal_flag"] = False

    member_cols = [col for col in rm_gal.colnames if col != "ID"]
    for col in member_cols:
        out[col] = np.full(len(out), -1.0)

    if len(out) == 0 or len(rm_gal) == 0:
        return out

    rm_coord = SkyCoord(ra=rm_gal["RA_member"] * u.deg, dec=rm_gal["DEC_member"] * u.deg)
    bgs_coord = SkyCoord(ra=out["RA_BGS"] * u.deg, dec=out["DEC_BGS"] * u.deg)
    idx_rm, idx_bgs, d2d, _ = bgs_coord.search_around_sky(rm_coord, max_sep)

    if len(idx_rm) == 0:
        return out

    same_cluster = np.asarray(rm_gal["ID"][idx_rm]) == np.asarray(out["ID"][idx_bgs])
    idx_rm = idx_rm[same_cluster]
    idx_bgs = idx_bgs[same_cluster]
    d2d = d2d[same_cluster]

    order = np.argsort(d2d)
    filled = set()
    for i in order:
        bgs_i = int(idx_bgs[i])
        if bgs_i in filled:
            continue
        rm_i = int(idx_rm[i])
        out["RM_gal_flag"][bgs_i] = True
        for col in member_cols:
            out[col][bgs_i] = rm_gal[col][rm_i]
        filled.add(bgs_i)

    return out


# -----------------------------------------------------------------------------
# Parallel geometric fraction
# -----------------------------------------------------------------------------

def discover_random_files(path: Path = RANDOM_DIR) -> list[Path]:
    """Discover random catalogs for geometric-fraction calculations."""
    random_files = discover_files(path, RANDOM_GLOB_PATTERNS)
    if len(random_files) == 0:
        legacy_files = [
            Path(path) / RANDOM_PATTERN.format(random_index)
            for random_index in range(N_RANDOM_FILES)
            if (Path(path) / RANDOM_PATTERN.format(random_index)).exists()
        ]
        random_files = legacy_files
    if len(random_files) == 0:
        raise FileNotFoundError(
            f"No random catalogs found in {path}. "
            f"Tried patterns: {RANDOM_GLOB_PATTERNS} and {RANDOM_PATTERN}"
        )
    return random_files


def assigned_random_files(random_files: list[Path], rank: int, size: int):
    """Return the random catalog paths assigned to this MPI rank."""
    return random_files[rank::size]


def count_randoms_for_rank(cluster_xyz, aperture_chord_radius, random_files, rank: int, size: int):
    """Count random points around every cluster for the files assigned to rank."""
    local_counts = np.zeros(len(cluster_xyz), dtype=np.float64)
    local_files_read = 0

    for random_path in assigned_random_files(random_files, rank, size):
        if not random_path.exists():
            mpi_print(rank, f"missing random catalog: {random_path}")
            continue

        mpi_print(rank, f"reading random catalog {random_path.name}")
        ran = Table.read(random_path)
        random_xyz = spherical_to_cartesian(ran["RA"], ran["DEC"])
        tree = KDTree(random_xyz)
        counts = tree.query_ball_point(
            cluster_xyz,
            aperture_chord_radius,
            workers=N_KDTREE_WORKERS,
            return_length=True,
        )
        local_counts += np.asarray(counts, dtype=np.float64)
        local_files_read += 1
        mpi_print(rank, f"counted {len(ran):,} random points in {random_path.name}")

        del ran, random_xyz, tree, counts

    return local_counts, local_files_read


def build_geo_table(rm_clus: Table, total_counts, n_files_read: int) -> Table:
    """Build a cluster-level geometric-fraction table."""
    theta_deg = angular_radius_deg_from_hmpc(PROJECTED_APERTURE_HMPC, rm_clus["Z_SPEC_central"])
    area_deg2 = np.pi * theta_deg**2
    expected_per_file = area_deg2 * RANDOM_DENSITY_PER_DEG2
    expected_total = expected_per_file * n_files_read

    with np.errstate(divide="ignore", invalid="ignore"):
        geo_frac = np.asarray(total_counts, dtype=float) / expected_total
    geo_frac[~np.isfinite(geo_frac)] = np.nan

    out = Table()
    out["ID"] = rm_clus["ID"]
    out["RA_central"] = rm_clus["RA_central"]
    out["DEC_central"] = rm_clus["DEC_central"]
    out["Z_SPEC_central"] = rm_clus["Z_SPEC_central"]
    out["angRad_deg"] = theta_deg
    out["sq_deg"] = area_deg2
    out[f"Nr_{PROJECTED_APERTURE_HMPC:g}hmpc_expected_per_file"] = expected_per_file
    out["N_random_total"] = total_counts
    out["N_random_files_GEOMETRIC_FRACTION"] = n_files_read
    out["GEOMETRIC_FRACTION"] = geo_frac
    return out


def compute_geo_fraction_parallel(rm_clus: Table, comm, rank: int, size: int):
    """
    Compute geometric fraction with MPI.

    The random catalogs are split by file index across ranks.  Each rank counts
    randoms around every cluster for its assigned files, then rank 0 sums the
    counts and converts them into a footprint fraction.
    """
    if rank == 0:
        theta_deg = angular_radius_deg_from_hmpc(PROJECTED_APERTURE_HMPC, rm_clus["Z_SPEC_central"])
        theta_rad = np.deg2rad(theta_deg)
        aperture_chord_radius = 2.0 * np.sin(0.5 * theta_rad)
        cluster_xyz = spherical_to_cartesian(rm_clus["RA_central"], rm_clus["DEC_central"])
        random_files = discover_random_files(RANDOM_DIR)
        print(f"Discovered random catalogs: {len(random_files):,}")
    else:
        aperture_chord_radius = None
        cluster_xyz = None
        random_files = None

    if comm is not None:
        cluster_xyz = comm.bcast(cluster_xyz, root=0)
        aperture_chord_radius = comm.bcast(aperture_chord_radius, root=0)
        random_files = comm.bcast(random_files, root=0)

    local_counts, local_files_read = count_randoms_for_rank(
        cluster_xyz,
        aperture_chord_radius,
        random_files,
        rank,
        size,
    )

    if comm is None:
        total_counts = local_counts
        total_files_read = local_files_read
    else:
        total_counts = np.zeros_like(local_counts)
        comm.Reduce(local_counts, total_counts, op=MPI.SUM, root=0)
        total_files_read = comm.reduce(local_files_read, op=MPI.SUM, root=0)

    if rank != 0:
        return None

    if total_files_read == 0:
        raise RuntimeError(f"No random catalogs were read from {RANDOM_DIR}")

    return build_geo_table(rm_clus, total_counts, total_files_read)


def main() -> int:
    comm, rank, size = mpi_context()
    if rank == 0:
        print(f"Running with {size} MPI rank(s)")
        print("Loading redMaPPer catalog")
        rm_clus, rm_gal = load_redmapper_catalog(RM_PICKLE)
        print(f"redMaPPer clusters: {len(rm_clus):,}")
        print(f"redMaPPer member rows: {len(rm_gal):,}")

        print("Loading BGS catalog")
        bgs = load_bgs_catalog(BGS_CATALOG)
        print(f"BGS rows: {len(bgs):,}")

        print("Matching BGS galaxies to redMaPPer cluster centers")
        bgs_matched = match_bgs_to_clusters_projected(rm_clus, bgs)
        print(f"Matched cluster/BGS pairs: {len(bgs_matched):,}")
        print(
            "Matched redMaPPer cluster fraction: "
            f"{len(unique(bgs_matched, keys='ID')) / len(rm_clus):.3f}"
        )

        print("Matching BGS rows to redMaPPer member-galaxy rows")
        bgs_matched = attach_redmapper_member_match(bgs_matched, rm_gal)
        print(
            "Rows flagged as redMaPPer members: "
            f"{np.count_nonzero(bgs_matched['RM_gal_flag']):,}"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    else:
        rm_clus = None
        bgs_matched = None

    if comm is not None:
        rm_clus = comm.bcast(rm_clus, root=0)

    geo = compute_geo_fraction_parallel(rm_clus, comm, rank, size)

    if rank == 0:
        geo_join = geo[
            [
                "ID",
                "angRad_deg",
                "sq_deg",
                f"Nr_{PROJECTED_APERTURE_HMPC:g}hmpc_expected_per_file",
                "N_random_total",
                "N_random_files_GEOMETRIC_FRACTION",
                "GEOMETRIC_FRACTION",
            ]
        ]
        bgs_matched = join(bgs_matched, geo_join, keys="ID", join_type="left")
        geo_values = np.asarray(bgs_matched["GEOMETRIC_FRACTION"], dtype=float)
        n_missing_geo = np.count_nonzero(~np.isfinite(geo_values))
        print(f"Rows missing GEOMETRIC_FRACTION after join: {n_missing_geo:,}")
        print("Applying cluster-level LF weights and recomputing TOTAL_WEIGHT")
        bgs_matched = add_lf_weight_columns(bgs_matched)
        print(
            "LF_WEIGHT percentiles: "
            f"{np.nanpercentile(np.asarray(bgs_matched['LF_WEIGHT'], dtype=float), [0, 16, 50, 84, 100])}"
        )
        if ADD_SPEC_RICHNESS_COLUMNS:
            print("Computing lambda_spec_proj/noproj richness columns")
            bgs_matched = add_spectroscopic_richness_columns(bgs_matched)

        with OUTPUT_PICKLE.open("wb") as handle:
            pickle.dump(bgs_matched, handle, protocol=pickle.HIGHEST_PROTOCOL)
        bgs_matched.write(OUTPUT_FITS, overwrite=True)
        print(f"Saved final matched catalog: {OUTPUT_PICKLE}")
        print(f"Saved final matched catalog: {OUTPUT_FITS}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
