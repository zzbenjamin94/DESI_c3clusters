"""
Shared setup utilities for DESI_c3clusters notebooks.

The goal is to replace repeated notebook boilerplate with a few explicit calls:

    import sys
    sys.path.append("/global/homes/z/zzhang13/DESI")

    from tools.notebook_setup import *

    apply_plot_style()
    bgs_matched = load_bgs_matched_catalog()
    bgs_matched = add_dereddened_magnitudes(bgs_matched)
    zbin = make_redshift_offset_bins()
    lambda_bins = make_lambda_bins()
    z_bins = make_z_bins()

This module also re-exports the common scientific imports used at the top of
the older notebooks, so ``from tools.notebook_setup import *`` gives those
notebooks the usual names: ``np``, ``pd``, ``plt``, ``Table``, ``stats``, etc.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import pickle
import warnings

from astropy import table
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from astropy.io import ascii
from astropy.table import Table, join, unique
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy import stats
from scipy.interpolate import BSpline, make_interp_spline, UnivariateSpline
from scipy.special import erf
from scipy.stats import kde
import seaborn as sns
import tables
from tqdm import tqdm

from astropy.cosmology import Planck18
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
import numpy as np
import astropy.units as u
import astropy.cosmology.units as cu
from astropy.cosmology import Planck18
from astropy.cosmology import z_at_value
cosmo = Planck18

import sys
sys.path.append("/global/homes/z/zzhang13/DESI")
from setup import *
## Functions for computing spectroscopic richness and profiles
from tools.projection_functions import *


def _optional_import(module_name: str, alias: str | None = None):
    """Import optional notebook dependencies without breaking lightweight use."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        warnings.warn(
            f"Optional notebook dependency {module_name!r} could not be imported: {exc}",
            ImportWarning,
            stacklevel=2,
        )
        return None


emcee = _optional_import("emcee")
cr = _optional_import("incredible")
_specutils = _optional_import("specutils")
SpectralRegion = None if _specutils is None else _specutils.SpectralRegion

try:
    from setup import data_dir
except ImportError:  # pragma: no cover - depends on notebook sys.path
    data_dir = None


PLOT_PARAMS = {
    "axes.linewidth": 1.25,
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "xtick.minor.width": 1,
    "ytick.minor.width": 1,
    "xtick.major.size": 12,
    "ytick.major.size": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.pad": 6,
    "xtick.minor.pad": 6,
    "figure.constrained_layout.use": True,
    "font.family": "Serif",
    "font.size": 14,
    "legend.fontsize": 11,
}


@dataclass(frozen=True)
class RedshiftOffsetBins:
    bin_boundaries: np.ndarray
    bin_centers: np.ndarray
    bin_widths: np.ndarray
    micro_bin_boundaries: np.ndarray
    micro_bin_centers: np.ndarray


def apply_plot_style(extra_params: dict | None = None) -> None:
    """Apply the plotting style used by the richness notebooks."""

    params = dict(PLOT_PARAMS)
    if extra_params:
        params.update(extra_params)
    plt.rcParams.update(params)


def load_bgs_matched_catalog(
    filename: str = "bgs_clus_RM_gal_matched.pickle",
    catalog_dir: str | Path | None = None,
):
    """
    Load the matched BGS-redMaPPer catalog.

    If ``catalog_dir`` is not supplied, this uses ``setup.data_dir()`` from the
    existing repo configuration.
    """

    if catalog_dir is None:
        if data_dir is None:
            raise ImportError(
                "Could not import setup.data_dir. Add the repo root to sys.path "
                "or pass catalog_dir explicitly."
            )
        catalog_dir = data_dir()

    path = Path(catalog_dir) / filename
    with path.open("rb") as handle:
        return pickle.load(handle)


def add_dereddened_magnitudes(table):
    """
    Add ``r_dered``, ``g_dered``, and ``gmr`` columns in-place and return table.
    """

    table["r_dered"] = 22.5 - 2.5 * np.log10(table["flux_r_dered"])
    table["g_dered"] = 22.5 - 2.5 * np.log10(table["flux_g_dered"])
    table["gmr"] = table["g_dered"] - table["r_dered"]
    return table


def add_absolute_magnitude(
    table,
    z_col: str = "Z_BGS",
    apparent_mag_col: str = "r_dered",
    output_col: str = "M_r",
    cosmo=Planck18,
    h: float = 0.67,
):
    """Add the h-scaled absolute magnitude column used in the richness notebooks."""

    table[output_col] = (
        table[apparent_mag_col] - cosmo.distmod(table[z_col]).value - 5 * np.log10(h)
    )
    return table


def apply_default_richness_cuts(
    table,
    r_dered_limit: float | None = 19.5,
    z_max: float | None = None,
    m_r_limit: float | None = None,
    lambda_min: float | None = None,
):
    """Apply common optional cuts and return the filtered table."""

    mask = np.ones(len(table), dtype=bool)
    if r_dered_limit is not None:
        mask &= np.asarray(table["r_dered"]) < r_dered_limit
    if z_max is not None:
        mask &= np.asarray(table["Z_BGS"]) < z_max
    if m_r_limit is not None:
        mask &= np.asarray(table["M_r"]) < m_r_limit
    if lambda_min is not None:
        mask &= np.asarray(table["LAMBDA"]) > lambda_min
    return table[np.where(mask)]


def make_redshift_offset_bins() -> RedshiftOffsetBins:
    """Return the non-uniform redshift-offset bins used for richness profiles."""

    wide_bin_1 = np.linspace(-0.1, -0.005, 21, endpoint=False)
    small_bin = np.linspace(-0.005, 0.005, 21, endpoint=False)
    micro_bin = np.linspace(-0.005, 0.005, 31)
    wide_bin_2 = np.linspace(0.005, 0.1, 21, endpoint=False)

    bin_boundaries = np.hstack((wide_bin_1, small_bin, wide_bin_2))
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    bin_widths = np.diff(bin_boundaries)
    micro_bin_centers = 0.5 * (micro_bin[:-1] + micro_bin[1:])

    assert len(set(bin_centers)) == len(bin_centers), "Overlapping bins"
    assert len(set(bin_boundaries)) == len(bin_boundaries), "Overlapping bins"

    return RedshiftOffsetBins(
        bin_boundaries=bin_boundaries,
        bin_centers=bin_centers,
        bin_widths=bin_widths,
        micro_bin_boundaries=micro_bin,
        micro_bin_centers=micro_bin_centers,
    )


def make_lambda_bins(
    min_lambda: float = 20,
    max_lambda: float = 100,
    n_bins: int = 10,
    log: bool = True,
) -> list[list[float]]:
    """Return richness bin pairs, matching the notebook's default log bins."""

    if log:
        edges = np.logspace(np.log10(min_lambda), np.log10(max_lambda), n_bins + 1)
    else:
        edges = np.linspace(min_lambda, max_lambda, n_bins + 1)
    return [[float(edges[i]), float(edges[i + 1])] for i in range(len(edges) - 1)]


def make_z_bins(
    edges: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4)
) -> list[list[float]]:
    """Return redshift bin pairs."""

    return [[float(edges[i]), float(edges[i + 1])] for i in range(len(edges) - 1)]


def prepare_default_richness_inputs(
    catalog_filename: str = "bgs_clus_RM_gal_matched.pickle",
    catalog_dir: str | Path | None = None,
    apply_cuts: bool = True,
):
    """
    Convenience wrapper for the common richness-notebook setup.

    Returns
    -------
    bgs_matched, zbin, lambda_bins, z_bins
    """

    apply_plot_style()
    bgs_matched = load_bgs_matched_catalog(catalog_filename, catalog_dir=catalog_dir)
    bgs_matched = add_dereddened_magnitudes(bgs_matched)
    bgs_matched = add_absolute_magnitude(bgs_matched)
    if apply_cuts:
        bgs_matched = apply_default_richness_cuts(bgs_matched)

    zbin = make_redshift_offset_bins()
    lambda_bins = make_lambda_bins()
    z_bins = make_z_bins()
    return bgs_matched, zbin, lambda_bins, z_bins
