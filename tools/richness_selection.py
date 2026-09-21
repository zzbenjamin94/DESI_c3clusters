"""Shared postprocessing selections for spectroscopic-richness analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BCG_Z_RANGE = (0.10, 0.35)
BGS_Z_RANGE = (0.05, 0.40)
DELTA_Z_RANGE = (-0.05, 0.05)
BCG_Z_BIN_EDGES = (0.10, 0.18, 0.24, 0.35)


@dataclass(frozen=True)
class RedshiftOffsetBins:
    bin_boundaries: np.ndarray
    bin_centers: np.ndarray
    bin_widths: np.ndarray
    micro_bin_boundaries: np.ndarray
    micro_bin_centers: np.ndarray


def _column_name(table, preferred: str, fallback: str | None = None) -> str:
    names = table.colnames if hasattr(table, "colnames") else table.columns
    if preferred in names:
        return preferred
    if fallback is not None and fallback in names:
        return fallback
    choices = repr(preferred) if fallback is None else f"{preferred!r} or {fallback!r}"
    raise KeyError(f"Missing required column {choices}")


def normalized_redshift_offset(
    table,
    bgs_z_col: str = "Z_BGS",
    central_z_col: str | None = None,
) -> np.ndarray:
    """Return Delta z = (z_BGS - z_BCG) / (1 + z_BCG)."""
    if central_z_col is None:
        central_z_col = _column_name(table, "Z_SPEC_central", "Z_SPEC_x")
    z_bgs = np.asarray(table[bgs_z_col], dtype=float)
    z_bcg = np.asarray(table[central_z_col], dtype=float)
    return (z_bgs - z_bcg) / (1.0 + z_bcg)


def richness_analysis_mask(
    table,
    bcg_z_range: tuple[float, float] = BCG_Z_RANGE,
    bgs_z_range: tuple[float, float] = BGS_Z_RANGE,
    delta_z_range: tuple[float, float] = DELTA_Z_RANGE,
) -> np.ndarray:
    """Select the analysis domain without modifying the broad parent catalog."""
    central_z_col = _column_name(table, "Z_SPEC_central", "Z_SPEC_x")
    z_bcg = np.asarray(table[central_z_col], dtype=float)
    z_bgs = np.asarray(table["Z_BGS"], dtype=float)
    delta_z = normalized_redshift_offset(table, central_z_col=central_z_col)

    return (
        np.isfinite(z_bcg)
        & np.isfinite(z_bgs)
        & np.isfinite(delta_z)
        & (z_bcg >= bcg_z_range[0])
        & (z_bcg < bcg_z_range[1])
        & (z_bgs >= bgs_z_range[0])
        & (z_bgs < bgs_z_range[1])
        & (delta_z >= delta_z_range[0])
        & (delta_z <= delta_z_range[1])
    )


def select_richness_analysis_sample(table):
    """Return rows in the adopted BCG, BGS, and normalized-offset ranges."""
    selected = table[richness_analysis_mask(table)]
    selected = selected.copy()
    selected["DZ_CLUSTER"] = normalized_redshift_offset(selected)
    return selected


def make_redshift_offset_bins() -> RedshiftOffsetBins:
    """Return non-uniform Delta-z bins spanning exactly -0.05 to 0.05."""
    left = np.linspace(DELTA_Z_RANGE[0], -0.005, 22)
    core = np.linspace(-0.005, 0.005, 22)
    right = np.linspace(0.005, DELTA_Z_RANGE[1], 22)
    bin_boundaries = np.concatenate((left[:-1], core[:-1], right))
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    bin_widths = np.diff(bin_boundaries)

    micro_bin_boundaries = np.linspace(-0.005, 0.005, 31)
    micro_bin_centers = 0.5 * (
        micro_bin_boundaries[:-1] + micro_bin_boundaries[1:]
    )

    if not np.all(np.diff(bin_boundaries) > 0):
        raise ValueError("Redshift-offset bin boundaries must be strictly increasing")

    return RedshiftOffsetBins(
        bin_boundaries=bin_boundaries,
        bin_centers=bin_centers,
        bin_widths=bin_widths,
        micro_bin_boundaries=micro_bin_boundaries,
        micro_bin_centers=micro_bin_centers,
    )


def make_z_bins(
    edges: tuple[float, ...] = BCG_Z_BIN_EDGES,
) -> list[list[float]]:
    """Return the three approximately equal-count BCG redshift bins."""
    return [[float(lo), float(hi)] for lo, hi in zip(edges[:-1], edges[1:])]
