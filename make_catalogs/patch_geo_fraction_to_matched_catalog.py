"""
Patch precomputed geometric fractions onto the matched BGS/redMaPPer catalog.

This script joins the cluster-level output from ``compute_geo_fraction_mpi.py``
onto the galaxy-level matched catalog from ``projection_match_catalogs.py``.
It writes a new pickle so the original matched catalog remains available.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from astropy.table import Table, join

from projection_match_catalogs import CATALOG_DIR, add_weight_columns


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MATCHED_PICKLE = CATALOG_DIR / "bgs_clus_RM_gal_matched_no_geoFrac.pickle"
GEO_PICKLE = CATALOG_DIR / "rm_cluster_geo_fraction_1p5hmpc.pickle"
OUTPUT_PICKLE = CATALOG_DIR / "bgs_clus_RM_gal_matched_with_geoFrac.pickle"

GEOMETRY_COLUMNS = {
    "geoFrac",
    "angRad_deg",
    "sq_deg",
    "N_random_files_geoFrac",
    "N_random_total",
    "Nr_1.5hmpc_expected_per_file",
    "Nr_1.5hmpc_expected",
}


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


def drop_stale_geometry_columns(table: Table) -> Table:
    """Remove existing geometric-fraction columns before patching fresh values."""
    out = table.copy()
    for col in list(out.colnames):
        if col in GEOMETRY_COLUMNS:
            out.remove_column(col)
    return out


def main() -> int:
    print(f"Reading matched catalog: {MATCHED_PICKLE}")
    matched = read_pickle_table(MATCHED_PICKLE)
    print(f"Matched rows: {len(matched):,}")

    print(f"Reading geometric fractions: {GEO_PICKLE}")
    geo = read_pickle_table(GEO_PICKLE)
    print(f"Geometry rows: {len(geo):,}")

    matched = drop_stale_geometry_columns(matched)
    keep_geo_cols = [
        col
        for col in geo.colnames
        if col
        in {
            "ID",
            "geoFrac",
            "angRad_deg",
            "sq_deg",
            "N_random_files_geoFrac",
            "N_random_total",
            "Nr_1.5hmpc_expected_per_file",
        }
    ]
    geo = geo[keep_geo_cols]

    patched = join(matched, geo, keys="ID", join_type="left")
    patched = add_weight_columns(patched)
    geo_values = np.asarray(patched["geoFrac"], dtype=float)
    n_missing = np.count_nonzero(~np.isfinite(geo_values))
    print(f"Patched rows: {len(patched):,}")
    print(f"Rows missing geoFrac after join: {n_missing:,}")

    with OUTPUT_PICKLE.open("wb") as handle:
        pickle.dump(patched, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {OUTPUT_PICKLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
