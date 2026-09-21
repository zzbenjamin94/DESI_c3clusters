"""Select the analysis sample and append spectroscopic-richness columns.

Run this after ``projection_match_catalogs.py``. The input parent catalog is
left unchanged; this script writes a separate, analysis-ready catalog.
"""

from __future__ import annotations

import pickle
from pathlib import Path
import sys

import numpy as np
from astropy.table import Table, unique


REPO_ROOT = Path("/global/homes/z/zzhang13/DESI/Projection")
if not REPO_ROOT.exists():
    REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.projection_functions import append_spec_richness_columns
from tools.richness_selection import (
    BCG_Z_RANGE,
    BGS_Z_RANGE,
    DELTA_Z_RANGE,
    make_redshift_offset_bins,
    select_richness_analysis_sample,
)


CATALOG_DIR = REPO_ROOT / "catalogs"
INPUT_PICKLE = CATALOG_DIR / "bgs_clus_RM_gal_matched_with_weights.pickle"
OUTPUT_PICKLE = (
    CATALOG_DIR / "bgs_clus_RM_gal_matched_with_spec_richness_lfweighted.pickle"
)
OUTPUT_FITS = (
    CATALOG_DIR / "bgs_clus_RM_gal_matched_with_spec_richness_lfweighted.fits"
)


def read_pickle_table(path: Path) -> Table:
    """Read a pickled Astropy table or pandas DataFrame as an Astropy Table."""
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, Table):
        return obj
    if obj.__class__.__module__.startswith("pandas"):
        return Table.from_pandas(obj)
    if hasattr(obj, "to_pandas"):
        return Table.from_pandas(obj.to_pandas())
    return Table(obj)


def main() -> int:
    print(f"Reading broad parent catalog: {INPUT_PICKLE}")
    parent = read_pickle_table(INPUT_PICKLE)
    n_parent_clusters = len(unique(parent, keys="ID"))

    selected = select_richness_analysis_sample(parent)
    n_selected_clusters = len(unique(selected, keys="ID"))
    if len(selected) == 0:
        raise RuntimeError("The postprocessing cuts selected no candidate galaxies")

    print(
        "Postprocessing cuts: "
        f"{BCG_Z_RANGE[0]:.2f} <= z_BCG < {BCG_Z_RANGE[1]:.2f}, "
        f"{BGS_Z_RANGE[0]:.2f} <= z_BGS < {BGS_Z_RANGE[1]:.2f}, "
        f"{DELTA_Z_RANGE[0]:.2f} <= Delta_z <= {DELTA_Z_RANGE[1]:.2f}"
    )
    print(f"Rows: {len(parent):,} -> {len(selected):,}")
    print(f"Clusters: {n_parent_clusters:,} -> {n_selected_clusters:,}")

    zbin = make_redshift_offset_bins()
    selected = append_spec_richness_columns(
        selected,
        zbin.bin_boundaries,
        zbin.bin_centers,
        total_weight_col="TOTAL_WEIGHT",
    )

    OUTPUT_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PICKLE.open("wb") as handle:
        pickle.dump(selected, handle, protocol=pickle.HIGHEST_PROTOCOL)
    selected.write(OUTPUT_FITS, overwrite=True)

    delta_z = np.asarray(selected["DZ_CLUSTER"], dtype=float)
    print(f"Selected Delta_z range: [{np.nanmin(delta_z):.5f}, {np.nanmax(delta_z):.5f}]")
    print(f"Saved analysis catalog: {OUTPUT_PICKLE}")
    print(f"Saved analysis catalog: {OUTPUT_FITS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
