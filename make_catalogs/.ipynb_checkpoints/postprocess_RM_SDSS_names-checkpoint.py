"""
Postprocess an existing RM_SDSS_df.pkl into readable column names.

This is for the case where the raw redMaPPer FITS catalogs are no longer
available, but the merged pickle made by mem_galaxy_2ddist_SDSS.ipynb already
exists.

Preferred convention after this script:

    *_central  = cluster/BCG/central quantities
    *_member   = redMaPPer member-galaxy quantities

For backward compatibility, legacy aliases such as RA_x, DEC_x, Z_SPEC_x,
RA_y, DEC_y, Z_SPEC_y, R, and P are also kept.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path("/global/homes/z/zzhang13/DESI/Projection")
if not REPO_ROOT.exists():
    REPO_ROOT = Path(__file__).resolve().parents[1]

CATALOG_DIR = REPO_ROOT / "catalogs"
INPUT_PICKLE = CATALOG_DIR / "RM_SDSS_df.pkl"
BACKUP_PICKLE = CATALOG_DIR / "RM_SDSS_df_legacy_xy.pkl"
OUTPUT_PICKLE = INPUT_PICKLE


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def read_pickle_as_dataframe(path: Path) -> pd.DataFrame:
    """Read a pickle as a pandas DataFrame, accepting common table objects."""
    with path.open("rb") as handle:
        obj = pickle.load(handle)

    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, Table):
        return obj.to_pandas()
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()

    return pd.DataFrame(obj)


def write_pickle(obj, path: Path) -> None:
    """Write a pickle using the highest available protocol."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


# -----------------------------------------------------------------------------
# Naming and filtering
# -----------------------------------------------------------------------------

def add_readable_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add *_central and *_member aliases from legacy pandas _x/_y columns.

    This keeps the old columns too.  The science-facing columns are the new
    aliases; the old names are retained so older notebooks do not immediately
    break.
    """
    df = df.copy()

    for col in list(df.columns):
        if col.endswith("_x"):
            preferred = f"{col[:-2]}_central"
            if preferred not in df.columns:
                df[preferred] = df[col]
        elif col.endswith("_y"):
            preferred = f"{col[:-2]}_member"
            if preferred not in df.columns:
                df[preferred] = df[col]

    member_only_aliases = {
        "R": "R_member",
        "P": "P_member",
        "P_FREE": "P_FREE_member",
        "THETA_I": "THETA_I_member",
        "THETA_R": "THETA_R_member",
    }
    for legacy, preferred in member_only_aliases.items():
        if legacy in df.columns and preferred not in df.columns:
            df[preferred] = df[legacy]

    # Legacy aliases copied from preferred names.  This is intentionally after
    # the readable aliases are made, so the preferred columns are authoritative.
    legacy_aliases = {
        "RA_x": "RA_central",
        "DEC_x": "DEC_central",
        "Z_SPEC_x": "Z_SPEC_central",
        "OBJID_x": "OBJID_central",
        "MODEL_MAG_R_x": "MODEL_MAG_R_central",
        "MODEL_MAGERR_R_x": "MODEL_MAGERR_R_central",
        "RA_y": "RA_member",
        "DEC_y": "DEC_member",
        "Z_SPEC_y": "Z_SPEC_member",
        "OBJID_y": "OBJID_member",
        "MODEL_MAG_R_y": "MODEL_MAG_R_member",
        "MODEL_MAGERR_R_y": "MODEL_MAGERR_R_member",
        "R": "R_member",
        "P": "P_member",
    }
    for legacy, preferred in legacy_aliases.items():
        if preferred in df.columns:
            df[legacy] = df[preferred]

    return df


def remove_repeated_central_member_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the redMaPPer member is the central galaxy itself.

    This preserves the original logic:

        merge_df = merge_df.iloc[np.where(
            merge_df["Z_SPEC_y"] != merge_df["Z_SPEC_x"]
        )]

    but uses the readable names when available.
    """
    df = df.copy()

    z_central = "Z_SPEC_central" if "Z_SPEC_central" in df.columns else "Z_SPEC_x"
    mask = np.isfinite(np.asarray(df[z_central], dtype=float))
    mask &= np.asarray(df[z_central], dtype=float) > 0.0

    if {"OBJID_central", "OBJID_member"}.issubset(df.columns):
        mask &= np.asarray(df["OBJID_central"]) != np.asarray(df["OBJID_member"])
    elif {"OBJID_x", "OBJID_y"}.issubset(df.columns):
        mask &= np.asarray(df["OBJID_x"]) != np.asarray(df["OBJID_y"])

    if {"Z_SPEC_central", "Z_SPEC_member"}.issubset(df.columns):
        mask &= np.asarray(df["Z_SPEC_member"]) != np.asarray(df["Z_SPEC_central"])
    elif {"Z_SPEC_x", "Z_SPEC_y"}.issubset(df.columns):
        mask &= np.asarray(df["Z_SPEC_y"]) != np.asarray(df["Z_SPEC_x"])

    return df.loc[mask].reset_index(drop=True)


def summarize(df: pd.DataFrame, label: str) -> None:
    """Print a compact summary useful for checking the postprocess step."""
    print(f"\n{label}")
    print("-" * len(label))
    print(f"rows: {len(df):,}")
    if "ID" in df.columns:
        print(f"unique clusters: {df['ID'].nunique():,}")
    for col in [
        "RA_central",
        "DEC_central",
        "Z_SPEC_central",
        "RA_member",
        "DEC_member",
        "Z_SPEC_member",
        "R_member",
        "P_member",
    ]:
        if col in df.columns:
            n_finite = np.isfinite(np.asarray(df[col], dtype=float)).sum()
            print(f"{col:16s}: {n_finite:,} finite")


def main() -> int:
    if not INPUT_PICKLE.exists():
        raise FileNotFoundError(f"Input pickle does not exist: {INPUT_PICKLE}")

    df = read_pickle_as_dataframe(INPUT_PICKLE)
    summarize(df, "Input")

    if not BACKUP_PICKLE.exists():
        write_pickle(df, BACKUP_PICKLE)
        print(f"\nBacked up legacy input to: {BACKUP_PICKLE}")
    else:
        print(f"\nBackup already exists, leaving it untouched: {BACKUP_PICKLE}")

    df = add_readable_column_aliases(df)
    before_cut = len(df)
    df = remove_repeated_central_member_rows(df)
    removed = before_cut - len(df)

    summarize(df, "Standardized output")
    print(f"\nRemoved repeated central/member rows: {removed:,}")

    write_pickle(df, OUTPUT_PICKLE)
    print(f"Saved standardized catalog to: {OUTPUT_PICKLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
