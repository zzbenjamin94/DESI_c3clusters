"""Validate and reduce an existing richness catalog to one row per cluster.

Does not recompute richness or weights. Never overwrites the input catalog.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.richness_selection import BCG_Z_RANGE, BCG_Z_BIN_EDGES

DEFAULT_INPUT = REPO_ROOT / "catalogs/bgs_clus_RM_gal_matched_with_spec_richness_lfweighted.pickle"
DEFAULT_OUTPUT = REPO_ROOT / "catalogs/conditional_richness/sample.npz"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sample(path):
    """Read a prepared, checksum-validated sample without importing a fitter."""
    path = Path(path)
    audit = json.loads(path.with_suffix(".json").read_text())
    if file_sha256(path) != audit["sample_sha256"]:
        raise ValueError("Sample checksum differs from preparation audit")
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    n = len(data["ID"])
    for key in ("lambda_rm", "lambda_spec", "z"):
        if data[key].shape != (n,) or np.any(~np.isfinite(data[key])):
            raise ValueError(f"Invalid shape or nonfinite {key} in prepared sample")
    if (n == 0 or np.any(data["lambda_spec"] <= 0) or np.any(data["lambda_rm"] <= 0)
            or len(np.unique(data["ID"])) != n):
        raise ValueError("Expected nonempty sample, positive richnesses and unique IDs")
    return data, audit


def read_catalog(path):
    path = Path(path)
    if path.suffix in {".pickle", ".pkl"}:
        with path.open("rb") as handle:
            obj = pickle.load(handle)
        if isinstance(obj, Table):
            return obj
        if obj.__class__.__module__.startswith("pandas"):
            return Table.from_pandas(obj)
        return Table(obj)
    return Table.read(path)


def prepare(table, spec_col="lambda_spec_noproj_weighted", rm_min=20.0, rm_max=np.inf):
    if not (0 <= rm_min < rm_max):
        raise ValueError("Require 0 <= rm_min < rm_max")
    required = ["ID", "Z_SPEC_central", "LAMBDA", spec_col]
    missing = set(required) - set(table.colnames)
    if missing:
        raise KeyError(f"Missing {sorted(missing)}. Run postprocess_spectroscopic_richness.py first.")
    if np.any(np.ma.getmaskarray(table["ID"])):
        raise ValueError("Masked cluster IDs are not allowed")
    ids, first, inverse = np.unique(np.asarray(table["ID"]).astype(str), return_index=True, return_inverse=True)
    values = {}
    optional = [c for c in ("LF_WEIGHT", "GEOMETRIC_FRACTION", "lf_z_cluster", "lambda_spec_noproj")
                if c in table.colnames]
    for col in required[1:] + optional:
        arr = np.ma.asarray(table[col], dtype=float).filled(np.nan)
        reference = arr[first][inverse]
        consistent = np.isclose(arr, reference, rtol=1e-10, atol=1e-12, equal_nan=True)
        if not np.all(consistent):
            bad = ids[np.unique(inverse[~consistent])][:5]
            raise ValueError(f"{col} varies within cluster IDs {bad.tolist()}; cannot silently deduplicate")
        values[col] = arr[first]
    z, rm, spec = values["Z_SPEC_central"], values["LAMBDA"], values[spec_col]
    keep = np.ones(len(ids), dtype=bool)
    cutflow = {"input_rows": len(table), "unique_clusters": len(ids)}
    conditions = [
        ("finite_cluster_measurements", np.isfinite(z) & np.isfinite(rm) & np.isfinite(spec)),
        ("bcg_redshift_range", (z >= BCG_Z_RANGE[0]) & (z < BCG_Z_RANGE[1])),
        ("positive_lambda_spec", spec > 0),
        ("rm_selection", (rm > 0) & (rm >= rm_min) & (rm < rm_max)),
    ]
    for name, mask in conditions:
        before = int(keep.sum())
        keep &= mask
        cutflow[name] = {"remaining": int(keep.sum()), "removed": before - int(keep.sum())}
    if not np.any(keep):
        raise ValueError(f"No clusters pass selection: {cutflow}")
    sample = {"ID": ids[keep], "lambda_rm": rm[keep], "lambda_spec": spec[keep], "z": z[keep]}
    for col in optional:
        sample[col] = values[col][keep]
    audit = {
        "schema_version": 1, "spec_column": spec_col, "cutflow": cutflow,
        "bcg_z_range": list(BCG_Z_RANGE), "z_bin_edges": list(BCG_Z_BIN_EDGES),
        "rm_min": rm_min, "rm_max": None if np.isinf(rm_max) else rm_max,
        "z_bin_counts": np.histogram(sample["z"], BCG_Z_BIN_EDGES)[0].tolist(),
        "warnings": [
            "Models condition on measured lambda_spec; scatter is not deconvolved intrinsic scatter.",
            "Only the explicit RM interval is modeled as selection; BGS matching/completeness may add selection.",
            "Current upstream estimator shares continuum and mean offset-bin weights across clusters. "
            "Validate cluster-level LF scaling before physical interpretation; this preparation does not repair it.",
            "Nonpositive lambda_spec are excluded and counted; no log floor is substituted.",
        ],
    }
    if "lf_z_cluster" in sample and not np.allclose(sample["lf_z_cluster"], sample["z"], equal_nan=False):
        raise ValueError("LF weights were evaluated at redshifts inconsistent with Z_SPEC_central")
    return sample, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spec-col", default="lambda_spec_noproj_weighted")
    parser.add_argument("--rm-min", type=float, default=20.0)
    parser.add_argument("--rm-max", type=float, default=np.inf)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    audit_path = args.output.with_suffix(".json")
    if (args.output.exists() or audit_path.exists()) and not args.overwrite:
        parser.error("Output exists; select a new output or explicitly use --overwrite")
    if args.input.resolve() in {args.output.resolve(), audit_path.resolve()}:
        parser.error("Input and output must differ")
    sample, audit = prepare(read_catalog(args.input), args.spec_col, args.rm_min, args.rm_max)
    audit.update(input_path=str(args.input.resolve()), input_sha256=file_sha256(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **sample)
    audit["sample_sha256"] = file_sha256(args.output)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
