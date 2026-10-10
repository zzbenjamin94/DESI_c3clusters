"""Explicit configuration and input contracts for Section 4 measurements."""

import json
from pathlib import Path


def load_config(path, tracer):
    c = json.loads(Path(path).read_text())
    for key in ("cluster_catalog", "output_dir", "gaussian_width_hmpc", "chunk_rows"):
        if c.get(key) is None:
            raise ValueError(f"Set {key} in the configuration")
    if c["gaussian_width_hmpc"] <= 0 or c["chunk_rows"] < 1:
        raise ValueError("Gaussian width and chunk_rows must be positive")
    t = c[tracer]
    required = ["random_globs", "random_measure_column", "input_selection_confirmed"]
    required += (["sweep_dirs", "photoz_dirs", "lf_curve", "r_mag_limit", "maskbits_reject"]
                 if tracer == "dr9" else ["data_globs", "quality_column", "quality_values",
                                           "pi_half_hmpc", "lf_curve", "nbar_curve"])
    for key in required:
        if t.get(key) is None:
            raise ValueError(f"Set {tracer}.{key}; see ENVIRONMENT_FILTERS.md")
    if t["input_selection_confirmed"] is not True:
        raise ValueError("Confirm random footprint/selection and unique-object ownership before production")
    if tracer == "dr9" and len(t["sweep_dirs"]) != len(t["photoz_dirs"]):
        raise ValueError("Supply one photo-z sidecar directory per sweep directory")
    if tracer == "bgs" and t["pi_half_hmpc"] <= 0:
        raise ValueError("Cylinder half-length must be positive")
    return c
