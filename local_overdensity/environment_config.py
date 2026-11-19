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
    native = tracer == "dr9" and t.get("photoz_mode") == "gaussian_fits"
    required += (["sweep_dirs", "photoz_dirs", "r_mag_limit", "maskbits_reject"] + ([] if native else ["lf_curve"])
                 if tracer == "dr9" else ["data_globs", "quality_column", "quality_values",
                                           "pi_half_hmpc"])
    for key in required:
        if t.get(key) is None:
            raise ValueError(f"Set {tracer}.{key}; see ENVIRONMENT_FILTERS.md")
    if t["input_selection_confirmed"] is not True:
        raise ValueError("Confirm random footprint/selection and unique-object ownership before production")
    if tracer == "dr9" and len(t["sweep_dirs"]) != len(t["photoz_dirs"]):
        raise ValueError("Supply one photo-z sidecar directory per sweep directory")
    if native:
        if t.get("regions") != ["north", "south"] or len(t["sweep_dirs"]) != 2 or len(t["random_globs"]) != 2:
            raise ValueError("Native DR9 expects north/south directory and random patterns in that order")
        if t.get("random_density_per_realization", 0) <= 0 or t.get("photoz_half_width", 0) <= 0:
            raise ValueError("Set positive random density and photo-z half width")
    if tracer == "bgs" and t["pi_half_hmpc"] <= 0:
        raise ValueError("Cylinder half-length must be positive")
    if tracer == "bgs":
        if t.get("quality_mode", "include") not in ("include", "exclude"):
            raise ValueError("quality_mode must be include or exclude")
        if t.get("random_mode", "volume") not in ("angular", "volume"):
            raise ValueError("random_mode must be angular or volume")
        if t.get("random_mode") == "angular" and t.get("random_density_per_realization", 0) <= 0:
            raise ValueError("Set positive BGS random density per realization")
    return c
