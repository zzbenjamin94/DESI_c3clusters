"""Recompute three weighting stages once; never modify the matched catalog.

The continuum prescription is a degree-6 Chebyshev/specutils fit with a
five-bin median smoothing window.
Only preparation requires specutils; plotting cached products does not.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from richness_relation.prepare_conditional_richness import DEFAULT_INPUT, file_sha256, read_catalog
from tools.richness_selection import (
    BCG_Z_BIN_EDGES, make_redshift_offset_bins, richness_analysis_mask,
    normalized_redshift_offset,
)

DEFAULT_OUTPUT = ROOT / "catalogs/richness_weighting/cluster_weighting_stages.ecsv"
STAGES = ("lambda_spec_unweighted", "lambda_spec_geo_comp", "lambda_spec_geo_comp_lf")
CONTINUUM_DEGREE = 6
CONTINUUM_MEDIAN_WINDOW = 5


def floats(table, key):
    return np.ma.asarray(table[key], dtype=float).filled(np.nan)


def legacy_continuum(binBoundaries, binCent, offsets):
    """Legacy fitting procedure with updated degree and median-window settings."""
    from astropy import units as u
    from astropy.modeling.models import Chebyshev1D
    from specutils import SpectralRegion
    from specutils.spectra import Spectrum1D
    from specutils.fitting import fit_generic_continuum

    pdf, _ = np.histogram(offsets, bins=binBoundaries, density=True)
    spectrum = Spectrum1D(flux=pdf * u.Jy, spectral_axis=binCent * u.um)
    model = fit_generic_continuum(
        spectrum, model=Chebyshev1D(CONTINUUM_DEGREE), median_window=CONTINUUM_MEDIAN_WINDOW,
        exclude_regions=[SpectralRegion(-0.02 * u.um, 0.02 * u.um)],
    )
    return np.asarray(model(binCent * u.um).value, dtype=float)


def prepare_stages(table, continuum_estimator=None):
    required = {"ID", "Z_SPEC_central", "Z_BGS", "LAMBDA",
                "COMP_WEIGHT", "GEOMETRIC_WEIGHT", "LF_WEIGHT"}
    missing = required - set(table.colnames)
    if missing:
        raise KeyError(f"Missing weight-stage inputs: {sorted(missing)}")
    if np.any(np.ma.getmaskarray(table["ID"])):
        raise ValueError("Masked cluster IDs are not allowed")
    # Validate cluster-level fields BEFORE selecting individual candidate rows.
    ids, first, inverse = np.unique(np.asarray(table["ID"]).astype(str),
                                    return_index=True, return_inverse=True)
    for key in ("Z_SPEC_central", "LAMBDA", "GEOMETRIC_WEIGHT", "LF_WEIGHT"):
        values = floats(table, key)
        bad = ~np.isclose(values, values[first][inverse], rtol=1e-10, atol=1e-12, equal_nan=True)
        if np.any(bad):
            raise ValueError(f"{key} varies within cluster IDs {ids[np.unique(inverse[bad])][:5]}")
    science = richness_analysis_mask(table)
    tab = table[science]
    if not len(tab):
        raise ValueError("No candidate rows pass the science redshift/offset cuts")
    weights = {k: floats(tab, k) for k in ("COMP_WEIGHT", "GEOMETRIC_WEIGHT", "LF_WEIGHT")}
    for key, values in weights.items():
        if np.any(~np.isfinite(values) | (values <= 0)):
            raise ValueError(f"Invalid {key} in selected candidates; refusing to silently drop or zero weights")
    checks = {}
    for column, expected in (
        ("TOTAL_WEIGHT", weights["COMP_WEIGHT"] * weights["GEOMETRIC_WEIGHT"] * weights["LF_WEIGHT"]),
        ("PROB_OBS", 1 / weights["COMP_WEIGHT"]),
        ("GEOMETRIC_FRACTION", 1 / weights["GEOMETRIC_WEIGHT"]),
    ):
        if column in tab.colnames:
            if not np.allclose(floats(tab, column), expected, rtol=1e-6, atol=1e-10):
                raise ValueError(f"{column} is inconsistent with the separate correction weights")
            checks[column] = "consistent"
    bins = make_redshift_offset_bins()
    offsets = normalized_redshift_offset(tab)
    custom_continuum = continuum_estimator is not None
    continuum_estimator = continuum_estimator or legacy_continuum
    continuum = continuum_estimator(bins.bin_boundaries, bins.bin_centers, offsets)
    if continuum.shape != bins.bin_centers.shape or not np.all(np.isfinite(continuum)):
        raise ValueError("Continuum estimator returned invalid values")
    mass = continuum * bins.bin_widths
    counts, _ = np.histogram(offsets, bins.bin_boundaries)
    weighted, _ = np.histogram(offsets, bins.bin_boundaries, weights=weights["COMP_WEIGHT"])
    # Only the BACKGROUND uses a pooled offset-dependent completeness weight.
    # For empty bins, use the sample mean rather than silently zeroing background.
    comp_background = np.full(len(counts), weights["COMP_WEIGHT"].mean())
    np.divide(weighted, counts, out=comp_background, where=counts > 0)
    ids, first, inverse = np.unique(np.asarray(tab["ID"]).astype(str),
                                    return_index=True, return_inverse=True)
    n = np.bincount(inverse).astype(float)
    signal_comp = np.bincount(inverse, weights=weights["COMP_WEIGHT"])
    geo, lf = weights["GEOMETRIC_WEIGHT"][first], weights["LF_WEIGHT"][first]
    background = n * mass.sum()
    background_gc = geo * n * np.dot(mass, comp_background)
    out = Table({"ID": ids, "Z_SPEC_central": floats(tab, "Z_SPEC_central")[first],
                 "LAMBDA": floats(tab, "LAMBDA")[first], "N_candidates": n.astype(int),
                 "GEOMETRIC_WEIGHT": geo, "LF_WEIGHT": lf,
                 "COMP_WEIGHT_mean": signal_comp / n,
                 "lambda_spec_proj_unweighted": n,
                 "lambda_spec_proj_geo_comp": geo * signal_comp,
                 "lambda_spec_proj_geo_comp_lf": lf * geo * signal_comp,
                 "continuum_unweighted": background,
                 "continuum_geo_comp": background_gc,
                 "lambda_spec_unweighted": n - background,
                 "lambda_spec_geo_comp": geo * signal_comp - background_gc})
    out[STAGES[2]] = lf * np.asarray(out[STAGES[1]])
    # Preserve old richnesses solely for auditing; never use them for the stages.
    for key in ("lambda_spec_noproj", "lambda_spec_noproj_weighted"):
        if key in tab.colnames:
            values = floats(tab, key)
            if not np.all(np.isclose(values, values[first][inverse], equal_nan=True)):
                raise ValueError(f"Stored {key} is not constant within each ID")
            out["stored_" + key] = values[first]
    all_ids = np.unique(np.asarray(table["ID"]).astype(str))
    audit = {
        "continuum_settings": {
            "estimator": "custom" if custom_continuum else "specutils_chebyshev",
            "degree": None if custom_continuum else CONTINUUM_DEGREE,
            "median_window_bins": None if custom_continuum else CONTINUUM_MEDIAN_WINDOW,
            "excluded_offset_range": None if custom_continuum else [-0.02, 0.02],
        },
        "input_rows": len(table), "selected_candidate_rows": len(tab),
        "input_clusters": len(all_ids), "clusters_with_selected_candidates": len(ids),
        "clusters_without_selected_candidates": int(len(all_ids) - len(ids)),
        "z_bin_edges": list(BCG_Z_BIN_EDGES), "weight_checks": checks,
        "binBoundaries": bins.bin_boundaries.tolist(), "binCent": bins.bin_centers.tolist(),
        "continuum_pdf": continuum.tolist(), "continuum_integral": float(mass.sum()),
        "background_comp_mean_by_offset": comp_background.tolist(),
        "empty_offset_bins": int(np.sum(counts == 0)),
        "nonpositive_stage_counts": {key: int(np.sum(out[key] <= 0)) for key in STAGES},
        "method": "Shared unweighted continuum; individual completeness-weighted signal; "
                  "pooled offset-bin completeness for background; cluster geometry; cluster LF last.",
        "warnings": [
            "Background completeness uses pooled offset-bin means (sample mean in empty bins). "
            "Validate their applicability to each cluster/redshift before scientific interpretation.",
            "No LF cap and no clipping of negative richness or continuum values.",
            "Clusters with no selected candidate galaxies are not assigned a fabricated richness.",
            "Stored old richnesses may differ because the old signal used pooled total weights.",
            "The shared continuum induces covariance not included in plotted SEM.",
        ],
    }
    return out, audit


def prepare_file(input_path=DEFAULT_INPUT, output=DEFAULT_OUTPUT, overwrite=False):
    input_path, output = Path(input_path), Path(output)
    if output.suffix != ".ecsv":
        raise ValueError("Output must end in .ecsv")
    audit_path = output.with_suffix(".json")
    if input_path.resolve() in (output.resolve(), audit_path.resolve()):
        raise ValueError("Never overwrite the input catalog")
    if not overwrite and (output.exists() or audit_path.exists()):
        raise FileExistsError(f"{output} or its audit exists; use --overwrite to regenerate")
    clusters, audit = prepare_stages(read_catalog(input_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    clusters.write(output, format="ascii.ecsv", overwrite=overwrite)
    audit.update(input_path=str(input_path.resolve()), input_sha256=file_sha256(input_path),
                 sample_sha256=file_sha256(output), code_sha256=file_sha256(Path(__file__)))
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(clusters)} clusters to {output}")
    return clusters, audit


def load_stages(path=DEFAULT_OUTPUT):
    path = Path(path)
    audit = json.loads(path.with_suffix(".json").read_text())
    if file_sha256(path) != audit["sample_sha256"]:
        raise ValueError("Weight-stage sample checksum differs from its audit")
    return Table.read(path, format="ascii.ecsv"), audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare_file(args.input, args.output, args.overwrite)
