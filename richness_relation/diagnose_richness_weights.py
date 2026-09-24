"""Read-only weight diagnostics; no weight repair or scientific-quality cuts."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from tools.richness_selection import richness_analysis_mask


def numeric(table, name):
    return np.ma.asarray(table[name], dtype=float).filled(np.nan)


def categories(values):
    return {"zero": int(np.sum(values == 0)),
            "negative_finite": int(np.sum(np.isfinite(values) & (values < 0))),
            "nan_or_masked": int(np.isnan(values).sum()),
            "infinite": int(np.isinf(values).sum()),
            "invalid_total": int(np.sum(~np.isfinite(values) | (values <= 0)))}


def diagnose_weights(table):
    """Return JSON summary and affected selected rows/clusters without mutating input."""
    selected = np.asarray(richness_analysis_mask(table), dtype=bool)
    ids = np.asarray(table["ID"]).astype(str)
    flags, weights = {}, {}
    for key in ("COMP_WEIGHT", "GEOMETRIC_WEIGHT", "LF_WEIGHT"):
        values = numeric(table, key)
        flags["INVALID_" + key] = ~np.isfinite(values) | (values <= 0)
        weights[key] = {"all_rows": categories(values), "selected_rows": categories(values[selected])}
    bad_comp = flags["INVALID_COMP_WEIGHT"]
    probability = {"available": "PROB_OBS" in table.colnames}
    if probability["available"]:
        prob = numeric(table, "PROB_OBS")
        valid = np.isfinite(prob) & (prob > 0) & (prob <= 1)
        flags["INVALID_PROB_OBS"] = ~valid
        expected = np.full(len(table), np.nan)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            np.divide(1.0, prob, out=expected, where=valid)
        flags["COMP_PROB_MISMATCH"] = valid & ~bad_comp & ~np.isclose(
            numeric(table, "COMP_WEIGHT"), expected, rtol=1e-6, atol=1e-10)
        probability.update(
            invalid_selected=int(np.sum(selected & ~valid)),
            invalid_comp_with_valid_probability=int(np.sum(selected & bad_comp & valid)),
            invalid_comp_with_invalid_probability=int(np.sum(selected & bad_comp & ~valid)),
            valid_weight_probability_mismatches=int(np.sum(selected & flags["COMP_PROB_MISMATCH"])),
        )
    bad = selected & np.logical_or.reduce(list(flags.values()))
    quality_names = [name for name in table.colnames
                     if any(token in name.upper() for token in
                            ("ZWARN", "DELTACHI2", "SPECTYPE", "ZERR", "GOODZ", "QSO", "COADD_FIBERSTATUS"))
                     and np.asarray(table[name]).ndim == 1]
    requested = ["ID", "TARGETID", "Z_SPEC_central", "Z_BGS", "LAMBDA", "PROB_OBS",
                 "COMP_WEIGHT", "GEOMETRIC_FRACTION", "GEOMETRIC_WEIGHT", "LF_WEIGHT", "TOTAL_WEIGHT"]
    columns = list(dict.fromkeys([k for k in requested if k in table.colnames] + quality_names))
    rows = table[bad][columns].copy()
    rows.meta.clear()
    rows["INPUT_ROW_INDEX"] = np.flatnonzero(bad)
    for key, flag in flags.items():
        rows[key] = flag[bad]
    cluster_ids, inverse = np.unique(ids[selected], return_inverse=True)
    total = np.bincount(inverse, minlength=len(cluster_ids))
    affected = np.bincount(inverse, weights=bad[selected], minlength=len(cluster_ids)).astype(int)
    clusters = Table({"ID": cluster_ids, "N_selected_rows": total, "N_flagged_rows": affected})
    for key, flag in flags.items():
        clusters["N_" + key] = np.bincount(inverse, weights=flag[selected], minlength=len(cluster_ids)).astype(int)
    clusters = clusters[affected > 0]
    quality = {}
    for key in quality_names:
        quality[key] = {}
        for label, mask in (("all_selected", selected), ("flagged_selected", bad)):
            vals, counts = np.unique(np.ma.asarray(table[key])[mask].astype(str).filled("MASKED"), return_counts=True)
            order = np.argsort(-counts)[:20]
            quality[key][label] = {"distinct_values": len(vals),
                                   "top_values": {str(vals[i]): int(counts[i]) for i in order}}
    summary = {
        "input_rows": len(table), "selected_rows": int(selected.sum()),
        "selected_clusters": len(cluster_ids), "flagged_selected_rows": int(bad.sum()),
        "affected_clusters": len(clusters), "weights": weights, "probability": probability,
        "quality_columns_available": quality_names, "quality_value_counts": quality,
        "selection": "BCG z [0.1,0.35), BGS z [0.05,0.4), normalized offset [-0.05,0.05]",
        "note": "Counts are matched-catalog rows, not unique galaxies. A galaxy may appear in several clusters. "
                "Valid probability alone does not establish trustworthy redshift. No rows or weights changed. "
                "Probability must be finite and in (0,1]. Quality flags are reported, not interpreted as cuts.",
    }
    return summary, rows, clusters


def write_report(summary, rows, clusters, directory):
    directory = Path(directory) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    rows.write(directory / "flagged_rows.ecsv", format="ascii.ecsv")
    clusters.write(directory / "affected_clusters.ecsv", format="ascii.ecsv")
    print(f"Weight diagnostics: {summary['flagged_selected_rows']} flagged selected rows; "
          f"{summary['affected_clusters']} affected clusters.")
    for key, counts in summary["weights"].items():
        print(f"  {key}: {counts['selected_rows']}")
    print("  PROB_OBS:", summary["probability"])
    print(f"Diagnostic report saved to {directory.resolve()}")
    return directory
