"""Downsample cluster centers and run Section 4 serially, without mpi4py.

By default all tracer/random tasks are processed for 25 random cluster centers.
Optional task caps are execution smoke tests ONLY: incomplete footprints and
random normalization can bias every derived statistic. Galaxies are not thinned.
Requires the same calibrated input products as the production pipeline.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_overdensity.environment_config import load_config
from local_overdensity.environment_catalogs import load_clusters
from local_overdensity.environment_runner import jobs_for, process, signature
from local_overdensity.finalize_environment import finalize


def select_jobs(jobs, max_data_jobs=0, max_random_jobs=0):
    if max_data_jobs < 0 or max_random_jobs < 0:
        raise ValueError("Task limits must be >=0 (0 means all)")
    selected = []
    for kind, limit in (("data", max_data_jobs), ("random", max_random_jobs)):
        subset = [j for j in jobs if j["kind"] == kind]
        if not subset:
            raise ValueError(f"No {kind} tasks")
        selected.extend(subset[:limit] if limit else subset)
    return selected


def targeted_north_inputs(clusters, c):
    """Complete apertures inside the known-good 130--140, 45--50 degree sweep."""
    from astropy.coordinates import SkyCoord
    from astropy.cosmology import Planck18
    from astropy.io import fits
    t = c["dr9"]
    if t.get("photoz_mode") != "gaussian_fits" or t.get("regions") != ["north", "south"]:
        raise ValueError("Targeted test requires the native north/south DR9 configuration")
    ra = np.asarray(clusters["RA_central"], float)
    dec = np.asarray(clusters["DEC_central"], float)
    z = np.asarray(clusters["Z_SPEC_central"], float)
    theta = 10 / (Planck18.comoving_transverse_distance(z).value * Planck18.h)
    delta_dec = np.rad2deg(theta)
    delta_ra = np.rad2deg(np.arcsin(np.clip(np.sin(theta)/np.cos(np.deg2rad(dec)), -1, 1)))
    b = SkyCoord(ra, dec, unit="deg").galactic.b.deg
    keep = ((ra-delta_ra > 130) & (ra+delta_ra < 140) &
            (dec-delta_dec > 45) & (dec+delta_dec < 50) &
            (dec-delta_dec >= 32.375) & (b > delta_dec))
    eligible = np.flatnonzero(keep)
    if not len(eligible):
        raise ValueError("No clusters have a complete 10 h^-1 Mpc aperture inside the test sweep")
    path = Path(t["sweep_dirs"][0]) / "sweep-130p045-140p050.fits"
    photoz = Path(t["photoz_dirs"][0]) / "sweep-130p045-140p050-pz.fits"
    random = Path(t["random_globs"][0]).parent / "randoms-north-1-0.fits"
    for file in (path, photoz, random):
        if not file.is_file():
            raise FileNotFoundError(file)
    jobs = [dict(kind="data", path=str(path), photoz=str(photoz), region="north")]
    with fits.open(random, memmap=True) as hdul:
        n = hdul[1].header["NAXIS2"]
    if n == 0:
        raise ValueError("Targeted random realization is empty")
    for start in range(0, n, c["chunk_rows"]):
        jobs.append(dict(kind="random", path=str(random), start=start,
                         stop=min(start+c["chunk_rows"], n), region="north",
                         density=t["random_density_per_realization"]))
    return eligible, jobs


def smoke_run(config, tracer, output, max_clusters=25, seed=42,
              max_data_jobs=0, max_random_jobs=0, dry_run=False, targeted_north=False):
    # Reject a multi-process launch, but permit a plain Python process inside
    # an interactive Slurm allocation that happens to reserve multiple CPUs.
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE"):
        if int(os.environ.get(key, "1")) > 1:
            raise ValueError("Run once with python, not an MPI launcher")
    if max_clusters < 1:
        raise ValueError("max_clusters must be positive")
    c = load_config(config, tracer)
    clusters = load_clusters(c["cluster_catalog"])
    available_clusters = len(clusters)
    if targeted_north:
        if tracer != "dr9" or max_data_jobs or max_random_jobs:
            raise ValueError("--targeted-north requires dr9 and no task limits")
        eligible, all_jobs = targeted_north_inputs(clusters, c)
    else:
        eligible = np.arange(len(clusters))
        all_jobs = jobs_for(c, tracer)
    indices = np.sort(np.random.default_rng(seed).choice(
        eligible, size=min(max_clusters, len(eligible)), replace=False))
    clusters = clusters[indices]
    jobs = select_jobs(all_jobs, max_data_jobs, max_random_jobs)
    partial = len(jobs) != len(all_jobs)
    output = Path(output).resolve()
    production = Path(c["output_dir"]).resolve()
    if output == production or production in output.parents or output in production.parents:
        raise ValueError("Smoke output must be separate from the production output tree")
    c["output_dir"] = str(output)
    c["serial_test"] = dict(seed=seed, cluster_indices=indices.tolist(),
                            max_data_jobs=max_data_jobs, max_random_jobs=max_random_jobs,
                            targeted_north=targeted_north)
    sig, manifest = signature(c, jobs, tracer)
    report = dict(signature=sig, tracer=tracer, seed=seed,
                  clusters_available=available_clusters, clusters_selected=len(clusters),
                  cluster_ids=[str(x) for x in clusters["ID"]],
                  available_tasks=len(all_jobs), selected_tasks=len(jobs),
                  partial_inputs=partial, dry_run=dry_run)
    report.update(targeted_north=targeted_north, clusters_eligible=len(eligible))
    print(json.dumps(report, indent=2), flush=True)
    if partial:
        print("WARNING: PARTIAL INPUTS. Counts, coverage and densities are NOT "
              "scientific measurements; zeros may reflect unread files.", flush=True)
    if dry_run:
        return report
    # Atomic directory creation prevents accidental replacement or concurrent use.
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(
        {**manifest, "test_selection": report}, indent=2) + "\n")
    total = np.zeros((len(clusters), 3, 5))
    audits = []
    started = time.monotonic()
    try:
        for i, job in enumerate(jobs):
            task_start = time.monotonic()
            stats, audit = process(job, clusters, c, tracer)
            total += stats
            audit.update(kind=job["kind"], path=job["path"],
                         elapsed_seconds=time.monotonic() - task_start)
            if "start" in job:
                audit.update(start=job["start"], stop=job["stop"])
            audits.append(audit)
            print(f"Task {i+1}/{len(jobs)} {job['kind']}: "
                  f"{audit['input_rows']} rows, {audit['selected_rows']} selected, "
                  f"{audit['elapsed_seconds']:.2f}s", flush=True)
        result = finalize(clusters, total, c, tracer)
        result.meta["SMOKETST"] = True
        result.meta["PARTIAL"] = partial
        result.write(output / "smoke_environment.fits")
        np.savez_compressed(output / "smoke_accumulators.npz", stats=total)
        selected_rows = sum(a["selected_rows"] for a in audits if a["kind"] == "data")
        warnings = ["Serial validation sample, not a production result."]
        if partial:
            warnings.append("Input tasks truncated; no normalization or footprint completeness guaranteed.")
        if selected_rows == 0:
            warnings.append("No selected data rows: inspect cuts, input selection and task limits.")
        if not np.any(total[:, :, 0]):
            warnings.append("No galaxies near selected clusters; test may lack sky/redshift overlap.")
        report.update(status="completed", elapsed_seconds=time.monotonic()-started,
                      tasks=audits, selected_data_rows=selected_rows, warnings=warnings)
        (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"Saved {output / 'smoke_environment.fits'}", flush=True)
        for warning in warnings:
            print("WARNING:", warning)
        return report
    except Exception as error:
        (output / "failure.json").write_text(json.dumps(
            dict(error=str(error), completed_tasks=len(audits),
                 elapsed_seconds=time.monotonic()-started), indent=2) + "\n")
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--tracer", choices=("dr9", "bgs"), required=True)
    p.add_argument("--max-clusters", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-data-jobs", type=int, default=0, help="0=all; DR9 files or BGS chunks")
    p.add_argument("--max-random-jobs", type=int, default=0, help="0=all random chunks")
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--targeted-north", action="store_true",
                   help="Use known-good north sweep, interior clusters and one complete random realization")
    a = p.parse_args()
    output = a.output or ROOT / "local_overdensity/dr9_outputs/section4_smoke" / (
        a.tracer + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f"))
    smoke_run(a.config, a.tracer, output, a.max_clusters, a.seed,
              a.max_data_jobs, a.max_random_jobs, a.dry_run, a.targeted_north)


if __name__ == "__main__":
    main()
