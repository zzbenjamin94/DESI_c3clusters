"""Restartable serial/MPI Section 4 engine. One checkpoint per independent task."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback
import numpy as np
from astropy.io import fits
from astropy.table import Table
from local_overdensity.environment_config import load_config
from local_overdensity.environment_catalogs import (
    discover, fits_chunks, load_clusters, load_photoz, photo_indices, photo_selection, curve,
    imaging_selection,
)
from local_overdensity.environment_accumulators import Accumulator, select_bgs
from local_overdensity.finalize_environment import finalize


def jobs_for(c, tracer):
    jobs = []
    t = c[tracer]
    native = tracer == "dr9" and t.get("photoz_mode") == "gaussian_fits"
    if tracer == "dr9":
        for region_index, (directory, pdf_dir) in enumerate(zip(t["sweep_dirs"], t["photoz_dirs"])):
            for path in discover([str(Path(directory) / "sweep-*.fits")]):
                if native:
                    jobs.append({"kind": "data", "path": path,
                                 "region": t["regions"][region_index],
                                 "photoz": str(Path(pdf_dir) / (Path(path).stem + "-pz.fits"))})
                    continue
                jobs.append({"kind": "data", "path": path,
                             "pdf": str(Path(pdf_dir) / (Path(path).stem + ".npz"))})
    else:
        for path in discover(t["data_globs"]):
            with fits.open(path, memmap=True) as f:
                n = len(f[1].data)
            for start in range(0, n, c["chunk_rows"]):
                jobs.append({"kind": "data", "path": path, "start": start,
                             "stop": min(start+c["chunk_rows"], n)})
    random_info = {}
    if native:
        import re
        for region, pattern in zip(t["regions"], t["random_globs"]):
            paths = discover([pattern])
            if any(re.fullmatch(f"randoms-{region}-1-[0-9]+.fits", Path(p).name) is None for p in paths):
                raise ValueError("Only documented DR9 seed-1 complete realizations are supported")
            for p in paths:
                random_info[p] = (region, t["random_density_per_realization"] * len(paths))
    for path in discover(t["random_globs"]):
        with fits.open(path, memmap=True) as f:
            n = len(f[1].data)
        for start in range(0, n, c["chunk_rows"]):
            jobs.append({"kind": "random", "path": path, "start": start,
                         "stop": min(start+c["chunk_rows"], n)})
            if native:
                jobs[-1].update(region=random_info[path][0], density=random_info[path][1])
    if not jobs:
        raise ValueError("Empty input manifest")
    return jobs


def signature(c, jobs, tracer):
    paths = {str(Path(c["cluster_catalog"]).resolve())}
    if c[tracer].get("lf_curve"):
        paths.add(c[tracer]["lf_curve"])
    if tracer == "bgs":
        paths.add(c[tracer]["nbar_curve"])
    paths.update(j["path"] for j in jobs)
    paths.update(j["pdf"] for j in jobs if "pdf" in j)
    paths.update(j["photoz"] for j in jobs if "photoz" in j)
    inputs = {str(Path(p).resolve()): [Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
              for p in sorted(paths)}
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("environment_*.py")}
    p = Path(__file__).with_name("finalize_environment.py")
    code[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    p = Path(__file__).with_name("dr9_photometry.py")
    code[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {"configuration": c, "tracer": tracer, "inputs_size_mtime": inputs,
                "jobs": jobs, "code_sha256": code}
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(), manifest


def process(job, clusters, c, tracer):
    accumulator = Accumulator(clusters, c, tracer)
    audit = {"input_rows": 0, "selected_rows": 0}
    native = tracer == "dr9" and c["dr9"].get("photoz_mode") == "gaussian_fits"
    photo, lookup, lf = None, None, None
    if "pdf" in job:
        photo, lookup = load_photoz(job["pdf"])
        lf = curve(c["dr9"]["lf_curve"], photo["z_grid"], "LF_WEIGHT")

    def chunks():
        if "photoz" in job:
            from local_overdensity.environment_dr9_native import paired_chunks
            yield from paired_chunks(job, c["chunk_rows"])
            return
        if "start" in job:
            with fits.open(job["path"], memmap=True) as f:
                yield Table(f[1].data[job["start"]:job["stop"]], copy=True)
        else:
            for _, table in fits_chunks(job["path"], c["chunk_rows"]):
                yield table

    for table in chunks():
        audit["input_rows"] += len(table)
        if native:
            from local_overdensity.environment_dr9_native import ownership, coverage
            keep = ownership(table, job["region"]) & coverage(table)
            audit["outside_ownership_or_coverage"] = audit.get("outside_ownership_or_coverage", 0) + int((~keep).sum())
            table = table[keep]
        if job["kind"] == "random":
            if tracer == "dr9":
                table = table[imaging_selection(table, c["dr9"])]
                if native:
                    table[c["dr9"]["random_measure_column"]] = np.full(len(table), 1/job["density"])
            if tracer == "bgs":
                z = np.asarray(table["Z"], float)
                if np.any(~np.isfinite(z)) or np.any(z <= 0):
                    raise ValueError("Spectroscopic randoms need finite positive Z")
            accumulator.add(table, random=True)
            audit["selected_rows"] += len(table)
        elif tracer == "dr9":
            table = table[photo_selection(table, c["dr9"])]
            if native:
                mean, sigma = np.asarray(table["Z_PHOT_MEAN"], float), np.asarray(table["Z_PHOT_STD"], float)
                valid = np.isfinite(mean) & (mean >= 0) & np.isfinite(sigma) & (sigma > 0)
                audit["photoz_eligible_rows"] = audit.get("photoz_eligible_rows", 0) + len(table)
                audit["invalid_photoz_rows"] = audit.get("invalid_photoz_rows", 0) + int((~valid).sum())
                table = table[valid]
                table["ENV_WEIGHT"] = np.ones(len(table))
                accumulator.add(table)
                audit["selected_rows"] += len(table)
                continue
            idx = photo_indices(table, lookup)
            primary = photo["PRIMARY"][idx]
            table, idx = table[primary], idx[primary]
            table["ENV_WEIGHT"] = photo["SYS_WEIGHT"][idx]
            if np.any(~np.isfinite(table["ENV_WEIGHT"])) or np.any(table["ENV_WEIGHT"] <= 0):
                raise ValueError("Invalid photometric systematics weight")
            accumulator.add(table, pdf=photo["pdf"][idx], z_grid=photo["z_grid"], lf_pdf=lf)
            audit["selected_rows"] += len(table)
        else:
            # Discard only objects beyond every cylinder before LF interpolation.
            z = np.asarray(table[c["bgs"].get("redshift_column", "Z")], float)
            from astropy.cosmology import Planck18
            finite = np.isfinite(z) & (z > 0)
            distance = np.full(len(z), np.inf)
            distance[finite] = Planck18.comoving_distance(z[finite]).value * Planck18.h
            reach = c["bgs"]["pi_half_hmpc"]
            keep = (distance >= accumulator.chi.min()-reach) & (distance <= accumulator.chi.max()+reach)
            selected = select_bgs(table[keep], c)
            accumulator.add(selected)
            audit["selected_rows"] += len(selected)
    return accumulator.stats, audit


def run(tracer):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    comm, rank, size = None, 0, 1
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, size = comm.rank, comm.size
    elif int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        raise ValueError("Use --mpi with multiple Slurm ranks")
    lock = None
    try:
        c = load_config(args.config, tracer)
        clusters = load_clusters(c["cluster_catalog"])
        jobs = jobs_for(c, tracer)
        sig, manifest = signature(c, jobs, tracer)
        if args.dry_run:
            if rank == 0:
                print(json.dumps({"clusters": len(clusters), "tasks": len(jobs),
                                  "signature": sig, "configuration": c}, indent=2))
            return
        output = Path(c["output_dir"]) / tracer
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            candidate = output / "RUNNING.lock"
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            lock = candidate
            manifest_path = output / "manifest.json"
            if manifest_path.exists():
                old = json.loads(manifest_path.read_text())
                if not args.resume or old["signature"] != sig:
                    raise ValueError("Existing output differs or --resume missing; choose a new output_dir")
            manifest_path.write_text(json.dumps({"signature": sig, **manifest}, indent=2))
            (output / "checkpoints").mkdir(exist_ok=True)
        if comm:
            comm.Barrier()
        for index in range(rank, len(jobs), size):
            checkpoint = output / "checkpoints" / f"{index:07d}.npz"
            if args.resume and checkpoint.exists():
                continue
            stats, audit = process(jobs[index], clusters, c, tracer)
            temporary = checkpoint.with_suffix(".tmp")
            with temporary.open("wb") as f:
                np.savez_compressed(f, stats=stats, signature=sig, audit=json.dumps(audit))
            os.replace(temporary, checkpoint)
            print(f"rank {rank}: completed task {index+1}/{len(jobs)} {jobs[index]['path']}", flush=True)
        if comm:
            comm.Barrier()
        if rank == 0:
            total = np.zeros((len(clusters), 3, 5))
            audits = []
            for index in range(len(jobs)):
                with np.load(output / "checkpoints" / f"{index:07d}.npz", allow_pickle=False) as f:
                    if str(f["signature"]) != sig or f["stats"].shape != total.shape:
                        raise ValueError("Checkpoint mismatch")
                    total += f["stats"]
                    audits.append(json.loads(str(f["audit"])))
            selected_data = sum(a["selected_rows"] for j, a in zip(jobs, audits) if j["kind"] == "data")
            if selected_data == 0:
                raise ValueError("No eligible tracer rows across all inputs; refusing to finalize empty data")
            result = finalize(clusters, total, c, tracer)
            result.write(output / "environment.tmp.fits", overwrite=True)
            os.replace(output / "environment.tmp.fits", output / "environment.fits")
            report = {"signature": sig, "completed_tasks": len(jobs), "tasks": audits,
                      "selected_data_rows": selected_data,
                      "density_units": result.meta["DENSUNIT"],
                      "warnings": ["Random measures require independent footprint/selection calibration.",
                                   "Poisson errors exclude clustering, random noise and weight uncertainty.",
                                   "PRIMARY ownership must resolve DR9 north/south overlap."]}
            temporary = output / "summary.tmp.json"
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
            os.replace(temporary, output / "summary.json")
            print(f"Saved {output / 'environment.fits'}", flush=True)
    except Exception:
        traceback.print_exc()
        if comm:
            comm.Abort(1)
        raise
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)
