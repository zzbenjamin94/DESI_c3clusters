"""Targeted DR9 MPI validation; complete apertures and one full random realization.

Uses the production process/finalize functions and checkpoint aggregation pattern.
Optional serial replay compares all additive statistics for exactly the same inputs.
No production outputs are modified; each run requires a new output directory.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_overdensity.environment_catalogs import load_clusters
from local_overdensity.environment_config import load_config
from local_overdensity.environment_runner import process, signature
from local_overdensity.finalize_environment import finalize
from local_overdensity.run_environment_serial_test import targeted_north_inputs


def compare_statistics(total, reference):
    np.testing.assert_allclose(total, reference, rtol=1e-10, atol=1e-10)
    return float(np.max(np.abs(total - reference)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-clusters', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compare-serial', action='store_true')
    args = p.parse_args()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    created = False
    started = time.monotonic()
    try:
        if size < 2 or args.max_clusters < 1:
            raise ValueError('Use at least two MPI ranks and a positive cluster limit')
        # Verify communication before reading catalogs.
        ranks = comm.allgather(rank)
        if ranks != list(range(size)):
            raise RuntimeError('MPI communication check failed')
        if rank == 0:
            print(f'MPI communication OK: ranks={ranks}', flush=True)
            c = load_config(args.config, 'dr9')
            output = args.output.resolve()
            production = Path(c['output_dir']).resolve()
            if output == production or production in output.parents or output in production.parents:
                raise ValueError('Test output must be separate from production')
            clusters = load_clusters(c['cluster_catalog'])
            eligible, jobs = targeted_north_inputs(clusters, c)
            indices = np.sort(np.random.default_rng(args.seed).choice(
                eligible, size=min(args.max_clusters, len(eligible)), replace=False))
            clusters = clusters[indices]
            c['output_dir'] = str(output)
            c['mpi_test'] = dict(seed=args.seed, cluster_indices=indices.tolist())
            sig, manifest = signature(c, jobs, 'dr9')
            output.mkdir(parents=True, exist_ok=False)
            created = True
            (output / 'checkpoints').mkdir()
            report = dict(signature=sig, ranks=size, clusters_selected=len(clusters),
                          cluster_ids=[str(x) for x in clusters['ID']], tasks_expected=len(jobs),
                          targeted_north=True, partial_inputs=False,
                          warning='Small validation sample, not a production measurement.')
            (output / 'manifest.json').write_text(json.dumps(
                {**manifest, 'test_selection': report}, indent=2) + '\n')
            print(json.dumps(report, indent=2), flush=True)
            payload = c, clusters, jobs, sig, output
        else:
            payload = None
        c, clusters, jobs, sig, output = comm.bcast(payload, root=0)
        for index in range(rank, len(jobs), size):
            task_start = time.monotonic()
            stats, audit = process(jobs[index], clusters, c, 'dr9')
            audit.update(index=index, rank=rank, kind=jobs[index]['kind'],
                         path=jobs[index]['path'], elapsed_seconds=time.monotonic()-task_start)
            checkpoint = output / 'checkpoints' / f'{index:07d}.npz'
            temporary = checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as f:
                np.savez_compressed(f, stats=stats, signature=sig, audit=json.dumps(audit))
            temporary.replace(checkpoint)
            print(f'rank {rank}: completed {index+1}/{len(jobs)} in '
                  f'{audit["elapsed_seconds"]:.2f}s', flush=True)
        comm.Barrier()
        if rank == 0:
            mpi_seconds = time.monotonic() - started
            total = np.zeros((len(clusters), 3, 5))
            audits = []
            for index in range(len(jobs)):
                with np.load(output / 'checkpoints' / f'{index:07d}.npz', allow_pickle=False) as f:
                    if str(f['signature']) != sig or f['stats'].shape != total.shape:
                        raise ValueError('Checkpoint mismatch')
                    total += f['stats']
                    audits.append(json.loads(str(f['audit'])))
            if not np.all(np.isfinite(total)) or not np.any(total[:, :, 0]):
                raise ValueError('Nonfinite statistics or no nearby galaxies')
            if np.any(total[:, :, 3] <= 0):
                raise ValueError('Some test apertures have no effective area')
            result = finalize(clusters, total, c, 'dr9')
            result.meta['SMOKETST'] = True
            result.write(output / 'smoke_environment.fits')
            np.savez_compressed(output / 'smoke_accumulators.npz', stats=total)
            comparison = None
            if args.compare_serial:
                print('Replaying identical inputs serially for comparison', flush=True)
                reference = np.zeros_like(total)
                for job in jobs:
                    stats, _ = process(job, clusters, c, 'dr9')
                    reference += stats
                comparison = dict(passed=True, max_absolute_difference=compare_statistics(total, reference))
            report.update(status='completed', completed_tasks=len(audits), tasks=audits,
                          mpi_elapsed_seconds=mpi_seconds, elapsed_seconds=time.monotonic()-started,
                          serial_comparison=comparison,
                          box_valid=int(np.sum(result['dr9_box_valid'])),
                          gaussian_valid=int(np.sum(result['dr9_gaussian_valid'])))
            (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
            print(f'PASS: {len(jobs)} tasks; serial comparison={comparison}; output={output}', flush=True)
        comm.Barrier()
    except Exception as error:
        traceback.print_exc()
        if rank == 0 and created:
            (output / 'failure.json').write_text(json.dumps(dict(error=str(error)), indent=2))
        comm.Abort(1)
        raise


if __name__ == '__main__':
    main()
