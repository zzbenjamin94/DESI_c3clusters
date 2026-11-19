# DR9 MPI startup and measurement test

Run from the NERSC login node in `/global/homes/z/zzhang13/DESI/Projection`:

```bash
git pull --ff-only origin richness-trend-analysis
sbatch local_overdensity/run_environment_mpi_test.slurm
```

This requests one CPU node, four MPI ranks and a hard 15-minute time limit.
It tests MPI communication, then uses up to six clusters with complete 10 Mpc/h
apertures inside sweep-130p045-140p050, its photo-z sidecar, and the complete
randoms-north-1-0 realization. No random chunks are truncated; its density is
2500/deg^2, not the combined density of all configured realizations.

The workers call production `process()` and write per-task checkpoints. Rank 0
combines them, calls production `finalize()`, and replays the same tasks serially
to check every accumulator element (rtol=atol=1e-10). This validates task splitting
and accumulation, not the scientific correctness of shared processing functions.
It is a small northern sample, not a full-footprint or scaling benchmark.

```bash
sacct -j JOBID --format=JobID,State,Elapsed,Timelimit,ExitCode
tail -n 100 environment-mpi-test-JOBID.log
cat local_overdensity/dr9_outputs/section4_mpi_test/job_JOBID/summary.json
```

Require COMPLETED/0:0, the log's PASS line, all expected tasks completed, and
`serial_comparison.passed: true`. Inspect the FITS counts, coverage and valid flags
before scaling up. No summary/timeout is not a pass. The job cannot exceed 15
minutes, but completion within that budget depends on filesystem performance.
Outputs include smoke_environment.fits, smoke_accumulators.npz, manifests and
checkpoints in a new job-specific directory. Existing directories are refused;
production outputs and checkpoints are untouched.

Production run_environment.slurm now traces startup, sources the setup script
by absolute path and checks Python/MPI imports before launching. Its existing
32 ranks, 14-hour budget, config and resume settings are unchanged. Do not submit
production again until the test and the original failure log have been reviewed.
