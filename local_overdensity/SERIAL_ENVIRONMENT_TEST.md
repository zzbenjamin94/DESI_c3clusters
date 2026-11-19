# Serial Environment Smoke Test

Run one Python process, without srun, mpiexec or --mpi. This reuses the production
Section 4 filters, selections, weights, coverage calculation and finalizer.

## Inputs must be prepared first

Use a completed environment configuration (see ENVIRONMENT_FILTERS.md and
environment_config.example.json). The example deliberately contains unset fields;
the test does not invent Gaussian width, cylinder depth or photometric PDFs.
DR9 needs keyed photo-z PDF sidecars, an LF curve, explicit cuts and calibrated
angular random measures. BGS needs explicit quality cuts, cylinder half-depth,
LF and nbar curves and calibrated volume random measures. Raw survey random
files alone do not satisfy these contracts.

## Fast execution test

From the NERSC Projection repository, with the scientific Python environment active:

```bash
python local_overdensity/run_environment_serial_test.py \
  --config local_overdensity/environment_config.json \
  --tracer dr9 --max-clusters 25 \
  --max-data-jobs 2 --max-random-jobs 2 --dry-run
```

Remove --dry-run to execute. Use --tracer bgs for the DR2 spectroscopic test.
A DR9 data job is one sweep; a BGS data job or random job is one configured
chunk. Thus this limits execution units, not necessarily files. DR9 PDF loading
still loads an entire sidecar, so a small test is not a strict RAM limit.

The capped run is only a plumbing/timing test. File order can give little overlap
with the sampled clusters; zeros are not evidence of physical emptiness or bad
coverage. Derived densities and coverage from truncated random/data tasks are
not unbiased estimates. No inverse sampling factor is applied.

## Meaningful small-cluster validation

```bash
python local_overdensity/run_environment_serial_test.py \
  --config local_overdensity/environment_config.json \
  --tracer bgs --max-clusters 25
```

This randomly selects up to 25 unique science-range cluster centers, but reads
ALL configured data and random tasks. No tracer galaxies are downsampled.
Use this mode for comparing selected clusters against production results.
It reduces cluster work, not full-catalog I/O. The seed defaults to 42.

## Output and safeguards

A fresh timestamped folder under local_overdensity/dr9_outputs/section4_smoke/
contains smoke_environment.fits, raw smoke_accumulators.npz, manifest.json and
summary.json (per-task row counts and wall times). A failed run writes
failure.json; restart into a fresh directory. There is no resume or overwrite.
--output may specify another new directory outside the production output tree.

FITS flags SMOKETST and PARTIAL identify test products. A manifest records exact
selected IDs and input tasks. A completed test is not a scientific validation:
inspect finite coverage/densities and signal/background counts, then repeat
without task caps before production. Cluster catalog deduplication still reads
the parent catalog; prepare a cluster-level input if that is the RAM bottleneck.
