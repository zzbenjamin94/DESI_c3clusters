# DR9 native FITS test

The example JSON is runnable for DR9 (BGS still needs its own configuration).
It reads row-matched 9.0-photo-z FITS sidecars in chunks, checks all three
object keys, and uses a Gaussian conditional on z >= 0, integrated over
cluster z +/- 0.05. There is no DR9 LF correction.

Galaxies use non-PSF morphology, positive GRZ flux inverse variance, and
extinction-corrected r < 20.5. Both catalogs require NOBS_G/R/Z >= 1,
the standard DR9 north/south split, and MASKBITS veto 12290.
Galaxy and random NOBS definitions are not identical in DR9; this baseline
requires a coverage-systematics check before publication. Invalid photo-z
fractions are recorded separately; randoms do not correct photo-z failures.

Random files must be complete documented seed-1 realizations. Each regional
file contributes 2500 points/deg^2. Area per accepted point is
1 / (2500 * number_of_files_in_that_region). Duplicate copies, pre-thinned
files, or spatial partitions are not supported by that normalization.
The input selection and file list are preserved in the run manifest.

Signal radii are 1.5--5 and background radii 5--10 comoving h^-1 Mpc.
The Gaussian width is sqrt((5^2-1.5^2)/(2 ln(4))).
The accumulator converts angular area to comoving area: density/excess
columns have units h^2 Mpc^-2, while E_local is dimensionless.

## NERSC serial execution

### Recommended targeted test

Use --targeted-north WITHOUT task limits. It selects clusters with their
entire 10 h^-1 Mpc spherical aperture inside RA 130--140, Dec 45--50,
and inside the standard northern ownership region. It reads that one
sweep and its photo-z file, plus every chunk of randoms-north-1-0.fits.
Random density is 2500 deg^-2, not the density of all discovered files.
The actual number of clusters may be below --max-clusters; zero eligible
clusters is an explicit error. No southern files are needed for this test.
Rectangle containment does not guarantee unmasked coverage: inspect the
measured coverage and valid flags. The test remains a validation sample.

```bash
time srun -n 1 -c 1 "$CONDA_PREFIX/bin/python" \
  local_overdensity/run_environment_serial_test.py \
  --config local_overdensity/environment_config.example.json \
  --tracer dr9 --targeted-north --max-clusters 25 --seed 42
```

Add --dry-run to inspect selected cluster IDs and task counts before running.

### Untargeted execution-only test

From /global/homes/z/zzhang13/DESI/Projection:

```bash
source ./setup_nersc_mpi_myEnv_v39.sh
srun -n 1 -c 1 python local_overdensity/run_environment_serial_test.py \
  --config local_overdensity/environment_config.example.json \
  --tracer dr9 --max-clusters 25 \
  --max-data-jobs 2 --max-random-jobs 2 --seed 42
```

This checks execution only: limited files can miss every selected cluster.
Counts and areas are deliberately not renormalized for missing jobs.
For scientifically interpretable environments for the same small cluster
sample, remove both job limits (all sweeps/randoms are then read).
Time the limited test first; a full serial scan can exceed an hour.
Each run creates a new timestamped directory in
local_overdensity/dr9_outputs/section4_smoke with smoke_environment.fits,
smoke_accumulators.npz, manifest.json and summary.json.
Read per-task audits and validity flags before interpreting zero counts.
