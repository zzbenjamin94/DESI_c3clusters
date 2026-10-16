# Nine independent richness/redshift cells

This is a NEW analysis, not global posterior re-evaluation. It fits
P(lambda_RM | lambda_spec in bin, z in bin, RM selection). The prepared sample
is read-only. No weights, continuum fits or cluster richnesses are recomputed.

## Model and binning

Use full-sample lambda_spec terciles, shared across z=[.10,.18), [.18,.24),
[.24,.35). The top richness bin includes its upper edge; other bins exclude it.
--richness-edges accepts manually chosen common edges spanning the sample
(four edges for three bins). Terciles balance full-sample statistical power,
without choosing boundaries based on the observed lambda_RM outcome. Common
boundaries make redshift comparisons meaningful; individual cells need not have
equal counts. Broad bins mix underlying trends into the fitted scatter.
These are spectroscopic-richness bins, because the fitted response is lambda_RM.
The dry run prints the exact counts for YOUR input sample.

Each cell samples only:
- Log-normal: ln(mu_RM), ln(sigma0), where sigma0 is scatter in natural log.
- Mixture: ln(mu_RM), ln(sigma_lambda), f_proj, ln(tau), with tau an exponential
  RATE. The tail is a Gaussian convolved with positive exponential projection.

Both use the original selection-normalized density in linear lambda_RM,
including the lower RM threshold and any configured upper threshold. Priors are
uniform in the stated sampled coordinates with the same location/shape bounds
as the global models. Bounds are saved in bestfit.json and require sensitivity
checks, especially for cells piled up at the selection limit.
There is NO B or C parameter inside a cell. The nine local locations are free,
regardless of the global baseline decision. Scatter includes finite-bin mixing
and measurement noise; it is not deconvolved intrinsic scatter. The fitted cell
PDF is not a global-model quantile residual distribution.

## Preflight on NERSC

```bash
cd /global/homes/z/zzhang13/DESI/Projection
git pull --ff-only origin richness-trend-analysis
source ./setup_nersc_mpi_myEnv_v39.sh
python richness_relation/fit_richness_cells_mpi.py \
  --output catalogs/conditional_richness/cell_fits_3x3 --dry-run
```

## Overnight batch (recommended when leaving the terminal)

```bash
sbatch richness_relation/run_richness_cells.slurm
```

One node, 18 ranks, one independent fit per rank, 48 walkers and two independent
ensembles per fit, 5000 steps with 1000 burn-in. Four hours is a requested budget,
NOT a benchmarked runtime or convergence guarantee. On successful sampling,
the job saves quick plots/tables, then runs 200 refitted bootstrap simulations
per fit using 18 local workers to populate p-values in the final tables.
Check the job exit and logs; the bootstrap also consumes the four-hour budget.
The new default cell_fits_3x3 directory avoids overwriting older 12-cell runs.
Changing CELL_OUTPUT via sbatch --export=ALL,CELL_OUTPUT=... chooses a fresh root.

## Interactive alternative

```bash
salloc --account=desi --constraint=cpu --qos=interactive \
  --nodes=1 --ntasks=18 --cpus-per-task=1 --time=01:00:00
cd /global/homes/z/zzhang13/DESI/Projection
source ./setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --export=ALL -n 18 -c 1 "$CONDA_PREFIX/bin/python" -u \
  richness_relation/fit_richness_cells_mpi.py --mpi \
  --output catalogs/conditional_richness/cell_fits_3x3 \
  --steps 5000 --burn 1000 --walkers 48 --ensembles 2 --resume
```

For a benchmark use --steps 200 --burn 50 and a DIFFERENT output root such as
cell_pilot. Do not compare short pilot posteriors as science results. Burn-in,
walkers, ensemble count, sample and code hashes must match on resume. Increase
--steps (e.g. 10000) to extend a production run without discarding samples.

MPI divides independent cell/model tasks via rank-strided indexing. There is no
MPI communication inside a fit; >18 ranks cannot help for nine cells. A RUNNING.lock prevents
concurrent writers. After a timeout/MPI abort, remove that file ONLY after squeue
confirms no process is writing that output root, then resume with identical code.
HDF5 backends checkpoint every step; summary generation is separate from sampling.

## Quick figures and statistics

```bash
python richness_relation/plot_richness_cells.py \
  --fits catalogs/conditional_richness/cell_fits_3x3 \
  --output "plots/conditional_richness/cells_$(date +%Y%m%d_%H%M%S)"
```

Outputs: conditional_cell_distributions.pdf/png (3x3),
lognormal_posterior_trends.pdf/png, mixture_posterior_trends.pdf/png,
cell_counts.csv, fit_statistics.csv, posterior_parameters.csv and three .tex
tables. Existing plot output directories are refused. Missing cells are logged;
unconverged fits are visibly provisional. Trend points use posterior medians,
16th--84th intervals, D/p/+ markers and multiplicative 5% horizontal offsets,
following wl_covariance/compute/correlation/read_chains_posterior.ipynb.

Distribution errors are Garwood Poisson count intervals divided by N*linear-bin
width (including empty-bin upper errors); normalized histogram bins are correlated.
The fits use individual unbinned observations, not those display error bars.
Chi-square uses a fixed base histogram with expected counts merged to >=5.
AIC/BIC use the unbinned maximum likelihood. Differences compare ONLY models
within the same cell; never compare raw AIC values between different cells.

## Calibrated goodness-of-fit p-values (automatic in batch; manual rerun below)

Request one task with eight cores, then run this ONCE (not with eight MPI ranks):

```bash
srun --export=ALL -n 1 -c 8 "$CONDA_PREFIX/bin/python" -u \
  richness_relation/plot_richness_cells.py \
  --fits catalogs/conditional_richness/cell_fits_3x3 \
  --output "plots/conditional_richness/cells_gof_$(date +%Y%m%d_%H%M%S)" \
  --gof-bootstrap 200 --workers 8
```

The allocation must permit eight cores for that task. For example a fresh
salloc -A desi -C cpu -q interactive -N 1 -n 1 -c 8 -t 01:00:00.
Each bootstrap simulates from the selected fitted PDF, refits by multi-start
MLE, repeats the expected-count merging, and compares Pearson discrepancies.
Any failed refit makes that cell's p-value unavailable rather than silently
dropping it. Increase replicate count near an interpretation threshold; 200
replicates have coarse Monte Carlo precision. No naive analytic chi-square
p-value is provided (mixture nonregularity and fitted parameters matter).

## Convergence and publication

Each summary records 50-tau checks and the maximum classical R-hat across
fixed-walker trajectories from independent ensembles (not walkers treated as
independent chains). Both must pass for convergence_pass. Inspect traces and
prior-boundary sensitivity as well; two ensembles cannot establish exploration
of every mode. The generic checker also supports these chain directories:

```bash
python richness_relation/check_richness_convergence.py \
  --fits catalogs/conditional_richness/cell_fits_3x3
```

For Overleaf, review the provisional flags before including cell_statistics.tex,
lognormal_cell_parameters.tex and mixture_cell_parameters.tex in tables/.
No automatic manuscript changes or uploads are performed.
