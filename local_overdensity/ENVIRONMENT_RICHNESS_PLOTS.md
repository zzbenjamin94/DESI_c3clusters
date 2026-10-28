# Saved environment versus richness diagnostics

## Data-only conditional RM residuals

Open `plot_environment_residual_quartiles.ipynb` locally, or run from the repo:

```bash
python local_overdensity/plot_environment_richness.py --residual-quartiles \
  --output "plots/environment_residual_quartiles_$(date +%Y%m%d_%H%M%S)"
```

This joins the prepared richness sample to the environment catalog by unique ID,
verifies richness/redshift agreement, and merges the highest two of the original
four global equal-count spec bins, giving three spec bins and three redshift bins.
Each histogram has eight bins. Gaussian E_local quartiles are defined within each
parent cell. Residuals subtract that parent's observed arithmetic RM mean,
shared across quartiles. The code does not fit models or recenter each quartile.
It saves normalized histograms, empirical CDFs, assignments, bin boundaries,
quartile statistics, and settings. Error bars are 68% Wilson binomial intervals
at fixed measured center; they ignore spatial covariance and uncertainty in the
center. Density points/error bars are shown for nonempty bins, while steps also
show empty bins. All observed objects enter the histogram normalization.
The code requires a fresh output directory and does not overwrite older plots.

## Four-parameter environmental mixture fits

The 36 independent fits use the exact IDs and quartiles saved in
`plots/environment_residual_quartiles_merged_20261027/cluster_assignments.csv`.
All individual RM richnesses enter the unbinned normal-plus-positive-EMG
likelihood, normalized on RM > 20. The shared parent mean is a plotting offset,
not a fitted-mean constraint. The Gaussian core is not forced to zero residual.

```bash
python richness_relation/fit_global_spec_richness.py \
  --environment-assignments plots/environment_residual_quartiles_merged_20261027/cluster_assignments.csv \
  --output catalogs/environment_mixture_4param_20261027 \
  --steps 5000 --burn 1000 --walkers 32 --ensembles 4 --workers 4

python local_overdensity/plot_environment_richness.py \
  --mixture-fits catalogs/environment_mixture_4param_20261027 \
  --measurements plots/environment_residual_quartiles_merged_20261027 \
  --output plots/environment_mixture_4param_20261027
```

Use `--resume` with a larger `--steps` to extend unchanged chains. Do not launch
two writers on the same output directory. Plot outputs require a fresh directory.
The notebook's final cells display saved fits and contours without running MCMC.

Priors are uniform in mu, ln(sigma_lambda), f_proj, ln(tau), respectively within
(-300,300), (ln(0.1),ln(300)), (0.0001,0.9999), (ln(1/500),ln(10)).
These are finite analysis assumptions, not astrophysical measurements. The
strongly truncated data may not identify a core below the cutoff, its intrinsic
tail fraction, or a near-absent tail's scale. Prior sensitivity is essential.
In particular, when only the far exponential tail survives, the selected density
approaches `tau * exp[-tau * (lambda_RM - 20)]`: mu, sigma_lambda and f_proj
cancel out. More MCMC steps cannot recover information absent from the selected
data. A physically justified prior or a different selected-distribution
parameterization would be a separate scientific choice, not an automatic fix.
Four independently seeded ensembles start around independently found optima.
Passing requires retained steps >= 50 autocorrelation times and classical Rhat
< 1.1 for every parameter, using matching walker indices across independent
ensembles (reporting the worst index), never treating interacting walkers as
independent chains. Passing is not proof of convergence.

Outputs include all HDF5 chains, thinned posterior draws, provenance, diagnostics,
four 3x3 component plots, 36 contour plots, parameter trends, and CSV summaries.
Grey bands represent pointwise 68% posterior density intervals. Failed checks
are labelled on plots and faded in trends; do not interpret those as final
posterior constraints. PIT chi-squared is a discrepancy, not a nominal p-value;
the reported PPC compares it with replicated uniform-PIT histograms. AIC/BIC
are only comparable between models fitted to identical observations, not between
different environment quartiles. No model-comparison claim is made here.

## Adaptive pooled highest-richness test

The first test pools all environments and 0.10 <= z < 0.35 in the original
highest spectroscopic-richness bin (118.7517 through 796.0114; 3673 clusters).
It does not run the proposed 12 fits yet. The unbinned likelihood and priors
are unchanged. Histograms plot RM richness directly, with no mean subtraction.

```bash
python richness_relation/fit_global_spec_richness.py \
  --environment-assignments plots/environment_residual_quartiles_merged_20261027/cluster_assignments.csv \
  --highest-richness-pooled --adaptive \
  --output catalogs/environment_mixture_highest_pooled_adaptive_20261027 \
  --steps 10000 --block-steps 5000 --max-steps 100000 --max-seconds 3600 \
  --burn 5000 --walkers 32 --ensembles 4 --workers 4

python local_overdensity/plot_environment_richness.py \
  --pooled-mixture-fit catalogs/environment_mixture_highest_pooled_adaptive_20261027/pooled_s3_Q0 \
  --output plots/environment_mixture_highest_pooled_adaptive_20261027
```

Four independent ensembles run concurrently, each writing its own HDF5 file.
Every checkpoint saves posterior summaries and an atomic diagnostic-history JSON.
Stopping requires retained length >= 50 tau and worst matching-walker classical
Rhat < 1.1 for all parameters, plus <10% fractional change in every tau estimate
from the previous checkpoint. All three conditions must pass twice consecutively.
Burn-in is fixed at 5000 throughout, not reset for each block. The wall-time limit
is soft and checked between blocks. Hitting a limit is not labelled convergence.
The final decision is `adaptive_convergence_pass`; `convergence_pass` only covers
the instantaneous length and Rhat checks. No same-time reading/writing of chains.

To resume an interrupted run, use the identical command with `--resume`; saved
positions and RNG states are restored. A `RUNNING.lock` prevents concurrent
writers. After a hard crash, remove a stale lock only after verifying that no
sampler process is still writing. Configuration/provenance must match.

To intentionally extend a run that hit its limit, increase `--max-steps` and
add `--resume --extend-budget`. Both computational limits must be nondecreasing.
This records old/new metadata in `budget_revisions.json`; data, model checksum,
priors, burn-in, walker counts and convergence rules must remain unchanged. It
does not reset chains or relax convergence criteria. The wall-time budget is
per invocation, so a resumed run receives a fresh soft wall-time allowance.

After reviewing this test, `--pooled-redshift` instead of
`--highest-richness-pooled` prepares the 12-fit layout: three original richness
bins, with environment quartiles recomputed within each pooled richness bin.
Use a different output root. Do not reuse old redshift-cell quartile labels.

### Test result, 2026-10-27

The highest-richness pooled test completed 100000 steps in each of four
32-walker ensembles (5000 burn-in), saving 19 checkpoints from 10000 onward.
It stopped at the step safety limit, not convergence: minimum retained N/tau
20.28 (required 50), maximum Rhat 1.391 (required <1.1), maximum fractional
tau change 0.1089 (required <0.1). Runtime was about 14.8 minutes.
No 12-fit production run was launched. Chains remain resumable. The notebook
displays the direct RM distribution, contour, trace and convergence-history
plots under `plots/environment_mixture_highest_pooled_adaptive_20261027`.
These are diagnostic plots, not final posterior constraints.

## Environment versus richness correlations

From the NERSC repository root:

```bash
source ./setup_nersc_mpi_myEnv_v39.sh
python local_overdensity/plot_environment_richness.py
```

No MPI, raw sweeps, or new environment measurements are needed. The default
input is local_overdensity/dr9_outputs/section4/dr9/environment.fits.
Outputs go to plots/environment_richness (rerunning replaces these plots).

Four PNG/PDF figures compare box/Gaussian excess surface density and E_local
against RM and fully weighted continuum-subtracted spectroscopic richness.
lambda_spec_geo_comp_lf is the catalog builder's source for the alias
lambda_spec_noproj_weighted and is retained in the environment file.

Colors denote z=[0.10,0.18), [0.18,0.24), [0.24,0.35). Hollow markers show
linear-space mean richness and mean environment, with SEM of the latter.
Up to seven quantile bins per redshift/richness sample have at least 20 objects;
ties can reduce the number of bins or leave undersized bins unplotted.
These bin choices do not affect correlations, which use all eligible individual
clusters. Default selection uses finite values, positive richness and stored
validity flags, with no additional coverage cut. For a sensitivity test, use
--coverage-min 0.8 --output plots/environment_richness_cov08.

Legend brackets give 16th-84th percentile cluster-pair bootstrap intervals for
Spearman rho (1000 draws, seed 42); they are not scatter or SEM. CSV reports
asymptotic p-values separately. Both bootstrap and p-values ignore spatial
dependence and multiple testing; use spatial resampling for publication claims.
Negative excess densities are retained. Density units are h^2 Mpc^-2 (comoving);
E_local is dimensionless. Diagnostic ratios can be unstable at small backgrounds.

spearman_summary.csv, binned_means_sem.csv and settings.json record results and
settings. --bootstrap 20 is a quick rendering check, not a reliable uncertainty.
# Fixed-global-mean three-parameter pilot

`environment_mixture_fixed_global_highest_20261027` repeats the previous highest
merged richness-bin test (118.75--796.01, 3673 clusters), pooling all environments
and 0.10 <= z < 0.35. It does not fit the newer, narrower top-quartile sample.
The global no-evolution forward posterior supplies the median pre-selection
arithmetic mean separately for each cluster. The Gaussian core is derived as
`mu_core_i = global_mean_i - f_proj/tau`, not fixed to the global mean itself.
Only `ln_sigma_lambda`, `f_proj`, and `ln_tau` are sampled. Their priors are the
same as in the four-parameter test. Global uncertainty and possible global-model
misspecification are deliberately not propagated in this conditional pilot.

The likelihood explicitly multiplies by Theta(RM-20) and divides by each
cluster's mixture survival probability at 20. Means are constrained before
selection; the selected means remain free to differ from them. Histogram curves
average the separately normalized conditional densities over the actual sample.

```bash
python richness_relation/fit_global_spec_richness.py \
  --environment-assignments plots/environment_residual_quartiles_merged_20261027/cluster_assignments.csv \
  --highest-richness-pooled \
  --fixed-global-mean catalogs/global_rm_richness_scatter_20261027/noz \
  --adaptive --output catalogs/environment_mixture_fixed_global_highest_20261027 \
  --steps 5000 --block-steps 5000 --max-steps 100000 --max-seconds 1800 \
  --burn 1000 --walkers 32 --ensembles 4 --workers 4

python local_overdensity/plot_environment_richness.py \
  --pooled-mixture-fit catalogs/environment_mixture_fixed_global_highest_20261027/pooled_s3_Q0 \
  --histogram-bins 15 --output plots/environment_mixture_fixed_global_highest_20261027
```

The final section of `plot_environment_residual_quartiles.ipynb` displays these
plots. Check adaptive convergence and predictive adequacy separately.

### Pilot outcome

The run reached 100000 steps per ensemble. Original burn=1000 diagnostics
failed (worst Rhat=1.316). One walker explored a low-likelihood near-Gaussian
branch until approximately step 44660. A separate, explicitly post-hoc analysis
discarding 50000 steps passed the same thresholds at eight consecutive late
checkpoints: worst Rhat=1.00221, minimum retained N/tau=856.0, final maximum
relative tau change=0.01397. Original chains and failed reports remain intact.

Run the existing summarizer with `--environment-burn 50000 --fits` pointing at
the `pooled_s3_Q0` directory. It writes a separate `diagnostics_burn_50000`
subdirectory, which can be passed to `--pooled-mixture-fit`. Final figures are
in `plots/environment_mixture_fixed_global_highest_20261027_burn50000`.
The posterior medians are sigma_lambda=12.819, f_proj=0.5861, tau=0.07015.
The 15-bin conditional PIT discrepancy is about 182.5; zero of 100 posterior
predictive replicates exceeded the observed discrepancy. This is finite-Monte-
Carlo evidence of model mismatch, not an exactly zero probability. The model
underpredicts the counts near the selection threshold. Convergence of the late
segment is not evidence that fixing the global mean gives an adequate model.

## Differential-evolution richness-redshift grid

The nine-cell run pools all environments and uses three global equal-count
spectroscopic-richness bins crossed with [0.10,0.18), [0.18,0.24), [0.24,0.35).
Common richness edges permit comparisons between redshifts; individual cells
are not equal in size. The three-parameter likelihood and fixed per-object
global arithmetic means are unchanged from the pilot above.

```bash
python richness_relation/fit_global_spec_richness.py \
  --environment-assignments plots/environment_residual_quartiles_merged_20261027/cluster_assignments.csv \
  --richness-redshift-grid \
  --fixed-global-mean catalogs/global_rm_richness_scatter_20261027/noz \
  --proposal de_snooker --adaptive \
  --output catalogs/mixture_spec_tertiles_z_de_20261027 \
  --steps 10000 --block-steps 5000 --max-steps 100000 --max-seconds 900 \
  --burn 5000 --walkers 32 --ensembles 4 --workers 3

python local_overdensity/plot_environment_richness.py \
  --cell-mixture-fits catalogs/mixture_spec_tertiles_z_de_20261027 \
  --histogram-bins 15 --output plots/mixture_spec_tertiles_z_de_20261027
```

Three cells run concurrently; each has four independent ensembles with 32
walkers. Proposals mix emcee DEMove (80%) and DESnookerMove (20%). HDF5 backends
save partial chains; each 5000-step checkpoint checks retained length >=50 tau,
classical Rhat <1.1 across independent ensembles at matching walker indices,
and maximum fractional change in tau <0.1. All checks must pass twice in a row.
Burn-in remains 5000 throughout. Limits are 100000 steps or 900 seconds per cell;
the time cap is soft, checked only between complete blocks. Rerunning the same
command with `--resume` resumes compatible incomplete checkpoints, without
resetting chains.

The plotting command exports conditional-distribution panels, parameter trends,
posterior intervals, histogram measurements, and convergence/PPC statistics.
Histogram bars are 68% binomial Wilson intervals, not SEM. Trend bars are
16th--84th posterior percentiles. The PIT discrepancy uses 15 equal-probability
bins and 100 posterior-predictive replicates: a reported zero means 0/100
replicates exceeded the observed discrepancy, not an exactly zero probability.
Inspect the final section of `plot_environment_residual_quartiles.ipynb`.

### Nine-cell outcome

The global tertile edges are 6.87649969, 95.17748854, 145.03350667, 796.01137005.
Counts by redshift row and richness column are (349,402,251), (492,638,551),
and (1607,1408,1647). The richness totals are 2448,2448,2449.
Eight cells passed all adaptive checks at 20000--30000 steps. Cell z2_s2
stopped at its soft time cap after 55000 steps: worst Rhat=1.00794 and minimum
retained length/tau=524.2 pass, but the latest maximum fractional tau change
was 0.1608, above 0.1. Its plotted estimates are faded and remain provisional.
No extra burn-in was selected retrospectively for these nine runs.

The highest-richness/highest-redshift cell z3_s3 converged but fails the
predictive adequacy diagnostic: PIT discrepancy about 151.6, with 0/100
replicates exceeding the observed discrepancy. Its conditional-density plot
underpredicts the concentration near the richness cut. The fixed global mean
and chosen mixture family should be revisited before interpreting parameter
trends physically. Other cells have broad tail-parameter intervals; convergence
does not establish that those parameters are well identified. All results
condition on fixed global predictions and do not propagate their uncertainty.

## Four pooled richness bins with regularizing priors

The 2026-10-28 run uses four equal-count spectroscopic-richness bins, pooling
0.10 <= z < 0.35 and all environments. Priors: f_proj ~ Beta(2,2),
sigma_lambda ~ Uniform(0,30), and 1/tau ~ LogUniform(0.1,500). The sampler
still uses ln_sigma_lambda: its log prior includes the +ln_sigma_lambda
Jacobian. A numerical floor of 1e-12 excludes a singular Gaussian, rather than
retaining the former physical lower bound of 0.1. The optimizer now finds a
MAP point, recorded as such rather than called a maximum-likelihood estimate.
The global mean and selected likelihood are unchanged. Verified per-object
mean predictions are reused from the nine-cell run by ID and checksum.

```bash
python richness_relation/fit_global_spec_richness.py \
  --environment-assignments plots/environment_residual_quartiles_merged_20261027/cluster_assignments.csv \
  --pooled-spec-quartiles \
  --fixed-global-mean catalogs/global_rm_richness_scatter_20261027/noz \
  --fixed-mean-cache catalogs/mixture_spec_tertiles_z_de_20261027 \
  --mixture-prior beta22_uniform_sigma --proposal de_snooker --adaptive \
  --output catalogs/mixture_spec_quartiles_beta22_20261028 \
  --steps 10000 --block-steps 5000 --max-steps 100000 --max-seconds 1800 \
  --burn 5000 --walkers 32 --ensembles 4 --workers 4

python local_overdensity/plot_environment_richness.py \
  --cell-mixture-fits catalogs/mixture_spec_quartiles_beta22_20261028 \
  --histogram-bins 15 --output plots/mixture_spec_quartiles_beta22_20261028
```

The final notebook section displays this run. Four cells run concurrently;
limits are 100000 steps or 30 minutes per cell, checked between complete
blocks. The same two-consecutive-checkpoint stopping criteria apply. Never
resume old-prior chains as though they had the new prior.

Outcome: all four cells pass the specified adaptive criteria. Steps are
30000, 20000, 20000, 30000; worst Rhat values are 1.01035, 1.00177, 1.00210,
1.07421. The highest bin is less well mixed than the others despite passing
the configured 1.1 threshold. PIT posterior-predictive upper-tail fractions
are 0.43, 0.06, 0.16, and 0/100. The latter is evidence of persistent model
mismatch, not an exactly zero probability. These diagnostics use 100 draws
and condition on the fixed global mean. They are not classical chi-square
p-values. Bin changes and prior changes were made together, so comparisons
against the older nine-cell results do not isolate a prior effect.

The posterior median (sigma_lambda, f_proj, tau), from lowest to highest bin,
is (6.880,0.389,0.1546), (7.186,0.708,0.1346), (8.186,0.792,0.09847),
(15.417,0.766,0.05598). The plotted tau is a rate; the mean tail boost is 1/tau.
See posterior_parameters.csv for 16th--84th percentile intervals. Contours
are in the plots directory's contours/ subfolder and the final notebook cell.
