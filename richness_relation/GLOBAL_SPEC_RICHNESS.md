# Global reverse-conditional richness baseline

## Forward global fit added 2026-10-27

The same scripts now support `--direction forward`, fitting
**P(lambda_RM | lambda_spec,z)** directly. Existing reverse chains are preserved.
Forward coefficients are not inverses of the reverse-regression coefficients:

```text
ln(mu_RM) = A + B ln(lambda_spec/40) + C ln[(1+z)/1.24]
ell = ln(mu_RM) - sigma_ln_RM^2/2
ln(lambda_RM) ~ Normal(ell, sigma_ln_RM^2), selected at lambda_RM > 20
log L = sum(lognormal logpdf - log survival_probability_at_20)
```

Here mu_RM is the pre-selection arithmetic mean. The selection normalizer is
evaluated separately for every cluster, with stable Gaussian log-survival
probabilities. The lower cut is read from the sample audit; finite upper cuts
are explicitly rejected by this implementation. Display limits never truncate
the fitted sample. This does not model any additional BGS selection function.
Priors are uniform in A (-5,10), B (-5,5), C (-10,10), ln_sigma_ln_RM (-7,1).

```bash
PYTHON=/Users/zzbenjamin94/Documents/Codex/envs/desi-local/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PYTHON" richness_relation/fit_global_spec_richness.py \
  --direction forward --model both --output catalogs/global_rm_richness_20261027 \
  --steps 5000 --burn 1000 --walkers 32 --ensembles 4
"$PYTHON" richness_relation/summarize_global_spec_richness.py \
  --fits catalogs/global_rm_richness_20261027
"$PYTHON" richness_relation/plot_global_richness_relation.py \
  --fits catalogs/global_rm_richness_20261027/z \
  --output plots/global_rm_richness_20261027/z
"$PYTHON" richness_relation/plot_global_richness_relation.py \
  --fits catalogs/global_rm_richness_20261027/noz \
  --output plots/global_rm_richness_20261027/noz
```

Use `--resume` only to extend the same chains; choose a new plot output directory
if it already exists. The figure uses spec on x, RM on y, density-colored points,
arithmetic binned means with SEM, and 68% pointwise posterior bands. Solid curves
are selected-sample means; dashed curves are pre-selection arithmetic means.
For evolving fits curves use the median z of each interval, whereas markers
average all objects in that interval. X/Y display limits are 35--600 / 20--200.

Forward summaries include convergence diagnostics and selection-aware PPCs using
the probability-integral transform (PIT), with histogram chi-square and
Cramer-von Mises discrepancies. `chi2_mle` is the forward PIT histogram
discrepancy, NOT the reverse Gaussian sum of squared residuals; no nominal
chi-square p-value is assigned. PPC values are upper-tail probabilities, not
probabilities that the model is true. AIC/BIC compare the same selected data.
The 36 environment-cell fits have not been launched by these commands.

## Original reverse model

Model Section 5's **P(lambda_spec | lambda_RM,z)**:

    mu = a_rev + b_rev ln(lambda_RM/40)
         + c_rev ln[(1+z)/(1+0.24)]
    ln(lambda_spec) ~ Normal(mu, s_rev^2)

The no-z model fixes c_rev=0. exp(mu) is the conditional geometric mean
(also the lognormal median), not the arithmetic mean exp(mu+s_rev^2/2).
Scatter includes measurement noise and is not a deconvolved intrinsic scatter.
No environment term, individual cell refitting, or galaxy-weight recomputation.

The checksum-validated prepared sample is used unchanged: one row per cluster,
fully weighted continuum-subtracted lambda_spec, positive richnesses. Check its
adjacent JSON for selections. Positive-response selection and measurement errors
can affect inference; this baseline does not infer the excluded negative estimates.
An RM cut selects the predictor, so the reverse likelihood has no forward-model
RM truncation denominator. Plot limits are not sample cuts.

## Run locally from the Git repository

```bash
PYTHON=/Users/zzbenjamin94/Documents/Codex/envs/desi-local/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$PYTHON" richness_relation/fit_global_spec_richness.py \
  --data catalogs/conditional_richness/overnight_sample.npz \
  --output catalogs/global_spec_richness \
  --model both --steps 5000 --burn 1000 --walkers 32 --ensembles 4
"$PYTHON" richness_relation/summarize_global_spec_richness.py \
  --fits catalogs/global_spec_richness
```

Open `plot_global_spec_richness.ipynb` and
`plot_conditional_spec_residuals.ipynb` with **Python (DESI local)**.
The latter's MODEL setting is your explicit choice after reviewing evidence.
Both notebooks expose bins, paths and style settings. They read chains without
modifying them and save figures/tables locally, not to Git.

To extend chains, repeat the fitting command with a larger **total** --steps and
--resume. Keep all other settings identical. Rerun summaries and notebook cells.
An interrupted ensemble resumes from its last HDF checkpoint. Do not run two
writers against the same output folder.

## Priors and convergence

Uniform in a_rev (0,10), b_rev (-5,5), c_rev (-10,10), ln_s_rev (-7,2).
Uniform ln_s corresponds to the usual scale prior inside the finite bounds.
Bounds, pivots, sample hash and random seeds are recorded in each run.json.
Check posterior proximity to bounds; change priors only with a fresh output path.

Independent ensembles use independent seeds. We require retained steps >=50 tau
for every parameter/ensemble (emcee integrated autocorrelation time) and classical
Rhat <1.1. Rhat compares a fixed walker index across independent ensembles, then
reports the worst index. Walkers within an ensemble are not independent chains.
These checks are necessary diagnostics, not a guarantee against undiscovered modes.
Inspect traces, ensemble contour overlap and effective lengths.

References: Gelman & Rubin (1992), Statistical Science 7, 457;
Goodman & Weare (2010), CAMCoS 5, 65; Foreman-Mackey et al. (2013), PASP 125, 306.

AIC/BIC use the exact Gaussian maximum likelihood, including the lognormal
Jacobian, on the same sample. Delta values mean z minus no-z. Parameter tables
report posterior medians, 16th/84th intervals and MLEs. Chi-square at the fitted
Gaussian-scatter MLE equals N: **do not interpret it as independent evidence of
goodness of fit**. Inspect conditional distributions, QQ plots and posterior
predictive skewness/tail checks instead. Predictions in each cell average over
its observed RM richness and redshift distribution.

## Independent 3x3 residual fits (Section 5.2)

The existing fitter/summary scripts also support `--cells`; no second pipeline
is required. The conditional-residual notebook now displays these cell fits,
not the old re-evaluated global posterior. The global-fit notebook is unchanged.

Use the converged no-z global fit to fix a_rev,b_rev at their MLE. Every cluster
retains its own prediction, even within a richness bin:

    r_i = ln(lambda_spec_i) - a_rev - b_rev ln(lambda_RM_i/40)
    Log-normal: r_i ~ N(0, s_rev^2)
    Mixture: r_i = G - J*T + f_rev/tau_rev
    G ~ N(0, sigma_rev^2), J ~ Bernoulli(f_rev), T ~ Exp(rate=tau_rev)

All three draws are independent. The additive shift enforces E[r]=0; it is
not an additional parameter. This is a negative-tail mixture in LOG richness,
not the manuscript's positive projection kernel in linear RM richness.
The mixture has 3 free parameters (sigma_rev,f_rev,tau_rev); log-normal has 1.
Priors uniform in ln_sigma_rev in (-6,1), f_rev in (.001,.999),
ln_tau_rev in (-3,5); ln_s_rev in (-6,1). Check boundary sensitivity,
especially in weakly identified near-Gaussian cells.

Cell bins: RM [20,30,50,100], z [.10,.18,.24,.35], left inclusive/right exclusive.
RM >=100 is excluded from these cell fits, NOT from the global relation.
Global mean uncertainty is not propagated. Cell mean offsets remain testable
model discrepancies; they are not removed by recentering data.

```bash
"$PYTHON" richness_relation/fit_global_spec_richness.py --cells \
  --steps 5000 --burn 1000 --walkers 32 --ensembles 4 --workers 4
"$PYTHON" richness_relation/summarize_global_spec_richness.py --cells \
  --fits catalogs/spec_residual_cells --ppc-draws 2000
```

Then run `plot_conditional_spec_residuals.ipynb`. Outputs go to
catalogs/spec_residual_cells and plots/spec_residual_cells. The notebook shows
residual histograms, model posterior bands, log-normal scatter posteriors and
nine pyGTC mixture corner plots overlaying independent ensembles.
Tables contain parameter intervals, AIC/BIC, convergence and PPCs for residual
mean, standard deviation, skewness and excess kurtosis. Histogram chi-square
includes overflow bins and merges expected counts below 5; it is descriptive
and has no uncalibrated nominal p-value. Information criteria are conditional
on the fixed shared mean and not directly comparable to global-fit criteria.

As for global fits, `--resume --steps LARGER_TOTAL` extends existing ensembles.
Do not interpret provisional cells as converged, even when curves look smooth.
Add `--only-unconverged` to extend only cells flagged by the preceding summary.
The parameter table also records the posterior fraction within 1% of a sampled
prior-coordinate boundary. Large fractions warrant a prior-sensitivity check;
they are not automatically cured by longer chains.

## Forward lognormal baseline, 2026-10-27

The forward run `catalogs/global_rm_richness_scatter_20261027` fits RM richness
conditional on measured spec richness, with and without redshift evolution.
Its density is `Theta(RM-20) * LogNormal(RM; ell, sigma) / P(RM>20)`.
The predictor is `ell=A+B*ln(spec/40)+C*ln((1+z)/1.24)`; set C=0 for noz.
The implementation retains the older arithmetic-mean intercept internally:
`A_log = A_internal - sigma**2/2`. The summarizer transforms each posterior
draw and exports `parameters_mean_log.csv`; `parameters.csv` retains the
internal arithmetic-mean convention. The stored bounds define the prior in
the internal coordinates, not independent rectangular bounds on A_log.

Run the existing fitter with `--direction forward --model both`, then the
existing summarizer with `--fits catalogs/global_rm_richness_scatter_20261027`.
Plot each model with `plot_global_richness_relation.py --scatter-contours`,
passing its noz/z subdirectory via --fits and a fresh --output directory.
The last cell of `plot_global_spec_richness.ipynb` displays these forward results
independently of the earlier reverse-regression cells.

These plots show density-colored scatter, KDE contour levels at 10, 30, 60,
and 90 percent of the maximum density on the displayed log-coordinate grid
(not enclosed-probability contours), and the unselected median exp(ell) with
pointwise 68 percent posterior bands. No binned means or error bars are shown.
The median is straight in log-log coordinates; it is not the selected-sample
arithmetic mean. The latter is generally curved near the cutoff. Error bars
on a selected-sample mean are not intrinsically biased, but that mean must not
be treated as an estimate of the unselected relation.
