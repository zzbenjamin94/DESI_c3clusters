# Section 5: conditional richness

This workflow fits P(lambda_RM | measured lambda_spec, z), with no environment
terms. It uses the Section 5 manuscript notation. It does not recompute the
galaxy matching, weights, or continuum.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| `prepare_conditional_richness.py` | Load, validate, deduplicate, and select the cluster sample. |
| `conditional_richness_models.py` | Common likelihoods, priors, conditional CDFs and predictive sampling. |
| `fit_conditional_richness_mpi.py` | Multi-start optimization, emcee, checkpoints, fit statistics and predictive checks. |
| `plot_conditional_richness.py` | Reusable plotting functions and terminal entry point; no fitting. |
| `plot_conditional_richness.ipynb` | Data-only plots by default; optional saved fits, tables and diagnostics. |
| `run_conditional_richness.slurm` | One-node, eight-rank NERSC example under account `desi`. |

## Data-only plots (no fitting)

Prepare the sample once, then run:

```bash
python richness_relation/prepare_conditional_richness.py
python richness_relation/plot_conditional_richness.py --data-only
```

Alternatively, run `plot_conditional_richness.ipynb` with `DATA_ONLY = True`
(the default). Neither route imports the MCMC runner or reads fit products.
Outputs are PNG and PDF files in `plots/conditional_richness/data_only/`:

- `richness_relation_redshift_bins`: individual richnesses and geometric means
  with log-space SEM transformed to asymmetric linear error bars, in the three
  existing redshift bins.
- `richness_logratio_offsets`: d = log10(lambda_RM / lambda_spec) versus
  spectroscopic richness and BCG redshift, plus its distribution in each
  redshift bin. Points show arithmetic means of d with SEM = std(d, ddof=1)/sqrt(N),
  in dex, without connecting lines. Quantile bins have at most seven bins per
  redshift interval; bins with fewer than 20 clusters are not plotted.

This offset is relative to equality, not a fitted residual or a direct measure
of projection: differences in estimator normalization and selection also matter.
SEM describes uncertainty in the mean, not cluster-to-cluster scatter, and
assumes independent clusters. Data-only outputs are separated from fit outputs
so the notebook cannot accidentally display stale fitted curves or tables.
Set `DATA_ONLY = False` or omit `--data-only` to display saved model results.

## Input and selection

Default input:

`catalogs/bgs_clus_RM_gal_matched_with_weights.fits`

This is the combined matched catalog containing weights and precomputed
spectroscopic richness. Preparation reads those measurements; it does not
recompute weights or richness. Another catalog can be passed using `--input`;
verify that its richness uses the intended postprocessing selection.

The fields are `ID`, `lambda_spec_noproj_weighted`, `LAMBDA`, and
`Z_SPEC_central`. All repeated values must agree within each cluster ID before
deduplication. Optional cluster-level LF/geometry fields are checked as well.

The prepared sample retains 0.10 <= z_BCG < 0.35, finite measurements,
lambda_spec > 0, and lambda_RM >= 20. There is no default upper richness cut.
Use `--rm-min` / `--rm-max` to match the actual sample selection, not the desired
plot limits. Excluded counts are written to the preparation JSON.

The three redshift bins are imported from `tools/richness_selection.py`:
[0.10, 0.18), [0.18, 0.24), [0.24, 0.35). No new galaxy-level redshift cuts are
applied to an already calculated cluster richness. Zero and negative estimates
are counted and excluded; they are never replaced by a small positive floor.

**Upstream validation required for physical interpretation:** the current
`calc_specRichness_individual` shares a continuum and mean redshift-offset-bin
weights across the entire supplied sample. A cluster's own `LF_WEIGHT` is
therefore not automatically preserved as its exact richness multiplier. The
new workflow records this limitation and uses the saved measurements as given.
It does not silently change the estimator. It also has no validated per-cluster
richness errors, so scatter is conditional observed dispersion, not deconvolved
intrinsic scatter. Shared-continuum/LF uncertainty is not propagated here.

## Equations and conventions

All logarithms in the likelihood are natural logarithms:

\[
x_i=\ln(\lambda_{{\rm spec},i}/\lambda_{\rm piv}),\quad
\zeta_i=\ln[(1+z_i)/(1+z_{\rm piv})],\quad
\ln\mu_{{\rm RM},i}=\ln A+B x_i+C\zeta_i.
\]

The fixed pivots default to lambda_piv = 40 and z_piv = 0.24 and can be changed
at the fitting stage. They set parameter reference points, not observed results.
Every fit in a comparison must use consistent pivots.

The lognormal model is

\[
p_{\rm LN}(r_i)=\frac{1}{r_i\sigma_0\sqrt{2\pi}}
\exp\left[-\frac{(\ln r_i-\ln\mu_{{\rm RM},i})^2}{2\sigma_0^2}\right],
\qquad r_i=\lambda_{{\rm RM},i}.
\]

Here mu_RM is the unselected median. The unselected arithmetic mean is
mu_RM exp(sigma0^2/2). Scatter in dex is sigma0 / ln(10). Neither relation should
be used unchanged for the selected distribution; its quantiles come from the
normalized selected CDF.

The mixture is

\[
\lambda_{{\rm RM},i}=\mu_{{\rm RM},i}+\Delta_{{\rm bkg},i}+\Delta_{{\rm proj},i},
\quad \Delta_{\rm bkg}\sim\mathcal N(0,\sigma_\lambda^2),
\]
\[
p(\Delta_{\rm proj})=(1-f_{\rm proj})\delta_{\rm D}(\Delta_{\rm proj})
+f_{\rm proj}\tau e^{-\tau\Delta_{\rm proj}}\Theta(\Delta_{\rm proj}),
\]
\[
p_{\rm mix}(r_i)=(1-f_{\rm proj})\mathcal N(r_i;\mu_{{\rm RM},i},\sigma_\lambda^2)
+f_{\rm proj}p_{\rm EMG}(r_i;\mu_{{\rm RM},i},\sigma_\lambda,\tau).
\]

This is the manuscript EMG kernel, evaluated using SciPy `exponnorm` with
K = 1 / (tau sigma_lambda). Tau is a RATE; tau_inverse is the tail length.
Delta_mu is fixed to zero to avoid initially trading a background shift against
the core normalization. f_proj, tau and sigma_lambda are constant within a fit.
The unselected mixture mean is mu_RM + f_proj/tau. These are phenomenological
parameters, not proof that individual objects are projected.

Every likelihood includes the configured observed RM interval:

\[
p_{\rm sel}(r_i)=p(r_i)/[F(r_{\max})-F(r_{\min})].
\]

The RM lower threshold is essential for a conditional model whose response is
RM richness. This corrects only that explicit threshold; redMaPPer detection,
BGS matching and other sample-completeness effects are not modeled. The Gaussian
mixture is normalized over the positive selected range even though its
untruncated core has support below zero.

Both likelihoods are densities in **linear RM richness**, so comparisons include
the lognormal Jacobian and use the same data and selection normalization.
There is no extra multiplication of the cluster likelihood by galaxy weights.

## Prior specification

Independent proper uniform priors in sampled coordinates are:

| Coordinate | Lower | Upper |
| --- | --- | --- |
| ln A | ln 1 | ln 1000 |
| B | -3 | 3 |
| C (when fitted) | -30 | 30 |
| ln sigma0 (lognormal) | ln 0.01 | ln 2 |
| ln sigma_lambda (mixture) | ln 0.1 | ln 300 |
| f_proj (mixture) | 0 | 1 |
| ln tau (mixture) | ln(1/500) | ln 10 |

These are starting assumptions, not literature measurements. Log-uniform scale
priors can matter strongly when the tail is weakly identified. Check boundary
occupation and rerun with scientifically justified alternative bounds before
reporting a projected fraction or tail scale.

## NERSC: start with the baseline

From the repository root, use the existing NERSC Python/MPI environment:

```bash
cd /global/homes/z/zzhang13/DESI/Projection
source setup_nersc_mpi_myEnv_v39.sh
python -c "import numpy, scipy, astropy, pandas, matplotlib, h5py, emcee, schwimmbad; from mpi4py import MPI"
python -m unittest discover -s tests -p test_conditional_richness.py -v
python richness_relation/prepare_conditional_richness.py
```

If a Python package is missing, the separate requirements file lists it. Reuse
the working Cray-compatible MPI installation; do not replace it with a generic
MPI package. The input pickle must be a trusted local catalog.

For a quick first look, fit maximum likelihood without sampling:

```bash
python richness_relation/fit_conditional_richness_mpi.py --model lognormal --optimize-only
python richness_relation/plot_conditional_richness.py
```

The default `--scope suite` fits combined data with C=0, combined data with C
free, and each of the three redshift bins separately with C=0. Set `--scope
combined` or `--scope bins` for a subset. The combined C comparison tests
normalization evolution; this first implementation does not fit evolving slopes
or scatter. Remaining residual trends indicate whether those extensions are needed.

For an interactive MPI smoke test (use a separate output from production):

```bash
salloc -N 1 -C cpu -q interactive -t 01:00:00 -A desi
cd /global/homes/z/zzhang13/DESI/Projection
source setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun -n 8 -c 1 python richness_relation/fit_conditional_richness_mpi.py \
  --mpi --model lognormal --scope combined --steps 200 --burn 50 \
  --output catalogs/conditional_richness/smoke_fits
```

For production, the scheduler template runs the lognormal suite and resumes
existing compatible chains:

```bash
sbatch richness_relation/run_conditional_richness.slurm
```

Within a suitable allocation, the corresponding mixture command is:

```bash
srun -n 8 -c 1 python richness_relation/fit_conditional_richness_mpi.py \
  --mpi --model mixture --scope suite --steps 3000 --burn 750 --resume
```

`--model both` fits both families. `--resume` adds steps up to the requested
TOTAL `--steps`, never resets a chain, and checks data/code hashes, pivots,
walker count, burn-in and other fixed settings. To change those settings, use a
different output directory. `--seed` identifies an independent run. Do not run
two simultaneous jobs against the same output/seed directories. Checkpoints
are stored each step; a hard timeout can still interrupt a filesystem write.

The default two ensembles have independent random streams. Length > 50 tau is
only a heuristic, not a convergence certificate. Compare independent ensemble
posteriors and individual walker traces, not just acceptance fractions. ESS
reported from walker count/tau is approximate. More MPI ranks than active
walkers will not provide proportional speedup. This likelihood may be fast
enough that serial execution is competitive; benchmark first.

## Output and interpretation

The preparation writes `catalogs/conditional_richness/sample.npz` and its audit
JSON, including source checksum, cut counts and warnings. No source catalog is
overwritten. Per-task outputs live in `catalogs/conditional_richness/fits/`:

- `manifest.json`: input and code hashes, cluster IDs, exact run configuration.
- `bestfit.json`: bounded multi-start maximum likelihood, AIC/BIC/AICc.
- `chain_0.h5`, `chain_1.h5`: resumable emcee chains.
- `summary.json`: physical-parameter 16/50/84 percentiles, ensemble diagnostics, PPCs.
- `posterior.npz`: a bounded random subset of draws and per-cluster log likelihoods.

Use `--cv-folds 5` for an optional refitted, held-out MLE log-density check. This
is not posterior predictive evidence or spatial cross-validation. All model
comparisons must share the same held-out objects. AIC/BIC are supporting metrics
for a mixture, not Bayes factors; emcee does not calculate evidence.

PPC discrepancies use the selection-aware probability integral transform (PIT),
upper/lower tails, normal-score sum of squares, and correlations with richness
and redshift. The reported PPC probability is the fraction of replicated
discrepancies exceeding the observed discrepancy at the SAME posterior draw.
It is not a classical goodness-of-fit p-value. No nominal chi-squared p-value
is assigned to an asymmetric/truncated mixture or an in-sample fitted KS test.

Run `plot_conditional_richness.py` or open `plot_conditional_richness.ipynb` to
regenerate plots without MCMC. Data points summarize mean ln(lambda_RM) at
binned lambda_spec, exponentiated, with asymmetric transformed log-SEM bars.
Those bins are only visualization; all individual clusters enter the fits.
Scatter plots distinguish the underlying median, selected median, fixed-MLE
predictive 16--84% interval and posterior uncertainty of the selected median.
Conditional histograms average each model over the actual covariates of the
clusters in each panel, rather than evaluating just at a bin center. Density
is per unit LINEAR lambda_RM even when the horizontal axis is logarithmic.

PNG/PDF figures and comparison CSVs go under `plots/conditional_richness/`.
Both `plots/` and `catalogs/` are ignored by Git; notebook outputs are empty in
the supplied notebook. New code has not changed the upstream richness estimator.
