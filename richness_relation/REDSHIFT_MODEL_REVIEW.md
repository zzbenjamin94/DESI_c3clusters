# Review redshift evolution before plotting

This workflow reads existing fits; it never runs MCMC or fits independent cells.
Run from the repository root. Use the same input sample used by the chains.

```bash
source ./setup_nersc_mpi_myEnv_v39.sh
python richness_relation/check_richness_convergence.py --fits catalogs/conditional_richness/lognormal_fits
python richness_relation/check_richness_convergence.py --fits catalogs/conditional_richness/mixture_fits
python richness_relation/review_redshift_models.py \
  --data catalogs/conditional_richness/overnight_sample.npz \
  --fits catalogs/conditional_richness/lognormal_fits catalogs/conditional_richness/mixture_fits \
  --output plots/conditional_richness/redshift_review
```

Review convergence first, then C's posterior and delta AIC/BIC (z minus noz).
There is no automatic significance threshold or automatic model selection.
Choose separately for each distribution family. Share the report before choosing
if the posterior, convergence checks, and information criteria disagree.

After deciding, repeat the review command with a NEW output directory and both:
`--lognormal-choice noz --mixture-choice z` (example only, not a recommendation).
Either flag accepts `noz` or `z`.

Outputs include model_comparison.csv, parameters.csv, redshift_comparison.tex,
and, after choosing, conditional_distributions.pdf/png,
binned_quantile_residuals.pdf/png, bin_counts.csv and model_choices.json.
Richness boundaries are full-sample quartiles shared across redshifts. Cells with
fewer than 20 objects are not plotted. Redshift boundaries come from sample.json.
Histogram errors are approximate 68.27% Garwood Poisson intervals divided by
N times the linear bin width, including upper errors for empty bins. Normalized
histogram bins are correlated; these display errors are not used to refit models.

All panel curves average selected conditional PDFs over actual cluster richnesses
and redshifts, using the selected global MLE. No bin-specific parameters are inferred.
Residuals are selected-CDF normal scores; their histograms use total cell N even
when scores fall outside the displayed range.

The table reports posterior medians and equal-tail errors. CSV also includes MLEs.
The chi-square is a descriptive ten-bin PIT Pearson discrepancy, NOT a chi-square
of binned means. No classical p-value is assigned: fitted-model calibration would
require simulation and refitting. Do not describe it as reduced chi-square.
The model location is a lognormal median or mixture core, not generally a mean.
Convergence review is mandatory before using posterior errors in a publication.

Copy redshift_comparison.tex into the Overleaf tables directory and include it
with `\input{tables/redshift_comparison}`. This is a standalone table* without
additional package dependencies. Review results before publishing; the script
does not alter the manuscript or push generated tables to Git.
