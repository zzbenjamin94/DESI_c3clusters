# BGS cylindrical environment

Final collaborator-supplied measurement: `bgs/environment.fits`, with original
`manifest.json` and `summary.json`. Intermediate checkpoints are not published.
This small final catalog is an explicit exception to the usual no-catalog policy.

The run uses DR2 BGS Bright, a 60 comoving h^-1 Mpc cylinder half-length,
1.5--5 signal and 5--10 background transverse radii, angular random coverage,
and no LF correction. The plotted local statistic is
`E_local = n_signal / n_background - 1`, using random-corrected effective volumes.
It is not a BCG-only tracer or a global-nbar normalization; some draft wording
still describes that different estimator.

With the existing prepared richness sample available locally, run from repo root:

```bash
python local_overdensity/plot_environment_richness.py \
  --tracer bgs \
  --input local_overdensity/bgs_env_output/bgs/environment.fits \
  --sample catalogs/conditional_richness/overnight_sample.npz \
  --output plots/bgs_environment_richness
```

Richnesses are joined by unique cluster ID and redshifts are checked.
The prepared sample matches 7345 of 7391 environment rows. Saved validity flags
and redshift intervals [0.10,0.18), [0.18,0.24), [0.24,0.35) are applied.
No extra coverage threshold is imposed. Scatter points show individual clusters;
hollow markers show quantile-binned means with SEM. Spearman intervals use
1000 cluster bootstrap draws, ignoring spatial covariance. This is an exploratory
correlation plot, not a causal or fixed-richness environment test.

Dimensionless-statistic Spearman coefficients (in increasing redshift order):
RM: 0.152, 0.105, 0.075; fully weighted spectroscopic: 0.242, 0.247, 0.105.
The script also exports excess-density plots, correlation statistics, and binned
means. Generated plots remain local, following the repository policy.
