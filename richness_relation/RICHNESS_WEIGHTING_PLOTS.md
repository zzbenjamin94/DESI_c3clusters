# Three-stage, data-only richness plots

## Run on NERSC

From `/global/homes/z/zzhang13/DESI/Projection`, activate your normal analysis
environment with `source setup_nersc_mpi_myEnv_v39.sh`, then run:

```bash
python richness_relation/prepare_richness_weighting.py
python richness_relation/plot_richness_weighting_comparison.py
python richness_relation/plot_spectroscopic_richness_residuals.py
```

Or run the two correspondingly named notebooks after the preparation command.
The notebooks only read cached measurements and plot. MPI and MCMC are not
needed. Preparation needs numpy, astropy, scipy and specutils (as in the legacy
continuum method); plotting needs numpy, astropy and matplotlib.

Input is `catalogs/bgs_clus_RM_gal_matched_with_weights.fits`. Preparation saves
`catalogs/richness_weighting/cluster_weighting_stages.ecsv` and a JSON audit.
Use `--overwrite` only to regenerate these derived products. The parent FITS
is never overwritten. This is NOT the old conditional-richness `sample.npz`.

The two notebooks replace the data-only work formerly combined in
`plot_conditional_richness.ipynb`. Existing conditional-model fitting and the
`plot_conditional_richness.py` saved-fit visualizer remain unchanged.

## Estimator and important assumptions

### Diagnose invalid weights first

```bash
python richness_relation/prepare_richness_weighting.py --diagnose-only
```

This does not require specutils, fit a continuum, or overwrite the catalog or
existing richness products. A timestamped folder under
`catalogs/richness_weighting/weight_diagnostics/` contains `summary.json`,
`flagged_rows.ecsv`, and `affected_clusters.ecsv`. The summary distinguishes
zero, negative, NaN/masked, and infinite weights, checks probability validity,
and counts affected clusters. The row table includes original zero-based row
indices, weights, probabilities, redshifts and any available quality columns.
Quality flags are reported without assuming a catalog-specific quality cut.
Invalid stored weights with valid probabilities are identified for investigation,
not automatically repaired. Counts refer to matched rows, not unique TARGETIDs.
The normal preparation command also writes this report and stops when invalid
selected weights/probabilities are found. No rows are silently excluded.

For cluster i, N_i is the number of selected candidate galaxies, C_g is its
galaxy's COMP_WEIGHT, G_i is GEOMETRIC_WEIGHT, and F_i is LF_WEIGHT. Candidate
selection uses the shared science cuts: BCG z in [0.10,0.35), BGS z in
[0.05,0.40), normalized offset (z_BGS-z_BCG)/(1+z_BCG) in [-0.05,0.05].
The aperture remains the one used to create the matched input; this script
does not rematch galaxies or change that aperture.

The unweighted pooled offset PDF is fitted with a specutils
median-window-5, degree-6 Chebyshev continuum, excluding [-0.02,0.02].
These defaults are CONTINUUM_DEGREE and CONTINUUM_MEDIAN_WINDOW in
`prepare_richness_weighting.py` and are recorded in the output JSON audit.
The window counts histogram bins, not a fixed redshift width. Let
c_b be its value at offset-bin center b, and let q_b = c_b * bin_width_b.
Let Cbar_b be the mean COMP_WEIGHT of selected galaxies in offset bin b.
Empty bins use the whole selected-sample mean COMP_WEIGHT.

The three continuum-subtracted estimates are:

```text
unweighted_i = N_i - N_i * sum_b(q_b)
geometry_completeness_i = G_i * [sum_g(C_g) - N_i * sum_b(q_b * Cbar_b)]
fully_weighted_i = F_i * geometry_completeness_i
```

The signal is now a per-galaxy sum, not a pooled mean weight multiplied by
each cluster histogram. The background assumes the offset-bin mean
completeness weights represent continuum galaxies for each cluster. This is
an explicit approximation, requiring validation against redshift-dependent
continuum/selection changes; LF scaling alone does not validate it. The same
continuum shape is used in all three stages. Geometry and LF multiply both
signal and background for each cluster. These recomputed values need not
equal previously stored richnesses, which are copied as `stored_*` columns
for auditing when present. No existing scientific richness column is replaced.

No negative continuum or richness is clipped. Nonpositive results are saved
and counted in the audit, but excluded from all logarithmic panels using a
common three-stage mask. Clusters without selected candidates are counted in
the audit but cannot be assigned richness from this galaxy-only parent table.
Invalid weights, varying cluster-level weights and inconsistent TOTAL_WEIGHT
cause explicit errors rather than silently becoming unit weights.

## Plot conventions

The relation is lambda_RM on x, lambda_spec on y. Stage panels share axes;
redshift colors are crimson, darkorange and royalblue as in the archived
Scaling_Relations_Fit notebook. Points overlay the scatter with no connecting
lines. Default RM edges are eight logarithmic bins from 20 to 100; edit
RM_BINS in the notebook. MIN_COUNT defaults to 20. Means are geometric;
SEM = sample_std(log10 richness)/sqrt(N) is transformed into asymmetric
linear error bars. SEM excludes correlated shared-continuum uncertainty and
is not individual richness uncertainty or intrinsic scatter.

The archived residual notebook actually shows conditional distributions.
The new residual notebook shows those distributions, plus the explicit
offset log10(lambda_spec/lambda_RM) from equality in dex, not from a fitted
relation. This reverses the sign of the earlier log10(lambda_RM/lambda_spec)
diagnostic. No fitted Gaussian, lognormal, power-law or MCMC curves are used.
Histograms are normalized separately per redshift/stage/RM cell, using common
edges across stages in each RM bin. RM bins use lower-inclusive, upper-exclusive
edges. BCG redshift bins are [0.10,0.18), [0.18,0.24), [0.24,0.35).

Outputs are PNG/PDF files under `plots/richness_weighting/`, excluded from Git.
