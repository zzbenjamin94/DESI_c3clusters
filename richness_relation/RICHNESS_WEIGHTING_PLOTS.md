# Three-stage, data-only richness plots

## Run on NERSC

From `/global/homes/z/zzhang13/DESI/Projection`, activate your normal analysis
environment with `source setup_nersc_mpi_myEnv_v39.sh`. Inside a CPU allocation,
generate the matched catalogs (including all richness stages):

```bash
srun -n 8 -c 1 python make_catalogs/projection_match_catalogs.py
```

This rebuilds the configured matched FITS/pickle outputs. Back up existing
catalogs before regeneration if you want to retain previous versions.
After catalog generation finishes, plot using:

```bash
python richness_relation/plot_richness_weighting_comparison.py
python richness_relation/plot_spectroscopic_richness_residuals.py
```

Or run the two correspondingly named notebooks. The notebooks read only stored
columns, deduplicate by cluster ID, validate repeated values, and plot. MPI and
MCMC are not needed for plotting. Catalog generation needs the existing matching
dependencies and specutils; plotting needs numpy, astropy and matplotlib.

Input is `catalogs/bgs_clus_RM_gal_matched_with_weights.fits`. There is no
intermediate ECSV or preparation script. Catalog generation writes a companion
`bgs_clus_RM_gal_matched_with_weights_richness_audit.json` containing the fitted
continuum, settings and selection counts. Metadata CONTDEG/CONTWIN records the
polynomial order and smoothing window in FITS/pickle. All broad parent rows
are preserved; RICHNESS_ANALYSIS_ROW marks selected candidates, and
RICHNESS_AVAILABLE marks clusters with computed richness. Clusters without
selected candidates have NaN richness, not fabricated zeros.

Saved continuum-subtracted stage columns are `lambda_spec_unweighted`,
`lambda_spec_geo_comp`, `lambda_spec_geo_comp_lf`. The matching projected
columns are `lambda_spec_proj_unweighted`, `lambda_spec_proj_geo_comp`,
`lambda_spec_proj_geo_comp_lf`. `continuum_unweighted`, `continuum_geo_comp`
and `continuum_geo_comp_lf` store the subtracted counts. For compatibility, `lambda_spec_noproj` and
`lambda_spec_noproj_weighted` alias the first and last continuum-subtracted
stages; `lambda_spec_proj` and `lambda_spec_proj_weighted` alias their projected
counterparts. These values repeat on every galaxy row in each cluster.

The two notebooks replace the data-only work formerly combined in
`plot_conditional_richness.ipynb`. Existing conditional-model fitting and the
`plot_conditional_richness.py` saved-fit visualizer remain unchanged.

## Estimator and important assumptions

### Diagnose invalid weights first

```bash
python make_catalogs/diagnose_richness_weights.py
```

This does not require specutils, fit a continuum, or overwrite the catalog or
existing richness products. A timestamped folder under
`catalogs/weight_diagnostics/` contains `summary.json`,
`flagged_rows.ecsv`, and `affected_clusters.ecsv`. The summary distinguishes
zero, negative, NaN/masked, and infinite weights, checks probability validity,
and counts affected clusters. The row table includes original zero-based row
indices, weights, probabilities, redshifts and any available quality columns.
Quality flags are reported without assuming a catalog-specific quality cut.
Invalid stored weights with valid probabilities are identified for investigation,
not automatically repaired. Counts refer to matched rows, not unique TARGETIDs.
The catalog generator also writes this report and stops before saving when invalid
selected weights/probabilities are found. No rows are silently excluded.

For cluster i, N_i is the number of selected candidate galaxies, C_g is its
galaxy's COMP_WEIGHT, G_i is GEOMETRIC_WEIGHT, and F_i is LF_WEIGHT. Candidate
selection uses the shared science cuts: BCG z in [0.10,0.35), BGS z in
[0.05,0.40), normalized offset (z_BGS-z_BCG)/(1+z_BCG) in [-0.05,0.05].
The aperture is PROJECTED_APERTURE_HMPC in the matcher (currently 1.5 Mpc/h).
Science cuts affect richness only; they do not trim the broad parent catalog.

The unweighted pooled offset PDF is fitted with a specutils
median-window-5, degree-6 Chebyshev continuum, excluding [-0.02,0.02].
These defaults are CONTINUUM_DEGREE and CONTINUUM_MEDIAN_WINDOW in
`tools/richness_schema.py` and are recorded in the output JSON audit.
Both `make_catalogs/spectroscopic_richness.py` and the older shared
`tools/projection_functions.py` continuum function read these defaults.
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
for auditing when present. Regenerating the catalog updates its richness columns.

No negative continuum or richness is clipped. Nonpositive results are saved
and counted in the audit, but excluded from all logarithmic panels using a
common three-stage mask. Clusters without selected candidates are counted in
the audit and retained with NaN richness in this galaxy-level parent table.
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
