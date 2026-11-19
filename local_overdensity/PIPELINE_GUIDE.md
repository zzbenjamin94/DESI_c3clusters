# Local Environment Pipeline: Reading and Using the DR9 Results

This guide describes the existing `local_overdensity` pipeline and the downloaded
production results inspected on 2026-10-27. The project folder is named
`local_overdensity`, not `local_environments`. No measurement rerun is needed to
use the saved catalog on the local computer.

## 1. Start With the Production Output

The main result, relative to the repository root, is:

```text
local_overdensity/dr9_outputs/section4/dr9/
    environment.fits    # One row per cluster; richness and environment together
    manifest.json       # Configuration and input provenance for this run
    summary.json        # Completed tasks, selection counts, and warnings
    checkpoints/        # Per-task additive statistics, if retained
```

The inspected FITS table contains **7,391 clusters**. Its summary reports
**11,578 completed tasks** and **47,766,756 selected DR9 input rows**. The latter
is a tracer-selection count, not a count of galaxies assigned to clusters.

Do not substitute outputs under `section4_smoke` or `section4_mpi_test` for the
production catalog. Limited-file tests can have missing sky coverage and
incorrect full-survey normalization.

The saved `manifest.json`, rather than today's example configuration, is the
record of the settings used to generate this particular result. Its NERSC input
paths are provenance; they do not need to exist locally to read the FITS output.

## 2. Pipeline Architecture

```text
run_environment.slurm
  -> measure_dr9_environment_mpi.py
     -> environment_runner.run("dr9")
        -> load_config() and load_clusters()
        -> jobs_for(): enumerate data files and random chunks
        -> process(): read, select, and accumulate each task
           -> Accumulator.add()
        -> sum task checkpoints
        -> finalize(): densities, excesses, errors, validity flags
        -> environment.fits + summary.json
```

| File | Responsibility |
|---|---|
| `run_environment.slurm` | Allocate resources, activate the NERSC environment, limit numerical-library threads, and launch MPI. The current local launcher requests 4 nodes and 64 ranks; this is not proof of the historical run's allocation. |
| `environment_config.py` | Load and validate configuration. |
| `environment_catalogs.py` | Load cluster centers and catalog helpers. |
| `environment_runner.py` | Build tasks, distribute work, write checkpoints, and combine results. |
| `environment_dr9_native.py` | Pair sweeps with photo-z sidecars; implement ownership and Gaussian photo-z probabilities. |
| `environment_filters.py` | Define radial kernels and their full geometric integrals. |
| `environment_accumulators.py` | Accumulate counts, weights, squared weights, and random-derived effective areas. |
| `finalize_environment.py` | Turn accumulated sums into the final per-cluster measurements. |

### MPI and checkpointing

Rank `r` processes task indices `r, r + size, r + 2*size, ...`. Tasks partition
the input data, not the cluster sample: each task can contribute to many clusters.
The accumulator has shape `(number_of_clusters, 3, 5)`. Its three windows are
box, Gaussian, and background; its five statistics are:

1. Number of contributing galaxies.
2. Sum of kernel-weighted galaxy weights.
3. Sum of squared kernel-weighted galaxy weights.
4. Random-derived effective area.
5. Number of supporting random points.

These quantities are additive. Rank zero sums them across all tasks before
forming ratios or subtracting background. It does not average per-file densities.
Task checkpoints are written through temporary files and renamed into place.
`--resume` requires a matching run signature; the final aggregation also checks
checkpoint signatures and array shapes. `RUNNING.lock` guards against concurrent
writes to the same run directory. Final products replace previous products in
that output directory, so use a new output directory for a new measurement.

## 3. DR9 Selections and Filters

The production manifest specifies north and south DR9 `sweep/9.0` catalogs and
their `9.0-photo-z` sidecars. Matching row counts and identifiers
`RELEASE`, `BRICKID`, and `OBJID` are checked before attaching photo-z values.
Invalid photo-z entries such as `-99` must not enter the estimator.

The run uses an r-band magnitude limit of **20.5**, imaging-mask rejection value
**12290**, and **2500 random points per square degree per realization**. North
owns positions at Dec >= 32.375 degrees and Galactic latitude > 0; south owns the
complement. Imaging ownership and coverage selections avoid double counting.
No DR9 luminosity-function correction is enabled (`lf_curve: null`).

Distances use Planck18 and are comoving, in `h^-1 Mpc`:

- Signal box: `1.5 <= R < 5`.
- Truncated Gaussian: `exp[-R^2/(2*sigma_R^2)]` over the same signal annulus.
- Background box: `5 <= R < 10`.
- Saved Gaussian width: `sigma_R = 2.8644943786043604 h^-1 Mpc`.

The sky tree uses unit vectors and chord distances to find angular neighbors.
Angular separations are converted into projected comoving radii at each cluster's
redshift; photometric galaxies are not required to have the cluster's distance.

The photo-z weight is the probability in `z_cluster +/- 0.05`, using each
galaxy's Gaussian mean and standard deviation and normalizing its PDF on
nonnegative redshift. It is an integrated probability, **not** a Gaussian
similarity weight forced to equal one at zero redshift offset.

## 4. Effective Area and Environment Definitions

Randoms describe angular sampling and masked area, not additional galaxies.
Their angular quadrature measures are converted to comoving area at the cluster
redshift. Applying the same radial kernel gives its effective area, A_eff.
For the Gaussian filter this is a kernel-weighted area, not simply the annulus
area. The radial normalization cancels in the weighted-count/weighted-area ratio.

For each signal or background window:

```text
weighted_count = sum(kernel * photo-z weight * other enabled galaxy weights)
density        = weighted_count / effective_area
coverage       = effective_area / full_unmasked_kernel_area
excess         = signal_density - background_density
E_local        = excess / background_density
```

`density` and `excess` have units **h^2 Mpc^-2**; `E_local` is dimensionless.
Finite random sampling can produce coverage slightly above one.

Conditional Poisson errors are computed as:

```text
variance(density) = sum((kernel * galaxy_weight)^2) / effective_area^2
error(excess)    = sqrt(variance(signal_density) + variance(background_density))
```

These errors exclude galaxy clustering, random-area uncertainty, photo-z
uncertainty beyond the assigned weights, and uncertainty in other corrections.
They are not total scientific error bars.

## 5. Output Columns

| Column or suffix | Meaning |
|---|---|
| `ID` | Cluster identifier; use this for joins, not row order. |
| `RA_central`, `DEC_central`, `Z_SPEC_central` | BCG position and spectroscopic redshift. |
| `LAMBDA` | redMaPPer richness. |
| `lambda_spec_geo_comp_lf` | Saved fully weighted spectroscopic richness used by the correlation script. |
| `dr9_box_*` | Box-filter measurements. |
| `dr9_gaussian_*` | Truncated-Gaussian measurements. |
| `dr9_background_*` | Shared background-annulus measurements. |
| `*_N` | Contributing-galaxy count, not the weighted count. |
| `*_weighted_count`, `*_sum_w2` | Weight sum and squared-weight sum. |
| `*_effective_measure`, `*_N_random`, `*_coverage` | Effective area and random support. |
| `*_density`, `*_density_shot_err` | Density and conditional Poisson uncertainty. |
| `*_excess`, `*_excess_shot_err` | Background-subtracted density and shot uncertainty. |
| `*_E_local` | Dimensionless local overdensity. |
| `*_valid` | Finite excess and nonzero signal/background random support. |
| `*_E_local_valid` | Also requires positive background density. |

The validity flags do **not** enforce an 80% coverage threshold. Inspect coverage
separately and document any additional selection. Do not replace invalid values
with zero.

## 6. Read the Output in Jupyter

Use the local DESI Python kernel. This cell reads only saved outputs:

```python
from pathlib import Path
import json
import numpy as np
from astropy.table import Table

ROOT = Path(
    '/Users/zzbenjamin94/Documents/Codex/2026-05-11/'
    'github-plugin-github-openai-curated-can/DESI_c3clusters'
)
RUN = ROOT / 'local_overdensity/dr9_outputs/section4/dr9'
environment = Table.read(RUN / 'environment.fits')
summary = json.loads((RUN / 'summary.json').read_text())
manifest = json.loads((RUN / 'manifest.json').read_text())

assert summary['signature'] == manifest['signature']
assert len(np.unique(environment['ID'])) == len(environment)
print('Clusters:', len(environment))
print('Completed tasks:', summary['completed_tasks'])
print('Units:', environment.meta['DENSUNIT'])
display(environment[[
    'ID', 'Z_SPEC_central', 'LAMBDA', 'lambda_spec_geo_comp_lf',
    'dr9_box_excess', 'dr9_gaussian_excess',
    'dr9_box_E_local', 'dr9_gaussian_E_local',
]][:10])
```

## 7. Coverage and Measurement Diagnostics

`plot_environment_diagnostics.ipynb` calls `plot_environment_diagnostics.py`.
It shows radial windows, counts, coverage, sky positions, and excess versus
redshift/richness. This version does not compute richness-environment Spearman
coefficients. The notebook currently expects `environment_config.json`, which
is not present in this checkout; use the saved manifest and local output root
instead of its configuration-loading cell:

```python
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT))
from local_overdensity.plot_environment_diagnostics import plot_diagnostics

figures = plot_diagnostics(
    RUN.parent,
    manifest['configuration']['gaussian_width_hmpc'],
    ROOT / 'plots/section4_environment_local',
)
plt.show()
for fig in figures:
    plt.close(fig)
```

The function expects the parent directory containing `dr9/environment.fits`.
It skips BGS if its output is absent. Choose a fresh plot directory when retaining
earlier figures: plotting routines can overwrite files of the same name.

## 8. Richness Correlations and Binned Plots

The matching script is now available in this repository:

```text
/Users/zzbenjamin94/Documents/Codex/2026-05-11/github-plugin-github-openai-curated-can/DESI_c3clusters/local_overdensity/plot_environment_richness.py
```

Copied from `publish-analysis-20260924` together with its usage notes and tests;
the originals are preserved. The older `plot_local_overdensity_relations.py`
expects different columns and should not be run unchanged on this output.

The matching script produces four figures: box/Gaussian crossed with
excess/E_local. Each has redMaPPer and spectroscopic-richness panels, scatter
points, binned means with SEM, and Spearman coefficients. Its settings are:

- Redshift intervals `[0.10, 0.18)`, `[0.18, 0.24)`, `[0.24, 0.35)`.
- At most seven quantile bins per redshift interval, with at least 20 clusters
  retained per displayed bin.
- SEM = sample standard deviation / sqrt(N), in linear environment units.
- Spearman correlations use individual valid clusters, not binned means.
- 1,000 cluster-pair bootstrap resamples give 16th/84th percentile intervals.
- No extra coverage cut unless `--coverage-min` is specified.
- Spectroscopic richness uses `lambda_spec_noproj_weighted` if available,
  otherwise `lambda_spec_geo_comp_lf` (the column in this production file).

To run the script locally:

```bash
BASE='/Users/zzbenjamin94/Documents/Codex/2026-05-11/github-plugin-github-openai-curated-can'
REPO="$BASE/DESI_c3clusters"
PYTHON='/Users/zzbenjamin94/Documents/Codex/envs/desi-local/bin/python'
OUT="$REPO/plots/environment_richness_local_$(date +%Y%m%d_%H%M%S)"

"$PYTHON" "$REPO/local_overdensity/plot_environment_richness.py" \
  --input "$REPO/local_overdensity/dr9_outputs/section4/dr9/environment.fits" \
  --output "$OUT" \
  --bootstrap 1000 --seed 42 --max-bins 7 --min-count 20
```

The script's default root is now this repository. The explicit timestamped output
directory preserves existing plots. MPI and the original DR9 sweeps are not needed.

Existing downloaded results are already under `plots/environment_richness/`:

```text
dr9_box_excess_richness.png / .pdf
dr9_gaussian_excess_richness.png / .pdf
dr9_box_E_local_richness.png / .pdf
dr9_gaussian_E_local_richness.png / .pdf
spearman_summary.csv     # Correlations, bootstrap intervals, asymptotic p-values
binned_means_sem.csv     # Bin edges, counts, means, SEM
settings.json           # Input path, column choice, cuts, and bootstrap settings
```

Cluster-pair bootstrap intervals ignore spatial correlations. Small asymptotic
p-values and positive correlations alone do not establish projection-driven
richness inflation: mass, redshift, selection, and spatial covariance remain
possible contributors. Validate these before interpreting the results as an
environment-dependent conditional-richness effect.

## 9. Scope and Preservation

This guide documents the saved DR9 run. The BGS cylindrical path is separate;
do not assume its configuration or outputs are complete based on DR9 success.
Keep FITS results, their manifest, and summary together. Catalogs and output
plots are local data products, not intended for Git tracking. The Markdown
guide and analysis source code can be version-controlled independently.
