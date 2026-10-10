# Section 4 environment measurements

New code is isolated from the existing overdensity and richness pipelines. Input catalogs are never overwritten. Center on RA_central, DEC_central, Z_SPEC_central, with one validated row per ID. Select cluster centers at 0.1 <= z < 0.35. Use Planck18 and **comoving h^-1 Mpc** throughout. Signal: [1.5,5); background: [5,10). Spectroscopic line-of-sight separation is chi(z_g)-chi(z_cluster), not a photometric width.

## Required decisions before a production run

Copy `environment_config.example.json` to `environment_config.json` and fill nulls. Nothing chooses the Gaussian width, cylinder depth, magnitude limit, quality cuts or LF calibration for you. `input_selection_confirmed` is an explicit acknowledgement of the input contracts below, not a substitute for checking them. The example is intentionally not runnable as science data.

Dependencies: numpy, scipy, astropy, pandas, matplotlib; use your existing NERSC mpi4py stack. Notebook additionally needs IPython/Jupyter. No specutils or MCMC is used here.

## DR9 input contract

- Sweeps are discovered only directly in the supplied north/south **9.0** directories; no recursive search over other releases. Each file is read once per fresh run, processing both filters and background together in chunks.
- Select non-PSF sources, positive finite GRZ flux inverse variances, a specified extinction-corrected r-band limit, and configured MASKBITS rejection. Optional g-r upper cut. Missing required columns cause errors.
- The actual photo-z product has **not yet been supplied**. Its adapter must produce one NPZ per sweep with the same stem in the corresponding photoz_dirs entry. Arrays: RELEASE, BRICKID, OBJID, PRIMARY (boolean), SYS_WEIGHT, z_grid and pdf (Ngal,Nz). Join by the three ID columns, not row order. Include every selected sweep object, even when PRIMARY=False. All PDFs must be normalized to unity over their declared grid; probability outside the grid is assumed zero. Missing PDFs abort the run. No Gaussian PDF is invented from point estimates.
- PRIMARY must resolve the union of north and south photometry, including duplicate ownership across files. BRICK_PRIMARY alone is not guaranteed to achieve that. Upstream ownership verification remains required; this engine detects duplicate sidecar keys within each file, not duplicate sky objects across unrelated IDs/files.
- `lf_curve`: CSV with `z,LF_WEIGHT`, calibrated for the DR9 magnitude/quality sample and desired luminosity threshold. It must span the PDF grid; no extrapolation, no reuse of BGS LF by default. Ones can be supplied only for an explicitly uncorrected validation experiment. SYS_WEIGHT similarly must be explicitly supplied, even if unity.
- PDF weight is integral p(z) LF_WEIGHT(z) dz over cluster z +/- 0.05 (absolute redshift, not normalized offset). Gaussian is a transverse radial window only.

## Coverage randoms: required independent inputs

Raw survey randoms generally need calibration/preparation before use here. They must represent the same resolved footprint, vetoes, depth eligibility and radial support as the selected data. Empty galaxy pixels are NOT used as a mask, and sweep bounding boxes are NOT treated as coverage.

- DR9 random FITS: RA, DEC, and AREA_DEG2. Each point carries its angular quadrature area, e.g. 1/rho_total for uniform random density rho_total per deg^2 **summed over all random realizations**. Do not give each seed the full survey area: that multiplies coverage by the number of seeds. Variable-density randoms need corresponding local inverse sampling densities.
- BGS random FITS: RA, DEC, Z, VOLUME_HMPC3. Each weight is inverse local random sampling density in comoving (h^-1 Mpc)^3, with normalization across all random files. A random Z distribution alone does not specify this normalization. Do not substitute an angular random catalog for 3D quadrature.
- Measures must be positive finite. Gaussian effective area uses the same radial kernel as counts. Coverage is the random estimate divided by analytic full area/volume; it may exceed one through random noise. It is not silently clipped, and no 0.8 cut is imposed. Test random-density convergence and inspect sparse support.

## DR2 BGS contract

Use the full tracer catalog, not the cluster-aperture matched table. Specify actual redshift column and a boolean eligible column (or a categorical quality column plus allowed values); a derived boolean column should encode any multi-column spectroscopic/imaging selection. It is NOT sufficient to assume PROB_OBS establishes redshift quality. Retain finite 0 < PROB_OBS <= 1, finite positive redshift, and the quality selection. Invalid rows are excluded before measurement. The input tracer catalog must have unique targets across data files/chunks.

ENV_WEIGHT = LF_WEIGHT(z_g)/PROB_OBS, optionally multiplied by an explicitly specified additional weight. This does not claim that PROB_OBS includes redshift-success corrections. BGS LF CSV has z,LF_WEIGHT. `nbar_curve` CSV has z,NBAR, with **weighted** mean tracer density in h^3 Mpc^-3 for the same selection and weights. It supports the Section 4 global E = n/nbar-1; the local-background E is also saved separately. Both use the configured cylinder half-length. Random coverage corrects density at the aperture level, not with each cluster's inner-aperture geometric weight.

## Run and restart

From the repository root, after configuring and preparing the required products:

```bash
python local_overdensity/measure_dr9_environment_mpi.py --config local_overdensity/environment_config.json --dry-run
python local_overdensity/measure_bgs_environment_mpi.py --config local_overdensity/environment_config.json --dry-run
# Within an allocated CPU job, after sourcing setup_nersc_mpi_myEnv_v39.sh:
srun -n 8 -c 1 python -u local_overdensity/measure_dr9_environment_mpi.py --config local_overdensity/environment_config.json --mpi --resume
srun -n 8 -c 1 python -u local_overdensity/measure_bgs_environment_mpi.py --config local_overdensity/environment_config.json --mpi --resume
```

Use a SMALL explicitly selected file manifest and a separate output directory first. A small-file run does not establish complete coverage. No automatic max-files option hides an incomplete footprint. Dry-run validates configuration and file availability, not every row or PDF. Runtime is not yet benchmarked on NERSC.

After a successful small test, the batch template can be submitted with `sbatch --export=ALL,ENV_TRACER=dr9 local_overdensity/run_environment.slurm` (or `ENV_TRACER=bgs`). Its two-hour walltime is only an initial budget. The two tracer jobs write different output subdirectories and can run separately. Run synthetic tests with `python -m unittest discover -s tests -p test_environment_measurements.py -v`.

Work is distributed as tasks[rank::size]. DR9 tasks are whole sweeps (chunked internally); BGS and random tasks are FITS row chunks, so a single BGS file can use multiple ranks. MPI workers write independent atomic checkpoints; rank 0 sums checkpoints only after all workers finish. **Never average per-file ratios.** Restarts validate configuration, input size/mtime, code hashes and checkpoint shape. This is not a cryptographic checksum of large input catalogs. Changing rank count is allowed; changing chunk size/configuration requires a new output directory. Never change inputs during a run.

Outputs: `output_dir/dr9/` and `output_dir/bgs/`, each containing environment.fits, manifest.json, summary.json, checkpoints/. A RUNNING.lock prevents concurrent writes. After a timeout/crash, confirm no job remains before manually removing a stale lock and resuming. Normal exceptions abort MPI rather than leave workers waiting. Environment FITS is atomically replaced only after every checkpoint is present; summary.json is the completion report. Keep outputs out of Git.

## Statistics and limitations

Counts, sum(w), sum(w^2), effective measure and random support counts are retained per cluster/filter, followed by density, excess, relative local E and validity flags. Zero background yields NaN relative E, not zero. BGS additionally saves E_global relative to supplied nbar. Densities are h^2 Mpc^-2 (DR9) or h^3 Mpc^-3 (BGS), not deg^-2. Gaussian counts are weighted, not literal integer galaxy counts.

Conditional Poisson variance of weighted counts is sum(w^2 K^2); excess variance adds signal and disjoint-background terms after dividing by squared effective measures. These errors omit clustering covariance, random quadrature noise, PDF uncertainty, LF uncertainty and overlap between different cluster apertures. They are NOT final total uncertainties. Shared background also correlates filters; no independent-filter assumption is made for the empirical covariance plots.

The notebook plots normalized filters, count/coverage distributions, excess versus redshift/richness, cluster sky coverage, paired tracer environments and their empirical covariance. It saves a coarse sky-block jackknife covariance of the **sample mean**, clearly distinct from per-cluster environment covariance; test spatial patch size/occupancy before using it. There are no likelihood fits or calibrated theory covariance calculations in this step.
