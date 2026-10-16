# Saved environment versus richness diagnostics

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
