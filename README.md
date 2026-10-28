# Minimal BGS cylindrical environment run

Clone only this branch (independent history, no large analysis history):
```bash
git clone --single-branch --branch shared_NERSC https://github.com/zzbenjamin94/DESI_c3clusters.git
cd DESI_c3clusters
```

The four-column cluster FITS is included. BGS data and randoms are read directly
from the shared NERSC paths in local_overdensity/environment_config.example.json.
Confirm access to these files and the data redshift column Z_not4clus. Set
output_dir to a writable scratch directory if desired. No DR9 files are needed.

Use your own NERSC-compatible Python/MPI environment with numpy, scipy, astropy
and mpi4py. Put the commands to load/activate it in a shell script. Do not replace
NERSC MPI with a generic MPI binary. Submit from the repository root:
```bash
export ENV_SETUP=/absolute/path/to/your/env_setup.sh
sbatch --account=desi run_bgs_environment.slurm
```
Replace the account if needed. The batch script sources your setup itself.
Allocation: 4 CPU nodes, 64 ranks, 3 hours; runtime is not guaranteed.
```bash
squeue -j JOBID
tail -f bgs-environment-JOBID.log
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
```

Return output_dir/bgs/environment.fits, summary.json, manifest.json, the exact
configuration and batch log. Keep checkpoints until validation. Resubmit the same
configuration to resume. Remove a stale RUNNING.lock only after verifying no job
still uses that output. Never reuse checkpoints with changed inputs/configuration.

## Estimator

BCG redshifts: 0.10 <= z < 0.35. Signal annulus 1.5--5, background annulus 5--10,
cylinder half-length 60, all comoving h^-1 Mpc, Planck18 cosmology. Galaxy weights
are 1/PROB_OBS only, no LF weighting. The inherited quality selection excludes
ZWARN 99 and 999999; it does NOT require ZWARN=0. Review for your intended sample.
Vetoed BGS full data and full randoms are used. Nominal random density is 2500
deg^-2 per realization. Angular random area is multiplied by the cylinder length
(120) for effective volume; no random redshifts are needed.

The main dimensionless statistic bgs_box_E_local is (signal density - background
density)/background density. Footprint/random-density calibration is an assumption;
shot errors exclude clustering and random uncertainty. Some unused DR9 helper
functions remain in shared modules to preserve the tested implementation.

Run unit tests with python -m pytest tests -q. Full NERSC execution cannot be
tested locally. This branch includes no richness chains, plots or LF tables.
