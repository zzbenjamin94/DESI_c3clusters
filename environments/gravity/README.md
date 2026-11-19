# Gravity environment

## Install the DESI baseline

From the repository root on Gravity, after loading its conda module:

```bash
bash environments/gravity/install.sh
conda activate desi-gravity
```

Choose **Python (DESI Gravity)** in Jupyter. The installer refuses to replace an
existing environment. If installation stops, inspect the error before deleting
or recreating anything. No catalogs or existing environments are modified.

## MPI (after confirming Gravity's module names)

Load the site-approved MPI/compiler module and activate `desi-gravity`, then:

```bash
bash environments/gravity/install_mpi.sh
```

Use the same MPI module in every batch job. Within a PBS allocation, use its
matching launcher to test, for example if the site module provides `mpiexec`:

```bash
mpiexec -n 2 "$CONDA_PREFIX/bin/python" -c 'from mpi4py import MPI; c=MPI.COMM_WORLD; assert c.size == 2; assert c.allreduce(1) == 2; print(c.rank, MPI.Get_library_version())'
```

Do not source `setup_nersc_mpi_myEnv_v39.sh` on Gravity. Its Cray modules are
NERSC-specific. Existing Slurm launchers and catalog paths also require a
separate Gravity/PBS adaptation; installing Python does not migrate those jobs.

## Provenance and scope

Based on the NERSC exports saved 2026-10-16 in the local backup:
`NERSC_backup/zzhang13/DESI/Projection/environment_backup/`.
The original files remain unchanged. `environment-full.yml` and `pip-freeze.txt`
provide package versions; `environment-portable.yml` alone is incomplete.
The exports contain inaccessible conda build paths and NERSC MPI libraries,
so neither the raw pip freeze nor Linux explicit file is installed directly.

This is a curated Python 3.11 baseline for richness fitting, environment
estimation, plotting and Jupyter, not a byte-for-byte recreation of the large
NERSC environment. Python is upgraded from NERSC's 3.9 to 3.11 at the user's
request. Core scientific versions follow the export; Python patch, Jupyter
and transitive dependencies are resolved on Gravity. This deliberately targets
3.11 rather than the newest Python release to limit scientific-package changes.
Specialist packages such as pyccl, dsigma, pymaster, pymultinest, specutils,
linetools, GCR and JAX are not installed by this baseline. Add them separately
when resuming those projects, after checking their native-library requirements.

After successful installation, record the actually resolved environment:

```bash
conda env export > "$HOME/desi-gravity-resolved.yml"
python -m pip freeze > "$HOME/desi-gravity-pip.txt"
module list 2> "$HOME/desi-gravity-modules.txt"
```

MPI installation follows the official source-build guidance:
https://mpi4py.readthedocs.io/en/stable/install.html#building-from-sources

Validation status: shell/Python syntax can be checked locally. Linux dependency
resolution, native MPI linkage and multi-rank execution must be tested on Gravity.
