#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --job-name=bgs-environment
#SBATCH --output=bgs-environment-%j.log

set -exo pipefail
started=$SECONDS
trap 'status=$?; echo "BGS environment: elapsed=$((SECONDS-started))s exit=$status"' EXIT
cd /global/homes/z/zzhang13/DESI/Projection
source /global/homes/z/zzhang13/DESI/Projection/setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${CONDA_PREFIX:?Conda environment missing}/bin/python"
CONFIG="${ENV_CONFIG:-local_overdensity/environment_config.example.json}"
"$PYTHON" -c 'import numpy, scipy, astropy; from mpi4py import MPI; print(MPI.Get_library_version())'
# Fail before launching workers if configuration/selection remains unconfirmed.
"$PYTHON" local_overdensity/measure_bgs_environment_mpi.py --config "$CONFIG" --dry-run
srun --export=ALL -n "$SLURM_NTASKS" -c 1 "$PYTHON" -u \
  local_overdensity/measure_bgs_environment_mpi.py --config "$CONFIG" --mpi --resume
