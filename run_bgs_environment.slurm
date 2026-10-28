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

set -eo pipefail
started=$SECONDS
trap 'status=$?; echo "BGS environment: elapsed=$((SECONDS-started))s exit=$status"' EXIT
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source "${ENV_SETUP:?Set ENV_SETUP to your absolute environment setup script path}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${PYTHON_BIN:-$(command -v python)}"
CONFIG="${ENV_CONFIG:-local_overdensity/environment_config.example.json}"
"$PYTHON" -c 'import numpy, scipy, astropy; from mpi4py import MPI; print(MPI.Get_library_version())'
env SLURM_NTASKS=1 "$PYTHON" local_overdensity/measure_bgs_environment_mpi.py --config "$CONFIG" --dry-run
srun --export=ALL -n "$SLURM_NTASKS" -c 1 "$PYTHON" -u local_overdensity/measure_bgs_environment_mpi.py --config "$CONFIG" --mpi --resume
