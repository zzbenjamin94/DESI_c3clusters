#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=14:00:00
#SBATCH --job-name=section4-environment
#SBATCH --output=section4-environment-%j.log

set -exo pipefail
started=$SECONDS
trap 'status=$?; echo "environment: elapsed=$((SECONDS - started))s, exit=$status"' EXIT
echo "Starting on $(hostname) at $(date)"
cd /global/homes/z/zzhang13/DESI/Projection
source /global/homes/z/zzhang13/DESI/Projection/setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${CONDA_PREFIX:?Conda environment was not activated}/bin/python"
echo "Python: $PYTHON"
command -v srun
"$PYTHON" -c "import numpy, scipy, astropy; from mpi4py import MPI; print(MPI.Get_library_version())"
: "${ENV_TRACER:?Submit with --export=ALL,ENV_TRACER=dr9 or ENV_TRACER=bgs}"
case "$ENV_TRACER" in dr9|bgs) ;; *) echo "Invalid ENV_TRACER"; exit 2 ;; esac
# Walltime is a starting budget, not a benchmark or completeness guarantee.
srun --export=ALL -n 32 -c 1 "$PYTHON" -u \
  "local_overdensity/measure_${ENV_TRACER}_environment_mpi.py" \
  --config local_overdensity/environment_config.example.json \
  --mpi --resume
