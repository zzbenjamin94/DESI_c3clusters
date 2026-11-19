#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --job-name=environment-mpi-test
#SBATCH --output=environment-mpi-test-%j.log

set -exo pipefail
started=$SECONDS
trap 'status=$?; echo "MPI test: elapsed=$((SECONDS - started))s, exit=$status"' EXIT
echo "Starting on $(hostname) at $(date)"
cd /global/homes/z/zzhang13/DESI/Projection
source /global/homes/z/zzhang13/DESI/Projection/setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${CONDA_PREFIX:?Conda environment was not activated}/bin/python"
echo "Python: $PYTHON"
command -v srun
"$PYTHON" -c "import numpy, scipy, astropy; from mpi4py import MPI; print(MPI.Get_library_version())"
srun --export=ALL -n 4 -c 1 "$PYTHON" -u \
  local_overdensity/run_environment_mpi_test.py \
  --config local_overdensity/environment_config.example.json \
  --max-clusters 6 --seed 42 --compare-serial \
  --output "local_overdensity/dr9_outputs/section4_mpi_test/job_${SLURM_JOB_ID}"
