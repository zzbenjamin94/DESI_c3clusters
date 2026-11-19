#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=4:00:00
#SBATCH --job-name=richness-lognormal
#SBATCH --output=richness-lognormal-%j.log

set -exo pipefail
started=$SECONDS
trap 'status=$?; echo "lognormal: elapsed=$((SECONDS - started))s, exit=$status"' EXIT

echo "Starting on $(hostname) at $(date)"
cd /global/homes/z/zzhang13/DESI/Projection
source /global/homes/z/zzhang13/DESI/Projection/setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "CONDA_PREFIX=${CONDA_PREFIX:-unset}"
command -v python
command -v srun
PYTHON="${CONDA_PREFIX:?Conda environment was not activated}/bin/python"
"$PYTHON" -c "import numpy, emcee, schwimmbad, h5py; from mpi4py import MPI; print(MPI.Get_library_version())"

# Walltime is a resource budget, not a convergence or runtime guarantee.
# Never submit concurrent jobs writing to this same output directory.
srun --export=ALL -n 8 -c 1 "$PYTHON" -u richness_relation/fit_conditional_richness_mpi.py \
  --mpi --model lognormal --scope suite \
  --data catalogs/conditional_richness/overnight_sample.npz \
  --output catalogs/conditional_richness/lognormal_fits \
  --walkers 48 --steps 5000 --burn 1000 --ensembles 2 \
  --ppc-draws 200 --save-draws 1000 --seed 42 --resume
