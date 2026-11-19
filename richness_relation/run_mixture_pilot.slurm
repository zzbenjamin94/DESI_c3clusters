#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --job-name=richness-mixture_pilot
#SBATCH --output=richness-mixture_pilot-%j.log

set -eo pipefail
cd /global/homes/z/zzhang13/DESI/Projection
source /global/homes/z/zzhang13/DESI/Projection/setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Walltime is a resource budget, not a convergence or runtime guarantee.
# Never submit concurrent jobs writing to this same output directory.
started=$SECONDS
trap 'status=$?; echo "mixture_pilot: elapsed=$((SECONDS - started))s, exit=$status"' EXIT

srun -n 8 -c 1 python -u richness_relation/fit_conditional_richness_mpi.py \
  --mpi --model mixture --scope combined \
  --data catalogs/conditional_richness/overnight_sample.npz \
  --output catalogs/conditional_richness/mixture_pilot_1000 \
  --walkers 48 --steps 1000 --burn 50 --ensembles 1 \
  --ppc-draws 20 --save-draws 100 --seed 42 --resume
