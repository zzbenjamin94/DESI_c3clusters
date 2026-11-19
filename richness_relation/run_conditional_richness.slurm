#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --job-name=conditional-richness
#SBATCH --output=conditional-richness-%j.log

set -eo pipefail
cd /global/homes/z/zzhang13/DESI/Projection
source /global/homes/z/zzhang13/DESI/Projection/setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
started=$SECONDS
trap 'echo "Elapsed seconds: $((SECONDS - started))"' EXIT

# Prepare sample.npz once before submitting. The walltime is an initial budget,
# not an estimate of scientific convergence. Change model to mixture when ready.
srun -n 8 -c 1 python richness_relation/fit_conditional_richness_mpi.py \
    --mpi --model lognormal --scope suite \
    --walkers 48 --steps 3000 --burn 750 --ensembles 2 --seed 42 --resume
