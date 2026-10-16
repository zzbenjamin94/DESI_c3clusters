#!/bin/bash
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --nodes=2
#SBATCH --ntasks=36
#SBATCH --ntasks-per-node=18
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --job-name=richness-cells
#SBATCH --output=richness-cells-%j.log

set -exo pipefail
started=$SECONDS
trap 'status=$?; echo "cell fits elapsed=$((SECONDS-started))s exit=$status"' EXIT
cd /global/homes/z/zzhang13/DESI/Projection
source /global/homes/z/zzhang13/DESI/Projection/setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON="${CONDA_PREFIX:?Conda environment missing}/bin/python"
"$PYTHON" -c 'import emcee, h5py, scipy; from mpi4py import MPI; print(MPI.Get_library_version())'
OUT="${CELL_OUTPUT:-catalogs/conditional_richness/cell_fits_3x3}"
srun --export=ALL -n "$SLURM_NTASKS" -c 1 "$PYTHON" -u \
  richness_relation/fit_richness_cells_mpi.py --mpi --model both \
  --data catalogs/conditional_richness/overnight_sample.npz \
  --output "$OUT" --steps 5000 --burn 1000 --walkers 48 --ensembles 2 --resume
# Save quick figures first, then calibrate goodness of fit with refitting.
srun --export=ALL --exact -N 1 -n 1 -c 1 "$PYTHON" -u richness_relation/plot_richness_cells.py \
  --fits "$OUT" --output "plots/conditional_richness/cells_${SLURM_JOB_ID}_quick" --gof-bootstrap 0
# The bootstrap pool is local to one node; do not launch duplicate writers.
srun --export=ALL --exact -N 1 -n 1 -c 18 "$PYTHON" -u richness_relation/plot_richness_cells.py \
  --fits "$OUT" --output "plots/conditional_richness/cells_${SLURM_JOB_ID}" \
  --gof-bootstrap 200 --workers 18
