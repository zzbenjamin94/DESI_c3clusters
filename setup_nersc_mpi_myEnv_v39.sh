#!/usr/bin/env bash

# Source this file before running mpi4py jobs on NERSC Perlmutter:
#
#   source /global/homes/z/zzhang13/DESI/Projection/setup_nersc_mpi_myEnv_v39.sh
#
# Then test:
#
#   python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
#
# This intentionally lives outside .bashrc so it only affects sessions where
# you explicitly want the Cray MPI Python stack.

# Keep the executable available after module unload removes Conda shell functions.
CONDA_BASE=/global/common/software/nersc/pe/conda/26.1.0/Miniforge3-25.11.0-1
if [ ! -x "$CONDA_BASE/condabin/conda" ]; then
    echo "Conda executable not found: $CONDA_BASE/condabin/conda" >&2
    return 1
fi
export PATH="$CONDA_BASE/condabin:$PATH"

module purge
module load PrgEnv-gnu
module load cray-mpich
module load cudatoolkit
module load python

conda activate myEnv_v39
unset MPI4PY_LIBMPI

echo "Loaded NERSC MPI stack for conda env: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
