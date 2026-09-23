#!/usr/bin/env bash

# Source this file before running mpi4py jobs on NERSC Perlmutter:
#
#   source local_overdensity/setup_nersc_mpi_myEnv_v39.sh
#
# Then test:
#
#   python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
#
# This intentionally lives outside .bashrc so it only affects sessions where
# you explicitly want the Cray MPI Python stack.

module purge
module load PrgEnv-gnu
module load cray-mpich
module load cudatoolkit
module load python

conda activate myEnv_v39
unset MPI4PY_LIBMPI

echo "Loaded NERSC MPI stack for conda env: ${CONDA_DEFAULT_ENV}"
echo "Python: $(which python)"
