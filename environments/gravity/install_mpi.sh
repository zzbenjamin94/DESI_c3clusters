#!/usr/bin/env bash
set -eo pipefail
[[ "${CONDA_DEFAULT_ENV:-}" == desi-gravity ]] || {
    echo 'Activate desi-gravity first.' >&2; exit 1;
}
command -v mpicc >/dev/null || {
    echo 'Load the approved Gravity MPI/compiler module first.' >&2; exit 1;
}
export MPI4PY_BUILD_MPICC="$(command -v mpicc)"
echo "Building against $MPI4PY_BUILD_MPICC"
python -m pip install --no-cache-dir --no-binary=mpi4py --force-reinstall 'mpi4py==4.1.2'
python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'
python -m pip check
echo 'Next: test two ranks inside a PBS allocation using this same MPI module.'
