#!/usr/bin/env bash
# Run with bash; do not source. Never modifies an existing environment.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v conda >/dev/null || { echo 'Load Gravity conda first.' >&2; exit 1; }
[[ "$(uname -s)" == Linux ]] || { echo 'This installer targets Gravity/Linux.' >&2; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create --file "$HERE/environment.yml"
conda activate desi-gravity
python -m pip check
python "$HERE/check.py"
python -m ipykernel install --user --name desi-gravity --display-name 'Python (DESI Gravity)'
echo 'Core environment installed. MPI is a separate step; see README.md.'
