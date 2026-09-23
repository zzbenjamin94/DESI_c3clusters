"""
Compatibility wrapper for older workflows.

The end-to-end catalog builder now lives in ``projection_match_catalogs.py`` and
already applies LF weights plus spectroscopic-richness columns.  Use that script
for new runs.  This wrapper is kept so old commands fail gently with a useful
message instead of silently producing a stale intermediate product.
"""

from __future__ import annotations


def main() -> int:
    msg = (
        "LF weighting is now integrated into make_catalogs/projection_match_catalogs.py.\n"
        "Run:\n\n"
        "    python make_catalogs/projection_match_catalogs.py\n\n"
        "or with MPI on NERSC using srun."
    )
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
