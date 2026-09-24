"""Read and validate saved cluster richnesses; never calculate weights or continuum."""

from pathlib import Path
import numpy as np

from richness_relation.prepare_conditional_richness import read_catalog
from tools.richness_schema import STAGES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / 'catalogs/bgs_clus_RM_gal_matched_with_weights.fits'


def cluster_rows(table):
    required = ['ID', 'Z_SPEC_central', 'LAMBDA', 'LF_WEIGHT', *STAGES]
    missing = set(required) - set(table.colnames)
    if missing:
        raise KeyError(f'Missing {sorted(missing)}. Regenerate the matched catalog with '
                       'make_catalogs/projection_match_catalogs.py; plotting does not compute richness.')
    if np.any(np.ma.getmaskarray(table['ID'])):
        raise ValueError('Masked cluster IDs are not allowed')
    ids, first, inverse = np.unique(np.asarray(table['ID']).astype(str), return_index=True, return_inverse=True)
    for key in required[1:]:
        values = np.ma.asarray(table[key], dtype=float).filled(np.nan)
        if not np.all(np.isclose(values, values[first][inverse], rtol=1e-10, atol=1e-12, equal_nan=True)):
            raise ValueError(f'{key} varies within cluster IDs')
    out = table[first][required].copy()
    for key in required[1:]:
        out[key] = np.ma.asarray(out[key], dtype=float).filled(np.nan)
    if not np.allclose(out[STAGES[2]], out['LF_WEIGHT'] * out[STAGES[1]],
                       rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError('Saved LF richness does not equal LF_WEIGHT times geometry+completeness richness')
    return out


def load_stages(path=DEFAULT_CATALOG):
    table = read_catalog(Path(path))
    clusters = cluster_rows(table)
    return clusters, {'input_path': str(Path(path).resolve()), 'matched_rows': len(table),
                      'clusters': len(clusters), 'catalog_metadata': dict(table.meta),
                      'note': 'Only reading saved columns; no richness or continuum calculation.'}
