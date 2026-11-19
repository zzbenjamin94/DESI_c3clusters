"""Update the matched FITS catalog to the BGS 0.2 Lstar reference, with backup."""
from pathlib import Path
import argparse
import json
import os
import shutil

import numpy as np
from astropy.io import fits
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]


def prepare_sample(output):
    """Retain the original fit IDs while replacing richness from the new catalog."""
    import hashlib
    source = ROOT / 'catalogs/conditional_richness/overnight_sample.npz'
    catalog = ROOT / 'catalogs/bgs_clus_RM_gal_matched_with_weights.fits'
    with np.load(source, allow_pickle=False) as archive:
        sample = {k: archive[k].copy() for k in archive.files}
    with fits.open(catalog) as hdus:
        assert hdus[1].header['LFREFMOD'] == 'BGS_0.2LSTAR'
        table = hdus[1].data
        ids, first = np.unique(table['ID'].astype(str), return_index=True)
        lookup = dict(zip(ids, first))
        rows = np.array([lookup[str(v)] for v in sample['ID']])
        assert np.allclose(sample['lambda_rm'], table['LAMBDA'][rows])
        assert np.allclose(sample['z'], table['Z_SPEC_central'][rows])
        factor = np.asarray(table['lambda_spec_noproj_weighted'][rows]) / sample['lambda_spec']
        assert np.allclose(factor, np.median(factor), rtol=1e-6)
        sample['lambda_spec'] = np.array(table['lambda_spec_noproj_weighted'][rows])
        sample['LF_WEIGHT'] = np.array(table['LF_WEIGHT'][rows])
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **sample)
    audit = json.loads(source.with_suffix('.json').read_text())
    audit.update(sample_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                 source_sample=str(source), updated_catalog=str(catalog),
                 lf_reference='BGS r-band 0.2 Lstar', lf_scale=float(np.median(factor)),
                 same_cluster_ids=True, reverse_equivalent_spec_min=float(20*np.median(factor)))
    output.with_suffix('.json').write_text(json.dumps(audit, indent=2))
    print('Saved', output, 'N=',len(sample['ID']), 'scale=',audit['lf_scale'],
          'reverse cut=',audit['reverse_equivalent_spec_min'])


def main():
    path = ROOT / 'catalogs/bgs_clus_RM_gal_matched_with_weights.fits'
    backup = path.with_name(path.stem + '_before_rm_adjusted_lf.fits')
    with fits.open(path, memmap=False) as hdus:
        if hdus[1].header.get('LFREFMOD') == 'BGS_0.2LSTAR':
            raise RuntimeError('Already updated; refusing to apply correction twice.')
        data = hdus[1].data
        ids, first, inverse = np.unique(data['ID'], return_index=True, return_inverse=True)
        star = np.asarray(data['lf_log10_L_star'][first], float)
        alpha = np.asarray(data['lf_alpha'][first], float)
        limits = np.asarray(data['logL_lim_lf'][first], float)
        old = np.asarray(data['LF_WEIGHT'], float).copy()
        def integral(x, a):
            return quad(lambda t: t**a * np.exp(-t), x, np.inf,
                        epsabs=1e-10, epsrel=1e-8)[0]
        new = np.array([integral(.2, a) / integral(10**(lim-s), a)
                        if np.isfinite(lim+s+a) else np.nan
                        for lim, s, a in zip(limits, star, alpha)])[inverse]
        ratio = new / old
        changed = ['lambda_spec_proj_geo_comp_lf', 'continuum_geo_comp_lf',
                   'lambda_spec_geo_comp_lf', 'lambda_spec_noproj_weighted',
                   'lambda_spec_proj_weighted']
        for col in changed:
            data[col] = np.asarray(data[col], float) * ratio
        data['LF_WEIGHT'] = new
        data['lf_observed_fraction'] = 1 / new
        data['logL_ref_lf'] = (star + np.log10(.2))[inverse]
        data['TOTAL_WEIGHT'] = data['COMP_WEIGHT'] * data['GEOMETRIC_WEIGHT'] * new
        assert np.allclose(data['lambda_spec_geo_comp_lf'],
                           data['lambda_spec_geo_comp'] * new, equal_nan=True)
        assert np.allclose(data['continuum_geo_comp_lf'],
                           data['continuum_geo_comp'] * new, equal_nan=True)
        analysis = (data['Z_SPEC_central'] >= .1) & (data['Z_SPEC_central'] < .35)
        report = dict(reference='BGS r-band 0.2 Lstar proxy', floor_at_one=False,
                      rows=len(data), clusters=len(ids), updated_columns=changed,
                      analysis_ratio_percentiles=np.nanpercentile(ratio[analysis], [0,50,100]).tolist(),
                      backup=str(backup), downstream_fits_need_regeneration=True)
        hdus[1].header['LFREFMOD'] = 'BGS_0.2LSTAR'
        hdus[1].header['LFFLOOR'] = False
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(path, backup)
        temporary = path.with_suffix('.tmp.fits')
        hdus.writeto(temporary, overwrite=False, checksum=True)
    with fits.open(temporary, checksum=True) as check:
        assert len(check[1].data) == report['rows']
        assert check[1].header['LFREFMOD'] == 'BGS_0.2LSTAR'
    os.replace(temporary, path)
    path.with_name('rm_adjusted_lf_update.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-sample', type=Path)
    args = parser.parse_args()
    if args.prepare_sample:
        prepare_sample(args.prepare_sample)
    else:
        main()
