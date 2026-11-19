import unittest
import numpy as np
from astropy.table import Table
from local_overdensity.environment_accumulators import select_bgs, Accumulator


class BGSWeightTests(unittest.TestCase):
    def test_warning_exclusion(self):
        table = Table({'PROB_OBS': [1.] * 5, 'Z': [.2, .2, 999999., .2, .2],
                       'ZWARN': [0, 99, 0, 999999, 4]})
        c = {'bgs': {'quality_column': 'ZWARN', 'quality_values': [99, 999999],
                     'quality_mode': 'exclude'}}
        np.testing.assert_array_equal(select_bgs(table, c)['ZWARN'], [0, 4])

    def test_angular_random_volume_without_z(self):
        clusters = Table({'RA_central': [10.], 'DEC_central': [0.], 'Z_SPEC_central': [.2]})
        c = {'gaussian_width_hmpc': 2.86, 'bgs': {'pi_half_hmpc': 60.,
             'random_mode': 'angular', 'random_measure_column': 'AREA_DEG2'}}
        acc = Accumulator(clusters, c, 'bgs')
        randoms = Table({'RA': 10 + np.rad2deg(np.array([3., 7.])/acc.dm[0]),
                         'DEC': [0., 0.], 'AREA_DEG2': [1/45000., 1/45000.]})
        acc.add(randoms, random=True)
        expected = (np.pi/180*acc.dm[0])**2 * 120/45000
        np.testing.assert_allclose(acc.stats[0, [0, 2], 3], expected)

    def test_completeness_only_and_invalid_rows(self):
        table = Table({
            'PROB_OBS': [.5, 1., 0., np.nan, 1.1, .5],
            'Z': [.2] * 6,
            'GOOD': [True, True, True, True, True, False],
            'LF_WEIGHT': [10.] * 6,
            'WEIGHT_ZFAIL': [3.] * 6,
        })
        config = {'bgs': {'quality_column': 'GOOD', 'quality_values': [True],
                          'lf_curve': '/does/not/exist.csv',
                          'additional_weight_column': 'WEIGHT_ZFAIL'}}
        result = select_bgs(table, config)
        np.testing.assert_allclose(result['ENV_WEIGHT'], [2., 1.])


if __name__ == '__main__':
    unittest.main()
