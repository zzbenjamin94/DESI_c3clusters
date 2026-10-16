import unittest
import numpy as np
from local_overdensity.plot_environment_richness import binned_means, correlation


class SummaryTests(unittest.TestCase):
    def test_bins_cover_unique_sample(self):
        x = np.arange(140.) + 1
        bins = binned_means(x, x, 7, 20)
        self.assertEqual(len(bins), 7)
        self.assertEqual(sum(b['N'] for b in bins), 140)
        self.assertAlmostEqual(bins[0]['sem'], np.std(x[:20], ddof=1)/np.sqrt(20))

    def test_constant_and_small_samples(self):
        self.assertEqual(binned_means(np.arange(5.), np.arange(5.), 7, 20), [])
        self.assertTrue(np.isnan(correlation(np.ones(10), np.arange(10), 20, np.random.default_rng(1))[0]))

    def test_monotonic_bootstrap(self):
        x = np.arange(50.)
        rho, _, lo, hi, n = correlation(x, -x, 20, np.random.default_rng(1))
        self.assertAlmostEqual(rho, -1)
        self.assertAlmostEqual(lo, -1)
        self.assertAlmostEqual(hi, -1)
        self.assertEqual(n, 20)
