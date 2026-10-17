import unittest
import numpy as np
from richness_relation.fit_reverse_richness_relation import fit_posterior


class ReverseRichnessTests(unittest.TestCase):
    def test_recovers_simulated_relation(self):
        rng = np.random.default_rng(12)
        rm = np.exp(rng.uniform(np.log(20), np.log(100), 5000))
        spec = np.exp(4.5 + .7*np.log(rm/40) + rng.normal(0, .3, len(rm)))
        samples = fit_posterior(rm, spec)
        np.testing.assert_allclose(np.median(samples, axis=0), [4.5, .7, .3], atol=.025)
        np.testing.assert_array_equal(samples, fit_posterior(rm, spec))

    def test_rejects_degenerate_predictor(self):
        with self.assertRaises(ValueError):
            fit_posterior(np.full(10, 40.), np.arange(1, 11.))


if __name__ == '__main__':
    unittest.main()
