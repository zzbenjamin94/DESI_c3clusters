"""Focused tests; run with python -m unittest discover -s tests -p test_richness_convergence.py."""
import unittest
import numpy as np
from richness_relation.check_richness_convergence import classical_rhat


class ClassicalRhatTests(unittest.TestCase):
    def test_matching_and_shifted(self):
        x = np.random.default_rng(42).normal(size=(4, 10000, 2))
        self.assertTrue(np.all(classical_rhat(x) < 1.01))
        x[0, :, 0] += 3
        self.assertGreater(classical_rhat(x)[0], 1.1)

    def test_constant_unavailable(self):
        self.assertTrue(np.isnan(classical_rhat(np.ones((2, 100, 1)))).all())

    def test_hand_calculation(self):
        x = np.array([[[0.], [2.]], [[2.], [4.]]])
        np.testing.assert_allclose(classical_rhat(x), np.sqrt(1.5))

    def test_one_ensemble_rejected(self):
        with self.assertRaises(ValueError):
            classical_rhat(np.ones((1, 100, 1)))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            classical_rhat(np.full((2, 100, 1), np.nan))


if __name__ == "__main__":
    unittest.main()

