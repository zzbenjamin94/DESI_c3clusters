"""Synthetic Section 5 checks; no survey catalogs or MPI runtime required."""

import unittest

import numpy as np
from astropy.table import Table
from scipy.integrate import quad
from scipy.stats import kstest, lognorm, norm

from richness_relation.conditional_richness_models import (
    ModelConfig, log_probability, mu_rm, selected_cdf, selected_logpdf, selected_rvs,
)
from richness_relation.prepare_conditional_richness import prepare


class TestConditionalLikelihood(unittest.TestCase):
    def test_lognormal_jacobian_and_normalization(self):
        theta = [np.log(45), 0.9, np.log(0.3)]
        config = ModelConfig(rm_min=20, rm_max=100)
        value = selected_logpdf(theta, 50, 40, 0.24, config)
        expected = lognorm.logpdf(50, s=0.3, scale=45) - np.log(lognorm.cdf(100, s=0.3, scale=45) - lognorm.cdf(20, s=0.3, scale=45))
        self.assertAlmostEqual(float(value), expected, places=10)
        integral = quad(lambda r: np.exp(selected_logpdf(theta, r, 40, 0.24, config)), 20, 100)[0]
        self.assertAlmostEqual(integral, 1, places=7)

    def test_mixture_matches_direct_convolution(self):
        theta = [np.log(40), 1.0, np.log(5), 0.3, np.log(0.1)]
        config = ModelConfig("mixture", rm_min=0)
        r = 60
        numerator = 0.7 * norm.pdf(r, 40, 5) + 0.3 * quad(lambda d: norm.pdf(r, 40 + d, 5) * 0.1 * np.exp(-0.1 * d), 0, np.inf)[0]
        denominator = 0.7 * norm.sf(0, 40, 5) + 0.3 * quad(lambda d: norm.sf(0, 40 + d, 5) * 0.1 * np.exp(-0.1 * d), 0, np.inf)[0]
        self.assertAlmostEqual(float(np.exp(selected_logpdf(theta, r, 40, 0.24, config))), numerator / denominator, places=10)
        for upper in (100, np.inf):
            cfg = ModelConfig("mixture", rm_min=20, rm_max=upper)
            integral = quad(lambda v: np.exp(selected_logpdf(theta, v, 40, 0.24, cfg)), 20, upper)[0]
            self.assertAlmostEqual(integral, 1, places=7)

    def test_truncated_draws_match_selected_cdf(self):
        for model, theta in [("lognormal", [np.log(25), 0.8, np.log(0.4)]),
                             ("mixture", [np.log(25), 0.8, np.log(12), 0.4, np.log(0.05)])]:
            config = ModelConfig(model, rm_min=20, rm_max=80)
            spec = np.geomspace(10, 150, 5000)
            z = np.full(len(spec), 0.2)
            draws = selected_rvs(theta, spec, z, config, np.random.default_rng(431))
            self.assertTrue(np.all((draws >= 20) & (draws < 80)))
            pit = selected_cdf(theta, draws, spec, z, config)
            self.assertLess(kstest(pit, "uniform").statistic, 0.025)

    def test_evolution_pivot_and_prior(self):
        cfg = ModelConfig(evolution=True)
        theta = [np.log(40), 0.8, 2.0, np.log(0.2)]
        self.assertAlmostEqual(float(mu_rm(theta, 40, 0.24, cfg)), 40)
        data = {"lambda_rm": np.array([45.0]), "lambda_spec": np.array([40.0]), "z": np.array([0.24])}
        self.assertTrue(np.isfinite(log_probability(theta, data, cfg)))
        theta[1] = 9
        self.assertEqual(log_probability(theta, data, cfg), -np.inf)

    def test_tail_numerics_and_outside_selection(self):
        cfg = ModelConfig("mixture", rm_min=20, rm_max=100)
        theta = [np.log(2), 0.1, np.log(0.11), 0.5, np.log(9.0)]
        self.assertTrue(np.isfinite(selected_logpdf(theta, 25, 40, 0.24, cfg)))
        self.assertEqual(selected_logpdf(theta, 10, 40, 0.24, cfg), -np.inf)

    def test_selected_synthetic_parameter_recovery(self):
        from richness_relation.fit_conditional_richness_mpi import optimize
        rng = np.random.default_rng(763)
        cases = [
            (ModelConfig(evolution=True), [np.log(45), 0.8, -1.5, np.log(0.25)], [0.07, 0.1, 0.4, 0.15]),
            (ModelConfig("mixture"), [np.log(40), 0.8, np.log(4), 0.3, np.log(0.1)], [0.07, 0.1, 0.25, 0.12, 0.35]),
        ]
        for config, truth, tolerance in cases:
            spec = np.exp(rng.uniform(np.log(10), np.log(100), 1800))
            z = rng.uniform(0.1, 0.35, len(spec))
            rm = selected_rvs(truth, spec, z, config, rng)
            fitted = optimize({"lambda_spec": spec, "z": z, "lambda_rm": rm}, config, rng, 3)
            self.assertTrue(fitted.success)
            self.assertTrue(np.all(np.abs(fitted.x - truth) < tolerance), (config, fitted.x))


class TestClusterPreparation(unittest.TestCase):
    def table(self):
        return Table({"ID": [1, 1, 2, 3, 4, 5, 6],
                      "Z_SPEC_central": [0.1, 0.1, 0.18, 0.24, 0.35, 0.2, 0.09],
                      "LAMBDA": [20, 20, 40, 80, 40, 40, 40],
                      "lambda_spec_noproj_weighted": [12, 12, 30, 60, 20, -1, 20]})

    def test_unique_clusters_boundaries_and_nonpositive(self):
        sample, audit = prepare(self.table())
        self.assertEqual(sample["ID"].tolist(), ["1", "2", "3"])
        self.assertEqual(audit["z_bin_counts"], [1, 1, 1])
        self.assertEqual(audit["cutflow"]["positive_lambda_spec"]["removed"], 1)

    def test_inconsistent_repeated_cluster_is_rejected(self):
        table = self.table()
        table["LAMBDA"][1] = 21
        with self.assertRaisesRegex(ValueError, "varies within"):
            prepare(table)

    def test_existing_parent_without_richness_fails_clearly(self):
        table = self.table()
        table.remove_column("lambda_spec_noproj_weighted")
        with self.assertRaisesRegex(KeyError, "postprocess"):
            prepare(table)


if __name__ == "__main__":
    unittest.main()
