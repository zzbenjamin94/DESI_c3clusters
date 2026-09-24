"""Data-only plotting must work without saved fits or an MCMC installation."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np

from richness_relation import plot_conditional_richness as plotting
from richness_relation.prepare_conditional_richness import file_sha256, load_sample


class DataOnlyPlotTests(unittest.TestCase):
    def test_offset_mean_sem(self):
        x = np.arange(1, 41, dtype=float)
        offset = np.linspace(-0.2, 0.4, len(x))
        points = plotting.binned_offset_mean(x, offset, max_bins=1, log_x=True)
        np.testing.assert_allclose(points[0], [np.exp(np.log(x).mean()),
                                              offset.mean(), offset.std(ddof=1) / np.sqrt(len(x))])
        self.assertEqual(plotting.binned_offset_mean(x[:19], offset[:19]).shape, (0, 3))

    def test_data_only_bypasses_fit_products(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / "sample.npz"
            spec = np.exp(rng.uniform(np.log(5), np.log(150), 180))
            np.savez(sample, ID=np.arange(180), lambda_spec=spec,
                     lambda_rm=spec * 10 ** rng.normal(0.1, 0.15, 180),
                     z=np.concatenate([rng.uniform(lo, hi, 60) for lo, hi in
                                       [(0.1, 0.18), (0.18, 0.24), (0.24, 0.35)]]))
            sample.with_suffix(".json").write_text(json.dumps({
                "sample_sha256": file_sha256(sample), "z_bin_edges": [0.1, 0.18, 0.24, 0.35]}))
            with patch.object(plotting, "load_results", side_effect=AssertionError("Read fit products")):
                result = plotting.plot_all(sample, root / "missing_fits", root / "plots", data_only=True)
            self.assertTrue(result.empty)
            products = list((root / "plots/data_only").iterdir())
            self.assertEqual(len(products), 4)
            self.assertTrue(all(p.stat().st_size > 1000 for p in products))
            sample.with_suffix(".json").write_text(json.dumps({"sample_sha256": "incorrect"}))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_sample(sample)

    def test_no_mcmc_runner_import(self):
        code = """
import sys
from richness_relation import plot_conditional_richness
assert 'richness_relation.fit_conditional_richness_mpi' not in sys.modules
assert not any(k in sys.modules for k in ('emcee', 'mpi4py', 'schwimmbad'))
"""
        subprocess.run([sys.executable, "-c", code], check=True,
                       cwd=Path(__file__).resolve().parents[1],
                       env={**os.environ, "MPLBACKEND": "Agg"})


if __name__ == "__main__":
    unittest.main()
