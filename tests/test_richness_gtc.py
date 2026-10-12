"""Synthetic HDF5 tests and optional end-to-end pygtc rendering."""
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from richness_relation.plot_richness_mcmc_gtc import load_ensembles, plot_task


def fixture(root, model):
    names = ["ln_A", "B", "ln_sigma0"] if model == "lognormal" else [
        "ln_A", "B", "ln_sigma_lambda", "f_proj", "ln_tau"]
    directory = root / model / (model + "_combined_noz") / "seed_42"
    directory.mkdir(parents=True)
    config = {"model": model}
    (directory / "manifest.json").write_text(json.dumps(
        dict(burn=20, walkers=8, ensembles=2, config=config)))
    (directory / "summary.json").write_text(json.dumps(dict(
        parameter_names=names, config=config, task=model + "_combined_noz")))
    for i in range(2):
        x = np.random.default_rng(i).normal(size=(300, 8, len(names)))
        with h5py.File(directory / f"chain_{i}.h5", "w") as f:
            g = f.create_group("mcmc")
            g.attrs["iteration"] = 250
            g.create_dataset("chain", data=x)
    return directory


class PlotTests(unittest.TestCase):
    def test_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            d = fixture(Path(temp), "mixture")
            samples, labels, names, audit = load_ensembles(d, thin=2, max_samples=10000)
            self.assertEqual(samples[0].shape, (115 * 8, 5))
            self.assertEqual(len(labels), 2)
            self.assertEqual(audit["chains"][0]["burn"], 20)
            a = load_ensembles(d, max_samples=40)[0][0]
            b = load_ensembles(d, max_samples=40)[0][0]
            np.testing.assert_array_equal(a, b)
            with self.assertRaises(ValueError):
                load_ensembles(d, burn=250)
            with self.assertRaises(ValueError):
                load_ensembles(d, thin=0)

    def test_render_and_protect(self):
        with tempfile.TemporaryDirectory() as temp:
            for model in ("lognormal", "mixture"):
                d = fixture(Path(temp), model)
                outputs = plot_task(d, Path(temp) / "plots")
                for p in outputs:
                    self.assertGreater(p.stat().st_size, 0)
                with self.assertRaises(FileExistsError):
                    plot_task(d, Path(temp) / "plots")


if __name__ == "__main__":
    unittest.main()
