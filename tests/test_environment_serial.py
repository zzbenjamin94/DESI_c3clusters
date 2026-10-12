"""Serial smoke runner integration tests, using synthetic inputs only."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from astropy.table import Table
from test_environment_measurements import fixture
from local_overdensity.environment_catalogs import load_clusters
from local_overdensity.environment_runner import jobs_for, process
from local_overdensity.run_environment_serial_test import smoke_run, select_jobs


class SerialSmokeTests(unittest.TestCase):
    def test_full_sample_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            for tracer in ("dr9", "bgs"):
                output = root / ("smoke_" + tracer)
                report = smoke_run(root / "config.json", tracer, output, max_clusters=2)
                self.assertFalse(report["partial_inputs"])
                out = Table.read(output / "smoke_environment.fits")
                self.assertEqual(len(out), 2)
                clusters = load_clusters(c["cluster_catalog"])
                expected = sum((process(j, clusters, c, tracer)[0]
                                for j in jobs_for(c, tracer)), np.zeros((3, 3, 5)))
                indices = [list(clusters["ID"]).index(x) for x in out["ID"]]
                with np.load(output / "smoke_accumulators.npz") as f:
                    np.testing.assert_allclose(f["stats"], expected[indices])
                with self.assertRaises(FileExistsError):
                    smoke_run(root / "config.json", tracer, output, max_clusters=2)

    def test_partial_and_dryrun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            output = root / "partial"
            report = smoke_run(root / "config.json", "bgs", output,
                               max_data_jobs=1, max_random_jobs=1, dry_run=True)
            self.assertFalse(output.exists())
            self.assertTrue(report["partial_inputs"])
            smoke_run(root / "config.json", "bgs", output,
                      max_data_jobs=1, max_random_jobs=1)
            out = Table.read(output / "smoke_environment.fits")
            self.assertTrue(out.meta["PARTIAL"])
            with self.assertRaises(ValueError):
                smoke_run(root / "config.json", "bgs", Path(c["output_dir"]) / "oops")
            with self.assertRaises(ValueError):
                select_jobs([], -1, 0)

    def test_config_stays_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            c["dr9"]["photoz_dirs"] = None
            (root / "config.json").write_text(json.dumps(c))
            with self.assertRaises(ValueError):
                smoke_run(root / "config.json", "dr9", root / "smoke")


if __name__ == "__main__":
    unittest.main()
