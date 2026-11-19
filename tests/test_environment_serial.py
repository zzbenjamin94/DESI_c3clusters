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
    def test_targeted_north(self):
        from local_overdensity.run_environment_serial_test import targeted_north_inputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            t = c["dr9"]
            t.update(photoz_mode="gaussian_fits", regions=["north", "south"],
                     photoz_half_width=.05, random_density_per_realization=2500,
                     lf_curve=None, random_globs=[str(root/"north/randoms-north-1-*.fits"),
                                                 str(root/"south/randoms-south-1-*.fits")])
            clusters = Table.read(c["cluster_catalog"])
            clusters["RA_central"] = [135., 130.001, 139.999]
            clusters["DEC_central"] = [47., 47., 47.]
            clusters.write(c["cluster_catalog"], overwrite=True)
            data = Table.read(root/"north/sweep-test.fits")
            data["RA"] = np.full(len(data), 135.4)
            data["DEC"] = np.full(len(data), 47.)
            for b in "GRZ":
                data[f"NOBS_{b}"] = np.ones(len(data), dtype=int)
            data.write(root/"north/sweep-130p045-140p050.fits")
            photo = data[["RELEASE", "BRICKID", "OBJID"]].copy()
            photo["Z_PHOT_MEAN"] = np.full(len(data), .12)
            photo["Z_PHOT_STD"] = np.full(len(data), .02)
            photo.write(root/"north_pdf/sweep-130p045-140p050-pz.fits")
            data.write(root/"north/randoms-north-1-0.fits")
            eligible, jobs = targeted_north_inputs(clusters, c)
            np.testing.assert_array_equal(eligible, [0])
            random_jobs = jobs[1:]
            self.assertEqual(sum(j["stop"]-j["start"] for j in random_jobs), len(data))
            self.assertTrue(all(j["density"] == 2500 for j in random_jobs))
            (root/"config.json").write_text(json.dumps(c))
            report = smoke_run(root/"config.json", "dr9", root/"targeted",
                               targeted_north=True)
            self.assertEqual(report["clusters_selected"], 1)
            self.assertEqual(report["selected_data_rows"], len(data))
            self.assertFalse(report["partial_inputs"])
            with self.assertRaises(ValueError):
                smoke_run(root/"config.json", "dr9", root/"bad",
                          targeted_north=True, max_random_jobs=1)
            with self.assertRaisesRegex(ValueError, "No clusters"):
                targeted_north_inputs(clusters[1:], c)

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
