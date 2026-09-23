"""Synthetic unit tests for the DR2 BGS/redMaPPer catalog pipeline.

These tests do not require access to the NERSC DR2 catalogs. They exercise the
catalog loader, projected matching, geometric correction, luminosity-function
correction, and combined per-galaxy weights using small inputs with known
answers.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table

from make_catalogs import projection_match_catalogs as pipeline


class TestDR2CatalogLoading(unittest.TestCase):
    def test_load_bgs_catalog_standardizes_columns_and_computes_weight(self):
        source = Table()
        source["TARGETID"] = [101, 102, 103, 104, 105]
        source["RA"] = [10.0, 10.1, 10.2, 10.3, 10.4]
        source["DEC"] = [-1.0, -0.9, -0.8, -0.7, -0.6]
        source["Z"] = [0.10, 0.20, 0.30, 0.40, 0.50]
        source["PROB_OBS"] = [1.0, 0.5, 0.25, 0.0, np.nan]
        source["FLUX_R"] = [10.0, 20.0, 30.0, 40.0, 50.0]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "BGS_BRIGHT_full_noveto.dat.fits"
            source.write(path)
            loaded = pipeline.load_bgs_catalog(path)

        self.assertIn("RA_BGS", loaded.colnames)
        self.assertIn("DEC_BGS", loaded.colnames)
        self.assertIn("Z_BGS", loaded.colnames)
        self.assertIn("COMP_WEIGHT", loaded.colnames)
        np.testing.assert_allclose(
            loaded["COMP_WEIGHT"],
            [1.0, 2.0, 4.0, 0.0, 0.0],
        )

    def test_load_bgs_catalog_requires_prob_obs(self):
        source = Table()
        source["TARGETID"] = [101]
        source["RA"] = [10.0]
        source["DEC"] = [-1.0]
        source["Z"] = [0.2]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "BGS_BRIGHT_full_noveto.dat.fits"
            source.write(path)
            with self.assertRaisesRegex(KeyError, "PROB_OBS"):
                pipeline.load_bgs_catalog(path)


class TestDR2Weights(unittest.TestCase):
    def test_combined_weight_has_known_answer(self):
        table = Table()
        table["COMP_WEIGHT"] = [1.0, 2.0]
        table["GEOMETRIC_FRACTION"] = [0.5, 0.5]
        table["LF_WEIGHT"] = [3.0, 3.0]

        weighted = pipeline.add_weight_columns(table)

        np.testing.assert_allclose(weighted["GEOMETRIC_WEIGHT"], [2.0, 2.0])
        np.testing.assert_allclose(weighted["TOTAL_WEIGHT"], [6.0, 12.0])
        np.testing.assert_allclose(
            weighted["TOTAL_WEIGHT"],
            weighted["COMP_WEIGHT"]
            * weighted["GEOMETRIC_WEIGHT"]
            * weighted["LF_WEIGHT"],
        )

    def test_invalid_geometric_fraction_does_not_create_infinite_weight(self):
        table = Table()
        table["COMP_WEIGHT"] = [1.0, 1.0, 1.0]
        table["GEOMETRIC_FRACTION"] = [0.0, np.nan, 1.0]
        table["LF_WEIGHT"] = [1.0, 1.0, 1.0]

        weighted = pipeline.add_weight_columns(table)

        np.testing.assert_allclose(weighted["GEOMETRIC_WEIGHT"], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(weighted["TOTAL_WEIGHT"], [0.0, 0.0, 1.0])
        self.assertTrue(np.all(np.isfinite(weighted["TOTAL_WEIGHT"])))

    def test_lf_weight_is_cluster_level_and_increases_with_redshift(self):
        table = Table()
        table["ID"] = [1, 1, 2, 2]
        table["Z_SPEC_central"] = [0.1, 0.1, 0.35, 0.35]
        table["COMP_WEIGHT"] = [1.0, 2.0, 1.0, 2.0]
        table["GEOMETRIC_FRACTION"] = [1.0, 1.0, 1.0, 1.0]

        summary = pd.DataFrame(
            {
                "log10_L_star": [10.5],
                "alpha": [-0.8],
                "M_star_minus_5logh": [-20.5],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "lf_summary.csv"
            summary.to_csv(summary_path, index=False)
            weighted = pipeline.add_lf_weight_columns(table, summary_path)

        for group in weighted.group_by("ID").groups:
            np.testing.assert_allclose(group["LF_WEIGHT"], group["LF_WEIGHT"][0])

        cluster_weights = {
            int(group["ID"][0]): float(group["LF_WEIGHT"][0])
            for group in weighted.group_by("ID").groups
        }
        self.assertAlmostEqual(cluster_weights[1], 1.0, places=10)
        self.assertGreater(cluster_weights[2], cluster_weights[1])


class TestProjectedMatching(unittest.TestCase):
    def test_projected_and_redshift_cuts_and_pair_uniqueness(self):
        clusters = Table()
        clusters["ID"] = [11]
        clusters["RA_central"] = [150.0]
        clusters["DEC_central"] = [2.0]
        clusters["Z_SPEC_central"] = [0.2]
        clusters["LAMBDA"] = [40.0]

        bgs = Table()
        bgs["TARGETID"] = [1, 2, 3, 1]
        bgs["RA_BGS"] = [150.0, 151.0, 150.0, 150.0]
        bgs["DEC_BGS"] = [2.0, 2.0, 2.0, 2.0]
        bgs["Z_BGS"] = [0.2, 0.2, 0.8, 0.2]
        bgs["PROB_OBS"] = [1.0, 1.0, 1.0, 1.0]
        bgs["COMP_WEIGHT"] = [1.0, 1.0, 1.0, 1.0]

        matched = pipeline.match_bgs_to_clusters_projected(clusters, bgs)

        self.assertEqual(len(matched), 1)
        self.assertEqual(int(matched["TARGETID"][0]), 1)
        self.assertTrue(bool(matched["central_flag"][0]))
        self.assertLess(float(matched["R_PROJ_HMPC"][0]), pipeline.PROJECTED_APERTURE_HMPC)
        self.assertLessEqual(abs(float(matched["DZ_CLUSTER"][0])), pipeline.DZ_ABS_MAX)


class TestGeometricCoverage(unittest.TestCase):
    def test_geo_fraction_has_known_answer(self):
        clusters = Table()
        clusters["ID"] = [1, 2]
        clusters["RA_central"] = [10.0, 20.0]
        clusters["DEC_central"] = [0.0, 1.0]
        clusters["Z_SPEC_central"] = [0.2, 0.3]

        theta = pipeline.angular_radius_deg_from_hmpc(
            pipeline.PROJECTED_APERTURE_HMPC,
            clusters["Z_SPEC_central"],
        )
        expected_per_file = np.pi * theta**2 * pipeline.RANDOM_DENSITY_PER_DEG2
        n_files = 2
        desired_fraction = np.array([1.0, 0.5])
        counts = desired_fraction * expected_per_file * n_files

        geo = pipeline.build_geo_table(clusters, counts, n_files)

        np.testing.assert_allclose(geo["GEOMETRIC_FRACTION"], desired_fraction)

    def test_rank_file_assignment_is_complete_and_disjoint(self):
        files = [Path(f"random-{i}.fits") for i in range(11)]
        assignments = [pipeline.assigned_random_files(files, rank, 4) for rank in range(4)]
        flattened = [path for assignment in assignments for path in assignment]

        self.assertCountEqual(flattened, files)
        self.assertEqual(len(flattened), len(set(flattened)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
