"""Deterministic stage tests independent of the external continuum fitter."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from make_catalogs.spectroscopic_richness import (
    prepare_stages, STAGES, CONTINUUM_DEGREE, CONTINUUM_MEDIAN_WINDOW,
    append_richness_to_matched,
)
from richness_relation.weighting_plot_utils import common_sample, binned_log_mean
from richness_relation.plot_richness_weighting_comparison import plot_relation
from richness_relation.plot_spectroscopic_richness_residuals import plot_distributions
from make_catalogs.diagnose_richness_weights import diagnose_weights
from richness_relation.richness_catalog import load_stages, cluster_rows


def catalog():
    z = np.repeat([0.12, 0.22, 0.3], 4)
    geo = np.repeat([1.1, 1.2, 1.3], 4)
    lf = np.repeat([1.0, 2.0, 4.0], 4)
    comp = np.arange(1, 13, dtype=float)
    return Table({"ID": np.repeat([10, 20, 30], 4), "Z_SPEC_central": z,
                  "Z_BGS": z + np.tile([-0.025, -0.001, 0.001, 0.025], 3) * (1 + z),
                  "LAMBDA": np.repeat([25, 40, 80], 4),
                  "COMP_WEIGHT": comp, "GEOMETRIC_WEIGHT": geo, "LF_WEIGHT": lf,
                  "TOTAL_WEIGHT": comp * geo * lf, "PROB_OBS": 1 / comp,
                  "GEOMETRIC_FRACTION": 1 / geo})


def flat_continuum(edges, centers, offsets):
    return np.full(len(centers), 2.0)


class WeightingTests(unittest.TestCase):
    def test_matching_main_saves_all_stages(self):
        from make_catalogs import projection_match_catalogs as matching
        tab = catalog()
        tab.remove_column('GEOMETRIC_FRACTION')
        tab['RM_gal_flag'] = False
        rm = Table({'ID': [10, 20, 30]})
        geo = Table({'ID': [10, 20, 30], 'angRad_deg': [1., 1., 1.], 'sq_deg': [1., 1., 1.],
                     'Nr_1.5hmpc_expected_per_file': [1., 1., 1.], 'N_random_total': [10, 10, 10],
                     'N_random_files_GEOMETRIC_FRACTION': [1, 1, 1],
                     'GEOMETRIC_FRACTION': 1 / np.array([1.1, 1.2, 1.3])})
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            for name, value in [('OUTPUT_DIR', root), ('OUTPUT_PICKLE', root / 'matched.pickle'),
                                ('OUTPUT_FITS', root / 'matched.fits'),
                                ('OUTPUT_RICHNESS_AUDIT', root / 'audit.json')]:
                stack.enter_context(patch.object(matching, name, value))
            for name, value in [('mpi_context', (None, 0, 1)),
                                ('load_redmapper_catalog', (rm, rm)), ('load_bgs_catalog', tab),
                                ('match_bgs_to_clusters_projected', tab),
                                ('attach_redmapper_member_match', tab),
                                ('compute_geo_fraction_parallel', geo)]:
                stack.enter_context(patch.object(matching, name, return_value=value))
            stack.enter_context(patch.object(matching, 'add_lf_weight_columns', side_effect=lambda t: t))
            stack.enter_context(patch('make_catalogs.spectroscopic_richness.legacy_continuum',
                                      side_effect=flat_continuum))
            self.assertEqual(matching.main(), 0)
            saved = Table.read(root / 'matched.fits')
            self.assertEqual(len(saved), len(tab))
            self.assertEqual(saved.meta['CONTDEG'], 6)
            self.assertEqual(saved.meta['CONTWIN'], 5)
            self.assertTrue((root / 'audit.json').exists())
            clusters, _ = load_stages(root / 'matched.fits')
            np.testing.assert_allclose(clusters[STAGES[2]], clusters[STAGES[1]] * clusters['LF_WEIGHT'])

    def test_weight_diagnostics(self):
        tab = catalog()
        tab['COMP_WEIGHT'][:5] = [0, -1, np.nan, np.inf, 0]
        tab['PROB_OBS'][4] = 0
        tab['ZWARN'] = np.zeros(len(tab), dtype=int)
        tab['ZWARN'][4] = 999999
        tab['SPECTYPE'] = ['GALAXY'] * len(tab)
        tab['Z_BGS'][3] = 1.0  # Outside the selection: reported only in all-row counts.
        summary, rows, clusters = diagnose_weights(tab)
        self.assertEqual(summary['flagged_selected_rows'], 4)
        self.assertEqual(summary['affected_clusters'], 2)
        self.assertEqual(summary['probability']['invalid_comp_with_valid_probability'], 3)
        self.assertEqual(summary['probability']['invalid_comp_with_invalid_probability'], 1)
        self.assertIn('ZWARN', rows.colnames)
        self.assertEqual(summary['weights']['COMP_WEIGHT']['all_rows']['infinite'], 1)
        self.assertEqual(summary['weights']['COMP_WEIGHT']['selected_rows']['infinite'], 0)
        np.testing.assert_array_equal(rows['INPUT_ROW_INDEX'], [0, 1, 2, 4])
        self.assertEqual(tab['COMP_WEIGHT'][0], 0)
        tab.remove_column('PROB_OBS')
        self.assertFalse(diagnose_weights(tab)[0]['probability']['available'])

    def test_diagnostic_files_preserve_existing_outputs(self):
        tab = catalog()
        tab['COMP_WEIGHT'][0] = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.fits'
            tab.write(source)
            before = source.read_bytes()
            with patch('make_catalogs.spectroscopic_richness.legacy_continuum',
                       side_effect=AssertionError('Continuum must not run')):
                with self.assertRaisesRegex(ValueError, 'No rows were dropped'):
                    append_richness_to_matched(tab, root / 'diagnostics')
                report = next((root / 'diagnostics').iterdir())
                self.assertTrue((report / 'flagged_rows.ecsv').exists())
                self.assertTrue((report / 'affected_clusters.ecsv').exists())
            self.assertEqual(source.read_bytes(), before)

    def test_catalog_annotation_and_plot_reader(self):
        tab = catalog()
        tab['Z_SPEC_central'][:4] = 0.08
        with tempfile.TemporaryDirectory() as tmp:
            out, audit = append_richness_to_matched(tab, Path(tmp), flat_continuum)
            self.assertEqual(len(out), len(tab))
            np.testing.assert_array_equal(out['ID'], tab['ID'])
            self.assertTrue(np.all(np.isnan(out[STAGES[0]][:4])))
            self.assertFalse(np.any(out['RICHNESS_AVAILABLE'][:4]))
            self.assertEqual(np.sum(out['RICHNESS_ANALYSIS_ROW']), 8)
            np.testing.assert_allclose(out['lambda_spec_noproj_weighted'], out[STAGES[2]])
            path = Path(tmp) / 'matched.fits'
            out.write(path)
            clusters, info = load_stages(path)
            self.assertEqual(len(clusters), 3)
            np.testing.assert_allclose(clusters[STAGES[2]], clusters['LF_WEIGHT'] * clusters[STAGES[1]])
            self.assertEqual(info['matched_rows'], len(tab))
            with self.assertRaisesRegex(KeyError, 'Regenerate'):
                cluster_rows(tab)
            out[STAGES[1]][4] += 1
            with self.assertRaisesRegex(ValueError, 'varies within'):
                cluster_rows(out)

    def test_continuum_defaults_and_custom_audit(self):
        self.assertEqual(CONTINUUM_DEGREE, 6)
        self.assertEqual(CONTINUUM_MEDIAN_WINDOW, 5)
        _, audit = prepare_stages(catalog(), flat_continuum)
        self.assertEqual(audit["continuum_settings"]["estimator"], "custom")
        self.assertIsNone(audit["continuum_settings"]["degree"])

    def test_individual_signal_and_lf_factorization(self):
        tab = catalog()
        out, audit = prepare_stages(tab, flat_continuum)
        np.testing.assert_allclose(out[STAGES[0]], 4 * (1 - 0.2))
        np.testing.assert_allclose(out["lambda_spec_proj_geo_comp"], [11, 31.2, 54.6])
        np.testing.assert_allclose(out[STAGES[2]] / out[STAGES[1]], [1, 2, 4])
        self.assertEqual(audit["selected_candidate_rows"], 12)
        shuffled, _ = prepare_stages(tab[[7, 1, 3, 8, 0, 2, 9, 6, 4, 10, 5, 11]], flat_continuum)
        for key in STAGES:
            np.testing.assert_allclose(shuffled[key], out[key])

    def test_constant_weights_scale_continuum_and_signal(self):
        tab = catalog()
        tab.remove_columns(["TOTAL_WEIGHT", "PROB_OBS", "GEOMETRIC_FRACTION"])
        tab["COMP_WEIGHT"] = 2.0
        tab["GEOMETRIC_WEIGHT"] = 1.5
        out, _ = prepare_stages(tab, flat_continuum)
        np.testing.assert_allclose(out[STAGES[1]], 3 * out[STAGES[0]])

    def test_science_selection_and_invalid_weights(self):
        tab = catalog()
        tab["Z_BGS"][0] = 0.01
        out, audit = prepare_stages(tab, flat_continuum)
        self.assertEqual(audit["selected_candidate_rows"], 11)
        self.assertEqual(out["N_candidates"][0], 3)
        tab = catalog()
        tab["LF_WEIGHT"][0] = 7
        with self.assertRaisesRegex(ValueError, "varies within"):
            prepare_stages(tab, flat_continuum)
        tab = catalog()
        tab["COMP_WEIGHT"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "Invalid COMP_WEIGHT"):
            prepare_stages(tab, flat_continuum)
        tab = catalog()
        tab["TOTAL_WEIGHT"][0] *= 2
        with self.assertRaisesRegex(ValueError, "TOTAL_WEIGHT"):
            prepare_stages(tab, flat_continuum)

    def test_nonpositive_not_clipped(self):
        out, audit = prepare_stages(catalog(), lambda e, c, d: np.full(len(c), 20.0))
        self.assertTrue(np.all(out[STAGES[0]] < 0))
        self.assertEqual(audit["nonpositive_stage_counts"][STAGES[0]], 3)
        with self.assertRaisesRegex(ValueError, "No clusters"):
            common_sample(out)

    def test_log_sem_and_plots(self):
        rm = np.arange(20, 40, dtype=float)
        spec = np.geomspace(5, 50, 20)
        points = binned_log_mean(rm, spec, edges=[20, 40], min_count=20)
        np.testing.assert_allclose(points[0, 1:3], [np.log10(spec).mean(), np.log10(spec).std(ddof=1) / np.sqrt(20)])
        self.assertEqual(len(binned_log_mean(rm[:19], spec[:19], [20, 40])), 0)
        out, _ = prepare_stages(catalog(), flat_continuum)
        figs = [plot_relation(out, min_count=2),
                plot_distributions(out, min_count=1),
                plot_distributions(out, residual=False, min_count=1)]
        for fig in figs:
            fig.canvas.draw()
            self.assertTrue(np.any(np.asarray(fig.canvas.buffer_rgba())[:, :, :3] < 200))
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
