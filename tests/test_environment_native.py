"""Native FITS integration tests without access to NERSC."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from astropy.table import Table
from scipy.integrate import quad
from scipy.stats import norm
from test_environment_measurements import fixture
from local_overdensity.environment_config import load_config
from local_overdensity.environment_catalogs import load_clusters
from local_overdensity.environment_dr9_native import gaussian_probability, ownership
from local_overdensity.environment_runner import jobs_for, process


class NativeTests(unittest.TestCase):
    def test_probability(self):
        for mean, sigma, z in [(0.2, 0.01, 0.2), (0., 0.1, 0.02), (0.1, 0.2, 0.3)]:
            expected = quad(lambda x: norm.pdf(x, mean, sigma)/norm.cdf(mean/sigma),
                            max(0, z-.05), z+.05)[0]
            self.assertAlmostEqual(float(gaussian_probability(np.array([mean]), np.array([sigma]), z)[0]),
                                   expected, places=12)

    def test_ownership(self):
        t = Table({"RA": [180., 180., 0.], "DEC": [32.375, 32., -5.]})
        n, s = ownership(t, "north"), ownership(t, "south")
        np.testing.assert_array_equal(n, [True, False, False])
        np.testing.assert_array_equal(n | s, True)
        self.assertFalse(np.any(n & s))

    def test_native_processing_and_random_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = fixture(root)
            t = c["dr9"]
            t.update(photoz_mode="gaussian_fits", regions=["north", "south"],
                     photoz_half_width=.05, random_density_per_realization=2500,
                     lf_curve=None, maskbits_reject=12290)
            t["random_globs"] = []
            for region in t["regions"]:
                path = root/region/"sweep-test.fits"
                tab = Table.read(path)
                for b in "GRZ":
                    tab[f"NOBS_{b}"] = np.ones(len(tab), dtype=int)
                tab.write(path, overwrite=True)
                photo = tab[["RELEASE", "BRICKID", "OBJID"]].copy()
                photo["Z_PHOT_MEAN"] = np.repeat([.12, .22, .3], 4)
                photo["Z_PHOT_STD"] = np.full(len(tab), .02)
                photo["Z_PHOT_MEAN"][0] = -99
                photo.write(root/(region+"_pdf")/"sweep-test-pz.fits")
                rand = tab[["RA", "DEC", "MASKBITS", "NOBS_G", "NOBS_R", "NOBS_Z"]]
                rand["MASKBITS"][1] = 2
                rand.write(root/region/f"randoms-{region}-1-0.fits")
                t["random_globs"].append(str(root/region/f"randoms-{region}-1-*.fits"))
            (root/"config.json").write_text(json.dumps(c))
            c = load_config(root/"config.json", "dr9")
            clusters = load_clusters(c["cluster_catalog"])
            def run():
                results = [process(j, clusters, c, "dr9") for j in jobs_for(c, "dr9")]
                return np.sum([r[0] for r in results], axis=0), [r[1] for r in results]
            stats, audits = run()
            self.assertEqual(sum(a.get("invalid_photoz_rows", 0) for a in audits), 1)
            self.assertTrue(np.any(stats[:, :, 1] > 0))
            self.assertTrue(np.all(stats[:, :, 3] > 0))
            for region in t["regions"]:
                rand = Table.read(root/region/f"randoms-{region}-1-0.fits")
                rand.write(root/region/f"randoms-{region}-1-1.fits")
            doubled, _ = run()
            np.testing.assert_allclose(stats[:, :, 3], doubled[:, :, 3])
            np.testing.assert_allclose(2*stats[:, :, 4], doubled[:, :, 4])
            photo_path = root/"south_pdf"/"sweep-test-pz.fits"
            photo = Table.read(photo_path)
            photo["OBJID"][0] = 999
            photo.write(photo_path, overwrite=True)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                run()
