"""Synthetic Section 4 tests: no NERSC catalogs or MPI installation required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18
from scipy.integrate import quad
from local_overdensity.environment_filters import radial_kernels, kernel_areas, pdf_integral
from local_overdensity.environment_catalogs import load_clusters, load_photoz, photo_indices
from local_overdensity.environment_accumulators import Accumulator
from local_overdensity.environment_runner import jobs_for, process
from local_overdensity.finalize_environment import finalize

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    z = np.array([0.12, 0.22, 0.3])
    ra, dec = np.array([10., 130., 250.]), np.zeros(3)
    clusters = Table({"ID": [1,2,3], "RA_central": ra, "DEC_central": dec,
                      "Z_SPEC_central": z, "LAMBDA": [25.,45.,80.],
                      "lambda_spec_geo_comp_lf": [30.,50.,90.]})
    clusters.write(root / "clusters.fits", overwrite=True)
    radii = np.tile([2.,3.,7.,8.], 3)
    dm = Planck18.comoving_transverse_distance(z).value * Planck18.h
    galaxy_ra = np.repeat(ra,4) + np.rad2deg(radii/np.repeat(dm,4))
    n = len(galaxy_ra)
    phot = Table({"RA":galaxy_ra,"DEC":np.zeros(n),"RELEASE":np.ones(n,dtype=int),
                  "BRICKID":np.ones(n,dtype=int),"OBJID":np.arange(n),
                  "TYPE":["EXP"]*n,"FLUX_R":np.full(n,100.),
                  "MW_TRANSMISSION_R":np.ones(n),"MASKBITS":np.zeros(n,dtype=int)})
    for band in "GRZ":
        phot[f"FLUX_IVAR_{band}"] = np.ones(n)
    grid = np.linspace(0,1,101)
    for side in ("north","south"):
        (root/side).mkdir(exist_ok=True)
        (root/(side+"_pdf")).mkdir(exist_ok=True)
        phot.write(root/side/"sweep-test.fits", overwrite=True)
        np.savez(root/(side+"_pdf")/"sweep-test.npz", RELEASE=phot["RELEASE"],
                 BRICKID=phot["BRICKID"], OBJID=phot["OBJID"], z_grid=grid,
                 pdf=np.ones((n,len(grid))), PRIMARY=np.full(n,side=="north",dtype=bool),
                 SYS_WEIGHT=np.ones(n))
    bgs = Table({"RA":galaxy_ra,"DEC":np.zeros(n),"Z":np.repeat(z,4),
                 "TARGETID":np.arange(n),"PROB_OBS":np.ones(n),"GOOD":np.ones(n,bool)})
    bgs.write(root/"bgs.fits", overwrite=True)
    random = Table({"RA":galaxy_ra,"DEC":np.zeros(n),"Z":np.repeat(z,4),
                    "AREA_DEG2":np.full(n,0.01),"VOLUME_HMPC3":np.full(n,10.),
                    "MASKBITS":np.zeros(n,dtype=int)})
    random.write(root/"random.fits", overwrite=True)
    Table({"z":[0.,1.],"LF_WEIGHT":[1.,1.]}).write(root/"lf.csv",format="ascii.csv",overwrite=True)
    Table({"z":[0.,1.],"NBAR":[0.1,0.1]}).write(root/"nbar.csv",format="ascii.csv",overwrite=True)
    c = {"cluster_catalog":str(root/"clusters.fits"),"output_dir":str(root/"outputs"),
         "gaussian_width_hmpc":3.,"chunk_rows":5,
         "dr9":{"sweep_dirs":[str(root/"north"),str(root/"south")],
                "photoz_dirs":[str(root/"north_pdf"),str(root/"south_pdf")],
                "lf_curve":str(root/"lf.csv"),"r_mag_limit":22.,"maskbits_reject":0,
                "random_globs":[str(root/"random.fits")],"random_measure_column":"AREA_DEG2",
                "input_selection_confirmed":True},
         "bgs":{"data_globs":[str(root/"bgs.fits")],"quality_column":"GOOD","quality_values":[True],
                "pi_half_hmpc":20.,"lf_curve":str(root/"lf.csv"),"nbar_curve":str(root/"nbar.csv"),
                "random_globs":[str(root/"random.fits")],"random_measure_column":"VOLUME_HMPC3",
                "input_selection_confirmed":True}}
    (root/"config.json").write_text(json.dumps(c))
    return c


class EnvironmentTests(unittest.TestCase):
    def test_shared_imaging_mask(self):
        from local_overdensity.environment_catalogs import imaging_selection, photo_selection
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture(tmp)
            c["dr9"]["maskbits_reject"] = 12290
            flags = [0, 2, 4096, 8192, 4098, 4, 6, 0, 0, 0, 0, 0]
            expected = np.array([True, False, False, False, False, True,
                                 False, True, True, True, True, True])
            table = Table.read(Path(tmp)/"north/sweep-test.fits")
            table["MASKBITS"] = flags
            np.testing.assert_array_equal(photo_selection(table, c["dr9"]), expected)
            random = Table.read(Path(tmp)/"random.fits")
            random["MASKBITS"] = flags
            np.testing.assert_array_equal(imaging_selection(random, c["dr9"]), expected)
            random.write(Path(tmp)/"random.fits", overwrite=True)
            clusters = load_clusters(c["cluster_catalog"])
            for tracer in ("dr9", "bgs"):
                jobs = [j for j in jobs_for(c, tracer) if j["kind"] == "random"]
                count = sum(process(j, clusters, c, tracer)[1]["selected_rows"] for j in jobs)
                self.assertEqual(count, expected.sum() if tracer == "dr9" else len(random))
            random.remove_column("MASKBITS")
            with self.assertRaisesRegex(ValueError, "MASKBITS"):
                imaging_selection(random, c["dr9"])

    def test_normalization_and_edges(self):
        for width in (1.,3.,8.):
            area = kernel_areas(width)
            for k, (lo,hi) in enumerate(((1.5,5),(1.5,5),(5,10))):
                integral = quad(lambda r: 2*np.pi*r*radial_kernels(np.array([r]),width)[k,0],lo,hi)[0]
                self.assertAlmostEqual(integral/area[k],1,places=9)
        k = radial_kernels(np.array([1.49,1.5,4.99,5.,9.99,10.]),3)
        np.testing.assert_array_equal(k[0],[0,1,1,0,0,0])
        np.testing.assert_array_equal(k[2],[0,0,0,1,1,0])

    def test_pdf_and_lf(self):
        z = np.linspace(0,1,11)
        p = np.ones((2,11))
        np.testing.assert_allclose(pdf_integral(z,p,0.235),0.1)
        np.testing.assert_allclose(pdf_integral(z,p,0.235,lf_weight=2*np.ones(11)),0.2)
        np.testing.assert_allclose(pdf_integral(z,p,0.01),0.06)
        np.testing.assert_allclose(pdf_integral(z,p,2.),0.)
        with self.assertRaises(ValueError):
            pdf_integral(z,p*2,0.2)

    def test_uniform_and_zero_background(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=fixture(tmp)
            clusters=load_clusters(c["cluster_catalog"])
            s=np.zeros((3,3,5))
            s[:,:,3]=kernel_areas(3.)*0.7
            s[:,:,1]=s[:,:,3]*2
            s[:,:,2]=s[:,:,1]
            s[:,:,4]=100
            out=finalize(clusters,s,c,"dr9")
            np.testing.assert_allclose(out["dr9_box_E_local"],0)
            np.testing.assert_allclose(out["dr9_gaussian_E_local"],0)
            s[:,2,1]=0
            out=finalize(clusters,s,c,"dr9")
            self.assertTrue(np.all(np.isnan(out["dr9_box_E_local"])))
            s[:,:,3]=0
            out=finalize(clusters,s,c,"dr9")
            self.assertFalse(np.any(out["dr9_box_valid"]))

    def test_partition_equivalence_and_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=fixture(tmp)
            clusters=load_clusters(c["cluster_catalog"])
            for tracer in ("dr9","bgs"):
                jobs=jobs_for(c,tracer)
                arrays=[process(j,clusters,c,tracer)[0] for j in jobs]
                serial=np.sum(arrays,axis=0)
                distributed=sum((sum(arrays[r::3],np.zeros_like(serial)) for r in range(3)),np.zeros_like(serial))
                np.testing.assert_allclose(serial,distributed)
                np.testing.assert_allclose(serial[:,0,0],2)
                np.testing.assert_allclose(serial[:,2,0],2)
                c["chunk_rows"]=100
                unchunked=sum((process(j,clusters,c,tracer)[0] for j in jobs_for(c,tracer)),np.zeros_like(serial))
                np.testing.assert_allclose(serial,unchunked)
                c["chunk_rows"]=5

    def test_cylinder_excludes_los(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=fixture(tmp)
            clusters=load_clusters(c["cluster_catalog"])
            table=Table.read(Path(tmp)/"bgs.fits")
            table["ENV_WEIGHT"]=np.ones(len(table))
            table["Z"]=0.8
            acc=Accumulator(clusters,c,"bgs")
            acc.add(table)
            self.assertEqual(acc.stats.sum(),0)

    def test_missing_pdf_and_bad_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=fixture(tmp)
            table=Table.read(Path(tmp)/"north/sweep-test.fits")
            photo,lookup=load_photoz(Path(tmp)/"north_pdf/sweep-test.npz")
            table["OBJID"][0]=999
            with self.assertRaises(ValueError):
                photo_indices(table,lookup)
            table=Table.read(c["cluster_catalog"])
            table["ID"][1]=1
            table.write(c["cluster_catalog"],overwrite=True)
            with self.assertRaises(ValueError):
                load_clusters(c["cluster_catalog"])

    def test_serial_cli_resume_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=fixture(tmp)
            for tracer in ("dr9","bgs"):
                command=[sys.executable,str(ROOT/f"local_overdensity/measure_{tracer}_environment_mpi.py"),
                         "--config",str(Path(tmp)/"config.json")]
                for suffix in (["--dry-run"],[],["--resume"]):
                    result=subprocess.run(command+suffix,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                out=Table.read(Path(c["output_dir"])/tracer/"environment.fits")
                self.assertEqual(len(out),3)
                self.assertTrue(np.all(out[f"{tracer}_box_valid"]))
            c["gaussian_width_hmpc"]=4.
            (Path(tmp)/"config.json").write_text(json.dumps(c))
            result=subprocess.run(command+["--resume"],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)


if __name__ == "__main__":
    unittest.main()
