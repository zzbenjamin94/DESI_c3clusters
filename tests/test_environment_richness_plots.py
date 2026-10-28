import unittest
import pytest
import numpy as np
from local_overdensity.plot_environment_richness import binned_means, correlation


@pytest.mark.parametrize('nz,ns',[(3,3),(1,4)])
def test_nine_cell_trend_rendering(tmp_path,nz,ns):
    import json,hashlib
    import pandas as pd
    from local_overdensity.plot_environment_richness import plot_mixture_cell_trends
    source=tmp_path/'assignments.csv'
    pd.DataFrame(dict(ID=np.arange(20*nz*ns).astype(str),lambda_rm=np.tile(np.linspace(21,100,20),nz*ns))).to_csv(source,index=False)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    root=tmp_path/'fits';root.mkdir()
    edges=np.linspace(5,200,ns+1).tolist()
    for iz in range(nz):
        for ix in range(ns):
            cell=f'z{iz+1}_s{ix+1}'
            directory=root/f'{cell}_Q0';directory.mkdir()
            start=(ns*iz+ix)*20
            meta=dict(cell=cell,z_bin=iz+1,spec_bin=ix+1,assignments=str(source),assignments_sha256=digest,
                      cluster_ids=[str(i) for i in range(start,start+20)],fixed_global_means=[35.]*20,
                      spec_edges=edges,spec_range=edges[ix:ix+2],
                      z_range=[.1,.18,.24,.35][iz:iz+2],burn=5)
            summary=dict(adaptive_convergence_pass=ix!=0,steps=20,stop_reason='max_steps',
                         rhat_worst=[1.01]*3,steps_per_tau=[[60]*3]*4)
            (directory/'run.json').write_text(json.dumps(meta))
            (directory/'summary.json').write_text(json.dumps(summary))
            np.savez(directory/'posterior.npz',samples=np.tile([np.log(5),.3,np.log(.05)],(5,1)))
    output=tmp_path/'plots'
    plot_mixture_cell_trends(root,output,15)
    parameters=pd.read_csv(output/'posterior_parameters.csv')
    assert len(parameters)==3*nz*ns
    assert parameters.cell.nunique()==nz*ns
    assert (output/'mixture_posterior_trends.png').stat().st_size>1000
    assert (output/'conditional_cell_distributions.pdf').exists()


class SummaryTests(unittest.TestCase):
    def test_bins_cover_unique_sample(self):
        x = np.arange(140.) + 1
        bins = binned_means(x, x, 7, 20)
        self.assertEqual(len(bins), 7)
        self.assertEqual(sum(b['N'] for b in bins), 140)
        self.assertAlmostEqual(bins[0]['sem'], np.std(x[:20], ddof=1)/np.sqrt(20))

    def test_constant_and_small_samples(self):
        self.assertEqual(binned_means(np.arange(5.), np.arange(5.), 7, 20), [])
        self.assertTrue(np.isnan(correlation(np.ones(10), np.arange(10), 20, np.random.default_rng(1))[0]))

    def test_monotonic_bootstrap(self):
        x = np.arange(50.)
        rho, _, lo, hi, n = correlation(x, -x, 20, np.random.default_rng(1))
        self.assertAlmostEqual(rho, -1)
        self.assertAlmostEqual(lo, -1)
        self.assertAlmostEqual(hi, -1)
        self.assertEqual(n, 20)
