import unittest
import pytest
import numpy as np
from local_overdensity.plot_environment_richness import binned_means, correlation


def test_open_highest_richness_label():
    from local_overdensity.plot_environment_richness import spec_bin_label
    assert spec_bin_label(130.9, 473.5, highest=True) == r'$130.9<\lambda_{\rm spec}$'
    assert '473.5' not in spec_bin_label(130.9, 473.5, highest=True)
    assert '130.9' in spec_bin_label(105.6, 130.9)


@pytest.mark.parametrize('nz,ns,residual',[(3,3,False),(1,4,False),(4,4,False),(3,4,False),(3,4,True)])
def test_nine_cell_trend_rendering(tmp_path,nz,ns,residual):
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
            cell=f'z{1 if nz==4 else iz+1}_s{ix+1}'
            directory=root/f'{cell}_Q{iz+1 if nz==4 else 0}';directory.mkdir()
            start=(ns*iz+ix)*20
            meta=dict(cell=cell,z_bin=iz+1,spec_bin=ix+1,assignments=str(source),assignments_sha256=digest,
                      cluster_ids=[str(i) for i in range(start,start+20)],fixed_global_means=[35.]*20,
                      spec_edges=edges,spec_range=edges[ix:ix+2],
                      z_range=[.1,.35] if nz==4 else [.1,.18,.24,.35][iz:iz+2],burn=5,
                      quartile=iz+1 if nz==4 else 0)
            summary=dict(adaptive_convergence_pass=ix!=0,steps=20,stop_reason='max_steps',
                         rhat_worst=[1.01]*3,steps_per_tau=[[60]*3]*4)
            (directory/'run.json').write_text(json.dumps(meta))
            (directory/'summary.json').write_text(json.dumps(summary))
            np.savez(directory/'posterior.npz',samples=np.tile([np.log(5),.3,np.log(.05)],(5,1)))
    output=tmp_path/'plots'
    plot_mixture_cell_trends(root,output,15,residual)
    parameters=pd.read_csv(output/'posterior_parameters.csv')
    assert len(parameters)==3*nz*ns
    assert parameters.cell.nunique()==nz*ns
    assert (output/'mixture_posterior_trends.png').stat().st_size>1000
    assert (output/'conditional_cell_distributions.pdf').exists()
    hist=pd.read_csv(output/'histogram_measurements.csv')
    assert hist['count'].sum()==20*nz*ns
    assert np.allclose(hist.assign(mass=hist.density*(hist.hi-hist.lo)).groupby('cell').mass.sum(),1.)


@pytest.mark.parametrize('ns',[4,10])
def test_lognormal_comparison_rendering(tmp_path,ns):
    import json
    import pandas as pd
    from richness_relation.global_spec_richness_model import FixedMeanLognormal
    from local_overdensity.plot_environment_richness import plot_mixture_cell_trends
    test_nine_cell_trend_rendering(tmp_path,1,ns,False)
    root=tmp_path/'fits'; lnroot=tmp_path/'lnfits'; lnroot.mkdir()
    for path in root.iterdir():
        meta=json.loads((path/'run.json').read_text())
        target=lnroot/path.name;target.mkdir()
        (target/'run.json').write_text(json.dumps(meta))
        (target/'summary.json').write_text((path/'summary.json').read_text())
        reg=FixedMeanLognormal(np.linspace(21,100,20),np.full(20,35.))
        best=reg.optimize()[0].x
        np.savez(target/'posterior.npz',samples=np.tile(best,(5,1)))
        (target/'bestfit.json').write_text(json.dumps(dict(mle=best.tolist(),log_likelihood=float(reg.log_likelihood(best)[0]))))
        (path/'bestfit.json').write_text(json.dumps(dict(optima=[[np.log(5),.3,np.log(.05)]])))
    output=tmp_path/'comparison'
    plot_mixture_cell_trends(root,output,lognormal_fits=lnroot)
    table=pd.read_csv(output/'model_comparison.csv')
    assert len(table)==2*ns and set(table.family)=={'mixture','lognormal'}
    assert np.all(np.isfinite(table[['AIC','BIC','PPC_upper_tail']]))
    assert (output/'lognormal_posterior_marginals.png').exists()


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
