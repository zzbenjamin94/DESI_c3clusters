import unittest
import numpy as np
from scipy.integrate import quad
from richness_relation.cell_richness_models import (
    config,logpdf,cdf,draw,parameter_spec,make_cells,log_probability,
    posterior_summary,pearson_stat,
)


class CellTests(unittest.TestCase):
    def test_partition(self):
        data=dict(lambda_spec=np.tile(np.arange(1.,9.),3),z=np.repeat([.1,.18,.24],8))
        edges,cells=make_cells(data)
        self.assertEqual(len(cells),9)
        self.assertTrue(np.all(np.sum([m for c,m in cells],axis=0)==1))
        self.assertEqual(sum(c['N'] for c,m in cells),24)

    def test_tied_bins_rejected(self):
        with self.assertRaises(ValueError): make_cells(dict(lambda_spec=np.ones(10),z=np.full(10,.2)))

    def test_normalization_and_support(self):
        for model,theta in [('lognormal',[np.log(35),np.log(.4)]),
                            ('mixture',[np.log(35),np.log(8),.3,np.log(.06)])]:
            cfg=config(model,20,None)
            self.assertEqual(len(parameter_spec(model)),len(theta))
            integral=quad(lambda x:float(np.exp(logpdf(theta,x,cfg))),20,np.inf)[0]
            self.assertAlmostEqual(integral,1.,places=6)
            self.assertAlmostEqual(float(cdf(theta,20,cfg)),0.)
            self.assertAlmostEqual(float(cdf(theta,np.inf,cfg)),1.)
            sample=draw(theta,1000,cfg,np.random.default_rng(42))
            self.assertTrue(np.all(sample>=20))
            self.assertTrue(np.all(np.isfinite(sample)))
            self.assertTrue(np.isfinite(log_probability(theta,sample,cfg)))
            statistic,k=pearson_stat(sample,theta,cfg)
            self.assertTrue(np.isfinite(statistic)); self.assertGreater(k,1)

    def test_physical_summary(self):
        s=np.array([[np.log(20),np.log(.3)],[np.log(40),np.log(.5)]])
        out=posterior_summary(s,'lognormal')
        self.assertAlmostEqual(out['mu']['median'],30.)

    def test_bounded_selection(self):
        theta=[np.log(35),np.log(.4)]
        cfg=config('lognormal',20,100)
        self.assertEqual(float(logpdf(theta,100,cfg)),-np.inf)
        self.assertAlmostEqual(quad(lambda x:float(np.exp(logpdf(theta,x,cfg))),20,100)[0],1.)
