"""Focused mathematical checks for the reverse conditional regression."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import lognorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "richness_relation"))
from global_spec_richness_model import Regression, ForwardRegression, design, classical_rhat, residual_logpdf, residual_draw
from scipy.integrate import quad


class ReverseRegressionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(321)
        self.data = {"lambda_rm": np.exp(rng.uniform(np.log(20), np.log(200), 10000)),
                     "z": rng.uniform(.1, .35, 10000)}
        self.beta = np.array([4.9, .7, 1.5])
        self.data["lambda_spec"] = np.exp(design(self.data, "z") @ self.beta
                                         + rng.normal(0, .3, 10000))

    def test_recover_and_logpdf(self):
        reg = Regression(self.data, "z")
        np.testing.assert_allclose(reg.beta, self.beta, atol=.08)
        self.assertAlmostEqual(np.exp(reg.mle[-1]), .3, delta=.01)
        direct = lognorm.logpdf(self.data["lambda_spec"], np.exp(reg.mle[-1]),
                               scale=np.exp(reg.X @ reg.mle[:-1])).sum()
        self.assertAlmostEqual(reg.log_likelihood(reg.mle), direct, places=7)
        # Constant-scatter Gaussian MLE must have chi-square N.
        self.assertAlmostEqual(reg.sse / np.exp(2*reg.mle[-1]), reg.n, places=7)

    def test_noz_is_nested(self):
        no = Regression(self.data, "noz")
        yes = Regression(self.data, "z")
        theta = np.array([4.9, .7, np.log(.3)])
        self.assertAlmostEqual(no.log_likelihood(theta),
                               yes.log_likelihood(np.insert(theta, 2, 0.)), places=7)

    def test_independent_rhat(self):
        rng = np.random.default_rng(2)
        chains = rng.normal(size=(4, 2000, 3))
        self.assertTrue(np.all(classical_rhat(chains) < 1.02))
        chains[0, :, 0] += 3.
        self.assertGreater(classical_rhat(chains)[0], 1.1)

    def test_centered_negative_tail(self):
        theta=np.array([np.log(.25),.2,np.log(2.)])
        pdf=lambda r: float(np.exp(residual_logpdf(r,theta,"mixture")))
        self.assertAlmostEqual(quad(pdf,-30,10)[0],1.,places=7)
        self.assertAlmostEqual(quad(lambda r:r*pdf(r),-30,10)[0],0.,places=7)
        variance=.25**2+.2*(2-.2)/2.**2
        self.assertAlmostEqual(quad(lambda r:r*r*pdf(r),-30,10)[0],variance,places=7)
        self.assertLess(quad(lambda r:r**3*pdf(r),-30,10)[0],0.)
        draws=residual_draw(theta,"mixture",200000,np.random.default_rng(5))
        self.assertAlmostEqual(draws.mean(),0.,delta=.003)


class ForwardRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(132)
        spec=np.exp(rng.uniform(np.log(10),np.log(600),20000))
        z=rng.uniform(.1,.35,len(spec))
        cls.truth=np.array([2.5,.8,-.3])
        logmean=2.5+.8*np.log(spec/40)-.3*np.log((1+z)/1.24)
        rm=np.exp(logmean-.5*.45**2+rng.normal(0,.45,len(spec)))
        keep=rm>20
        cls.data=dict(lambda_rm=rm[keep],lambda_spec=spec[keep],z=z[keep])
        cls.reg=ForwardRegression(cls.data,'z')

    def test_selected_recovery(self):
        np.testing.assert_allclose(self.reg.mle[:2],self.truth[:2],atol=.06)
        self.assertAlmostEqual(np.exp(self.reg.mle[-1]),.45,delta=.02)

    def test_direct_density_and_nested_model(self):
        r=self.reg; t=np.array([2.5,.8,0.,np.log(.45)])
        loc=r.X@t[:-1]-.5*.45**2
        direct=(lognorm.logpdf(self.data['lambda_rm'],.45,scale=np.exp(loc))-
                lognorm.logsf(20,.45,scale=np.exp(loc))).sum()
        self.assertAlmostEqual(r.log_likelihood(t),direct,places=7)
        no=ForwardRegression(self.data,'noz')
        self.assertAlmostEqual(no.log_likelihood(t[[0,1,3]]),direct,places=7)

    def test_truncation_normalization_and_arithmetic_mean(self):
        for loc in (1.,3.,5.):
            d=lognorm(.45,scale=np.exp(loc)); survival=d.sf(20)
            self.assertAlmostEqual(quad(lambda x:d.pdf(x)/survival,20,np.inf)[0],1.,places=7)
        mu=50.
        d=lognorm(.45,scale=mu*np.exp(-.5*.45**2))
        self.assertAlmostEqual(d.mean(),mu,places=10)


if __name__ == "__main__":
    unittest.main()
