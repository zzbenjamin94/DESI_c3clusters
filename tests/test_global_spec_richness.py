"""Focused mathematical checks for the reverse conditional regression."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import lognorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "richness_relation"))
from global_spec_richness_model import Regression, ForwardRegression, design, classical_rhat, residual_logpdf, residual_draw
from scipy.integrate import quad


def test_curved_reverse_nesting_and_normalization():
    from global_spec_richness_model import CurvedReverseRegression, TruncatedReverseRegression
    rng=np.random.default_rng(92)
    rm=np.geomspace(20,150,150)
    data=dict(lambda_rm=rm,lambda_spec=np.exp(4.7+.7*np.log(rm/40)+rng.normal(0,.2,len(rm))),z=np.full(len(rm),.2))
    base=TruncatedReverseRegression(data,'noz')
    for model,t in [('quadratic',[4.7,.7,0,np.log(.2)]),('broken',[4.7,.7,.7,np.log(45/40),np.log(.2)])]:
        reg=CurvedReverseRegression(data,model)
        np.testing.assert_allclose(reg.log_likelihood(t),base.log_likelihood([4.7,.7,np.log(.2)]))
        np.testing.assert_allclose(reg.location(t,np.array([40])),[4.7])
        np.testing.assert_allclose(reg.pit(t),base.pit([4.7,.7,np.log(.2)]))
    t=[4.7,.5,1.1,np.log(45/40),np.log(.3)]
    assert abs(np.diff(reg.location(t,np.array([45-1e-7,45+1e-7])))[0])<1e-7
    ell=reg.location(t,np.array([40]))[0];s=np.exp(t[-1])
    integral=quad(lambda v:lognorm.pdf(v,s,scale=np.exp(ell))/lognorm.sf(20,s,scale=np.exp(ell)),20,np.inf)[0]
    assert np.isclose(integral,1.)


def test_forward_histogram_selected_mass_and_overflow():
    from summarize_global_spec_richness import forward_histogram_discrepancy
    rng=np.random.default_rng(19)
    data=dict(lambda_spec=rng.uniform(20,300,600),lambda_rm=rng.uniform(21,250,600),z=rng.uniform(.1,.35,600))
    reg=ForwardRegression(data,'z',scatter_model='poisson')
    cells=np.arange(600)%6
    stats,records=forward_histogram_discrepancy(data,reg,reg.mle,cells,np.r_[np.linspace(20,200,16),np.inf])
    assert np.isclose(stats[0],sum(stats[1:]))
    for cell in range(1,7):
        rows=[r for r in records if r['cell']==cell]
        assert sum(r['observed'] for r in rows)==100
        assert np.isclose(sum(r['expected'] for r in rows),100)
        assert all(r['expected']>=5 for r in rows)
        assert np.isinf(rows[-1]['right'])


def test_broken_forward_nests_single_power_law():
    from global_spec_richness_model import BrokenForwardRegression
    rng=np.random.default_rng(711)
    data=dict(lambda_spec=rng.uniform(20,300,150),lambda_rm=rng.uniform(21,100,150),z=rng.uniform(.1,.35,150))
    base=ForwardRegression(data,'z',scatter_model='poisson')
    broken=BrokenForwardRegression(data)
    a,b,c,s=base.mle;t=[a,b,b,np.log(100/40),c,s]
    np.testing.assert_allclose(base.location_scale(base.mle),broken.location_scale(t))
    np.testing.assert_allclose(base.log_likelihood(base.mle),broken.log_likelihood(t))
    assert broken.log_likelihood(broken.mle)>=base.log_likelihood(base.mle)-1e-5


def test_poisson_scatter_selected_likelihood():
    from scipy.stats import truncnorm
    rng=np.random.default_rng(52)
    spec=np.exp(rng.uniform(np.log(15),np.log(400),3000))
    z=rng.uniform(.1,.35,len(spec))
    ell=2.5+.7*np.log(spec/40)+.3*np.log((1+z)/1.24)
    scale=np.sqrt(.35**2+np.exp(-ell)-np.exp(-2*ell))
    y=truncnorm.rvs((np.log(20)-ell)/scale,np.inf,loc=ell,scale=scale,random_state=rng)
    data=dict(lambda_spec=spec,lambda_rm=np.exp(y),z=z)
    reg=ForwardRegression(data,'z',scatter_model='poisson')
    theta=np.array([2.5,.7,.3,np.log(.35)])
    loc,s=reg.location_scale(theta)
    np.testing.assert_allclose(loc,ell)
    np.testing.assert_allclose(s,scale)
    direct=(lognorm.logpdf(data['lambda_rm'],s,scale=np.exp(loc))
            -lognorm.logsf(20,s,scale=np.exp(loc))).sum()
    np.testing.assert_allclose(reg.log_likelihood(theta),direct)
    assert reg.log_likelihood(reg.mle)>=reg.log_likelihood(theta)
    for i in (0,100,1000):
        density=lambda v: lognorm.pdf(v,s[i],scale=np.exp(loc[i]))/lognorm.sf(20,s[i],scale=np.exp(loc[i]))
        assert np.isclose(quad(density,20,np.inf)[0],1.)
    no=ForwardRegression(data,'noz',scatter_model='poisson')
    np.testing.assert_allclose(no.log_likelihood(theta[[0,1,3]]),reg.log_likelihood([2.5,.7,0,np.log(.35)]))
    assert reg([-4,0,0,np.log(.35)]) == -np.inf


def test_forward_environment_nesting_and_normalization():
    rng=np.random.default_rng(813)
    data=dict(lambda_spec=rng.uniform(30,250,250),lambda_rm=rng.uniform(21,80,250),
              z=rng.uniform(.1,.35,250),environment=rng.normal(0,.3,250))
    base=ForwardRegression(data,'z',scatter_model='poisson')
    reg=ForwardRegression(data,'z',scatter_model='poisson',environment=True)
    theta=np.array([2.5,.7,-.4,0.,np.log(.35)])
    np.testing.assert_allclose(reg.log_likelihood(theta),base.log_likelihood(theta[[0,1,2,4]]))
    theta[3]=.2
    ell,s=reg.location_scale(theta)
    np.testing.assert_allclose(ell,base.X@theta[:3]+.2*(data['environment']-np.median(data['environment'])))
    expected=(lognorm.logpdf(data['lambda_rm'],s,scale=np.exp(ell))-lognorm.logsf(20,s,scale=np.exp(ell))).sum()
    np.testing.assert_allclose(reg.log_likelihood(theta),expected)
    assert np.isclose(quad(lambda v: lognorm.pdf(v,s[0],scale=np.exp(ell[0]))/lognorm.sf(20,s[0],scale=np.exp(ell[0])),20,np.inf)[0],1)
    assert reg.names==['A','B','C','D','ln_sigma0']


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
