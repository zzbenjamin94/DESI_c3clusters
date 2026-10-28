"""Checks for the four-parameter, selected environment-cell likelihood."""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm, exponnorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'richness_relation'))
from global_spec_richness_model import EnvironmentMixture, FixedMeanMixture


def test_beta_uniform_sigma_prior_and_jacobian():
    rm=np.linspace(21,80,30); means=np.full(30,35.)
    old=FixedMeanMixture(rm,means)
    new=FixedMeanMixture(rm,means,'beta22_uniform_sigma')
    theta=np.array([np.log(6.),.3,np.log(.05)])
    expected=np.log(6./30.)+np.log(6*.3*.7)-np.log(np.log(5000.))
    assert np.isclose(new(theta)-old(theta),expected)
    assert np.allclose(new(np.array([theta,theta])),new(theta))
    assert not np.isfinite(new([np.log(31.),.3,np.log(.05)]))
    for f in (0.,1.): assert not np.isfinite(new([np.log(6.),f,np.log(.05)]))
    assert np.allclose(new.components(theta,rm),old.components(theta,rm))


def test_pooled_quartiles():
    import pandas as pd
    from fit_global_spec_richness import richness_redshift_cells
    frame=pd.DataFrame(dict(ID=np.arange(80),lambda_spec=np.arange(1,81),
                            lambda_rm=np.linspace(21,90,80),z=np.linspace(.1,.349,80)))
    cells,edges,z=richness_redshift_cells(frame,True)
    assert cells.groupby('cell').size().tolist()==[20]*4
    assert len(edges)==5 and np.allclose(z,[.1,.35])


def test_fixed_global_mean_support_normalization_and_likelihood():
    theta=np.array([np.log(5.),.3,np.log(.05)])
    rm=np.linspace(21,100,40)
    means=np.linspace(12,70,40)
    reg=FixedMeanMixture(rm,means)
    mu=means-.3/.05
    assert np.allclose(mu+.3/.05,means)
    expected=(.7*norm.pdf(rm,mu,5)+.3*exponnorm.pdf(rm,4,loc=mu,scale=5))
    survival=.7*norm.sf(20,mu,5)+.3*exponnorm.sf(20,4,loc=mu,scale=5)
    assert np.isclose(reg(theta),np.log(expected/survival).sum())
    assert np.allclose(reg(np.stack([theta,theta])),reg(theta))
    for mean in (12.,30.,70.):
        integral=quad(lambda x:sum(np.exp(v) for v in reg.component_logpdf(theta,x,mean)),20,np.inf)[0]
        assert np.isclose(integral,1,atol=1e-7)
        g,e=reg.component_logpdf(theta,np.array([19.,20.]),mean)
        assert np.all(np.isneginf(g)) and np.all(np.isneginf(e))
    assert np.isclose(quad(lambda x:sum(reg.components(theta,np.array([x])))[0],20,np.inf)[0],1)
    assert len(reg.names)==3


def test_richness_tertiles_and_redshift_cells():
    import pandas as pd
    from fit_global_spec_richness import richness_redshift_cells
    frame=pd.DataFrame(dict(ID=np.arange(90),lambda_spec=np.arange(1,91),
                            lambda_rm=np.arange(21,111),z=np.tile([.1,.18,.24],30)))
    cells,edges,zedges=richness_redshift_cells(frame)
    assert cells.groupby('spec_bin').size().tolist()==[30,30,30]
    assert cells.groupby('cell').size().tolist()==[10]*9
    assert cells.quartile.eq(0).all()
    assert cells.ID.nunique()==90
    assert np.array_equal(zedges,[.1,.18,.24,.35])
    assert cells.loc[cells.z==.18,'z_bin'].eq(2).all()


def test_de_moves_checkpoint_resume(tmp_path):
    import emcee
    from fit_global_spec_richness import ensemble_moves
    meta={'proposal':'de_snooker'}
    rng=np.random.default_rng(8)
    start=rng.normal(size=(16,3))
    def logp(x): return -.5*np.sum(x*x,axis=-1)
    backend=emcee.backends.HDFBackend(str(tmp_path/'de.h5')); backend.reset(16,3)
    sampler=emcee.EnsembleSampler(16,3,logp,vectorize=True,backend=backend,moves=ensemble_moves(meta))
    sampler.random_state=np.random.RandomState(9).get_state()
    sampler.run_mcmc(start,20)
    prefix=backend.get_chain().copy()
    sampler=emcee.EnsembleSampler(16,3,logp,vectorize=True,backend=backend,moves=ensemble_moves(meta))
    sampler.run_mcmc(None,20)
    fresh=emcee.EnsembleSampler(16,3,logp,vectorize=True,moves=ensemble_moves(meta))
    fresh.random_state=np.random.RandomState(9).get_state()
    fresh.run_mcmc(start,40)
    assert np.array_equal(backend.get_chain()[:20],prefix)
    assert np.array_equal(backend.get_chain(),fresh.get_chain())


def test_normalization_and_components():
    x=np.linspace(20.1,150,200)
    for mu in (-30.,10.77,21.82,80.):
        theta=np.array([mu,np.log(5.),.1,np.log(.05)])
        normal,tail=EnvironmentMixture.components(theta,x)
        survival=.9*norm.sf(20,mu,5)+.1*exponnorm.sf(20,4,loc=mu,scale=5)
        expected=(.9*norm.pdf(x,mu,5)+.1*exponnorm.pdf(x,4,loc=mu,scale=5))/survival
        assert np.allclose(normal+tail,expected)
        integral=quad(lambda v:sum(EnvironmentMixture.components(theta,v)),20,np.inf)[0]
        assert np.isclose(integral,1.,atol=1e-7)
        assert sum(EnvironmentMixture.components(theta,19.))==0
        reg=EnvironmentMixture(x)
        assert np.isclose(reg(theta),np.log(expected).sum())
        assert np.allclose(reg(np.array([theta,theta])),[reg(theta),reg(theta)])


def test_no_parent_mean_constraint():
    reg=EnvironmentMixture(np.linspace(21,100,50))
    theta=np.array([-10.,np.log(5.),.2,np.log(.05)])
    assert np.isfinite(reg(theta))
    assert reg(np.array([301.,*theta[1:]])) == -np.inf
    assert len(reg.names)==4


def test_far_below_cut_exponential_limit():
    # Once only the exponential tail is selected, location and fraction cancel.
    x=np.linspace(21.,100.,100)
    expected=.05*np.exp(-.05*(x-20))
    for mu,f in [(-100.,.01),(-200.,.9)]:
        theta=np.array([mu,np.log(5.),f,np.log(.05)])
        assert np.allclose(sum(EnvironmentMixture.components(theta,x)),expected)


def test_adaptive_requires_stability_and_two_passes():
    from fit_global_spec_richness import adaptive_checkpoint
    report=dict(tau_steps=[[10.]*4]*4,steps_per_tau=[[100.]*4]*4,
                rhat_worst=[1.01]*4,convergence_pass=True,steps=10000)
    first=adaptive_checkpoint(report,None)
    assert not first['checkpoint_pass']
    report['steps']=15000
    second=adaptive_checkpoint(report,first)
    assert second['checkpoint_pass'] and not second['converged']
    report['steps']=20000
    third=adaptive_checkpoint(report,second)
    assert third['converged']
    report['tau_steps']=[[20.]*4]*4
    assert not adaptive_checkpoint(report,third)['checkpoint_pass']
    report['tau_steps']=[[10.]*4]*4
    report['convergence_pass']=False
    assert adaptive_checkpoint(report,third)['consecutive_passes']==0


def test_hdf_resume_preserves_prefix_and_random_state(tmp_path):
    import emcee
    from fit_global_spec_richness import advance_environment_ensemble
    rm=np.linspace(21,60,25)
    meta=dict(walkers=8,ensembles=2,seed=42)
    optima=np.array([[25.,np.log(5.),.2,np.log(.1)]])
    split=tmp_path/'split'; split.mkdir()
    whole=tmp_path/'whole'; whole.mkdir()
    advance_environment_ensemble((split,rm,meta,optima,10,0))
    prefix=emcee.backends.HDFBackend(str(split/'chain_0.h5'),read_only=True).get_chain()
    advance_environment_ensemble((split,rm,meta,optima,20,0))
    continued=emcee.backends.HDFBackend(str(split/'chain_0.h5'),read_only=True).get_chain()
    advance_environment_ensemble((whole,rm,meta,optima,20,0))
    uninterrupted=emcee.backends.HDFBackend(str(whole/'chain_0.h5'),read_only=True).get_chain()
    np.testing.assert_array_equal(continued[:10],prefix)
    np.testing.assert_array_equal(continued,uninterrupted)


def test_budget_extension_preserves_target(tmp_path):
    import json
    import pytest
    from fit_global_spec_richness import extend_environment_budget
    old=dict(model_sha256='same-model',burn=5000,runner_sha256='original-runner',
             adaptive=dict(max_steps=100000,max_seconds=3600,block_steps=5000,tau_stability=.1))
    path=tmp_path/'run.json';path.write_text(json.dumps(old))
    new=json.loads(json.dumps(old));new['adaptive']['max_steps']=200000
    new['runner_sha256']='budget-aware-runner'
    extend_environment_budget(tmp_path,new)
    assert json.loads(path.read_text())==new
    assert json.loads((tmp_path/'budget_revisions.json').read_text())[0]['previous']==old
    changed=json.loads(json.dumps(new));changed['model_sha256']='different-model'
    changed['adaptive']['max_steps']=300000
    with pytest.raises(ValueError):extend_environment_budget(tmp_path,changed)
