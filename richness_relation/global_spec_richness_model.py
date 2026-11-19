"""Section 5 reverse regression: P(lambda_spec | lambda_RM, z).

The location is the mean of ln(lambda_spec), not its arithmetic mean.
Scatter includes measurement noise; no errors-in-variables deconvolution.
"""
import numpy as np
from scipy.special import log_ndtr
from scipy.optimize import minimize

RM_PIVOT = 40.0
Z_PIVOT = 0.24


class NegativeTailMixture:
    """Linear Gaussian/negative-EMG mixture, selected at spec richness > 0."""
    names = ['mu_core', 'ln_sigma', 'f_low', 'ln_tail_scale']
    bounds = [(0., 1000.), (np.log(.1), np.log(500.)),
              (.00001, .99999), (np.log(.1), np.log(500.))]

    def __init__(self, values):
        self.values = np.asarray(values, float)
        if np.any(~np.isfinite(self.values)) or np.any(self.values <= 0):
            raise ValueError('Positive finite spectroscopic richness required')

    @staticmethod
    def log_survival(theta, x):
        mu, ls, f, lt = np.moveaxis(np.asarray(theta), -1, 0)
        sigma, scale = np.exp(ls), np.exp(lt)
        a = (mu-np.asarray(x))/sigma
        ratio = sigma/scale
        # S_mix = Phi(a) - f exp(-a*r+r^2/2) Phi(a-r).
        # Subtract in log space to avoid scipy EMG-CDF cancellation in tails.
        with np.errstate(invalid='ignore',divide='ignore'):
            loga = log_ndtr(a)
            logb = -a*ratio+.5*ratio**2+log_ndtr(a-ratio)
            result = loga+np.log1p(-f*np.exp(np.minimum(logb-loga,0.)))
        return np.where(np.isposinf(x),-np.inf,result)

    @staticmethod
    def log_components(theta, x):
        mu, ls, f, lt = np.moveaxis(np.asarray(theta), -1, 0)
        sigma, scale = np.exp(ls), np.exp(lt)
        delta = np.asarray(x)-mu
        g = np.log1p(-f)-ls-.5*np.log(2*np.pi)-.5*(delta/sigma)**2
        e = np.log(f)-lt+delta/scale+.5*(sigma/scale)**2+log_ndtr(-delta/sigma-sigma/scale)
        norm = NegativeTailMixture.log_survival(theta,0.)
        return g-norm, e-norm

    def log_likelihood(self, theta):
        t = np.atleast_2d(theta)
        g,e = self.log_components(t[:,None,:],self.values[None,:])
        return np.logaddexp(g,e).sum(axis=1)

    def __call__(self, theta):
        scalar = np.ndim(theta)==1
        t = np.atleast_2d(theta)
        bounds = np.asarray(self.bounds)
        good = np.all(np.isfinite(t)&(t>bounds[:,0])&(t<bounds[:,1]),axis=1)
        result = np.full(len(t),-np.inf)
        if np.any(good):
            v=t[good]
            # Uniform core location/linear sigma, Beta(2,2) fraction, log-uniform scale.
            result[good] = self.log_likelihood(v)+v[:,1]+np.log(v[:,2])+np.log1p(-v[:,2])
        return float(result[0]) if scalar else result

    @staticmethod
    def cdf(theta, x):
        logratio = NegativeTailMixture.log_survival(theta,np.asarray(x))-NegativeTailMixture.log_survival(theta,0.)
        return np.where(np.asarray(x)<=0,0.,np.clip(-np.expm1(logratio),0.,1.))


class EnvironmentMixture:
    """Free Gaussian core plus positive EMG tail, conditioned on RM > 20.

    Parameters are mu, ln(sigma_lambda), f_proj, ln(tau). The parent-cell
    mean is deliberately absent: it is only a plotting reference.
    """
    names = ['mu', 'ln_sigma_lambda', 'f_proj', 'ln_tau']
    bounds = [(-300., 300.), (np.log(.1), np.log(300.)),
              (.0001, .9999), (np.log(1/500.), np.log(10.))]

    def __init__(self, rm):
        self.rm = np.asarray(rm, dtype=float)
        if len(self.rm) < 20 or not np.all(np.isfinite(self.rm) & (self.rm > 20)):
            raise ValueError('Require >=20 finite RM richnesses strictly above 20')

    @staticmethod
    def log_survival(theta, x):
        mu, ls, f, lt = np.asarray(theta).T
        s, rate = np.exp(ls), np.exp(lt)
        a = (np.asarray(x)-mu)/s
        return np.logaddexp(log_ndtr(-a), np.log(f)-rate*(np.asarray(x)-mu)
                            + .5*(rate*s)**2+log_ndtr(a-rate*s))

    @staticmethod
    def components(theta, x):
        mu, ls, f, lt = np.asarray(theta).T
        s, rate = np.exp(ls), np.exp(lt)
        v = np.asarray(x)-mu
        g = np.log1p(-f)-ls-.5*np.log(2*np.pi)-.5*(v/s)**2
        e = np.log(f)+lt-rate*v+.5*(rate*s)**2+log_ndtr(v/s-rate*s)
        norm = EnvironmentMixture.log_survival(theta, 20.)
        support = np.asarray(x)>20
        return np.where(support, np.exp(g-norm), 0.), np.where(support, np.exp(e-norm), 0.)

    def __call__(self, theta):
        # Vectorize walkers, avoiding a Python call for every proposed walker.
        scalar = np.asarray(theta).ndim == 1
        t = np.atleast_2d(theta)
        bounds = np.asarray(self.bounds)
        good = np.all(np.isfinite(t) & (t>bounds[:,0]) & (t<bounds[:,1]), axis=1)
        result = np.full(len(t), -np.inf)
        if np.any(good):
            v = t[good]
            mu, ls, f, lt = v.T[:, :, None]
            s, rate = np.exp(ls), np.exp(lt)
            delta = self.rm[None,:]-mu
            g = np.log1p(-f)-ls-.5*np.log(2*np.pi)-.5*(delta/s)**2
            e = np.log(f)+lt-rate*delta+.5*(rate*s)**2+log_ndtr(delta/s-rate*s)
            ll = np.logaddexp(g,e).sum(axis=1)-len(self.rm)*self.log_survival(v,20.)
            result[good] = np.where(np.isfinite(ll), ll, -np.inf)
        return float(result[0]) if scalar else result

    def optimize(self, seed=42):
        rng = np.random.default_rng(seed)
        spread = max(float(np.std(self.rm)), 2.)
        starts = [[np.median(self.rm)-shift*spread, np.log(max(spread*.5,1.)), f,
                   np.log(1/spread)] for shift,f in [(0,.1),(1,.3),(2,.7),(4,.95)]]
        starts += [[rng.uniform(-60,30), np.log(rng.uniform(2,30)), rng.uniform(.02,.98),
                    np.log(1/rng.uniform(2,60))] for _ in range(4)]
        bounds = [(lo+1e-6, hi-1e-6) for lo,hi in self.bounds]
        fits = [minimize(lambda t:-self(t), np.clip(x,np.array(bounds)[:,0],np.array(bounds)[:,1]),
                         bounds=bounds, method='Nelder-Mead', options={'maxiter':2500}) for x in starts]
        fits = [r for r in fits if r.success and np.isfinite(r.fun)]
        if not fits:
            raise RuntimeError('No successful finite mixture optimum')
        return sorted(fits, key=lambda r:r.fun)

    def initial(self, rng, walkers, optimum):
        positions=[]
        while len(positions)<walkers:
            t=optimum+rng.normal(size=4)*[8.,.4,.12,.5]
            if np.isfinite(self(t)):
                positions.append(t)
        return np.asarray(positions)


class FixedMeanMixture(EnvironmentMixture):
    """Three-parameter RM mixture with a fixed per-object pre-selection mean.

    p(x|m,x>20) = Theta(x-20)*[(1-f) Normal + f EMG]/S_mix(20).
    mu_core=m-f/tau; the selected mean is not constrained to equal m.
    """
    names = ['ln_sigma_lambda', 'f_proj', 'ln_tau']
    bounds = EnvironmentMixture.bounds[1:]

    def __init__(self, rm, means, prior='legacy'):
        super().__init__(rm)
        if prior not in ('legacy', 'beta22_uniform_sigma'):
            raise ValueError('Unknown mixture prior')
        self.prior = prior
        if prior == 'beta22_uniform_sigma':
            # Numerical floor only: uniform sigma on (0,30), sampled in log sigma.
            self.bounds = [(np.log(1e-12), np.log(30.)), (0., 1.),
                           (np.log(1/500.), np.log(10.))]
        self.means = np.asarray(means, dtype=float)
        if self.means.shape != self.rm.shape or not np.all(np.isfinite(self.means) & (self.means > 0)):
            raise ValueError('Require a positive finite global mean per object')

    @staticmethod
    def component_logpdf(theta, x, means):
        ls, f, lt = theta
        sigma, rate = np.exp(ls), np.exp(lt)
        mu = np.asarray(means) - f/rate
        delta = np.asarray(x)-mu
        normal = np.log1p(-f)-ls-.5*np.log(2*np.pi)-.5*(delta/sigma)**2
        emg = np.log(f)+lt-rate*delta+.5*(rate*sigma)**2+log_ndtr(delta/sigma-rate*sigma)
        cut = (20.-mu)/sigma
        survival = np.logaddexp(log_ndtr(-cut), np.log(f)-rate*(20.-mu)
                               +.5*(rate*sigma)**2+log_ndtr(cut-rate*sigma))
        # log Theta(x-20): keep the support explicit, including at the boundary.
        support = np.where(np.asarray(x) > 20., 0., -np.inf)
        return normal-survival+support, emg-survival+support

    def __call__(self, theta):
        scalar = np.asarray(theta).ndim == 1
        t = np.atleast_2d(theta)
        bounds = np.asarray(self.bounds)
        good = np.all(np.isfinite(t) & (t>bounds[:,0]) & (t<bounds[:,1]), axis=1)
        result = np.full(len(t), -np.inf)
        if np.any(good):
            g,e = self.component_logpdf(t[good].T[:,:,None], self.rm[None,:], self.means[None,:])
            ll = np.logaddexp(g,e).sum(axis=1)
            if self.prior == 'beta22_uniform_sigma':
                ls, f, lt = t[good].T
                ll += ls - np.log(30.) + np.log(6.) + np.log(f) + np.log1p(-f)
                ll -= np.log(np.log(5000.))
            result[good] = np.where(np.isfinite(ll), ll, -np.inf)
        return float(result[0]) if scalar else result

    def components(self, theta, x):
        # Average normalized per-cluster predictions, not a model at bin center.
        g,e = self.component_logpdf(theta, np.asarray(x)[None,:], self.means[:,None])
        return np.exp(g).mean(axis=0), np.exp(e).mean(axis=0)

    def optimize(self, seed=42):
        starts = [[np.log(s), f, np.log(1/t)] for s,f,t in
                  [(3,.1,10),(10,.3,20),(20,.7,30),(5,.95,50),(30,.5,5),(1,.9,100)]]
        bounds = [(lo+1e-6, hi-1e-6) for lo,hi in self.bounds]
        fits = [minimize(lambda t:-self(t), x, bounds=bounds, method='Nelder-Mead',
                         options={'maxiter':2500}) for x in starts]
        fits = [r for r in fits if r.success and np.isfinite(r.fun)]
        if not fits:
            raise RuntimeError('No successful fixed-mean mixture optimum')
        return sorted(fits, key=lambda r:r.fun)

    def initial(self, rng, walkers, optimum):
        positions=[]
        while len(positions)<walkers:
            trial=optimum+rng.normal(size=3)*[.4,.12,.5]
            if np.isfinite(self(trial)):
                positions.append(trial)
        return np.asarray(positions)


class FixedMeanLognormal:
    """Selected RM lognormal with fixed arithmetic means and one total log scatter."""
    names = ['ln_sigma_ln_lambda']
    bounds = [(-6., 1.)]

    def __init__(self, rm, means):
        self.rm = np.asarray(rm, float)
        self.means = np.asarray(means, float)
        if (len(self.rm) < 20 or self.rm.shape != self.means.shape
                or not np.all(np.isfinite(self.rm) & (self.rm > 20))
                or not np.all(np.isfinite(self.means) & (self.means > 0))):
            raise ValueError('Require valid selected RM and positive per-object means')

    @staticmethod
    def logpdf(theta, x, means):
        ls = np.asarray(theta)[..., 0]
        s = np.exp(ls)
        ell = np.log(means) - .5*s*s
        x = np.asarray(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            density = (-np.log(x)-ls-.5*np.log(2*np.pi)
                       -.5*((np.log(x)-ell)/s)**2
                       -log_ndtr((ell-np.log(20.))/s))
        return np.where(x > 20., density, -np.inf)

    def log_likelihood(self, theta):
        t = np.atleast_2d(theta)
        return self.logpdf(t[:, None, :], self.rm[None, :], self.means[None, :]).sum(axis=1)

    def __call__(self, theta):
        t = np.atleast_2d(theta)
        good = np.isfinite(t[:, 0]) & (t[:, 0] > -6.) & (t[:, 0] < 1.)
        result = np.full(len(t), -np.inf)
        result[good] = self.log_likelihood(t[good])
        return float(result[0]) if np.ndim(theta) == 1 else result

    def density(self, theta, x):
        return np.exp(self.logpdf(theta, np.asarray(x)[None, :], self.means[:, None])).mean(axis=0)

    def cdf(self, theta, x):
        s = np.exp(theta[0]); ell = np.log(self.means)-.5*s*s
        logratio = log_ndtr((ell-np.log(x))/s)-log_ndtr((ell-np.log(20.))/s)
        return np.where(np.asarray(x) > 20., np.clip(-np.expm1(logratio), 0., 1.), 0.)

    def optimize(self, seed=42):
        results = [minimize(lambda t: -self(t), [v], method='Nelder-Mead',
                            bounds=[(-6.+1e-6, 1.-1e-6)]) for v in (-2., -1., 0.)]
        results = [r for r in results if r.success and np.isfinite(r.fun)]
        if not results: raise RuntimeError('No successful lognormal optimum')
        return sorted(results, key=lambda r: r.fun)

    def initial(self, rng, walkers, optimum):
        positions = []
        while len(positions) < walkers:
            trial = optimum+rng.normal(0., .3, 1)
            if np.isfinite(self(trial)): positions.append(trial)
        return np.asarray(positions)


class FractionalMeanMixture(FixedMeanMixture):
    """Zero preselection mean fractional residual; per-object RM selection.

    delta=RM/m-1, core=-f/tau_delta. Likelihood is expressed in RM units
    with Jacobian 1/m; plotted components are densities in delta units.
    """
    names = ['ln_sigma_delta', 'f_proj', 'ln_tau_delta']

    def __init__(self, rm, means, prior='fractional_beta22'):
        super().__init__(rm, means)
        if prior != 'fractional_beta22': raise ValueError('Unknown fractional prior')
        self.prior=prior
        self.bounds=[(np.log(1e-12),0.),(0.,1.),(np.log(.1),np.log(100.))]
        self.residual=self.rm/self.means-1
        self.threshold=20/self.means-1

    @staticmethod
    def fractional_log_survival(theta, value):
        ls,f,lt=theta
        sigma,rate=np.exp(ls),np.exp(lt)
        d=np.asarray(value)+f/rate
        return np.logaddexp(log_ndtr(-d/sigma),np.log(f)-rate*d+.5*(rate*sigma)**2+log_ndtr(d/sigma-rate*sigma))

    @staticmethod
    def fractional_component_logpdf(theta, value, threshold):
        ls,f,lt=theta
        sigma,rate=np.exp(ls),np.exp(lt)
        d=np.asarray(value)+f/rate
        g=np.log1p(-f)-ls-.5*np.log(2*np.pi)-.5*(d/sigma)**2
        e=np.log(f)+lt-rate*d+.5*(rate*sigma)**2+log_ndtr(d/sigma-rate*sigma)
        normalizer=FractionalMeanMixture.fractional_log_survival(theta,threshold)
        support=np.where(np.asarray(value)>threshold,0.,-np.inf)
        return g-normalizer+support,e-normalizer+support

    @staticmethod
    def component_logpdf(theta, x, means):
        means=np.asarray(means)
        g,e=FractionalMeanMixture.fractional_component_logpdf(theta,np.asarray(x)/means-1,20/means-1)
        return g-np.log(means),e-np.log(means)

    def __call__(self, theta):
        scalar=np.asarray(theta).ndim==1
        t=np.atleast_2d(theta); bounds=np.asarray(self.bounds)
        good=np.all(np.isfinite(t)&(t>bounds[:,0])&(t<bounds[:,1]),axis=1)
        result=np.full(len(t),-np.inf)
        if np.any(good):
            g,e=self.component_logpdf(t[good].T[:,:,None],self.rm[None,:],self.means[None,:])
            ls,f,lt=t[good].T
            posterior=np.logaddexp(g,e).sum(axis=1)+ls+np.log(6.)+np.log(f)+np.log1p(-f)-np.log(np.log(1000.))
            result[good]=np.where(np.isfinite(posterior),posterior,-np.inf)
        return float(result[0]) if scalar else result

    def components(self, theta, x):
        g,e=self.fractional_component_logpdf(theta,np.asarray(x)[None,:],self.threshold[:,None])
        return np.exp(g).mean(axis=0),np.exp(e).mean(axis=0)

    def optimize(self, seed=42):
        starts=[[np.log(s),f,np.log(1/b)] for s,f,b in
                [(.1,.1,.2),(.3,.3,.5),(.6,.7,1.),(.2,.9,2.),(.8,.5,.1),(.05,.9,5.)]]
        bounds=[(lo+1e-6,hi-1e-6) for lo,hi in self.bounds]
        fits=[minimize(lambda t:-self(t),x,bounds=bounds,method='Nelder-Mead',options={'maxiter':2500}) for x in starts]
        fits=[r for r in fits if r.success and np.isfinite(r.fun)]
        if not fits: raise RuntimeError('No finite successful fractional mixture optimum')
        return sorted(fits,key=lambda r:r.fun)


def design(data, model):
    if model not in ("noz", "z"):
        raise ValueError(model)
    cols = [np.ones(len(data["z"])), np.log(data["lambda_rm"] / RM_PIVOT)]
    if model == "z":
        cols.append(np.log((1 + data["z"]) / (1 + Z_PIVOT)))
    return np.column_stack(cols)


def parameter_names(model):
    return ["a_rev", "b_rev"] + (["c_rev"] if model == "z" else []) + ["ln_s_rev"]


class Regression:
    def __init__(self, data, model):
        self.model = model
        self.X = design(data, model)
        self.y = np.log(data["lambda_spec"])
        self.log_jacobian = float(self.y.sum())
        self.n = len(self.y)
        self.beta, _, rank, _ = np.linalg.lstsq(self.X, self.y, rcond=None)
        if rank != self.X.shape[1] or self.n <= rank:
            raise ValueError("Insufficient independent covariates")
        self.sse = float(np.sum((self.y - self.X @ self.beta) ** 2))
        if self.sse <= 0:
            raise ValueError("Zero residual variance")
        self.xtx = self.X.T @ self.X
        self.mle = np.r_[self.beta, 0.5 * np.log(self.sse / self.n)]
        self.bounds = [(0., 10.), (-5., 5.)]
        if model == "z":
            self.bounds.append((-10., 10.))
        self.bounds.append((-7., 2.))

    def log_likelihood(self, theta):
        # Centered sufficient statistics avoid repeated N-object evaluations.
        delta = np.asarray(theta[:-1]) - self.beta
        rss = self.sse + delta @ self.xtx @ delta
        logs = theta[-1]
        return float(-0.5 * rss * np.exp(-2 * logs)
                     - self.n * (logs + 0.5 * np.log(2 * np.pi)) - self.log_jacobian)

    def __call__(self, theta):
        if any(not lo < v < hi for v, (lo, hi) in zip(theta, self.bounds)):
            return -np.inf
        return self.log_likelihood(theta)

    def initial(self, rng, walkers):
        cov = np.linalg.inv(self.xtx) * np.exp(2 * self.mle[-1])
        positions = np.column_stack([
            rng.multivariate_normal(self.beta, cov, size=walkers),
            rng.normal(self.mle[-1], 1 / np.sqrt(2 * self.n), size=walkers),
        ])
        if not all(np.isfinite(self(t)) for t in positions):
            raise ValueError("Initial posterior region intersects prior bounds; inspect model/priors")
        return positions


class TruncatedReverseRegression(Regression):
    """Mean-log reverse regression with a lower spectroscopic-richness cut."""
    def __init__(self, data, model, lower=20.):
        if lower <= 0 or np.any(np.asarray(data['lambda_spec']) < lower):
            raise ValueError('Observations must satisfy the positive spec threshold')
        self.lower = float(lower)
        super().__init__(data, model)
        fits=[]
        for factor in (1.,1.3,1.8):
            start=self.mle.copy(); start[-1]+=np.log(factor)
            result=minimize(lambda t:-self.log_likelihood(t),start,method='L-BFGS-B',
                            bounds=self.bounds,options={'maxiter':2000,'ftol':1e-12})
            if result.success and np.isfinite(result.fun): fits.append(result)
        if not fits: raise RuntimeError('Truncated reverse optimization failed')
        self.mle=min(fits,key=lambda r:r.fun).x

    def log_likelihood(self, theta):
        location=self.X@np.asarray(theta[:-1])
        sigma=np.exp(theta[-1])
        return super().log_likelihood(theta)-float(log_ndtr((location-np.log(self.lower))/sigma).sum())

    def initial(self, rng, walkers):
        positions=[]
        while len(positions)<walkers:
            trial=self.mle+rng.normal(0,.008,len(self.mle))
            if np.isfinite(self(trial)): positions.append(trial)
        return np.array(positions)

    def selected_mean(self, theta):
        ell=self.X@np.asarray(theta[:-1]); sigma=np.exp(theta[-1])
        t=(ell-np.log(self.lower))/sigma
        return np.exp(ell+.5*sigma**2+log_ndtr(t+sigma)-log_ndtr(t))

    def pit(self, theta):
        ell=self.X@np.asarray(theta[:-1]); sigma=np.exp(theta[-1])
        return -np.expm1(log_ndtr((ell-self.y)/sigma)-log_ndtr((ell-np.log(self.lower))/sigma))


class CurvedReverseRegression(TruncatedReverseRegression):
    """Quadratic or continuous broken mean-log relation, selected at spec >= lower."""
    def __init__(self, data, model, lower=20.):
        if model not in ('quadratic','broken'): raise ValueError(model)
        self.model=model;self.lower=float(lower)
        self.x=np.log(np.asarray(data['lambda_rm'])/RM_PIVOT)
        self.y=np.log(np.asarray(data['lambda_spec']));self.n=len(self.y)
        if lower<=0 or not np.all(np.isfinite(self.x)&np.isfinite(self.y)) or np.any(np.exp(self.y)<lower-1e-10):
            raise ValueError('Invalid selected reverse sample')
        self.names=['A','B','D','ln_sigma_ln_spec'] if model=='quadratic' else ['A','B_low','B_high','ln_RM_break_over_40','ln_sigma_ln_spec']
        self.bounds=[(0.,10.),(-5.,5.),(-5.,5.)]+([] if model=='quadratic' else [(np.log(25/40),np.log(100/40))])+[(-7.,2.)]
        b,a=np.polyfit(self.x,self.y,1);s=np.log(np.std(self.y-a-b*self.x))
        starts=([a,b,d,s] for d in [-.2,0,.2]) if model=='quadratic' else ([a,b,b+.3,np.log(v/40),s] for v in [30,40,50,65,85])
        fits=[minimize(lambda t:-self.log_likelihood(t),start,method='Nelder-Mead',bounds=self.bounds,options={'maxiter':5000,'xatol':1e-8,'fatol':1e-8}) for start in starts]
        fits=[r for r in fits if r.success and np.isfinite(r.fun)]
        if not fits: raise RuntimeError('Curved reverse optimization failed')
        self.mle=min(fits,key=lambda r:r.fun).x

    def location(self,theta,rm=None):
        x=self.x if rm is None else np.log(np.asarray(rm)/RM_PIVOT)
        if self.model=='quadratic': return theta[0]+theta[1]*x+theta[2]*x*x
        a,bl,bh,xb=theta[:4]
        return a+bl*x+(bh-bl)*(np.maximum(x-xb,0)-max(-xb,0))

    def log_likelihood(self,theta):
        ell=self.location(theta);s=np.exp(theta[-1])
        return float(np.sum(-self.y-theta[-1]-.5*np.log(2*np.pi)-.5*((self.y-ell)/s)**2-log_ndtr((ell-np.log(self.lower))/s)))

    def pit(self,theta):
        ell=self.location(theta);s=np.exp(theta[-1])
        return -np.expm1(log_ndtr((ell-self.y)/s)-log_ndtr((ell-np.log(self.lower))/s))

    def selected_mean(self,theta):
        ell=self.location(theta);s=np.exp(theta[-1]);t=(ell-np.log(self.lower))/s
        return np.exp(ell+.5*s*s+log_ndtr(t+s)-log_ndtr(t))


class ForwardRegression:
    """Selected P(RM|spec,z); A is log pre-selection arithmetic mean at pivot.

    Only the explicit lower RM selection is modeled. Conditioning on measured
    spec does not deconvolve its measurement uncertainty or BGS selection.
    """
    def __init__(self, data, model, lower=20., upper=None, scatter_model='constant', environment=False, environment_pivot=None):
        if model not in ('noz', 'z') or lower <= 0 or upper is not None:
            raise ValueError('Forward fit currently requires a lower-only RM cut')
        self.model, self.lower = model, float(lower)
        if scatter_model not in ('constant','poisson'):
            raise ValueError('Unknown forward scatter model')
        self.scatter_model = scatter_model
        self.environment = environment
        cols = [np.ones(len(data['z'])), np.log(data['lambda_spec']/40.)]
        if model == 'z':
            cols.append(np.log((1+data['z'])/(1+Z_PIVOT)))
        if environment:
            values = np.asarray(data['environment'], dtype=float)
            if values.shape != np.asarray(data['z']).shape or not np.all(np.isfinite(values)):
                raise ValueError('Invalid environment covariate')
            self.environment_pivot = float(np.median(values) if environment_pivot is None else environment_pivot)
            cols.append(values-self.environment_pivot)
        self.X = np.column_stack(cols)
        self.y = np.log(data['lambda_rm'])
        self.n = len(self.y)
        if not np.all(np.isfinite(self.X)) or not np.all(np.isfinite(self.y)) or np.any(data['lambda_rm'] < lower):
            raise ValueError('Invalid values or observations outside RM selection')
        self.names = ['A', 'B'] + (['C'] if model == 'z' else []) + (['D'] if environment else []) + ['ln_sigma_ln_RM']
        if scatter_model == 'poisson': self.names[-1] = 'ln_sigma0'
        self.bounds = [(-5.,10.),(-5.,5.)] + ([(-10.,10.)] if model == 'z' else []) + ([(-5.,5.)] if environment else []) + [(-7.,1.)]
        self.beta, _, rank, _ = np.linalg.lstsq(self.X, self.y, rcond=None)
        if rank != self.X.shape[1] or self.n <= rank:
            raise ValueError('Insufficient independent covariates')
        self.sse = float(np.sum((self.y-self.X@self.beta)**2))
        self.xtx = self.X.T@self.X
        self.log_jacobian = float(self.y.sum())
        s = np.sqrt(self.sse/self.n)
        starts = []
        for factor in (1., 1.5, 2.):
            sigma = s*factor
            b = self.beta.copy()
            if scatter_model == 'constant': b[0] += .5*sigma**2
            start = np.r_[b, np.log(sigma)]
            if scatter_model == 'poisson':
                from scipy.optimize import LinearConstraint
                constraint=LinearConstraint(np.column_stack([self.X,np.zeros(self.n)]),1e-6,np.inf)
                starts.append(minimize(lambda t:-self.log_likelihood(t),start,
                                       bounds=self.bounds,method='SLSQP',constraints=[constraint],
                                       options={'maxiter':2000,'ftol':1e-9}))
            else:
                starts.append(minimize(lambda t: -self.log_likelihood(t), start,
                                       bounds=self.bounds, method='L-BFGS-B',
                                       options={'maxiter':2000, 'ftol':1e-12}))
        good = [r for r in starts if r.success and np.isfinite(r.fun)]
        if not good:
            raise RuntimeError('Forward truncated optimization failed')
        best = min(good, key=lambda r:r.fun)
        self.mle = best.x

    def log_likelihood(self, theta):
        if self.scatter_model == 'poisson':
            location, sigma = self.location_scale(theta)
            if np.any(location <= 0) or not np.all(np.isfinite(sigma)):
                return -np.inf
            log_survival=log_ndtr((location-np.log(self.lower))/sigma)
            return float(np.sum(-.5*((self.y-location)/sigma)**2-np.log(sigma)
                         -.5*np.log(2*np.pi)-self.y-log_survival))
        sigma = np.exp(theta[-1])
        beta = np.array(theta[:-1], copy=True)
        beta[0] -= .5*sigma**2
        delta = beta-self.beta
        rss = self.sse+delta@self.xtx@delta
        location = self.X@beta
        log_survival = log_ndtr((location-np.log(self.lower))/sigma)
        return float(-.5*rss/sigma**2-self.n*(np.log(sigma)+.5*np.log(2*np.pi))
                     -self.log_jacobian-log_survival.sum())

    def location_scale(self, theta):
        location=self.X@np.asarray(theta[:-1])
        sigma0=np.exp(theta[-1])
        if self.scatter_model == 'poisson':
            inverse=np.exp(-location)
            with np.errstate(invalid='ignore'):
                return location,np.sqrt(sigma0**2+inverse-inverse**2)
        return location-.5*sigma0**2,np.full(self.n,sigma0)

    def __call__(self, theta):
        if any(not lo < v < hi for v,(lo,hi) in zip(theta,self.bounds)):
            return -np.inf
        return self.log_likelihood(theta)

    def initial(self, rng, walkers):
        positions = []
        widths = np.array([.06,.06]+([.2] if self.model=='z' else [])+([.05] if self.environment else [])+[.04])
        while len(positions)<walkers:
            trial = self.mle+rng.normal(size=len(widths))*widths
            if np.isfinite(self(trial)):
                positions.append(trial)
        return np.array(positions)


class BrokenForwardRegression(ForwardRegression):
    """Continuous broken mean-log RM relation with z evolution and selected scatter."""
    def __init__(self,data):
        baseline=ForwardRegression(data,'z',20.,None,'poisson')
        self.model='z';self.lower=20.;self.scatter_model='poisson';self.environment=False
        self.x=np.log(np.asarray(data['lambda_spec'])/40.)
        self.zterm=np.log((1+np.asarray(data['z']))/1.24)
        self.y=np.log(data['lambda_rm']);self.n=len(self.y)
        self.names=['A','B_low','B_high','ln_spec_break_over_40','C','ln_sigma0']
        self.bounds=[(-5.,10.),(-5.,5.),(-5.,5.),(np.log(20/40),np.log(300/40)),(-10.,10.),(-7.,1.)]
        a,b,c,s=baseline.mle;fits=[]
        for value in (25,40,70,120,200):
            start=[a,b,b,np.log(value/40),c,s]
            r=minimize(lambda t:-self.log_likelihood(t),start,method='Nelder-Mead',bounds=self.bounds,options={'maxiter':6000,'xatol':1e-7,'fatol':1e-7})
            if r.success and np.isfinite(r.fun): fits.append(r)
        if not fits: raise RuntimeError('Broken forward optimizer failed')
        self.mle=min(fits,key=lambda r:r.fun).x

    def location_scale(self,theta):
        a,bl,bh,xb,c,logs=theta
        ell=a+bl*self.x+(bh-bl)*(np.maximum(self.x-xb,0)-max(-xb,0))+c*self.zterm
        inv=np.exp(-ell)
        with np.errstate(invalid='ignore'):
            s=np.sqrt(np.exp(2*logs)+inv-inv*inv)
        return ell,s

    def initial(self,rng,walkers):
        positions=[]
        while len(positions)<walkers:
            trial=self.mle+rng.normal(0,.008,len(self.mle))
            if np.isfinite(self(trial)): positions.append(trial)
        return np.asarray(positions)


def classical_rhat(chains):
    """Classical GR for independent trajectories (ensemble, step, parameter)."""
    m, n, _ = chains.shape
    if m < 2 or n < 2:
        return np.full(chains.shape[-1], np.nan)
    within = chains.var(axis=1, ddof=1).mean(axis=0)
    between = n * chains.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(((n - 1) / n * within + between / n) / within)


def residual_logpdf(residual, theta, family):
    """Density per natural-log richness: centered negative EMG mixture."""
    residual = np.asarray(residual)
    sigma = np.exp(theta[0])
    if family == "lognormal":
        return -.5*(residual/sigma)**2 - np.log(sigma) - .5*np.log(2*np.pi)
    fraction, rate = theta[1], np.exp(theta[2])
    g = residual - fraction/rate
    core = -.5*(g/sigma)**2 - np.log(sigma) - .5*np.log(2*np.pi)
    tail = (np.log(rate) + rate*g + .5*(rate*sigma)**2
            + log_ndtr(-g/sigma-rate*sigma))
    return np.logaddexp(np.log1p(-fraction)+core, np.log(fraction)+tail)


def residual_draw(theta, family, n, rng):
    sigma = np.exp(theta[0])
    r = rng.normal(0, sigma, n)
    if family == "mixture":
        fraction, rate = theta[1], np.exp(theta[2])
        r += fraction/rate - (rng.random(n) < fraction)*rng.exponential(1/rate, n)
    return r


class CellResidual:
    """Global log-location is fixed; the cell likelihood has no free intercept."""
    def __init__(self, residual, log_spec, family):
        self.residual = np.asarray(residual)
        self.log_spec = np.asarray(log_spec)
        self.family = family
        self.n = len(residual)
        self.log_jacobian = float(self.log_spec.sum())
        self.sse = float(np.sum(self.residual**2))
        self.bounds = [(-6., 1.)]
        self.names = ["ln_s_rev"]
        rms = np.sqrt(np.mean(self.residual**2))
        if rms <= 0 or self.n < 20:
            raise ValueError("Need >=20 objects and nonzero residual RMS")
        if family == "lognormal":
            self.mle = np.array([np.log(rms)])
            self.optimizer_success = True
        elif family == "mixture":
            self.bounds += [(0.001, 0.999), (-3., 5.)]
            self.names = ["ln_sigma_rev", "f_rev", "ln_tau_rev"]
            results = []
            for f, tail_scale in ((.1,.5),(.3,.5),(.6,.3),(.85,.5),(.05,1.)):
                start = [np.log(max(.05, rms*.7)), f, np.log(1/tail_scale)]
                results.append(minimize(lambda t: -self.log_likelihood(t), start,
                                        bounds=self.bounds, method="L-BFGS-B"))
            best = min(results, key=lambda r: r.fun)
            self.mle, self.optimizer_success = best.x, bool(best.success)
        else:
            raise ValueError(family)
        if any(not lo <= t <= hi for t,(lo,hi) in zip(self.mle,self.bounds)):
            raise ValueError("MLE outside adopted parameter bounds")

    def log_likelihood(self, theta):
        # Include the lambda_spec Jacobian, identical for both cell models.
        if self.family == "lognormal":
            return float(-.5*self.sse*np.exp(-2*theta[0])
                         -self.n*(theta[0]+.5*np.log(2*np.pi))-self.log_jacobian)
        return float(residual_logpdf(self.residual, theta, self.family).sum()
                     - self.log_jacobian)

    def __call__(self, theta):
        if any(not lo < t < hi for t,(lo,hi) in zip(theta,self.bounds)):
            return -np.inf
        return self.log_likelihood(theta)

    def initial(self, rng, walkers):
        positions = []
        # Independently dispersed ensembles, including starts away from the MLE.
        while len(positions) < walkers:
            if self.family == "lognormal":
                trial = self.mle + rng.normal(0, .15, 1)
            elif rng.random() < .7:
                trial = self.mle + rng.normal(0, [.25,.1,.5])
            else:
                trial = np.array([rng.uniform(-2,-.5),rng.uniform(.05,.8),
                                  rng.uniform(-.5,2.5)])
            if np.isfinite(self(trial)):
                positions.append(trial)
        return np.array(positions)
