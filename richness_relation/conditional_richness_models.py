"""Section 5 likelihoods in linear RM richness, with the manuscript notation.

No environmental terms or measurement-error deconvolution are included.
mu_RM is the lognormal median or the unboosted mixture core, respectively.
The mixture uses tau as an exponential RATE and fixes Delta_mu to zero.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import exponnorm, lognorm, norm


@dataclass(frozen=True)
class ModelConfig:
    model: str = "lognormal"
    evolution: bool = False
    lambda_piv: float = 40.0
    z_piv: float = 0.24
    rm_min: float = 20.0
    rm_max: float = float("inf")

    def __post_init__(self):
        if self.model not in {"lognormal", "mixture"}:
            raise ValueError("model must be lognormal or mixture")
        if not (self.lambda_piv > 0 and self.z_piv > -1):
            raise ValueError("Invalid pivots")
        if not (0 <= self.rm_min < self.rm_max):
            raise ValueError("Require 0 <= rm_min < rm_max")


def parameter_spec(config):
    """Proper independent uniform priors in these sampled coordinates.

    Bounds are deliberately explicit and must be checked for prior sensitivity.
    Log-scale parameters imply log-uniform priors on the physical scales.
    """
    spec = [("ln_A", np.log(1.0), np.log(1000.0)), ("B", -3.0, 3.0)]
    if config.evolution:
        spec.append(("C", -30.0, 30.0))
    if config.model == "lognormal":
        spec.append(("ln_sigma0", np.log(0.01), np.log(2.0)))
    else:
        spec.extend([
            ("ln_sigma_lambda", np.log(0.1), np.log(300.0)),
            ("f_proj", 0.0, 1.0),
            ("ln_tau", np.log(1.0 / 500.0), np.log(10.0)),
        ])
    return spec


def parameters(theta, config):
    p = dict(zip([s[0] for s in parameter_spec(config)], theta))
    p.setdefault("C", 0.0)
    return p


def mu_rm(theta, lambda_spec, z, config):
    p = parameters(theta, config)
    x = np.log(np.asarray(lambda_spec) / config.lambda_piv)
    zeta = np.log((1.0 + np.asarray(z)) / (1.0 + config.z_piv))
    return np.exp(p["ln_A"] + p["B"] * x + p["C"] * zeta)


def components(theta, lambda_spec, z, config):
    p = parameters(theta, config)
    mu = mu_rm(theta, lambda_spec, z, config)
    if config.model == "lognormal":
        return [(0.0, lognorm(s=np.exp(p["ln_sigma0"]), scale=mu))]
    sigma = np.exp(p["ln_sigma_lambda"])
    tau = np.exp(p["ln_tau"])
    f = p["f_proj"]
    with np.errstate(divide="ignore"):
        return [
            (np.log1p(-f), norm(loc=mu, scale=sigma)),
            (np.log(f), exponnorm(K=1.0 / (tau * sigma), loc=mu, scale=sigma)),
        ]


def log_difference(log_a, log_b):
    """log(exp(log_a) - exp(log_b)), including equal/zero-probability bounds."""
    log_a, log_b = np.broadcast_arrays(log_a, log_b)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = log_a + np.log(-np.expm1(np.minimum(log_b - log_a, 0.0)))
    return np.where(np.isneginf(log_b), log_a, out)


def interval_logprob(distribution, lower, upper):
    # Choose the CDF or survival representation to avoid cancellation in tails.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        log_cdf_hi = distribution.logcdf(upper)
        from_cdf = log_difference(log_cdf_hi, distribution.logcdf(lower))
        from_sf = log_difference(distribution.logsf(lower), distribution.logsf(upper))
    preferred = np.where(log_cdf_hi < np.log(0.5), from_cdf, from_sf)
    alternate = np.where(log_cdf_hi < np.log(0.5), from_sf, from_cdf)
    return np.where(np.isnan(preferred), alternate, preferred)


def selected_logpdf(theta, lambda_rm, lambda_spec, z, config):
    """Per-cluster log density in d(lambda_RM), including truncation and Jacobian."""
    comps = components(theta, lambda_spec, z, config)
    log_pdf = logsumexp([w + d.logpdf(lambda_rm) for w, d in comps], axis=0)
    log_selection = logsumexp([
        w + interval_logprob(d, config.rm_min, config.rm_max) for w, d in comps
    ], axis=0)
    rm = np.asarray(lambda_rm)
    with np.errstate(invalid="ignore"):
        result = log_pdf - log_selection
    return np.where((rm >= config.rm_min) & (rm < config.rm_max), result, -np.inf)


def selected_cdf(theta, lambda_rm, lambda_spec, z, config):
    comps = components(theta, lambda_spec, z, config)
    rm = np.clip(lambda_rm, config.rm_min, config.rm_max)
    numerator = logsumexp([
        w + interval_logprob(d, config.rm_min, rm) for w, d in comps
    ], axis=0)
    denominator = logsumexp([
        w + interval_logprob(d, config.rm_min, config.rm_max) for w, d in comps
    ], axis=0)
    return np.clip(np.exp(numerator - denominator), 0.0, 1.0)


def selected_rvs(theta, lambda_spec, z, config, rng):
    """Draw using selection-adjusted mixture probabilities, without rejection."""
    comps = components(theta, lambda_spec, z, config)
    log_mass = np.asarray([
        w + interval_logprob(d, config.rm_min, config.rm_max) for w, d in comps
    ])
    probabilities = np.exp(log_mass - logsumexp(log_mass, axis=0))
    shape = np.broadcast_arrays(lambda_spec, z)[0].shape
    component = rng.random(shape) > probabilities[0] if len(comps) == 2 else np.zeros(shape, bool)
    result = np.zeros(shape)
    for index, (_, dist) in enumerate(comps):
        u = rng.random(shape)
        # Interpolate using the more accurate tail of each component.
        cdf_lo, cdf_hi = dist.cdf(config.rm_min), dist.cdf(config.rm_max)
        sf_lo, sf_hi = dist.sf(config.rm_min), dist.sf(config.rm_max)
        q_cdf = np.clip(cdf_lo + u * (cdf_hi - cdf_lo), np.finfo(float).tiny, 1 - np.finfo(float).eps)
        q_sf = np.clip(sf_hi + u * (sf_lo - sf_hi), np.finfo(float).tiny, 1 - np.finfo(float).eps)
        draw = np.where(cdf_hi < 0.5, dist.ppf(q_cdf), dist.isf(q_sf))
        result = np.where(component == index, draw, result)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError("Nonfinite predictive draw; inspect parameter extremes")
    return result


def log_prior(theta, config):
    bounds = np.asarray([s[1:] for s in parameter_spec(config)])
    if np.any(~np.isfinite(theta)) or np.any(theta <= bounds[:, 0]) or np.any(theta >= bounds[:, 1]):
        return -np.inf
    return -float(np.log(bounds[:, 1] - bounds[:, 0]).sum())


def log_probability(theta, data, config):
    prior = log_prior(theta, config)
    if not np.isfinite(prior):
        return -np.inf
    ll = selected_logpdf(theta, data["lambda_rm"], data["lambda_spec"], data["z"], config)
    return float(prior + ll.sum()) if np.all(np.isfinite(ll)) else -np.inf


def physical_parameters(theta, config):
    """Reported symbols match Section 5; sigma0 is a natural-log dispersion."""
    p = parameters(theta, config)
    out = {"A": np.exp(p["ln_A"]), "B": p["B"]}
    if config.evolution:
        out["C"] = p["C"]
    if config.model == "lognormal":
        out["sigma0"] = np.exp(p["ln_sigma0"])
        out["sigma_log10_lambda"] = out["sigma0"] / np.log(10)
    else:
        out.update(sigma_lambda=np.exp(p["ln_sigma_lambda"]),
                   f_proj=p["f_proj"], tau=np.exp(p["ln_tau"]))
        out["tau_inverse"] = 1.0 / out["tau"]
        out["mean_projection_boost"] = out["f_proj"] / out["tau"]
    return out
