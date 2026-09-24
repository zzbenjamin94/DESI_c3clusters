"""
Trend analysis for spectroscopic richness versus redMaPPer richness.

This module is designed to be imported from
``richness_rm_vs_spec_individual.ipynb`` after the notebook has created the
cluster-level table ``rm_df`` with one row per cluster.

Typical notebook usage
----------------------

from richness_relation.richness_trend_analysis import *

analysis = make_analysis_table(
    rm_df,
    lambda_rm_col="LAMBDA",        # or "lambda_tot" if that is preferred
    lambda_spec_col="lambda_true",
    z_col="Z_LAMBDA",
)

lambda_bins = np.logspace(np.log10(20), np.log10(100), 7)
z_bins = np.array([0.1, 0.2, 0.3, 0.4])

binned = bin_lambda_z(analysis, lambda_bins=lambda_bins, z_bins=z_bins)
fig = plot_kde_binned_trends(analysis, binned, lambda_bins, z_bins)

fits = fit_models_by_redshift_bin(
    analysis,
    z_bins=z_bins,
    fit_to="individual",  # or "binned"
    binned=binned,
    nwalkers=64,
    nsteps=6000,
    burn=1500,
)

summary = summarize_fit_results(fits)
fig = plot_residual_trends(fits)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import gaussian_kde

try:
    import emcee
except ImportError as exc:  # pragma: no cover - exercised only in notebooks
    emcee = None
    _EMCEE_IMPORT_ERROR = exc


LN10 = np.log(10.0)


@dataclass
class FitResult:
    """Container for one MCMC model fit."""

    z_bin: tuple[float, float] | str
    model_name: str
    samples: np.ndarray
    best_params: dict[str, float]
    stats: dict[str, float]
    x: np.ndarray
    y: np.ndarray
    xerr: np.ndarray
    yerr: np.ndarray
    pivot_log_lambda: float
    model_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray]


def make_analysis_table(
    df: pd.DataFrame,
    lambda_rm_col: str = "LAMBDA",
    lambda_spec_col: str = "lambda_true",
    z_col: str = "Z_LAMBDA",
    lambda_rm_err_col: Optional[str] = None,
    lambda_spec_err_col: Optional[str] = None,
    min_lambda_rm: float = 0.0,
    min_lambda_spec: float = 0.0,
) -> pd.DataFrame:
    """
    Build the cluster-level table used by all later analysis.

    The output uses log10 variables:
    ``x = log10(lambda_RM)`` and ``y = log10(lambda_spec)``.
    """

    required = [lambda_rm_col, lambda_spec_col, z_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")

    out = pd.DataFrame(
        {
            "lambda_rm": np.asarray(df[lambda_rm_col], dtype=float),
            "lambda_spec": np.asarray(df[lambda_spec_col], dtype=float),
            "z": np.asarray(df[z_col], dtype=float),
        }
    )

    if "ID" in df.columns:
        out["ID"] = np.asarray(df["ID"])

    if lambda_rm_err_col is not None and lambda_rm_err_col in df.columns:
        out["lambda_rm_err"] = np.asarray(df[lambda_rm_err_col], dtype=float)
    else:
        out["lambda_rm_err"] = np.nan

    if lambda_spec_err_col is not None and lambda_spec_err_col in df.columns:
        out["lambda_spec_err"] = np.asarray(df[lambda_spec_err_col], dtype=float)
    else:
        out["lambda_spec_err"] = np.nan

    mask = (
        np.isfinite(out["lambda_rm"])
        & np.isfinite(out["lambda_spec"])
        & np.isfinite(out["z"])
        & (out["lambda_rm"] > min_lambda_rm)
        & (out["lambda_spec"] > min_lambda_spec)
    )
    out = out.loc[mask].copy()

    out["log_lambda_rm"] = np.log10(out["lambda_rm"])
    out["log_lambda_spec"] = np.log10(out["lambda_spec"])

    out["log_lambda_rm_err"] = _linear_err_to_log_err(
        out["lambda_rm"], out["lambda_rm_err"]
    )
    out["log_lambda_spec_err"] = _linear_err_to_log_err(
        out["lambda_spec"], out["lambda_spec_err"]
    )
    return out.reset_index(drop=True)


def _linear_err_to_log_err(value: pd.Series, err: pd.Series) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    err = np.asarray(err, dtype=float)
    log_err = err / (value * LN10)
    log_err[~np.isfinite(log_err) | (log_err <= 0)] = np.nan
    return log_err


def bin_lambda_z(
    table: pd.DataFrame,
    lambda_bins: np.ndarray,
    z_bins: np.ndarray,
    min_clusters_per_bin: int = 3,
) -> pd.DataFrame:
    """
    Bin clusters in redshift and redMaPPer richness.

    Summary points are computed in log space. ``yerr`` is the standard error of
    the mean log spectroscopic richness, while the 16-84 percentiles describe
    the population scatter in each bin.
    """

    rows = []
    for z_low, z_high in zip(z_bins[:-1], z_bins[1:]):
        z_sel = table[(table["z"] >= z_low) & (table["z"] < z_high)]
        for lm_low, lm_high in zip(lambda_bins[:-1], lambda_bins[1:]):
            group = z_sel[
                (z_sel["lambda_rm"] >= lm_low) & (z_sel["lambda_rm"] < lm_high)
            ]
            n = len(group)
            if n == 0:
                continue

            x = group["log_lambda_rm"].to_numpy()
            y = group["log_lambda_spec"].to_numpy()
            rows.append(
                {
                    "z_low": z_low,
                    "z_high": z_high,
                    "lambda_low": lm_low,
                    "lambda_high": lm_high,
                    "n_cluster": n,
                    "x": np.nanmean(x),
                    "y": np.nanmean(y),
                    "x_median": np.nanmedian(x),
                    "y_median": np.nanmedian(y),
                    "xerr": _sem_or_std_floor(x),
                    "yerr": _sem_or_std_floor(y),
                    "y_p16": np.nanpercentile(y, 16),
                    "y_p84": np.nanpercentile(y, 84),
                    "lambda_rm_center": np.sqrt(lm_low * lm_high),
                    "lambda_spec_mean": 10.0 ** np.nanmean(y),
                    "lambda_spec_median": 10.0 ** np.nanmedian(y),
                    "usable_for_fit": n >= min_clusters_per_bin,
                }
            )
    return pd.DataFrame(rows)


def _sem_or_std_floor(values: np.ndarray, floor: float = 0.03) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    sem = np.nanstd(values, ddof=1) / np.sqrt(len(values))
    return max(float(sem), floor / np.sqrt(len(values)))


def plot_kde_binned_trends(
    table: pd.DataFrame,
    binned: pd.DataFrame,
    lambda_bins: np.ndarray,
    z_bins: np.ndarray,
    min_lambda_spec: float = 1.0,
    max_lambda_spec: Optional[float] = None,
    kde_bw: float = 0.35,
    levels: int = 18,
    scatter_alpha: float = 0.20,
) -> plt.Figure:
    """
    Plot log-log KDEs and binned trends in each redshift slice.
    """

    n_z = len(z_bins) - 1
    fig, axes = plt.subplots(
        1, n_z, figsize=(5.2 * n_z, 4.8), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)

    xlim = (lambda_bins[0], lambda_bins[-1])
    ylim_low = min_lambda_spec
    ylim_high = max_lambda_spec or np.nanpercentile(table["lambda_spec"], 98)

    for ax, z_low, z_high in zip(axes, z_bins[:-1], z_bins[1:]):
        sub = table[(table["z"] >= z_low) & (table["z"] < z_high)].copy()
        sub = sub[
            (sub["lambda_rm"] > 0)
            & (sub["lambda_spec"] > 0)
            & np.isfinite(sub["log_lambda_rm"])
            & np.isfinite(sub["log_lambda_spec"])
        ]

        if len(sub) >= 5:
            _plot_log_kde(ax, sub, kde_bw=kde_bw, levels=levels)
            ax.scatter(
                sub["lambda_rm"],
                sub["lambda_spec"],
                s=8,
                color="white",
                edgecolor="black",
                linewidth=0.25,
                alpha=scatter_alpha,
                zorder=4,
            )

        trend = binned[
            (binned["z_low"] == z_low)
            & (binned["z_high"] == z_high)
            & binned["usable_for_fit"]
        ]
        if len(trend):
            x = 10.0 ** trend["x"].to_numpy()
            y = 10.0 ** trend["y"].to_numpy()
            y_low = 10.0 ** trend["y_p16"].to_numpy()
            y_high = 10.0 ** trend["y_p84"].to_numpy()
            yerr = np.vstack([y - y_low, y_high - y])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o-",
                color="crimson",
                ecolor="crimson",
                capsize=3,
                lw=1.8,
                ms=5,
                zorder=10,
                label="Binned log mean; 16-84%",
            )

        ax.plot(xlim, xlim, color="black", ls="--", lw=1, label="1:1")
        ax.set_title(rf"${z_low:.2f} \leq z < {z_high:.2f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim_low, ylim_high)
        ax.set_xlabel(r"$\lambda_{\rm RM}$")
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel(r"$\lambda_{\rm spec}$")
    return fig


def _plot_log_kde(
    ax: plt.Axes, table: pd.DataFrame, kde_bw: float, levels: int
) -> None:
    x = table["log_lambda_rm"].to_numpy()
    y = table["log_lambda_spec"].to_numpy()
    kde = gaussian_kde([x, y], bw_method=kde_bw)

    xgrid = np.linspace(np.nanmin(x), np.nanmax(x), 180)
    ygrid = np.linspace(np.nanmin(y), np.nanmax(y), 180)
    xx, yy = np.meshgrid(xgrid, ygrid)
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    density *= len(table)

    X = 10.0 ** xx
    Y = 10.0 ** yy
    positive = density[density > 0]
    vmin = np.nanpercentile(positive, 5) if len(positive) else 1e-6

    ax.contourf(
        X,
        Y,
        density,
        levels=levels,
        cmap="viridis",
        norm=colors.LogNorm(vmin=max(vmin, 1e-8), vmax=np.nanmax(density)),
        alpha=0.80,
        zorder=1,
    )
    ax.contour(X, Y, density, levels=8, colors="black", linewidths=0.55, zorder=2)


def loglinear_model(
    x: np.ndarray, theta: np.ndarray, pivot_log_lambda: float
) -> np.ndarray:
    alpha, beta, _log_sigma_int = theta
    return alpha + beta * (x - pivot_log_lambda)


def broken_loglinear_model(
    x: np.ndarray, theta: np.ndarray, pivot_log_lambda: float
) -> np.ndarray:
    alpha, beta_low, beta_high, x_break, _log_sigma_int = theta
    dx = x - pivot_log_lambda
    dx_break = x_break - pivot_log_lambda
    return alpha + beta_low * dx + (beta_high - beta_low) * np.maximum(
        0.0, x - x_break
    )


def fit_models_by_redshift_bin(
    table: pd.DataFrame,
    z_bins: np.ndarray,
    fit_to: str = "individual",
    binned: Optional[pd.DataFrame] = None,
    min_points: int = 6,
    nwalkers: int = 64,
    nsteps: int = 6000,
    burn: int = 1500,
    thin: int = 5,
    random_seed: int = 14,
) -> dict[tuple[float, float], dict[str, FitResult]]:
    """
    Fit log-linear and broken-log-linear relations in each redshift bin.

    Parameters
    ----------
    fit_to:
        ``"individual"`` fits all clusters. ``"binned"`` fits the 2D-binned
        summary points from ``bin_lambda_z``.
    """

    if emcee is None:
        raise ImportError("emcee is required for MCMC fitting") from _EMCEE_IMPORT_ERROR

    results = {}
    rng = np.random.default_rng(random_seed)

    for z_low, z_high in zip(z_bins[:-1], z_bins[1:]):
        z_key = (float(z_low), float(z_high))
        if fit_to == "individual":
            sub = table[(table["z"] >= z_low) & (table["z"] < z_high)]
            x = sub["log_lambda_rm"].to_numpy(dtype=float)
            y = sub["log_lambda_spec"].to_numpy(dtype=float)
            xerr = sub["log_lambda_rm_err"].to_numpy(dtype=float)
            yerr = sub["log_lambda_spec_err"].to_numpy(dtype=float)
        elif fit_to == "binned":
            if binned is None:
                raise ValueError("binned must be supplied when fit_to='binned'")
            sub = binned[
                (binned["z_low"] == z_low)
                & (binned["z_high"] == z_high)
                & binned["usable_for_fit"]
            ]
            x = sub["x"].to_numpy(dtype=float)
            y = sub["y"].to_numpy(dtype=float)
            xerr = sub["xerr"].to_numpy(dtype=float)
            yerr = sub["yerr"].to_numpy(dtype=float)
        else:
            raise ValueError("fit_to must be either 'individual' or 'binned'")

        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        xerr, yerr = _clean_log_errors(xerr[finite]), _clean_log_errors(yerr[finite])

        if len(x) < min_points:
            continue

        pivot = float(np.nanmedian(x))
        z_results = {}
        z_results["loglinear"] = run_mcmc_fit(
            x,
            y,
            xerr,
            yerr,
            model_name="loglinear",
            pivot_log_lambda=pivot,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burn=burn,
            thin=thin,
            rng=rng,
        )
        z_results["broken_loglinear"] = run_mcmc_fit(
            x,
            y,
            xerr,
            yerr,
            model_name="broken_loglinear",
            pivot_log_lambda=pivot,
            nwalkers=nwalkers,
            nsteps=nsteps,
            burn=burn,
            thin=thin,
            rng=rng,
        )
        results[z_key] = z_results

    return results


def _clean_log_errors(err: np.ndarray, default: float = 0.08) -> np.ndarray:
    err = np.asarray(err, dtype=float)
    err[~np.isfinite(err) | (err <= 0)] = default
    return err


def run_mcmc_fit(
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    model_name: str,
    pivot_log_lambda: float,
    nwalkers: int,
    nsteps: int,
    burn: int,
    thin: int,
    rng: np.random.Generator,
) -> FitResult:
    """Run one MCMC fit and return posterior samples plus fit statistics."""

    if model_name == "loglinear":
        names = ["alpha", "beta", "log_sigma_int"]
        model_func = loglinear_model
        initial = _initial_loglinear(x, y, pivot_log_lambda)
    elif model_name == "broken_loglinear":
        names = ["alpha", "beta_low", "beta_high", "x_break", "log_sigma_int"]
        model_func = broken_loglinear_model
        initial = _initial_broken(x, y, pivot_log_lambda)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    ndim = len(initial)
    p0 = initial + 1e-3 * rng.normal(size=(nwalkers, ndim))
    if model_name == "broken_loglinear":
        p0[:, 3] = rng.uniform(np.nanmin(x) + 0.05, np.nanmax(x) - 0.05, nwalkers)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        _log_probability,
        args=(x, y, xerr, yerr, model_name, pivot_log_lambda),
    )
    sampler.run_mcmc(p0, nsteps, progress=True)

    samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    log_probs = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    best = samples[np.argmax(log_probs)]
    best_params = {name: float(value) for name, value in zip(names, best)}

    stats = compute_model_statistics(
        x,
        y,
        xerr,
        yerr,
        best,
        model_name=model_name,
        pivot_log_lambda=pivot_log_lambda,
    )
    stats["max_log_likelihood"] = float(
        _log_likelihood(best, x, y, xerr, yerr, model_name, pivot_log_lambda)
    )

    return FitResult(
        z_bin="all",
        model_name=model_name,
        samples=samples,
        best_params=best_params,
        stats=stats,
        x=x,
        y=y,
        xerr=xerr,
        yerr=yerr,
        pivot_log_lambda=pivot_log_lambda,
        model_func=model_func,
    )


def _initial_loglinear(
    x: np.ndarray, y: np.ndarray, pivot_log_lambda: float
) -> np.ndarray:
    beta, intercept = np.polyfit(x - pivot_log_lambda, y, deg=1)
    resid = y - (intercept + beta * (x - pivot_log_lambda))
    sigma = max(np.nanstd(resid), 0.05)
    return np.array([intercept, beta, np.log(sigma)])


def _initial_broken(
    x: np.ndarray, y: np.ndarray, pivot_log_lambda: float
) -> np.ndarray:
    alpha, beta, log_sigma = _initial_loglinear(x, y, pivot_log_lambda)
    return np.array([alpha, beta, beta, np.nanmedian(x), log_sigma])


def _log_probability(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    model_name: str,
    pivot_log_lambda: float,
) -> float:
    lp = _log_prior(theta, x, y, model_name)
    if not np.isfinite(lp):
        return -np.inf
    return lp + _log_likelihood(theta, x, y, xerr, yerr, model_name, pivot_log_lambda)


def _log_prior(
    theta: np.ndarray, x: np.ndarray, y: np.ndarray, model_name: str
) -> float:
    if model_name == "loglinear":
        alpha, beta, log_sigma_int = theta
        if -2.0 < alpha < 4.0 and -5.0 < beta < 5.0 and -8.0 < log_sigma_int < 1.0:
            return 0.0
        return -np.inf

    alpha, beta_low, beta_high, x_break, log_sigma_int = theta
    if not (-2.0 < alpha < 4.0):
        return -np.inf
    if not (-5.0 < beta_low < 5.0 and -5.0 < beta_high < 5.0):
        return -np.inf
    if not (np.nanmin(x) + 0.03 < x_break < np.nanmax(x) - 0.03):
        return -np.inf
    if not (-8.0 < log_sigma_int < 1.0):
        return -np.inf
    return 0.0


def _log_likelihood(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    model_name: str,
    pivot_log_lambda: float,
) -> float:
    model_func = loglinear_model if model_name == "loglinear" else broken_loglinear_model
    y_model = model_func(x, theta, pivot_log_lambda)
    dydx = _model_slope(x, theta, model_name)
    sigma_int = np.exp(theta[-1])
    variance = yerr**2 + (dydx * xerr) ** 2 + sigma_int**2
    return -0.5 * np.sum((y - y_model) ** 2 / variance + np.log(2.0 * np.pi * variance))


def _model_slope(x: np.ndarray, theta: np.ndarray, model_name: str) -> np.ndarray:
    if model_name == "loglinear":
        return np.full_like(x, theta[1], dtype=float)
    beta_low, beta_high, x_break = theta[1], theta[2], theta[3]
    return np.where(x < x_break, beta_low, beta_high)


def compute_model_statistics(
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    theta: np.ndarray,
    model_name: str,
    pivot_log_lambda: float,
) -> dict[str, float]:
    """Compute chi-squared, reduced chi-squared, AIC, BIC, and residual trends."""

    model_func = loglinear_model if model_name == "loglinear" else broken_loglinear_model
    y_model = model_func(x, theta, pivot_log_lambda)
    dydx = _model_slope(x, theta, model_name)
    sigma_int = np.exp(theta[-1])
    variance = yerr**2 + (dydx * xerr) ** 2 + sigma_int**2

    residual = y - y_model
    chi2 = float(np.sum(residual**2 / variance))
    n = len(x)
    k = len(theta)
    dof = max(n - k, 1)
    log_likelihood = _log_likelihood(
        theta, x, y, xerr, yerr, model_name, pivot_log_lambda
    )

    # AIC/BIC use the full Gaussian likelihood, not only chi-squared.
    aic = float(2 * k - 2 * log_likelihood)
    bic = float(k * np.log(n) - 2 * log_likelihood)

    residual_slope, residual_intercept = np.polyfit(x - pivot_log_lambda, residual, 1)
    return {
        "n": float(n),
        "k": float(k),
        "chi2": chi2,
        "dof": float(dof),
        "reduced_chi2": chi2 / dof,
        "aic": aic,
        "bic": bic,
        "sigma_int_dex": float(sigma_int),
        "residual_mean_dex": float(np.nanmean(residual)),
        "residual_std_dex": float(np.nanstd(residual, ddof=1)),
        "residual_slope_vs_log_lambda": float(residual_slope),
        "residual_intercept": float(residual_intercept),
    }


def summarize_fit_results(
    fits: dict[tuple[float, float], dict[str, FitResult]]
) -> pd.DataFrame:
    """Return one compact table comparing models in each redshift bin."""

    rows = []
    for z_bin, model_results in fits.items():
        for model_name, result in model_results.items():
            row = {
                "z_low": z_bin[0],
                "z_high": z_bin[1],
                "model": model_name,
                **result.best_params,
                **result.stats,
            }
            rows.append(row)

        if {"loglinear", "broken_loglinear"}.issubset(model_results):
            simple = model_results["loglinear"].stats
            broken = model_results["broken_loglinear"].stats
            rows.append(
                {
                    "z_low": z_bin[0],
                    "z_high": z_bin[1],
                    "model": "delta_broken_minus_loglinear",
                    "delta_aic": broken["aic"] - simple["aic"],
                    "delta_bic": broken["bic"] - simple["bic"],
                    "delta_chi2": broken["chi2"] - simple["chi2"],
                }
            )
    return pd.DataFrame(rows)


def plot_model_fits(
    fits: dict[tuple[float, float], dict[str, FitResult]],
    model_names: tuple[str, ...] = ("loglinear", "broken_loglinear"),
) -> plt.Figure:
    """Plot data and posterior best-fit curves for each redshift bin."""

    n_z = len(fits)
    fig, axes = plt.subplots(1, n_z, figsize=(5.2 * n_z, 4.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (z_bin, model_results) in zip(axes, fits.items()):
        base = next(iter(model_results.values()))
        ax.errorbar(
            10.0**base.x,
            10.0**base.y,
            xerr=LN10 * 10.0**base.x * base.xerr,
            yerr=LN10 * 10.0**base.y * base.yerr,
            fmt=".",
            ms=4,
            alpha=0.35,
            color="0.35",
            ecolor="0.75",
            label="Data",
        )

        xline = np.linspace(np.nanmin(base.x), np.nanmax(base.x), 250)
        for model_name in model_names:
            if model_name not in model_results:
                continue
            result = model_results[model_name]
            theta = _theta_from_result(result)
            yline = result.model_func(xline, theta, result.pivot_log_lambda)
            label = _model_label(result)
            ax.plot(10.0**xline, 10.0**yline, lw=2, label=label)

        ax.plot(10.0**xline, 10.0**xline, color="black", ls="--", lw=1, label="1:1")
        ax.set_title(rf"${z_bin[0]:.2f} \leq z < {z_bin[1]:.2f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\lambda_{\rm RM}$")
        ax.legend(frameon=False, fontsize=8)

    axes[0].set_ylabel(r"$\lambda_{\rm spec}$")
    return fig


def _theta_from_result(result: FitResult) -> np.ndarray:
    if result.model_name == "loglinear":
        names = ["alpha", "beta", "log_sigma_int"]
    else:
        names = ["alpha", "beta_low", "beta_high", "x_break", "log_sigma_int"]
    return np.array([result.best_params[name] for name in names])


def _model_label(result: FitResult) -> str:
    stats = result.stats
    if result.model_name == "loglinear":
        return (
            "log-linear: "
            rf"$\beta={result.best_params['beta']:.2f}$, "
            rf"$\chi^2_\nu={stats['reduced_chi2']:.2f}$"
        )
    return (
        "broken: "
        rf"$\beta_1={result.best_params['beta_low']:.2f}$, "
        rf"$\beta_2={result.best_params['beta_high']:.2f}$"
    )


def plot_residual_trends(
    fits: dict[tuple[float, float], dict[str, FitResult]],
    model_name: str = "loglinear",
) -> plt.Figure:
    """Plot residuals versus richness for a chosen model."""

    n_z = len(fits)
    fig, axes = plt.subplots(1, n_z, figsize=(5.2 * n_z, 4.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (z_bin, model_results) in zip(axes, fits.items()):
        result = model_results[model_name]
        theta = _theta_from_result(result)
        y_model = result.model_func(result.x, theta, result.pivot_log_lambda)
        residual = result.y - y_model

        ax.axhline(0.0, color="black", ls="--", lw=1)
        ax.scatter(10.0**result.x, residual, s=14, alpha=0.55)

        slope = result.stats["residual_slope_vs_log_lambda"]
        ax.text(
            0.05,
            0.92,
            rf"$d\Delta/d\log\lambda={slope:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\lambda_{\rm RM}$")
        ax.set_title(rf"${z_bin[0]:.2f} \leq z < {z_bin[1]:.2f}$")

    axes[0].set_ylabel(
        r"$\Delta=\log_{10}\lambda_{\rm spec}-\log_{10}\lambda_{\rm model}$"
    )
    return fig


def posterior_summary(result: FitResult) -> pd.DataFrame:
    """Return median and 16/84 percentiles for a single fit result."""

    if result.model_name == "loglinear":
        names = ["alpha", "beta", "log_sigma_int"]
    else:
        names = ["alpha", "beta_low", "beta_high", "x_break", "log_sigma_int"]

    rows = []
    for idx, name in enumerate(names):
        q16, q50, q84 = np.nanpercentile(result.samples[:, idx], [16, 50, 84])
        rows.append(
            {
                "parameter": name,
                "median": q50,
                "minus_1sigma": q50 - q16,
                "plus_1sigma": q84 - q50,
            }
        )
    return pd.DataFrame(rows)
