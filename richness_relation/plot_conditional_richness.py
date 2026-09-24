"""Plot saved Section 5 fits; this module never launches optimization or MCMC."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from richness_relation.conditional_richness_models import (
    ModelConfig, mu_rm, selected_cdf, selected_logpdf,
)
from richness_relation.fit_conditional_richness_mpi import load_sample
from richness_relation.prepare_conditional_richness import DEFAULT_OUTPUT

COLORS = ["#176b93", "#b33b3b", "#27835e"]
MODEL_COLORS = {"lognormal": "#235ca3", "mixture": "#bb452d"}
PARAMETER_LABELS = {"ln_A": r"$\ln A$", "B": r"$B$", "C": r"$C$",
                    "ln_sigma0": r"$\ln\sigma_0$", "ln_sigma_lambda": r"$\ln\sigma_\lambda$",
                    "f_proj": r"$f_{\rm proj}$", "ln_tau": r"$\ln\tau$"}


def style(ax):
    ax.grid(False)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=250, bbox_inches="tight")
    plt.close(fig)


def load_results(fit_root, seed, audit):
    results = {}
    for directory in sorted(fit_root.glob(f"*/seed_{seed}")):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["data_sha256"] != audit["sample_sha256"]:
            raise ValueError(f"Input sample differs from fit in {directory}")
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            summary_path = directory / "bestfit.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        config = dict(summary["config"])
        config["rm_max"] = np.inf if config["rm_max"] is None else config["rm_max"]
        summary["model_config"] = ModelConfig(**config)
        summary["directory"] = directory
        summary["samples"] = None
        if (directory / "posterior.npz").exists() and "posterior" in summary:
            with np.load(directory / "posterior.npz") as source:
                summary["samples"] = source["samples"].copy()
        results[summary["task"]] = summary
    return results


def binned_mean_log(spec, rm, max_bins=7, min_count=20):
    nbin = min(max_bins, len(spec) // min_count)
    if nbin < 1:
        return np.empty((0, 4))
    edges = np.unique(np.quantile(spec, np.linspace(0, 1, nbin + 1)))
    points = []
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (spec >= lo) & ((spec <= hi) if index == len(edges) - 2 else (spec < hi))
        if mask.sum() < min_count:
            continue
        y = np.log(rm[mask])
        mean, sem = y.mean(), y.std(ddof=1) / np.sqrt(len(y))
        points.append([np.exp(np.mean(np.log(spec[mask]))), np.exp(mean),
                       np.exp(mean) - np.exp(mean - sem), np.exp(mean + sem) - np.exp(mean)])
    return np.asarray(points).reshape(-1, 4)


def selected_quantile(theta, spec, z, config, quantile=0.5):
    """Vector bisection for selected predictive quantiles of either model."""
    low = np.full_like(spec, max(config.rm_min, np.finfo(float).tiny))
    high = np.maximum(mu_rm(theta, spec, z, config) * 3, low + 50)
    if np.isfinite(config.rm_max):
        high[:] = config.rm_max
    else:
        for _ in range(60):
            need = selected_cdf(theta, high, spec, z, config) < quantile
            if not np.any(need):
                break
            high[need] *= 2
        else:
            raise FloatingPointError("Could not bracket predictive quantile")
    for _ in range(55):
        mid = 0.5 * (low + high)
        below = selected_cdf(theta, mid, spec, z, config) < quantile
        low, high = np.where(below, mid, low), np.where(below, high, mid)
    return 0.5 * (low + high)


def scatter_plot(data, audit, results, output, draws, rng):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True, sharey=True)
    edges = audit["z_bin_edges"]
    for i, (ax, color, lo, hi) in enumerate(zip(axes, COLORS, edges[:-1], edges[1:])):
        mask = (data["z"] >= lo) & (data["z"] < hi)
        spec, rm, z = [data[k][mask] for k in ("lambda_spec", "lambda_rm", "z")]
        ax.scatter(spec, rm, s=6, alpha=0.22, color=color, rasterized=True)
        points = binned_mean_log(spec, rm)
        if len(points):
            ax.errorbar(points[:, 0], points[:, 1], yerr=points[:, 2:].T,
                        fmt="o", ls="none", ms=5, color="black", label="Geometric mean / SEM")
        result = results.get(f"lognormal_zbin_{i + 1}")
        if result is not None and len(spec):
            config, theta = result["model_config"], result["mle_theta"]
            grid = np.geomspace(spec.min(), spec.max(), 70)
            zgrid = np.full_like(grid, np.median(z))
            ax.plot(grid, mu_rm(theta, grid, zgrid, config), color=color, ls="--", label="Underlying median")
            ax.plot(grid, selected_quantile(theta, grid, zgrid, config), color=color, label="Selected median")
            low = selected_quantile(theta, grid, zgrid, config, 0.16)
            high = selected_quantile(theta, grid, zgrid, config, 0.84)
            ax.fill_between(grid, low, high, color=color, alpha=0.09, label="Predictive 16--84% (MLE)")
            samples = result["samples"]
            if samples is not None:
                chosen = samples[rng.choice(len(samples), min(draws, len(samples)), replace=False)]
                curves = [selected_quantile(t, grid, zgrid, config) for t in chosen]
                lower, upper = np.percentile(curves, [16, 84], axis=0)
                ax.fill_between(grid, lower, upper, color=color, alpha=0.32, label="Median credible 16--84%")
        ax.text(0.04, 0.95, rf"${lo:.2f}\leq z<{hi:.2f}$", transform=ax.transAxes, va="top")
        ax.set(xscale="log", yscale="log", xlabel=r"$\lambda_{\rm spec}$")
        style(ax)
    axes[0].set_ylabel(r"$\lambda_{\rm RM}$")
    axes[0].set_xlim(data["lambda_spec"].min() * 0.8, data["lambda_spec"].max() * 1.2)
    axes[0].set_ylim(data["lambda_rm"].min() * 0.55, data["lambda_rm"].max() * 1.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.subplots_adjust(bottom=0.29, wspace=0.08)
    save(fig, output, "richness_relation_redshift_bins")


def distribution_plot(data, audit, results, output, draws, rng):
    z_edges = audit["z_bin_edges"]
    spec_edges = np.unique(np.quantile(data["lambda_spec"], [0, 1 / 3, 2 / 3, 1]))
    if len(spec_edges) < 2:
        return
    fig, axes = plt.subplots(3, len(spec_edges) - 1, squeeze=False,
                             figsize=(4.4 * (len(spec_edges) - 1), 10.5))
    for iz, (zlo, zhi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        for ix, (lo, hi) in enumerate(zip(spec_edges[:-1], spec_edges[1:])):
            ax = axes[iz, ix]
            mask = ((data["z"] >= zlo) & (data["z"] < zhi) & (data["lambda_spec"] >= lo)
                    & ((data["lambda_spec"] <= hi) if ix == len(spec_edges) - 2 else (data["lambda_spec"] < hi)))
            rm, spec, z = [data[k][mask] for k in ("lambda_rm", "lambda_spec", "z")]
            if len(rm) < 20:
                ax.text(0.5, 0.5, "Fewer than 20 clusters", transform=ax.transAxes, ha="center")
                continue
            bottom = max(float(audit["rm_min"]), rm.min() * 0.8, 1e-3)
            top = min(audit["rm_max"] or np.inf, rm.max() * 1.2)
            edges = np.geomspace(bottom, top, 18)
            # Normalize by ALL objects in the cell, not just the displayed bins.
            counts, _ = np.histogram(rm, edges)
            ax.stairs(counts / (len(rm) * np.diff(edges)), edges, color="0.35", label="Clusters")
            grid = np.geomspace(bottom, top, 100)
            for model, color in MODEL_COLORS.items():
                result = results.get(f"{model}_combined_z")
                if result is None:
                    continue
                config = result["model_config"]
                def curve(theta):
                    return np.mean(np.exp(selected_logpdf(theta, grid[:, None], spec[None, :], z[None, :], config)), axis=1)
                ax.plot(grid, curve(result["mle_theta"]), color=color,
                        ls="--" if model == "lognormal" else "-", label=model)
                if result["samples"] is not None:
                    samples = result["samples"]
                    selected = samples[rng.choice(len(samples), min(draws, len(samples)), replace=False)]
                    lower, upper = np.percentile([curve(t) for t in selected], [16, 84], axis=0)
                    ax.fill_between(grid, lower, upper, color=color, alpha=0.18)
            ax.set_xscale("log")
            ax.set_xlabel(r"$\lambda_{\rm RM}$")
            ax.set_ylabel(r"$p(\lambda_{\rm RM}\mid\lambda_{\rm spec},z,\mathrm{selected})$")
            ax.text(0.97, 0.95, rf"${zlo:.2f}\leq z<{zhi:.2f}$" + "\n" +
                    rf"${lo:.1f}\leq\lambda_{{\rm spec}}\leq {hi:.1f}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9)
            style(ax)
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
            break
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, output, "conditional_richness_distributions")


def residual_plot(data, results, output):
    for model in MODEL_COLORS:
        result = results.get(f"{model}_combined_z")
        if result is None:
            continue
        pit = selected_cdf(result["mle_theta"], data["lambda_rm"], data["lambda_spec"], data["z"], result["model_config"])
        score = norm.ppf(np.clip(pit, 1e-8, 1 - 1e-8))
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
        for ax, x, label in zip(axes[:2], [data["lambda_spec"], data["z"]],
                                [r"$\lambda_{\rm spec}$", r"$z_{\rm BCG}$"]):
            ax.scatter(x, score, c=data["z"], cmap="viridis", s=6, alpha=0.3, rasterized=True)
            ax.axhline(0, color="black", lw=1)
            ax.set_xlabel(label)
            ax.set_ylabel(r"$\Phi^{-1}(F_{\rm selected}(\lambda_{\rm RM}))$")
        axes[0].set_xscale("log")
        axes[2].hist(pit, bins=np.linspace(0, 1, 16), density=True, histtype="step", color=MODEL_COLORS[model])
        axes[2].axhline(1, color="black", ls="--")
        axes[2].set(xlabel="Selected conditional CDF", ylabel="Probability density")
        for ax in axes:
            style(ax)
        fig.tight_layout()
        save(fig, output, f"{model}_conditional_residuals")


def diagnostic_plots(results, output):
    import h5py
    for name, result in results.items():
        if result["samples"] is None:
            continue
        labels = result["parameter_names"]
        fig, axes = plt.subplots(len(labels), 1, figsize=(9, 1.8 * len(labels)), squeeze=False)
        for index, path in enumerate(sorted(result["directory"].glob("chain_*.h5"))):
            with h5py.File(path) as handle:
                n = int(handle["mcmc"].attrs["iteration"])
                chain = handle["mcmc/chain"][:n:max(1, n // 1500)]
            for j, ax in enumerate(axes[:, 0]):
                ax.plot(np.arange(len(chain)) * max(1, n // 1500), chain[:, :, j].mean(axis=1),
                        label=f"Ensemble {index}", lw=1)
                ax.set_ylabel(PARAMETER_LABELS[labels[j]])
                style(ax)
        axes[0, 0].legend()
        axes[-1, 0].set_xlabel("MCMC step (walker mean shown; inspect individual chains for detailed checks)")
        fig.tight_layout()
        save(fig, output, f"{name}_traces")
        fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 2.8), squeeze=False)
        for j, ax in enumerate(axes[0]):
            ax.hist(result["samples"][:, j], bins=35, density=True, histtype="step", color="black")
            ax.set_xlabel(PARAMETER_LABELS[labels[j]])
            style(ax)
        fig.tight_layout()
        save(fig, output, f"{name}_posterior_marginals")


def comparison_tables(results, output):
    rows, parameters = [], []
    for name, result in results.items():
        rows.append({"task": name, "N": result["n_clusters"],
                     "heldout_log_density": result.get("cross_validation", {}).get("total_heldout_log_density", np.nan),
                     **{k: result[k] for k in ("AIC", "BIC", "AICc", "log_likelihood_max")}})
        for parameter, value in result["mle_parameters"].items():
            row = {"task": name, "parameter": parameter, "MLE": value}
            row.update(result.get("posterior", {}).get(parameter, {}))
            parameters.append(row)
    summary = pd.DataFrame(rows)
    if len(summary):
        summary["comparison_group"] = summary["task"].str.replace(r"^(lognormal|mixture)_", "", regex=True)
        # All combined fits share observations; each redshift bin is its own comparison.
        summary["comparison_group"] = summary["comparison_group"].replace({"combined_noz": "combined", "combined_z": "combined"})
        for criterion in ("AIC", "BIC"):
            summary[f"delta_{criterion}"] = summary[criterion] - summary.groupby("comparison_group")[criterion].transform("min")
        output.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output / "model_comparison.csv", index=False)
        pd.DataFrame(parameters).to_csv(output / "parameter_summary.csv", index=False)
    return summary


def plot_all(data_path=DEFAULT_OUTPUT, fit_root=ROOT / "catalogs/conditional_richness/fits",
             output=ROOT / "plots/conditional_richness", seed=42, draws=30):
    data, audit = load_sample(Path(data_path))
    results = load_results(Path(fit_root), seed, audit)
    output = Path(output)
    rng = np.random.default_rng(seed)
    scatter_plot(data, audit, results, output, draws, rng)
    distribution_plot(data, audit, results, output, draws, rng)
    residual_plot(data, results, output)
    diagnostic_plots(results, output)
    table = comparison_tables(results, output)
    print(f"Read {len(results)} fits. Saved figures and comparison tables under {output}")
    for name, result in results.items():
        if "chain_diagnostics" in result:
            print(name, "50-tau length checks:", [d["length_exceeds_50_tau"] for d in result["chain_diagnostics"]])
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fits", type=Path, default=ROOT / "catalogs/conditional_richness/fits")
    parser.add_argument("--output", type=Path, default=ROOT / "plots/conditional_richness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=30)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    print(plot_all(args.data, args.fits, args.output, args.seed, args.draws).to_string(index=False))
