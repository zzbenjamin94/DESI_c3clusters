"""Fit Section 5 conditional richness models, serially or with MPI/emcee.

Run with --optimize-only for preliminary lines, then rerun with MCMC.
All paths default to the repository containing this script.
"""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from richness_relation.conditional_richness_models import (
    ModelConfig, log_probability, parameter_spec, physical_parameters,
    selected_cdf, selected_logpdf, selected_rvs,
)
from richness_relation.prepare_conditional_richness import DEFAULT_OUTPUT, file_sha256, load_sample

_TASKS = {}


def evaluate(theta, task_name):
    # Initialized once per worker, so a likelihood job transmits only theta.
    config, data = _TASKS[task_name]
    return log_probability(theta, data, config)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_tasks(args, data, audit):
    models = ["lognormal", "mixture"] if args.model == "both" else [args.model]
    selections = []
    if args.scope in {"suite", "combined"}:
        selections.extend([("combined_noz", False, np.ones(len(data["z"]), bool)),
                           ("combined_z", True, np.ones(len(data["z"]), bool))])
    if args.scope in {"suite", "bins"}:
        edges = audit["z_bin_edges"]
        selections.extend((f"zbin_{i + 1}", False, (data["z"] >= lo) & (data["z"] < hi))
                          for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])))
    for model in models:
        for name, evolution, mask in selections:
            config = ModelConfig(model, evolution, args.lambda_piv, args.z_piv,
                                 audit["rm_min"], audit["rm_max"] or np.inf)
            yield f"{model}_{name}", config, {k: v[mask] for k, v in data.items()}


def optimize(data, config, rng, starts):
    x = np.log(data["lambda_spec"] / config.lambda_piv)
    zeta = np.log((1 + data["z"]) / (1 + config.z_piv))
    design = np.column_stack([np.ones(len(x)), x] + ([zeta] if config.evolution else []))
    estimate = np.linalg.lstsq(design, np.log(data["lambda_rm"]), rcond=None)[0]
    scatter = max(float(np.std(np.log(data["lambda_rm"]) - design @ estimate)), 0.1)
    scales = [np.log(scatter)] if config.model == "lognormal" else [np.log(10.0), 0.2, np.log(1 / 20)]
    initial = np.concatenate([estimate, scales])
    bounds = np.asarray([p[1:] for p in parameter_spec(config)])
    width = bounds[:, 1] - bounds[:, 0]
    lower, upper = bounds[:, 0] + 1e-6 * width, bounds[:, 1] - 1e-6 * width
    initial = np.clip(initial, lower, upper)

    def objective(theta):
        ll = selected_logpdf(theta, data["lambda_rm"], data["lambda_spec"], data["z"], config)
        return -float(ll.sum()) if np.all(np.isfinite(ll)) else 1e100

    results = []
    for attempt in range(starts):
        guess = initial if attempt == 0 else np.clip(initial + rng.normal(0, 0.07, len(width)) * width, lower, upper)
        result = minimize(objective, guess, method="L-BFGS-B", bounds=list(zip(lower, upper)),
                          options={"maxiter": 3000, "ftol": 1e-10})
        if result.success and np.isfinite(result.fun) and result.fun < 1e99:
            results.append(result)
    if not results:
        raise RuntimeError("All optimizer starts failed; inspect selection, priors, and data")
    return min(results, key=lambda r: r.fun)


def posterior_summary(samples, config):
    values = [physical_parameters(t, config) for t in samples]
    return {name: dict(zip(["p16", "median", "p84"],
                          np.percentile([v[name] for v in values], [16, 50, 84]).tolist()))
            for name in values[0]}


def discrepancies(rm, spec, z, theta, config):
    # PIT accounts for both the asymmetric distribution and RM truncation.
    pit = np.clip(selected_cdf(theta, rm, spec, z, config), 1e-10, 1 - 1e-10)
    ordered = np.sort(pit)
    n = len(pit)
    score = norm.ppf(pit)
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
    return {
        "pit_ks_distance": float(max(np.max(np.arange(1, n + 1) / n - ordered),
                                      np.max(ordered - np.arange(n) / n))),
        "normal_score_chi2": float(score @ score),
        "upper_tail_fraction": float(np.mean(pit > 0.95)),
        "lower_tail_fraction": float(np.mean(pit < 0.05)),
        "abs_residual_richness_correlation": abs(corr(score, np.log(spec))),
        "abs_residual_redshift_correlation": abs(corr(score, z)),
    }


def predictive_checks(samples, data, config, rng, count):
    observed, replicated = [], []
    for index in rng.integers(len(samples), size=count):
        theta = samples[index]
        simulated = selected_rvs(theta, data["lambda_spec"], data["z"], config, rng)
        observed.append(discrepancies(data["lambda_rm"], data["lambda_spec"], data["z"], theta, config))
        replicated.append(discrepancies(simulated, data["lambda_spec"], data["z"], theta, config))
    return {name: {
        "observed_median": float(np.median([v[name] for v in observed])),
        "replicated_p16_p50_p84": np.percentile([v[name] for v in replicated], [16, 50, 84]).tolist(),
        "ppc_probability_rep_ge_obs": float(np.mean([r[name] >= o[name] for r, o in zip(replicated, observed)])),
    } for name in observed[0]}


def cross_validation(data, config, folds, seed, starts):
    """Optional held-out log density from refitted MLEs, not Bayesian evidence."""
    fold_rng = np.random.default_rng(seed)
    indices = np.array_split(fold_rng.permutation(len(data["z"])), folds)
    scores = np.empty(len(data["z"]))
    fold_means = []
    for held_out in indices:
        train = np.ones(len(scores), bool)
        train[held_out] = False
        fit = optimize({k: v[train] for k, v in data.items()}, config, fold_rng, starts)
        scores[held_out] = selected_logpdf(fit.x, data["lambda_rm"][held_out],
                                         data["lambda_spec"][held_out], data["z"][held_out], config)
        fold_means.append(float(scores[held_out].mean()))
    return scores, {"folds": folds, "total_heldout_log_density": float(scores.sum()),
                    "mean_heldout_log_density": float(scores.mean()), "fold_mean_log_densities": fold_means,
                    "note": "Random cluster folds and plug-in MLE predictions; shared estimator systematics are not independent across folds."}


def fit_task(name, config, data, audit, args, pool):
    import emcee
    directory = args.output / name / f"seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    encoded = asdict(config)
    encoded["rm_max"] = None if np.isinf(config.rm_max) else config.rm_max
    signature = {"data_sha256": audit["sample_sha256"], "config": encoded,
                 "IDs": data["ID"].tolist(), "walkers": args.walkers,
                 "seed": args.seed, "ensembles": args.ensembles,
                 "burn": args.burn, "thin": args.thin, "schema_version": 1,
                 "code_sha256": {p.name: file_sha256(p) for p in [
                     Path(__file__), Path(__file__).with_name("conditional_richness_models.py")]}}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{directory} exists; use --resume or a different --seed/--output")
        if json.loads(manifest_path.read_text()) != signature:
            raise ValueError("Resume configuration or input changed; use a new output directory")
    else:
        write_json(manifest_path, signature)
    rng = np.random.default_rng(args.seed)
    mle = optimize(data, config, rng, args.starts)
    ndim, n = len(mle.x), len(data["z"])
    ll_max = float(-mle.fun)
    summary = {
        "task": name, "n_clusters": n, "config": encoded,
        "parameter_names": [s[0] for s in parameter_spec(config)],
        "priors_uniform_in_sampled_coordinates": {s[0]: list(s[1:]) for s in parameter_spec(config)},
        "Delta_mu_fixed": 0.0, "mle_theta": mle.x.tolist(),
        "mle_parameters": physical_parameters(mle.x, config),
        "log_likelihood_max": ll_max, "AIC": 2 * ndim - 2 * ll_max,
        "BIC": ndim * np.log(n) - 2 * ll_max,
        "AICc": 2 * ndim - 2 * ll_max + 2 * ndim * (ndim + 1) / (n - ndim - 1) if n > ndim + 1 else None,
        "mle_discrepancies": discrepancies(data["lambda_rm"], data["lambda_spec"], data["z"], mle.x, config),
        "warnings": audit["warnings"] + [
            "AIC/BIC are supporting diagnostics for mixtures, not Bayes factors.",
            "Normal-score chi2 and PPC probabilities are not classical chi2 p-values.",
            "RM threshold normalization assumes a hard observed-richness selection.",
        ],
        "package_versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "emcee", "h5py")},
    }
    write_json(directory / "bestfit.json", summary)
    if args.cv_folds:
        scores, report = cross_validation(data, config, args.cv_folds, args.seed, args.starts)
        summary["cross_validation"] = report
        np.savez_compressed(directory / "cross_validation.npz", ID=data["ID"], log_density=scores)
        write_json(directory / "bestfit.json", summary)
    if args.optimize_only:
        print(f"{name}: MLE saved; no posterior sampled", flush=True)
        return
    retained, diagnostics, ensemble_summaries = [], [], []
    bounds = np.asarray([s[1:] for s in parameter_spec(config)])
    width = bounds[:, 1] - bounds[:, 0]
    for ensemble in range(args.ensembles):
        backend_path = directory / f"chain_{ensemble}.h5"
        backend = emcee.backends.HDFBackend(str(backend_path))
        sampler = emcee.EnsembleSampler(args.walkers, ndim, evaluate, args=(name,), pool=pool, backend=backend)
        if backend_path.exists() and backend.iteration > 0:
            if not args.resume:
                raise FileExistsError(str(backend_path))
            start = None
        else:
            backend.reset(args.walkers, ndim)
            sampler.random_state = np.random.RandomState(args.seed + ensemble).get_state()
            start = []
            for _ in range(args.walkers):
                for attempt in range(10000):
                    proposal = mle.x + rng.normal(0, 0.015, ndim) * width
                    if np.isfinite(evaluate(proposal, name)):
                        start.append(proposal)
                        break
                else:
                    raise RuntimeError("Could not initialize walkers within finite posterior support")
            start = np.asarray(start)
        remaining = args.steps - backend.iteration
        if remaining > 0:
            sampler.run_mcmc(start, remaining, progress=args.progress)
        samples = backend.get_chain(discard=args.burn, thin=args.thin, flat=True)
        retained.append(samples)
        ensemble_summaries.append(posterior_summary(samples, config))
        try:
            tau = backend.get_autocorr_time(discard=args.burn, tol=0)
            finite_tau = bool(np.all(np.isfinite(tau)) and np.all(tau > 0))
        except (emcee.autocorr.AutocorrError, ValueError):
            tau = np.full(ndim, np.nan)
            finite_tau = False
        diagnostics.append({
            "ensemble": ensemble, "steps": int(backend.iteration),
            "acceptance_mean": float(np.mean(sampler.acceptance_fraction)),
            "tau_steps": tau.tolist() if finite_tau else None,
            "approx_effective_samples": ((backend.iteration - args.burn) * args.walkers / tau).tolist() if finite_tau else None,
            "length_exceeds_50_tau": bool(finite_tau and np.all(backend.iteration - args.burn > 50 * tau)),
        })
    samples = np.concatenate(retained)
    summary["posterior"] = posterior_summary(samples, config)
    summary["ensemble_posteriors"] = ensemble_summaries
    summary["chain_diagnostics"] = diagnostics
    summary["convergence_note"] = "Length/tau is a heuristic. Compare independent ensembles and traces; this is not an automatic convergence certificate."
    summary["posterior_predictive"] = predictive_checks(samples, data, config, rng, args.ppc_draws)
    # Save a bounded collection of draws and pointwise densities for later checks.
    subset = samples[rng.choice(len(samples), size=min(len(samples), args.save_draws), replace=False)]
    pointwise = np.asarray([selected_logpdf(t, data["lambda_rm"], data["lambda_spec"], data["z"], config) for t in subset])
    np.savez_compressed(directory / "posterior.npz", samples=subset, log_likelihood=pointwise, ID=data["ID"])
    write_json(directory / "summary.json", summary)
    print(f"{name}: saved {directory}; long-chain checks {[d['length_exceeds_50_tau'] for d in diagnostics]}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "catalogs/conditional_richness/fits")
    parser.add_argument("--model", choices=["lognormal", "mixture", "both"], default="lognormal")
    parser.add_argument("--scope", choices=["suite", "combined", "bins"], default="suite")
    parser.add_argument("--lambda-piv", type=float, default=40.0)
    parser.add_argument("--z-piv", type=float, default=0.24)
    parser.add_argument("--walkers", type=int, default=48)
    parser.add_argument("--steps", type=int, default=3000, help="Total target steps per ensemble, also on resume")
    parser.add_argument("--burn", type=int, default=750)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--ensembles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--ppc-draws", type=int, default=200)
    parser.add_argument("--save-draws", type=int, default=1000)
    parser.add_argument("--cv-folds", type=int, default=0, help="Optional held-out MLE check (0 disables; 2--10 enables)")
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--optimize-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.steps <= args.burn or args.burn < 0:
        parser.error("Require steps > burn >= 0")
    if min(args.thin, args.ensembles, args.starts, args.ppc_draws, args.save_draws) < 1:
        parser.error("Counts must be positive")
    if args.walkers < 12 or args.walkers % 2:
        parser.error("Use an even walker count of at least 12 (twice the largest model dimension)")
    if args.cv_folds != 0 and not 2 <= args.cv_folds <= 10:
        parser.error("--cv-folds must be 0 or between 2 and 10")
    return args


def main():
    global _TASKS
    args = parse_args()
    if not args.mpi and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        raise RuntimeError("Multiple SLURM tasks require --mpi to avoid concurrent output writes")
    comm, rank = None, 0
    if args.mpi:
        from mpi4py import MPI
        from schwimmbad import MPIPool
        comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.rank
        if comm.size < 2:
            raise ValueError("--mpi requires at least two ranks")
    payload = None
    if rank == 0:
        try:
            data, audit = load_sample(args.data)
            payload = (data, audit, None)
        except Exception as error:
            payload = (None, None, str(error))
    if comm is not None:
        payload = comm.bcast(payload, root=0)
    data, audit, error = payload
    if error:
        raise RuntimeError(error)
    _TASKS = {name: (config, selected) for name, config, selected in make_tasks(args, data, audit)}
    # MPIPool workers enter their service loop in the constructor and exit when
    # closed. Build every task first, and reuse ONE pool for the whole suite.
    pool = MPIPool(comm=comm) if comm is not None else None
    try:
        run_tasks(args, audit, pool)
    finally:
        if pool is not None:
            pool.close()


def run_tasks(args, audit, pool):
    for name, (config, selected) in _TASKS.items():
        if len(selected["z"]) < 30:
            print(f"Skipping {name}: fewer than 30 clusters", flush=True)
            continue
        fit_task(name, config, selected, audit, args, pool)


if __name__ == "__main__":
    main()
