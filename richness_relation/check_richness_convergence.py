"""Read-only diagnostics for completed or stopped emcee runs.

References
----------
Gelman & Rubin (1992), Statistical Science 7, 457-472.
https://doi.org/10.1214/ss/1177011136
Sokal (1997), Monte Carlo Methods in Statistical Mechanics: Foundations
and New Algorithms, Functional Integration, pp. 131-192.
Goodman & Weare (2010), Ensemble Samplers with Affine Invariance.
https://doi.org/10.2140/camcos.2010.5.65

Uses the common uncorrected, unsplit classical R-hat:
sqrt(((n-1)/n * W + B/n) / W).
This is not the original finite-degrees-of-freedom corrected PSRF,
nor rank-normalized/split R-hat. The 50-tau cutoff is an emcee
heuristic, not a theorem from Sokal:
https://emcee.readthedocs.io/en/stable/tutorials/autocorr/

A fixed walker (default 0) from EACH independent ensemble supplies
one R-hat trajectory. Walkers within an ensemble are never treated
as independent chains. Do not flatten them. Only chain files belonging
to the same seed/task directory are compared.
Run only after the sampler stops; do not read an actively written HDF5.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def classical_rhat(chains):
    """Input shape: independent ensembles, ordered steps, parameters."""
    x = np.asarray(chains, dtype=float)
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("R-hat requires >=2 independent trajectories and >=2 steps")
    if not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite samples")
    n = x.shape[1]
    within = np.mean(np.var(x, axis=1, ddof=1), axis=0)
    between = n * np.var(np.mean(x, axis=1), axis=0, ddof=1)
    variance = (n - 1) / n * within + between / n
    # Even identical constant trajectories cannot establish convergence.
    result = np.full(x.shape[2], np.nan)
    np.sqrt(np.divide(variance, within, out=np.zeros_like(within),
                      where=within > 0), out=result, where=within > 0)
    return result


def diagnose(directory, burn=None, walker=0, tau_multiple=50, rhat_limit=1.1):
    import emcee
    import h5py

    manifest = json.loads((directory / "manifest.json").read_text())
    metadata_path = directory / "summary.json"
    if not metadata_path.exists():
        metadata_path = directory / "bestfit.json"
    metadata = json.loads(metadata_path.read_text())
    names = metadata["parameter_names"]
    discard = manifest["burn"] if burn is None else burn
    if discard < 0 or walker < 0 or tau_multiple <= 0 or rhat_limit <= 1:
        raise ValueError("Invalid burn, walker, or diagnostic threshold")
    paths = sorted(directory.glob("chain_*.h5"))
    if not paths:
        raise ValueError(f"No chains in {directory}")
    expected = {f"chain_{i}.h5" for i in range(manifest["ensembles"])}
    if not {p.name for p in paths}.issubset(expected):
        raise ValueError("Unexpected chain files; check run provenance")
    complete_set = {p.name for p in paths} == expected
    chains, taus, acceptance = [], [], []
    for path in paths:
        with h5py.File(path, "r") as handle:
            group = handle["mcmc"]
            steps = int(group.attrs["iteration"])
            if steps - discard < 2:
                raise ValueError(f"{path}: fewer than two retained steps")
            chain = group["chain"][discard:steps]
            acc = float(np.mean(group["accepted"][:]) / steps)
        if chain.ndim != 3 or chain.shape[2] != len(names):
            raise ValueError(f"{path}: parameter dimensions disagree with metadata")
        if chain.shape[1] != manifest["walkers"] or walker >= chain.shape[1]:
            raise ValueError(f"{path}: invalid walker count/index")
        if not np.all(np.isfinite(chain)):
            raise ValueError(f"{path}: nonfinite samples")
        with np.errstate(invalid="ignore", divide="ignore"):
            tau = emcee.autocorr.integrated_time(chain, tol=0)
        # A stuck walker is not repaired by averaging other walkers.
        stuck = np.any(np.var(chain, axis=0) == 0, axis=0)
        tau = np.where((tau > 0) & np.isfinite(tau) & ~stuck, tau, np.nan)
        chains.append(chain)
        taus.append(tau)
        acceptance.append(acc)
    n_common = min(len(c) for c in chains)
    rhat = np.full(len(names), np.nan)
    if len(chains) >= 2:
        # Equal-length chronological prefixes after burn-in; never flatten.
        rhat = classical_rhat(np.stack([c[:n_common, walker, :] for c in chains]))
    rows = []
    for j, name in enumerate(names):
        gr = None if not np.isfinite(rhat[j]) else bool(rhat[j] < rhat_limit)
        for i, chain in enumerate(chains):
            tau = taus[i][j]
            ok = bool(np.isfinite(tau) and len(chain) > tau_multiple * tau)
            rows.append(dict(
                task=directory.parent.name, seed=directory.name,
                chain=paths[i].name, parameter=name, burn=discard,
                n_post=len(chain), tau_steps=float(tau),
                length_over_tau=float(len(chain) / tau),
                autocorrelation_pass=ok, acceptance_mean=acceptance[i],
                rhat=float(rhat[j]), rhat_pass=gr,
                rhat_walker=walker, rhat_n_steps=n_common,
                ensembles_found=len(chains), expected_ensembles=manifest["ensembles"],
                both_pass=bool(ok and gr is True and complete_set),
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--burn", type=int, help="Default: burn saved in each manifest")
    parser.add_argument("--walker", type=int, default=0,
                        help="Predetermined walker for between-ensemble R-hat")
    parser.add_argument("--tau-multiple", type=float, default=50)
    parser.add_argument("--rhat-limit", type=float, default=1.1)
    parser.add_argument("--output", type=Path, help="Optional CSV; refuses overwrite")
    args = parser.parse_args()
    directories = sorted(args.fits.glob(f"*/seed_{args.seed}"))
    if not directories:
        parser.error("No task/seed directories found")
    rows = []
    for directory in directories:
        values = diagnose(directory, args.burn, args.walker,
                          args.tau_multiple, args.rhat_limit)
        rows.extend(values)
        print(f"\n{directory.parent.name}")
        for row in values:
            print(f"{row['chain']:12} {row['parameter']:18} "
                  f"N/tau={row['length_over_tau']:8.2f} "
                  f"Rhat={row['rhat']:7.3f} "
                  f"tau_pass={row['autocorrelation_pass']} "
                  f"Rhat_pass={row['rhat_pass']}")
        if values[0]["ensembles_found"] < 2:
            print("R-hat unavailable: need >=2 independent ensembles.")
        print("Both checks pass for every parameter/ensemble:",
              all(row["both_pass"] for row in values))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("\nPassing is not proof of convergence. Inspect walker traces and "
          "initialization; current production ensembles start near the same MLE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

