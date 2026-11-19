"""Plot saved richness MCMC ensembles with pygtc, without fitting.

Style reference: wl_covariance/compute/correlation/cov_cornerplot.ipynb.
Each task gets a separate PDF and PNG; independent ensembles are overlaid.
Run only after sampling stops. These plots do not certify convergence.
Parameters remain in sampled coordinates (natural logarithms where named ln_*).
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "ln_A": r"$\ln A$", "B": r"$B$", "C": r"$C$",
    "ln_sigma0": r"$\ln\sigma_0$",
    "ln_sigma_lambda": r"$\ln\sigma_\lambda$",
    "f_proj": r"$f_{\rm proj}$", "ln_tau": r"$\ln\tau$",
}


def load_ensembles(directory, burn=None, thin=1, max_samples=100000, random_seed=42):
    """Read actual HDF iterations, not unused capacity or posterior.npz subsets."""
    import h5py
    if thin < 1 or max_samples < 2 or (burn is not None and burn < 0):
        raise ValueError("Require thin >=1, max_samples >=2, burn >=0")
    manifest = json.loads((directory / "manifest.json").read_text())
    meta_path = directory / "summary.json"
    completed = meta_path.exists()
    if not completed:
        meta_path = directory / "bestfit.json"
    meta = json.loads(meta_path.read_text())
    if meta["config"] != manifest["config"]:
        raise ValueError(f"Inconsistent model metadata: {directory}")
    names = meta["parameter_names"]
    discard = manifest["burn"] if burn is None else burn
    if discard < 0:
        raise ValueError("Negative burn-in")
    unknown = set(names) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown parameter labels: {unknown}")
    paths = sorted(directory.glob("chain_*.h5"))
    if not paths:
        raise ValueError(f"No chains: {directory}")
    expected = {f"chain_{i}.h5" for i in range(manifest["ensembles"])}
    if not {p.name for p in paths}.issubset(expected):
        raise ValueError(f"Unexpected chain files: {directory}")
    rng = np.random.default_rng(random_seed)
    samples, labels, records = [], [], []
    for path in paths:
        with h5py.File(path, "r") as handle:
            group = handle["mcmc"]
            steps = int(group.attrs["iteration"])
            dataset = group["chain"]
            if steps <= discard or steps > dataset.shape[0]:
                raise ValueError(f"Invalid retained length: {path}")
            if dataset.shape[1:] != (manifest["walkers"], len(names)):
                raise ValueError(f"Chain dimensions disagree with metadata: {path}")
            chain = dataset[discard:steps:thin]
        flat = chain.reshape(-1, len(names))
        if len(flat) < 2 or not np.all(np.isfinite(flat)):
            raise ValueError(f"Insufficient or nonfinite samples: {path}")
        retained = len(flat)
        if retained > max_samples:
            flat = flat[rng.choice(retained, max_samples, replace=False)]
        if np.any(np.ptp(flat, axis=0) == 0):
            raise ValueError(f"Constant parameter cannot be contoured: {path}")
        samples.append(flat)
        labels.append(f"Ensemble {int(path.stem.split('_')[-1]) + 1}")
        records.append(dict(file=str(path.resolve()), steps=steps, burn=discard,
                            thin=thin, retained=retained, plotted=len(flat)))
    audit = dict(task=meta["task"], parameter_names=names, chains=records,
                 completed_summary=completed,
                 expected_ensembles=manifest["ensembles"],
                 random_seed=random_seed,
                 chain_diagnostics=meta.get("chain_diagnostics"),
                 note="Corner plots are not convergence tests. Missing ensembles "
                      "and unfinished summaries are explicitly reported.")
    return samples, labels, [LABELS[n] for n in names], audit


def plot_task(directory, output, burn=None, thin=1, max_samples=100000, overwrite=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pygtc

    samples, labels, names, audit = load_ensembles(directory, burn, thin, max_samples)
    stem = output / f"{directory.parent.name}_{directory.name}_gtc"
    targets = [stem.with_suffix(ext) for ext in (".pdf", ".png", ".json")]
    if not overwrite and any(p.exists() for p in targets):
        raise FileExistsError(f"Output exists: {stem}; use --overwrite intentionally")
    output.mkdir(parents=True, exist_ok=True)
    print(f"{audit['task']}: {len(samples)}/{audit['expected_ensembles']} ensembles; "
          f"completed summary={audit['completed_summary']}")
    print("Saved convergence diagnostics:", audit["chain_diagnostics"])
    # Preserve the original notebook typography and page format.
    fig = pygtc.plotGTC(
        chains=samples, paramNames=names, chainLabels=labels,
        figureSize="MNRAS_page",
        customLabelFont={"family": "DejaVu Sans", "size": 16},
        customLegendFont={"family": "DejaVu Sans", "size": 13},
        customTickFont={"family": "DejaVu Sans", "size": 10},
        plotName=str(targets[0]),
    )
    fig.savefig(targets[1], dpi=200, bbox_inches="tight")
    plt.close(fig)
    targets[2].write_text(json.dumps(audit, indent=2) + "\n")
    return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, nargs="+", default=[
        ROOT / "catalogs/conditional_richness/lognormal_fits",
        ROOT / "catalogs/conditional_richness/mixture_fits"])
    parser.add_argument("--output", type=Path, default=ROOT / "plots/conditional_richness/gtc")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--burn", type=int, help="Default: recorded burn-in")
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Maximum plotted samples per ensemble; no chain changes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    jobs = []
    for root in args.fits:
        directories = sorted(root.glob(f"*/seed_{args.seed}"))
        if not directories:
            parser.error(f"No task/seed directories in {root}")
        jobs.extend((d, args.output / root.name) for d in directories)
    identities = [(str(out.resolve()), d.parent.name, d.name) for d, out in jobs]
    if len(set(identities)) != len(identities):
        parser.error("Duplicate output names; run these fit roots separately")
    for directory, output in jobs:
        for path in plot_task(directory, output, args.burn, args.thin,
                              args.max_samples, args.overwrite):
            print(path)


if __name__ == "__main__":
    main()
