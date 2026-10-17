"""Fit ln(lambda_spec) = a + b ln(lambda_RM / 40), without redshift evolution.

Exact Gaussian-regression posterior under p(a,b,sigma^2) proportional to
1/sigma^2. Sigma describes measured conditional scatter, not deconvolved
intrinsic scatter. The positive-richness prepared sample defines the population.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from richness_relation.prepare_conditional_richness import load_sample


def fit_posterior(rm, spec, draws=10000, seed=42):
    rm, spec = np.asarray(rm), np.asarray(spec)
    if (rm.shape != spec.shape or rm.ndim != 1 or len(rm) <= 3
            or not np.all(np.isfinite(rm) & np.isfinite(spec) & (rm > 0) & (spec > 0))):
        raise ValueError('Require more than three finite positive richness pairs')
    X = np.column_stack([np.ones(len(rm)), np.log(rm / 40.)])
    y = np.log(spec)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank != 2:
        raise ValueError('RM richness must span a nonzero range')
    sse = np.sum((y - X @ beta)**2)
    if sse <= 0:
        raise ValueError('Zero residual variance gives an improper posterior')
    rng = np.random.default_rng(seed)
    # sigma^2 | data ~ inverse-gamma((N-2)/2, SSE/2).
    variance = sse / rng.chisquare(len(rm) - 2, size=draws)
    chol = np.linalg.cholesky(np.linalg.inv(X.T @ X))
    coefficients = beta + (rng.normal(size=(draws, 2)) @ chol.T) * np.sqrt(variance[:, None])
    return np.column_stack([coefficients, np.sqrt(variance)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'catalogs/conditional_richness/overnight_sample.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.draws < 100:
        p.error('Use at least 100 posterior draws')
    data, audit = load_sample(args.data)
    samples = fit_posterior(data['lambda_rm'], data['lambda_spec'], args.draws, args.seed)
    summary = dict(model='reverse_lognormal_noz', pivot=40., N=len(data['ID']),
                   sample_sha256=audit['sample_sha256'], seed=args.seed,
                   parameter_names=['a', 'b', 'sigma_ln_spec'],
                   prior='p(a,b,sigma_squared) proportional to 1/sigma_squared',
                   posterior_method='Exact independent Gaussian regression posterior draws',
                   percentiles_16_50_84=np.percentile(samples, [16, 50, 84], axis=0).tolist(),
                   warnings=['Conditions on measured RM; measurement errors are not deconvolved.',
                             'Descriptive fit to the prepared selected positive-richness sample.',
                             'Constant Gaussian scatter in ln(spec); no redshift evolution.'])
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'posterior.npz', samples=samples)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
