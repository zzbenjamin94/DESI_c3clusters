# Separate conditional-richness runs

These launchers use the existing prepared overnight_sample.npz and .json.
They do not regenerate catalogs or samples. Source the existing NERSC MPI
environment with its absolute path; schwimmbad, multiprocess, emcee and h5py
must already be importable there.

## Submit now

From /global/homes/z/zzhang13/DESI/Projection:

```bash
sbatch richness_relation/run_lognormal.slurm
sbatch richness_relation/run_mixture_pilot.slurm
```

These jobs can run independently. Do not also submit the production mixture
until the pilot timing is reviewed. There is no automatic dependent submission.

| Job | Scope | Steps / ensemble | Ensembles | Output under catalogs/conditional_richness/ |
| --- | --- | --- | --- | --- |
| lognormal | Combined without/with redshift + three redshift bins | 5000 | 2 | lognormal_fits |
| mixture pilot | Combined without/with redshift only | 1000 | 1 | mixture_pilot_1000 |
| mixture production | Combined without/with redshift + three redshift bins | 5000 | 2 | mixture_fits |

Slurm logs are separate richness-lognormal-JOBID.log,
richness-mixture_pilot-JOBID.log and richness-mixture-JOBID.log in the repo root.
Each fit continues to save individual HDF5 chains, posterior samples and JSON
summaries, rather than packing incompatible models into one output file.

The pilot matches the previous 200-step benchmark's 48 walkers, one ensemble,
50 burn-in steps, 20 posterior predictive draws, 100 saved draws, seed 42 and
combined scope. Only the target number of steps changes. Its burn-in is for a
timing comparison, not a validated choice for science inference.

## Runtime estimate

The previous two-fit mixture benchmark took T200 = 981 seconds. After the new
pilot finishes, record the mixture_pilot elapsed line. Only compare FRESH runs:
a resumed pilot measures remaining work, not 1000 full steps. A completed pilot
submitted again will mostly repeat optimization and diagnostics.

For a rough fixed-overhead plus linear-sampling model:
- seconds per ensemble step across the two combined fits:
  b = (T1000 - 981) / 800.
- estimated fixed overhead: a = 981 - 200*b.
- approximate two-combined-fit production time: a + 10000*b
  (5000 steps times two ensembles).

This approximation assumes comparable hardware/load and unchanged inputs/code.
Negative overhead or a nonpositive slope means this simple model is unreliable.
Production diagnostics use more draws; a second ensemble adds setup overhead.
The three redshift-bin fits add time and are not included in this estimate.
Do not interpret any timing extrapolation as a convergence guarantee.

## Production and restart

After assessing pilot timing and traces:

```bash
sbatch richness_relation/run_mixture.slurm
```

Both production jobs request 12 hours as initial budgets. Override with sbatch
--time if appropriate to allocation/queue limits. If interrupted, wait until the
old job has ended, then resubmit the same launcher. --resume preserves existing
chains; target steps are total steps, not additional steps. Keep data, code,
burn-in, walkers and other signature settings fixed, or choose a new output
directory. Never run two writers to the same model output. Inspect sacct status,
summary.json and chain diagnostics before interpreting fits.

## Plot each model

```bash
python richness_relation/plot_conditional_richness.py --data catalogs/conditional_richness/overnight_sample.npz --fits catalogs/conditional_richness/lognormal_fits --output plots/conditional_richness/lognormal
python richness_relation/plot_conditional_richness.py --data catalogs/conditional_richness/overnight_sample.npz --fits catalogs/conditional_richness/mixture_fits --output plots/conditional_richness/mixture
```

These commands plot each model separately. Cross-model comparison tables will
need both result sets presented to the plotting workflow together; do not
compare AIC/BIC between fits using different selected samples.
