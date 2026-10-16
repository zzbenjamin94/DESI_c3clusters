# Recover completed chains without refitting

Stop/confirm completion of the original sampler before reading its chains.
The finalizer processes exactly one global task, verifies sample IDs/checksum
and original fitting-code hashes, reads chains with recorded iteration >=5000,
and computes posterior summaries. MPI distributes predictive simulations and
pointwise likelihood evaluations; chain reading, autocorrelation and summary
aggregation remain on rank zero. More ranks do not speed up those serial stages.

```bash
salloc --account=desi --constraint=cpu --qos=interactive --nodes=1 --ntasks=8 --cpus-per-task=1 --time=01:00:00
cd /global/homes/z/zzhang13/DESI/Projection
source ./setup_nersc_mpi_myEnv_v39.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun --export=ALL -n 8 -c 1 "$CONDA_PREFIX/bin/python" -u \
  richness_relation/finalize_saved_richness_mpi.py --mpi \
  --fit catalogs/conditional_richness/mixture_fits/mixture_combined_z/seed_42 \
  --data catalogs/conditional_richness/overnight_sample.npz \
  --output catalogs/conditional_richness/mixture_finalized
```

An existing destination is refused; choose a new output root after a failed run.
Do not bypass a provenance mismatch or delete original manifests.
Recovered outputs contain no copies of large chain files: run convergence tests
against the original mixture_fits directory. Finalization is not convergence.
The random stream differs from the original finalizer but is reproducible and
independent of rank count. Default 200 PPC draws and 1000 saved posterior draws.

Review the completed global fits (order matters: recovered root LAST):

```bash
python richness_relation/review_redshift_models.py \
  --data catalogs/conditional_richness/overnight_sample.npz \
  --fits catalogs/conditional_richness/lognormal_fits \
         catalogs/conditional_richness/mixture_fits \
         catalogs/conditional_richness/mixture_finalized \
  --output "plots/conditional_richness/redshift_review_$(date +%Y%m%d_%H%M%S)"
```

Only the verified replacement of an incomplete original task is allowed.
Recover first, then decide the global baseline; this does not implement the
proposed independent-cell chains. Global location parameters depend on the chosen
conditional likelihood and are not generally arithmetic-mean parameters.
