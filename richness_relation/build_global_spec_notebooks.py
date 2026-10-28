"""Rebuild the two initial notebook templates; do not run after hand-editing them."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
SETUP = r'''
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import emcee
from scipy.stats import norm, chi2, gaussian_kde

ROOT = Path.cwd().resolve()
if ROOT.name == "richness_relation":
    ROOT = ROOT.parent
assert (ROOT / "richness_relation").is_dir(), "Open from the Git repository"
sys.path.insert(0, str(ROOT / "richness_relation"))
from prepare_conditional_richness import load_sample
from global_spec_richness_model import design
from summarize_global_spec_richness import summarize

FITS = ROOT / "catalogs/global_spec_richness"
OUTPUT = ROOT / "plots/global_spec_richness"
OUTPUT.mkdir(parents=True, exist_ok=True)
Z_EDGES = [0.10, 0.18, 0.24, 0.35]
RM_EDGES = [20, 30, 50, 100]  # Diagnostic cells only, not fitting cuts.
COLORS = ["crimson", "darkorange", "royalblue"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 12,
                     "axes.labelsize": 16, "axes.linewidth": 1.1})
def style(ax):
    ax.grid(False)
    ax.tick_params(which="both", direction="in", top=True, right=True)
def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")

comparison, parameters = summarize(FITS)
display(comparison.drop(columns="sample_sha256"), parameters)
meta = {m: json.loads((FITS / m / "run.json").read_text()) for m in ("noz", "z")}
data, audit = load_sample(meta["noz"]["data_path"])
posterior = {}
for m in meta:
    with np.load(FITS / m / "posterior.npz") as f:
        posterior[m] = f["samples"].copy()
    diag = json.loads((FITS / m / "diagnostics.json").read_text())
    print(m, "CONVERGED:", diag["converged"],
          "minimum N/tau:", np.min(diag["steps_per_tau"]),
          "worst Rhat:", max(diag["rhat_worst"]))
    if not diag["converged"]:
        print("PROVISIONAL: do not use this posterior for final inference.")
rng = np.random.default_rng(42)
'''

GLOBAL = r'''
X_LIM, Y_LIM = (20, 100), (20, 600)
MEAN_BINS = np.geomspace(*X_LIM, 8)
MIN_COUNT = 20
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
x, y, z = data["lambda_rm"], data["lambda_spec"], data["z"]
for ax, model in zip(axes, ("noz", "z")):
    for iz, (lo, hi) in enumerate(zip(Z_EDGES[:-1], Z_EDGES[1:])):
        keep = (z >= lo) & (z < hi)
        ax.scatter(x[keep], y[keep], s=3, alpha=0.09, color=COLORS[iz], rasterized=True)
        for left, right in zip(MEAN_BINS[:-1], MEAN_BINS[1:]):
            use = keep & (x >= left) & (x < right)
            if use.sum() < MIN_COUNT:
                continue
            ly = np.log(y[use])
            center, sem = ly.mean(), ly.std(ddof=1) / np.sqrt(use.sum())
            ax.errorbar(np.exp(np.log(x[use]).mean()), np.exp(center),
                        yerr=[[np.exp(center)-np.exp(center-sem)],
                              [np.exp(center+sem)-np.exp(center)]],
                        fmt="o", ms=3, mfc="none", color=COLORS[iz], capsize=2)
        if keep.sum() == 0:
            continue
        grid = np.geomspace(*X_LIM, 200)
        # Curves are at the median z of each redshift slice, not new cell fits.
        covariates = {"lambda_rm": grid, "z": np.full(len(grid), np.median(z[keep]))}
        draws = posterior[model][rng.choice(len(posterior[model]), 1500), :-1]
        predicted = np.exp(draws @ design(covariates, model).T)
        low, mid, high = np.percentile(predicted, [16, 50, 84], axis=0)
        ax.plot(grid, mid, color=COLORS[iz], lw=1,
                label=rf"${lo:.2f}\leq z<{hi:.2f}$")
        ax.fill_between(grid, low, high, color=COLORS[iz], alpha=0.18)
    ax.set(xscale="log", yscale="log", xlim=X_LIM, ylim=Y_LIM,
           xlabel=r"$\lambda_{\rm RM}$",
           title="No redshift evolution" if model == "noz" else "Redshift evolution")
    ax.legend(frameon=False, fontsize=10)
    style(ax)
axes[0].set_ylabel(r"$\lambda_{\rm spec}$")
fig.tight_layout()
save(fig, "global_relation")
plt.show()
'''
CHAINS = r'''
import pygtc
for model in ("noz", "z"):
    names = meta[model]["parameters"]
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 2*len(names)), sharex=True)
    ensembles = []
    for i in range(meta[model]["ensembles"]):
        backend = emcee.backends.HDFBackend(str(FITS/model/f"chain_{i}.h5"), read_only=True)
        chain = backend.get_chain()
        for j, ax in enumerate(axes):
            ax.plot(np.median(chain[:, :, j], axis=1), lw=0.7, label=f"ensemble {i}")
            ax.axvline(meta[model]["burn"], color="0.4", ls=":")
            ax.set_ylabel(names[j])
        retained = chain[meta[model]["burn"]:].reshape(-1, len(names))
        ensembles.append(retained[::max(1, len(retained)//20000)].copy())
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Saved step (ensemble median shown)")
    fig.tight_layout()
    save(fig, f"traces_{model}")
    plt.show()
    for sample in ensembles:
        sample[:, -1] = np.exp(sample[:, -1])
    labels = [r"$a_{\rm rev}$", r"$b_{\rm rev}$"]
    if model == "z":
        labels += [r"$c_{\rm rev}$"]
    labels += [r"$s_{\rm rev}$"]
    fig = pygtc.plotGTC(chains=ensembles, paramNames=labels,
        chainLabels=[f"ensemble {i}" for i in range(len(ensembles))],
        figureSize="MNRAS_page",
        customLabelFont={"family": "DejaVu Serif", "size": 14},
        customLegendFont={"family": "DejaVu Serif", "size": 10},
        customTickFont={"family": "DejaVu Serif", "size": 10})
    save(fig, f"corner_{model}")
    plt.show()
'''
RESIDUALS = r'''
MODEL = "noz"  # Your decision after comparison and convergence review; not automatic.
N_DRAWS = 200
draws = posterior[MODEL][rng.choice(len(posterior[MODEL]), N_DRAWS)]
central = np.median(posterior[MODEL], axis=0)
X = design(data, MODEL)
mu = X @ central[:-1]
log_residual = np.log(data["lambda_spec"]) - mu
s = np.exp(central[-1])

def density_errors(values, edges):
    counts, _ = np.histogram(values, edges)
    # Poisson count intervals: approximate for a fixed-N multinomial histogram.
    lower = np.zeros(len(counts))
    good = counts > 0
    lower[good] = 0.5 * chi2.ppf(0.16, 2*counts[good])
    upper = 0.5 * chi2.ppf(0.84, 2*(counts+1))
    scale = len(values) * np.diff(edges)  # Includes objects outside displayed bins.
    return counts/scale, np.array([counts-lower, upper-counts])/scale

for kind in ("conditional", "log_residual"):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for iz, (zl, zh) in enumerate(zip(Z_EDGES[:-1], Z_EDGES[1:])):
        for ir, (rl, rh) in enumerate(zip(RM_EDGES[:-1], RM_EDGES[1:])):
            ax = axes[iz, ir]
            keep = ((data["z"] >= zl) & (data["z"] < zh) &
                    (data["lambda_rm"] >= rl) & (data["lambda_rm"] < rh))
            style(ax)
            ax.text(.95,.95,rf"${rl:g}\leq\lambda_{{\rm RM}}<{rh:g}$"+"\n"+
                    rf"${zl:.2f}\leq z<{zh:.2f}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize=10)
            if keep.sum() < 20:
                ax.text(.5,.5,"Insufficient data", transform=ax.transAxes, ha="center")
                continue
            if kind == "conditional":
                edges = np.geomspace(5, 1000, 18)
                values = data["lambda_spec"][keep]
                grid = np.geomspace(edges[0], edges[-1], 250)
                centers = np.sqrt(edges[:-1]*edges[1:])
                ax.set_xscale("log")
            else:
                edges = np.linspace(-2, 2, 21)
                values = log_residual[keep]
                grid = np.linspace(-2, 2, 250)
                centers = .5*(edges[:-1]+edges[1:])
            density, errors = density_errors(values, edges)
            ax.errorbar(centers, density, yerr=errors, fmt="o", ms=3,
                        mfc="none", color=COLORS[iz], capsize=0)
            predictions = []
            for theta in draws:
                locations = X[keep] @ theta[:-1]
                if kind == "conditional":
                    prediction = norm.pdf(np.log(grid)[:,None], locations,
                                          np.exp(theta[-1])).mean(axis=1)/grid
                else:
                    prediction = norm.pdf(grid[:,None], locations-mu[keep],
                                          np.exp(theta[-1])).mean(axis=1)
                predictions.append(prediction)
            low, mid, high = np.percentile(predictions, [16,50,84], axis=0)
            ax.plot(grid, mid, color="0.25", lw=1.6)
            ax.fill_between(grid, low, high, color="0.5", alpha=.2)
            if iz == 2:
                ax.set_xlabel(r"$\lambda_{\rm spec}$" if kind == "conditional"
                              else r"$\ln\lambda_{\rm spec}-\mu$")
            if ir == 0:
                ax.set_ylabel("Probability density")
    fig.tight_layout()
    save(fig, f"{kind}_{MODEL}")
    plt.show()
'''
PPC = r'''
# Posterior predictive checks: replicate each cluster at its actual RM richness/z.
# These are diagnostic posterior predictive tail fractions, not frequentist GOF p-values.
from scipy.stats import skew, kurtosis
def statistics(residual):
    return np.array([skew(residual), kurtosis(residual),
                     np.mean(np.abs(residual) > 2)])
observed, replicated = [], []
for theta in draws:
    locations = X @ theta[:-1]
    sigma = np.exp(theta[-1])
    observed.append(statistics((np.log(data["lambda_spec"])-locations)/sigma))
    replicated.append(statistics(rng.normal(size=len(locations))))
observed, replicated = np.array(observed), np.array(replicated)
ppc = pd.DataFrame({
    "statistic": ["skewness", "excess kurtosis", "fraction beyond 2 sigma"],
    "observed_median": np.median(observed, axis=0),
    "replicate_median": np.median(replicated, axis=0),
    "posterior_predictive_upper_tail": np.mean(replicated >= observed, axis=0)})
display(ppc)
ppc.to_csv(OUTPUT / f"posterior_predictive_{MODEL}.csv", index=False)
fig, ax = plt.subplots(figsize=(6,5))
q = (np.arange(len(log_residual)) + .5)/len(log_residual)
ax.plot(norm.ppf(q), np.sort(log_residual/s), "o", ms=2, alpha=.4)
ax.plot([-4,4],[-4,4], color="black", lw=1)
ax.set(xlabel="Standard normal quantile", ylabel="Standardized log residual")
style(ax)
save(fig, f"qq_{MODEL}")
plt.show()
'''


CELL_SETUP = r'''
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from IPython.display import display
import emcee
import pygtc

ROOT = Path.cwd().resolve()
if ROOT.name == "richness_relation":
    ROOT = ROOT.parent
assert (ROOT/"richness_relation").is_dir()
sys.path.insert(0,str(ROOT/"richness_relation"))
from prepare_conditional_richness import load_sample
from global_spec_richness_model import residual_logpdf

FITS = ROOT/"catalogs/spec_residual_cells"
OUTPUT = ROOT/"plots/spec_residual_cells"
OUTPUT.mkdir(parents=True,exist_ok=True)
RM_EDGES = [20,30,50,100]
Z_EDGES = [.10,.18,.24,.35]
COLORS = ["crimson","darkorange","royalblue"]
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":12,"axes.labelsize":16,
                     "axes.linewidth":1.1})
comparison=pd.read_csv(FITS/"cell_comparison.csv")
parameters=pd.read_csv(FITS/"cell_parameters.csv")
display(comparison,parameters)
cache={}
for iz in range(3):
    for ir in range(3):
        for model in ("lognormal","mixture"):
            path=FITS/f"z{iz}_rm{ir}"/model
            meta=json.loads((path/"run.json").read_text())
            assert meta["rm_bounds"]==RM_EDGES[ir:ir+2] and meta["z_bounds"]==Z_EDGES[iz:iz+2], \
                "Plot labels must match fitted cell boundaries"
            with np.load(path/"posterior.npz") as f:
                samples=f["samples"].copy()
            diag=json.loads((path/"diagnostics.json").read_text())
            cache[iz,ir,model]=(meta,samples,diag)
meta=cache[0,0,"lognormal"][0]
data,audit=load_sample(meta["data_path"])
assert audit["sample_sha256"]==meta["sample_sha256"]
a,b=meta["fixed_global_mle"]
residual=np.log(data["lambda_spec"])-a-b*np.log(data["lambda_rm"]/40)
print("Fixed global MLE:",a,b)
print("All cell convergence checks pass:",bool(comparison.converged.all()))
print("Histogram chi-square is descriptive; no uncalibrated chi-square p-values are used.")
def style(ax):
    ax.grid(False)
    ax.tick_params(which="both",direction="in",top=True,right=True)
def save(fig,name):
    for ext in ("png","pdf"):
        fig.savefig(OUTPUT/f"{name}.{ext}",dpi=200,bbox_inches="tight")
'''
CELL_PLOT = r'''
# Residuals use natural logs. No hidden conversion to dex.
EDGES=np.linspace(-2.0,2.0,23)
GRID=np.linspace(EDGES[0],EDGES[-1],500)
N_DRAWS=300
rng=np.random.default_rng(42)
fig,axes=plt.subplots(3,3,figsize=(12,10),sharex=True)
for iz in range(3):
    for ir in range(3):
        ax=axes[iz,ir]
        meta,_,_=cache[iz,ir,"lognormal"]
        keep=np.isin(data["ID"].astype(str),meta["cluster_ids"])
        values=residual[keep]
        counts,_=np.histogram(values,EDGES)
        low=np.zeros_like(counts,dtype=float)
        nonzero=counts>0
        low[nonzero]=.5*chi2.ppf(.15865,2*counts[nonzero])
        high=.5*chi2.ppf(.84135,2*(counts+1))
        scale=len(values)*np.diff(EDGES)
        ax.errorbar((EDGES[1:]+EDGES[:-1])/2,counts/scale,
                    yerr=np.array([counts-low,high-counts])/scale,
                    fmt="o",ms=3.5,mfc="none",color=COLORS[iz],capsize=0,label="Data")
        for model,color,ls,label in [("lognormal","#2a6298","--","Log-normal"),
                                     ("mixture","0.25","-","Centered mixture")]:
            _,samples,diag=cache[iz,ir,model]
            draws=samples[rng.integers(len(samples),size=N_DRAWS)]
            densities=np.array([np.exp(residual_logpdf(GRID,t,model)) for t in draws])
            lo,mid,hi=np.percentile(densities,[16,50,84],axis=0)
            ax.plot(GRID,mid,color=color,ls=ls,lw=2,label=label)
            ax.fill_between(GRID,lo,hi,color=color,alpha=.13)
        ax.text(.97,.96,rf"${RM_EDGES[ir]}\leq\lambda_{{\rm RM}}<{RM_EDGES[ir+1]}$"
                +"\n"+rf"${Z_EDGES[iz]:.2f}\leq z<{Z_EDGES[iz+1]:.2f}$",
                transform=ax.transAxes,ha="right",va="top",fontsize=11)
        if not all(cache[iz,ir,m][2]["converged"] for m in ("lognormal","mixture")):
            ax.text(.03,.94,"Provisional",transform=ax.transAxes,va="top",fontsize=10)
        ax.set_xlim(EDGES[0],EDGES[-1])
        style(ax)
        if ir==0:ax.set_ylabel("Probability density")
        if iz==2:ax.set_xlabel(r"$\ln\lambda_{\rm spec}-m_{\rm spec}$")
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,0))
fig.tight_layout(rect=[0,.045,1,1])
save(fig,"conditional_reverse_residuals")
plt.show()
'''
CELL_CONTOURS = r'''
# One-dimensional scatter posteriors for the log-normal.
fig,axes=plt.subplots(3,3,figsize=(11,8))
for iz in range(3):
    for ir in range(3):
        ax=axes[iz,ir];meta,samples,diag=cache[iz,ir,"lognormal"]
        ax.hist(np.exp(samples[:,0]),bins=50,density=True,histtype="step",color=COLORS[iz])
        ax.set_title(rf"${RM_EDGES[ir]}\leq\lambda_{{\rm RM}}<{RM_EDGES[ir+1]}$, "
                     +rf"${Z_EDGES[iz]:.2f}\leq z<{Z_EDGES[iz+1]:.2f}$",fontsize=10)
        ax.set_xlabel(r"$s_{\rm rev}$")
        style(ax)
fig.tight_layout()
save(fig,"lognormal_cell_posteriors")
plt.show()

# Independent ensembles overlaid, not separate models on incompatible axes.
for iz in range(3):
    for ir in range(3):
        meta,posterior,diag=cache[iz,ir,"mixture"]
        ensembles=[]
        lengths=np.asarray(diag["retained_steps"])*meta["walkers"]
        assert lengths.sum()==len(posterior), "Rerun summaries; posterior metadata differ"
        # Summary preserves ensemble blocks; no live HDF files need to be opened.
        for samples in np.split(posterior,np.cumsum(lengths)[:-1]):
            samples=samples[::max(1,len(samples)//20000)].copy()
            samples[:,0]=np.exp(samples[:,0])
            ensembles.append(samples)
        fig=pygtc.plotGTC(chains=ensembles,
            paramNames=[r"$\sigma_{\rm rev}$",r"$f_{\rm rev}$",r"$\ln\tau_{\rm rev}$"],
            chainLabels=[f"ensemble {i}" for i in range(len(ensembles))],
            figureSize="MNRAS_page",
            customLabelFont={"family":"DejaVu Serif","size":14},
            customLegendFont={"family":"DejaVu Serif","size":10},
            customTickFont={"family":"DejaVu Serif","size":10})
        fig.suptitle(rf"${RM_EDGES[ir]}\leq\lambda_{{\rm RM}}<{RM_EDGES[ir+1]}$, "
                     +rf"${Z_EDGES[iz]:.2f}\leq z<{Z_EDGES[iz+1]:.2f}$",y=1.02,fontsize=12)
        save(fig,f"mixture_corner_z{iz}_rm{ir}")
        print(f"Cell z{iz}_rm{ir}: convergence={diag['converged']}")
        plt.show()
'''


def write(name, title, cells, overwrite=False):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = dict(display_name="Python (DESI local)",
                                       language="python", name="desi-local")
    book.cells = [nbf.v4.new_markdown_cell(title)]
    for text, code in cells:
        book.cells += [nbf.v4.new_markdown_cell(text), nbf.v4.new_code_cell(code.strip())]
    path = HERE / name
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite edited notebook: {path}")
    nbf.write(book, path)


if __name__ == "__main__":
    import sys
    if "--update-cells-only" in sys.argv:
        write("plot_conditional_spec_residuals.ipynb",
              "# Independent reverse-conditional residual fits\n"
              r"$P(\lambda_{\rm spec}\mid\lambda_{\rm RM},z)$, with the no-evolution "
              "global log-location fixed per cluster. One log-normal scatter or a "
              "three-parameter mean-centered negative-tail mixture in each cell. "
              "The forward projection fits are not used. Run the existing fitter with "
              "`--cells`, then the summary script with `--cells`, before running this notebook.",
              [("## Configuration, parameter tables and convergence",CELL_SETUP),
               ("## Residual distributions\nHollow points: histogram densities with approximate "
                "68.27% Poisson errors. Curves and shading: posterior median density and 68% "
                "pointwise interval, not data predictive intervals. No normalization to the visible range.",
                CELL_PLOT),
               ("## Posterior distributions and pyGTC contours",CELL_CONTOURS)],overwrite=True)
        raise SystemExit(0)
    write("plot_global_spec_richness.ipynb",
          "# Global spectroscopic-richness regression\n"
          "Section 5 reverse conditional, not an inversion of the forward projection model. "
          "Run the fitter first. These notebooks never run MCMC.",
          [("## Configuration, tables and convergence", SETUP),
           ("## Relation\nHollow markers: geometric means with SEM in log richness. "
            "Bands: 68% posterior intervals of the geometric-mean relation, not population scatter. "
            "Display limits do not affect the fit.", GLOBAL),
           ("## Independent ensemble traces and pyGTC contours", CHAINS)])
    write("plot_conditional_spec_residuals.ipynb",
          "# Conditional distributions and residuals\n"
          r"We model $P(\lambda_{\rm spec}\mid\lambda_{\rm RM},z)$, not a joint density "
          r"of all three quantities. Log residuals are Gaussian under this baseline.",
          [("## Configuration and saved posterior", SETUP),
           ("## Select the model explicitly\nPredictions average over every actual covariate "
            "in each diagnostic cell. No independent cell fits. Histogram errors are approximate "
            "68% Poisson intervals; zero-count upper limits are retained.", RESIDUALS),
           ("## Shape checks\nReplicated standardized residuals under a constant log scatter.", PPC)])
