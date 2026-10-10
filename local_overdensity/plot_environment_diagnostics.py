"""Read-only Section 4 diagnostics; no catalog generation or model fitting."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join
from local_overdensity.environment_filters import radial_kernels, kernel_areas


def spatial_mean_jackknife(values, ra, dec, nra=6, ndec=3):
    """Coarse sky-block jackknife covariance of the MEAN, not per-cluster scatter.

    This exploratory partition is not a validated survey jackknife. Stability
    against patch scale, geometry, occupancy and sample selection is required.
    """
    values = np.asarray(values)
    region = (np.floor((np.asarray(ra) % 360)/360*nra).astype(int)
              + nra*np.minimum(np.floor((np.sin(np.deg2rad(dec))+1)/2*ndec).astype(int), ndec-1))
    regions = np.unique(region)
    if len(regions) < 3:
        return None
    estimates = np.array([values[region != r].mean(axis=0) for r in regions])
    delta = estimates - estimates.mean(axis=0)
    return (len(regions)-1)/len(regions) * delta.T @ delta


def plot_diagnostics(root, width, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = []

    def save(fig, name):
        for ax in fig.axes:
            ax.tick_params(direction="in", top=True, right=True)
        for ext in ("png", "pdf"):
            fig.savefig(output / f"{name}.{ext}", dpi=160)
        saved.append(fig)

    radius = np.linspace(0, 11, 2000)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for k, label in enumerate(("Annular box", "Truncated Gaussian", "Background box")):
        ax.plot(radius, radial_kernels(radius, width)[k]/kernel_areas(width)[k], label=label)
    ax.set(xlabel=r"$R$ [comoving $h^{-1}\,\mathrm{Mpc}$]",
           ylabel=r"$W(R)$ [$h^2\,\mathrm{Mpc}^{-2}$]")
    ax.legend(frameon=False)
    save(fig, "filter_windows")
    tables = {}
    for tracer in ("dr9", "bgs"):
        path = Path(root) / tracer / "environment.fits"
        if not path.exists():
            print(f"Skipping absent output: {path}")
            continue
        tab = Table.read(path)
        tables[tracer] = tab
        filters = ("box", "gaussian") if tracer == "dr9" else ("box",)
        for filt in filters:
            p = f"{tracer}_{filt}"
            fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
            for ax, suffix, label in zip(axes[0], ("coverage", "N", "E_local"),
                                         ("Kernel coverage", "Contributing galaxies", r"$E_{a}^{\rm local}$")):
                v = np.asarray(tab[p+"_"+suffix], float)
                ax.hist(v[np.isfinite(v)], bins=30, histtype="step", color="teal")
                ax.set(xlabel=label, ylabel="Clusters")
            richness = tab["LAMBDA"] if "LAMBDA" in tab.colnames else tab["Z_SPEC_central"]
            for ax, x, label in zip(axes[1, :2], (tab["Z_SPEC_central"], richness),
                                    (r"$z_{\rm BCG}$", r"$\lambda_{\rm RM}$" if "LAMBDA" in tab.colnames else r"$z_{\rm BCG}$")):
                ax.scatter(x, tab[p+"_excess"], s=6, alpha=0.4)
                unit = r"h^2\,\mathrm{Mpc}^{-2}" if tracer == "dr9" else r"h^3\,\mathrm{Mpc}^{-3}"
                ax.set(xlabel=label, ylabel=r"Excess density [$"+unit+"$]")
                ax.axhline(0, color="0.5", lw=0.8)
            sc = axes[1, 2].scatter(tab["RA_central"], tab["DEC_central"], c=tab[p+"_coverage"], s=8, cmap="viridis")
            axes[1, 2].set(xlabel="RA [deg]", ylabel="Dec [deg]")
            fig.colorbar(sc, ax=axes[1, 2], label="Coverage")
            save(fig, p+"_diagnostics")
            print(p, "invalid local ratios:", np.sum(~np.asarray(tab[p+"_E_local_valid"], bool)))
    if set(tables) == {"dr9", "bgs"}:
        dr9, bgs = tables["dr9"], tables["bgs"]
        columns = ["dr9_box_E_local", "dr9_gaussian_E_local", "bgs_box_E_local"]
        paired = join(dr9[["ID", "RA_central", "DEC_central", *columns[:2]]],
                      bgs[["ID", columns[2]]], keys="ID", metadata_conflicts="silent")
        values = np.column_stack([np.asarray(paired[k], float) for k in columns])
        good = np.all(np.isfinite(values), axis=1)
        if good.sum() >= 3:
            values, paired = values[good], paired[good]
            np.savetxt(output / "environment_sample_covariance.csv", np.cov(values, rowvar=False), delimiter=",", header=",".join(columns))
            jk = spatial_mean_jackknife(values, paired["RA_central"], paired["DEC_central"])
            if jk is not None:
                np.savetxt(output / "exploratory_mean_jackknife_covariance.csv", jk, delimiter=",", header=",".join(columns))
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
            axes[0].scatter(values[:, 0], values[:, 2], s=8, alpha=0.5)
            axes[0].set(xlabel=r"$E_{\rm DR9,box}$", ylabel=r"$E_{\rm BGS,box}^{\rm local}$")
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(values, rowvar=False)
            im = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
            labels = ["DR9 box", "DR9 Gaussian", "BGS box"]
            axes[1].set_xticks(range(3), labels, rotation=30, ha="right")
            axes[1].set_yticks(range(3), labels)
            fig.colorbar(im, ax=axes[1], label="Sample correlation")
            save(fig, "environment_comparison")
    return saved
