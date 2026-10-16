"""Finalize count/measure sums; masked or unsupported environments remain NaN."""

import numpy as np
from local_overdensity.environment_filters import divide_or_nan, kernel_areas
from local_overdensity.environment_catalogs import curve


def finalize(clusters, stats, config, tracer):
    out = clusters.copy()
    names = ("box", "gaussian", "background") if tracer == "dr9" else ("box", "background")
    indices = (0, 1, 2) if tracer == "dr9" else (0, 2)
    full_measure = kernel_areas(config["gaussian_width_hmpc"])
    if tracer == "bgs":
        full_measure *= 2 * config["bgs"]["pi_half_hmpc"]
    density = divide_or_nan(stats[:, :, 1], stats[:, :, 3])
    variance = divide_or_nan(stats[:, :, 2], stats[:, :, 3]**2)
    for name, k in zip(names, indices):
        prefix = f"{tracer}_{name}"
        for label, col in (("N", 0), ("weighted_count", 1), ("sum_w2", 2),
                           ("effective_measure", 3), ("N_random", 4)):
            out[f"{prefix}_{label}"] = stats[:, k, col]
        coverage = stats[:, k, 3] / full_measure[k]
        out[f"{prefix}_coverage"] = coverage
        out[f"{prefix}_density"] = density[:, k]
        out[f"{prefix}_density_shot_err"] = np.sqrt(variance[:, k])
        if name != "background":
            excess = density[:, k] - density[:, 2]
            out[f"{prefix}_excess"] = excess
            out[f"{prefix}_excess_shot_err"] = np.sqrt(variance[:, k] + variance[:, 2])
            out[f"{prefix}_E_local"] = divide_or_nan(excess, density[:, 2])
            valid = np.isfinite(excess) & (stats[:, k, 4] > 0) & (stats[:, 2, 4] > 0)
            out[f"{prefix}_valid"] = valid
            out[f"{prefix}_E_local_valid"] = valid & (density[:, 2] > 0)
            if tracer == "bgs" and config["bgs"].get("nbar_curve"):
                nbar = curve(config["bgs"]["nbar_curve"], np.asarray(out["Z_SPEC_central"]), "NBAR")
                out[f"{prefix}_E_global"] = density[:, k]/nbar - 1
    out.meta["TRACER"] = tracer
    out.meta["RADUNIT"] = "comoving h^-1 Mpc"
    out.meta["DENSUNIT"] = "h^2 Mpc^-2" if tracer == "dr9" else "h^3 Mpc^-3"
    out.meta["SHOTNOTE"] = "Conditional Poisson only; excludes random/PDF/LF uncertainty and clustering"
    return out
