"""Additive statistics, summed before density ratios or background subtraction."""

import numpy as np
from astropy.cosmology import Planck18
from local_overdensity.environment_catalogs import sky_tree, xyz, curve
from local_overdensity.environment_filters import radial_kernels, pdf_integral


class Accumulator:
    # last axis: raw positive-overlap count, weighted count, sum(weight^2),
    # random-derived effective measure, number of supporting random points.
    def __init__(self, clusters, config, tracer):
        self.config, self.tracer = config, tracer
        self.z = np.asarray(clusters["Z_SPEC_central"], float)
        self.centers = xyz(clusters["RA_central"], clusters["DEC_central"])
        self.dm = Planck18.comoving_transverse_distance(self.z).value * Planck18.h
        self.chi = Planck18.comoving_distance(self.z).value * Planck18.h
        self.stats = np.zeros((len(clusters), 3, 5))

    def add(self, table, random=False, pdf=None, z_grid=None, lf_pdf=None):
        if len(table) == 0:
            return
        vectors, tree = sky_tree(table)
        t = self.config[self.tracer]
        if self.tracer == "bgs":
            chi = Planck18.comoving_distance(np.asarray(table["Z"], float)).value * Planck18.h
        if random:
            measure = np.asarray(table[t["random_measure_column"]], float)
            if np.any(~np.isfinite(measure)) or np.any(measure <= 0):
                raise ValueError("Random quadrature measures must be finite and positive")
        for i, center in enumerate(self.centers):
            indices = np.asarray(tree.query_ball_point(center, 2*np.sin(10/self.dm[i]/2)), dtype=int)
            if not len(indices):
                continue
            chord = np.linalg.norm(vectors[indices] - center, axis=1)
            radius = 2*np.arcsin(np.clip(chord/2, 0, 1)) * self.dm[i]
            kernel = radial_kernels(radius, self.config["gaussian_width_hmpc"])
            if self.tracer == "bgs":
                kernel *= (abs(chi[indices] - self.chi[i]) < t["pi_half_hmpc"])
            if random:
                # Photometric random weights represent deg^2; convert to comoving area.
                unit = (np.pi/180*self.dm[i])**2 if self.tracer == "dr9" else 1.0
                self.stats[i, :, 3] += kernel @ (measure[indices] * unit)
                self.stats[i, :, 4] += (kernel > 0).sum(axis=1)
                continue
            weight = np.asarray(table["ENV_WEIGHT"], float)[indices]
            if self.tracer == "dr9":
                if t.get("photoz_mode") == "gaussian_fits":
                    from local_overdensity.environment_dr9_native import gaussian_probability
                    weight *= gaussian_probability(
                        np.asarray(table["Z_PHOT_MEAN"], float)[indices],
                        np.asarray(table["Z_PHOT_STD"], float)[indices],
                        self.z[i], t["photoz_half_width"])
                else:
                    weight = weight * pdf_integral(z_grid, pdf[indices], self.z[i], lf_weight=lf_pdf)
            self.stats[i, :, 0] += ((kernel > 0) & (weight > 0)).sum(axis=1)
            weighted = kernel * weight
            self.stats[i, :, 1] += weighted.sum(axis=1)
            self.stats[i, :, 2] += (weighted**2).sum(axis=1)


def select_bgs(table, config):
    t = config["bgs"]
    p = np.ma.asarray(table["PROB_OBS"], float).filled(np.nan)
    z = np.ma.asarray(table[t.get("redshift_column", "Z")], float).filled(np.nan)
    quality = np.isin(table[t["quality_column"]], t["quality_values"])
    keep = np.isfinite(p) & (p > 0) & (p <= 1) & np.isfinite(z) & (z > 0) & quality
    out = table[keep].copy()
    out["Z"] = z[keep]
    # LF curve must cover retained tracer redshifts; caller should provide the
    # intended science tracer selection, not extrapolate beyond a calibrated LF.
    if len(out):
        lf = curve(t["lf_curve"], z[keep], "LF_WEIGHT")
        out["ENV_WEIGHT"] = lf / p[keep]
        extra = t.get("additional_weight_column")
        if extra:
            out["ENV_WEIGHT"] *= np.asarray(out[extra], float)
        if np.any(~np.isfinite(out["ENV_WEIGHT"])) or np.any(out["ENV_WEIGHT"] <= 0):
            raise ValueError("Invalid BGS environment weights")
    return out
