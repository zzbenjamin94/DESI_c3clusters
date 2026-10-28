"""Section 4 windows in comoving h^-1 Mpc; no catalog or MPI dependencies."""

import numpy as np
from scipy.integrate import cumulative_trapezoid


def radial_kernels(radius, gaussian_width):
    if not np.isfinite(gaussian_width) or gaussian_width <= 0:
        raise ValueError("Specify a positive Gaussian width; normalization does not set it")
    r = np.asarray(radius, dtype=float)
    signal = (r >= 1.5) & (r < 5.0)
    background = (r >= 5.0) & (r < 10.0)
    return np.array([signal.astype(float), np.exp(-0.5 * (r / gaussian_width)**2) * signal,
                     background.astype(float)])


def kernel_areas(width):
    radial_kernels(np.array([]), width)
    gaussian = 2 * np.pi * width**2 * (np.exp(-1.5**2 / (2 * width**2))
                                      - np.exp(-5.0**2 / (2 * width**2)))
    return np.array([np.pi * (5**2 - 1.5**2), gaussian, np.pi * (10**2 - 5**2)])


def pdf_integral(z_grid, pdf, center, half_width=0.05, lf_weight=None):
    """Exact integral of the piecewise-linear tabulated weighted PDF.

    PDF support outside the tabulated grid is explicitly zero. Input PDFs must
    already integrate to unity: reject, rather than silently renormalize, bad PDFs.
    """
    z = np.asarray(z_grid, dtype=float)
    p = np.asarray(pdf, dtype=float)
    if z.ndim != 1 or len(z) < 2 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("Photo-z grid must be finite and strictly increasing")
    if p.ndim != 2 or p.shape[1] != len(z) or np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("PDFs must be finite nonnegative arrays of shape (galaxy, redshift)")
    norm = cumulative_trapezoid(p, z, axis=1, initial=0)[:, -1]
    if not np.allclose(norm, 1, rtol=1e-3, atol=1e-6):
        raise ValueError("Photo-z PDFs are not normalized on their declared support")
    lf = np.ones(len(z)) if lf_weight is None else np.asarray(lf_weight, dtype=float)
    if lf.shape != z.shape or np.any(~np.isfinite(lf)) or np.any(lf <= 0):
        raise ValueError("LF weights must be finite and positive on the PDF grid")
    f = p * lf
    cumulative = cumulative_trapezoid(f, z, axis=1, initial=0)

    def primitive(x):
        x = np.clip(x, z[0], z[-1])
        k = min(np.searchsorted(z, x, side="right") - 1, len(z) - 2)
        dx = x - z[k]
        return cumulative[:, k] + f[:, k] * dx + 0.5 * (f[:, k+1] - f[:, k]) / (z[k+1] - z[k]) * dx**2

    return np.maximum(primitive(center + half_width) - primitive(center - half_width), 0)


def divide_or_nan(a, b):
    a, b = np.broadcast_arrays(a, b)
    out = np.full(a.shape, np.nan)
    np.divide(a, b, out=out, where=np.isfinite(b) & (b > 0))
    return out
