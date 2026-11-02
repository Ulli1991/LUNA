"""Validation metrics: L_p error norms, interpolation, radial profiles."""
import numpy as np


def lp_error(luna_field, ref_field, p=1):
    """Cell-averaged L_p norm of (luna - ref). Arrays must be on the same grid."""
    diff = np.abs(np.asarray(luna_field) - np.asarray(ref_field))
    return float((np.mean(diff**p))**(1.0 / p))


def l1_error(luna, ref):
    return lp_error(luna, ref, p=1)


def l2_error(luna, ref):
    return lp_error(luna, ref, p=2)


def interp_1d(x_src, f_src, x_dst):
    """Linear interpolation of f_src(x_src) onto x_dst."""
    return np.interp(np.asarray(x_dst), np.asarray(x_src), np.asarray(f_src))


def radial_profile(field, dx, dy, dz, center=None, nbins=None):
    """Spherically-averaged radial profile of a 3D field.

    field : (Nx, Ny, Nz) array
    dx, dy, dz : float, cell sizes
    center : (cx, cy, cz) in cell-index space; default is geometric center.
    nbins : int; default is min(Nx,Ny,Nz)//2.

    Returns (r_centers, profile).
    """
    f = np.asarray(field)
    nx, ny, nz = f.shape
    if center is None:
        center = ((nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0)
    cx, cy, cz = center

    ix = np.arange(nx)[:, None, None]
    iy = np.arange(ny)[None, :, None]
    iz = np.arange(nz)[None, None, :]
    r = np.sqrt(((ix - cx) * dx)**2
                + ((iy - cy) * dy)**2
                + ((iz - cz) * dz)**2).ravel()
    vals = f.ravel()

    if nbins is None:
        nbins = min(nx, ny, nz) // 2
    edges = np.linspace(0.0, r.max(), nbins + 1)

    sum_vals, _ = np.histogram(r, bins=edges, weights=vals)
    counts, _ = np.histogram(r, bins=edges)
    counts_safe = np.where(counts == 0, 1, counts)
    profile = sum_vals / counts_safe
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, profile


def convergence_slope(nx_values, errors):
    """Fit log(error) = a + b * log(Nx) and return the slope b.

    For a method of formal order p on a smooth problem, expect b ≈ -p.
    """
    nx_values = np.asarray(nx_values, dtype=float)
    errors = np.asarray(errors, dtype=float)
    log_nx = np.log(nx_values)
    log_err = np.log(errors)
    slope, _ = np.polyfit(log_nx, log_err, 1)
    return float(slope)
