"""Finite-difference cross-check for autodiff gradients.

Two utilities:

  central_diff(f, x, h)
      Central-difference Jacobian of a scalar-output `f: R^n -> R` at point `x`.
      Returns (∂f/∂x_i)_i. h is per-coordinate or scalar; for log-scaled inputs
      use h ≈ 1e-3 * |x_i|. Costs 2n forward evaluations.

  horizon_scan(f_for_horizon, params, horizons, h)
      For each integration horizon T in `horizons`, compute both the autodiff
      gradient and the FD gradient of `f_for_horizon(params, T)` w.r.t.
      `params`, and return arrays (grads_ad, grads_fd, err_rel). The single
      diagnostic that tells you when chaos / limiter kinks start eating the
      signal.

The autodiff side uses forward-mode (jacfwd) — correct for the small-parameter-
vector case our autodiff demos target (Sedov ∂R/∂E with 2 params, CC85 with
~4 params). For large parameter vectors, swap to jax.grad/reverse-mode.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


def central_diff(f, x, h):
    """Central-difference gradient of scalar f at point x.

    Parameters
    ----------
    f : Callable[[array], scalar]
    x : 1D array_like of length n
    h : scalar OR array_like of length n
        Step size per coordinate. Use h ~ 1e-3 * |x| for typical doubles
        (smaller and you hit roundoff; larger and the truncation error wins).

    Returns
    -------
    g : ndarray of length n, the gradient ∂f/∂x.
    """
    x = jnp.asarray(x)
    n = x.size
    h_arr = jnp.broadcast_to(jnp.asarray(h), x.shape)
    g = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ei = jnp.zeros(n).at[i].set(h_arr[i])
        f_plus = float(f(x + ei))
        f_minus = float(f(x - ei))
        g[i] = (f_plus - f_minus) / (2.0 * float(h_arr[i]))
    return g


def autodiff_grad(f, x):
    """Forward-mode gradient via jacfwd. Suits small parameter vectors."""
    g = jax.jacfwd(f)(jnp.asarray(x))
    return np.asarray(g)


def horizon_scan(f_for_horizon, params, horizons, h, log_relative=True):
    """Compare AD vs FD gradients of a sim-with-horizon objective.

    Parameters
    ----------
    f_for_horizon : Callable[[array, float], scalar]
        Closure: f(params, T) -> scalar observable at integration horizon T.
    params : 1D array_like
        Parameter vector to differentiate w.r.t.
    horizons : 1D array_like of horizon values
    h : scalar or per-coord array, step size for central diff
    log_relative : bool
        If True, return |g_ad - g_fd| / |g_fd| (componentwise max). False = abs.

    Returns
    -------
    grads_ad : (len(horizons), len(params)) ndarray
    grads_fd : (len(horizons), len(params)) ndarray
    err      : (len(horizons),) ndarray — per-horizon error metric.
    """
    horizons = np.asarray(horizons, dtype=np.float64)
    n_p = len(params)
    n_T = len(horizons)
    grads_ad = np.zeros((n_T, n_p), dtype=np.float64)
    grads_fd = np.zeros((n_T, n_p), dtype=np.float64)
    err = np.zeros(n_T, dtype=np.float64)
    for k, T in enumerate(horizons):
        f_T = lambda p, _T=float(T): f_for_horizon(p, _T)
        ga = autodiff_grad(f_T, params)
        gf = central_diff(f_T, params, h)
        grads_ad[k] = ga
        grads_fd[k] = gf
        if log_relative:
            denom = np.maximum(np.abs(gf), 1e-30)
            err[k] = float(np.max(np.abs(ga - gf) / denom))
        else:
            err[k] = float(np.max(np.abs(ga - gf)))
    return grads_ad, grads_fd, err


def plot_horizon_scan(horizons, err, out_path, title="AD vs FD: |Δg|/|g_FD|"):
    """Write a log-scale plot of the FD/AD error vs integration horizon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.semilogy(horizons, err, "o-", lw=1.5)
    ax.set_xlabel("integration horizon  T")
    ax.set_ylabel("|∇_AD − ∇_FD| / |∇_FD|  (max over params)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1e-3, color="gray", lw=0.8, ls="--")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
