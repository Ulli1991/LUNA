"""Gradient quality vs slope limiter (#37).

Limiters are piecewise functions (minmod: nested min/max; moncen/vanleer:
smoother but still kinked at sign changes). Their raw JAX derivative is
defined a.e. but switches branch discontinuously — the question is whether
that measurably corrupts gradients of integrated observables.

Two probes, both with known-good references:

  A. SEDOV (clean shock):  ∂lnR/∂lnE and ∂lnR/∂lnρ₀ vs the analytic ±1/5,
     and AD vs central FD. One row per scheme:
        order=0 (first-order Godunov — NO limiter, smoothest gradient path)
        order=1 + MinMod / MonCen / VanLeer.

  B. OU TURBULENCE (many kinks): loss-gradient AD vs FD at 5 random voxels
     of δρ after 100 forced steps at 64³. Turbulence exercises the limiter
     switching far more densely than a single shock front.

Output: a per-limiter table of gradient errors. If the limiter choice
matters for AD quality, it shows up here; if all rows agree, raw
differentiation through limiters is vindicated for these problem classes.
"""
from __future__ import annotations

import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")
jax.config.update("jax_enable_x64", True)

from scripts.autodiff.sedov_ad import simulate_state, shock_radius
from scripts.autodiff.fd_grad import central_diff, autodiff_grad
from scripts.autodiff.ou_inverse import low_k_field, _make_simulate
from hydro.turbulence import OUDriver, OUDriverConfig

LIMITER_NAMES = {0: "MinMod", 1: "MonCen", 2: "VanLeer"}


# ---------------------------------------------------------------------------
# Probe A: Sedov log-derivatives per scheme
# ---------------------------------------------------------------------------

def sedov_row(order, limiter):
    cfg = dict(Nx=32, L=1.0, n_steps=200, order=order, limiter=limiter,
               riemann_solver=1, gamma_static=5.0 / 3.0)
    t_end = 0.08
    params = jnp.array([0.5, 1.0])
    log_params = jnp.log(params)

    def f_log(u):
        p = jnp.exp(u)
        U = simulate_state(p, t_end, **cfg)
        return jnp.log(shock_radius(U, cfg["Nx"], cfg["L"]))

    g_ad = autodiff_grad(f_log, log_params)
    g_fd = central_diff(f_log, log_params, h=1e-3)
    err_analytic = max(abs(g_ad[0] - 0.2), abs(g_ad[1] + 0.2)) / 0.2
    err_fd = float(np.max(np.abs(g_ad - g_fd) / (np.abs(g_fd) + 1e-30)))
    return g_ad, err_analytic, err_fd


# ---------------------------------------------------------------------------
# Probe B: OU-turbulence loss-gradient AD vs FD per limiter
# ---------------------------------------------------------------------------

def ou_row(limiter, Nx=64, n_steps=100):
    L = 1.0
    gamma = 5.0 / 3.0
    rho_0, p_0 = 1.0, 1.0
    cs = float(np.sqrt(gamma * p_0 / rho_0))
    v_budget = 2.0
    dt = 0.3 * (L / Nx) / (cs + v_budget)

    key = jax.random.PRNGKey(7)
    k_true, k_ou0, k_walk = jax.random.split(key, 3)
    delta_rho_true = low_k_field(k_true, Nx, k_max=4, amp=0.05)
    ou_cfg = OUDriverConfig(Lx=L, Ly=L, Lz=L, k_min=1.0, k_max=4.0,
                             rms_acceleration=2.0, autocorrelation_time=0.5,
                             solenoidal_weight=1.0)
    ou = OUDriver(cfg=ou_cfg)
    ou_a0 = ou.init_state(k_ou0)
    ou_keys = jax.random.split(k_walk, n_steps)
    coord = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * (L / Nx)
    X, Y, Z = jnp.meshgrid(coord, coord, coord, indexing="ij")

    simulate = _make_simulate(Nx, L, gamma, rho_0, p_0, n_steps, dt,
                               order=2, limiter=limiter, ou_driver=ou,
                               X=X, Y=Y, Z=Z, ou_keys=ou_keys)
    U_obs = simulate(delta_rho_true, ou_a0)

    def loss(dr):
        return jnp.mean((simulate(dr, ou_a0) - U_obs) ** 2)

    g_ad = np.asarray(jax.grad(loss)(jnp.zeros_like(delta_rho_true)))

    rng = np.random.default_rng(0)
    h = 1e-4
    rels = []
    z = jnp.zeros_like(delta_rho_true)
    for flat in rng.choice(Nx ** 3, size=5, replace=False):
        i, j, k = np.unravel_index(flat, (Nx, Nx, Nx))
        Lp = float(loss(z.at[i, j, k].add(h)))
        Lm = float(loss(z.at[i, j, k].add(-h)))
        gfd = (Lp - Lm) / (2.0 * h)
        rels.append(abs(gfd - g_ad[i, j, k]) / max(abs(gfd), 1e-30))
    return float(np.max(rels)), float(np.mean(rels))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Gradient quality vs slope limiter")
    print("=" * 70)

    print("\nProbe A — Sedov ∂lnR/∂lnE (analytic +0.2), ∂lnR/∂lnρ₀ (analytic −0.2)")
    print(f"  {'scheme':<22s} {'∂lnE':>9s} {'∂lnρ₀':>9s} {'vs analytic':>12s} {'AD vs FD':>10s}")
    rows = [(0, 0, "Godunov (no limiter)")] + [
        (1, lim, f"PLM + {LIMITER_NAMES[lim]}") for lim in (0, 1, 2)
    ]
    for order, lim, name in rows:
        t0 = time.time()
        g, e_an, e_fd = sedov_row(order, lim)
        print(f"  {name:<22s} {g[0]:+9.4f} {g[1]:+9.4f} {e_an:12.2e} {e_fd:10.2e}"
              f"   ({time.time()-t0:.0f}s)")

    print("\nProbe B — OU turbulence 64³/100 steps: loss-grad AD vs FD (5 voxels)")
    print(f"  {'limiter':<22s} {'max rel':>10s} {'mean rel':>10s}")
    for lim in (0, 1, 2):
        t0 = time.time()
        mx, mn = ou_row(lim)
        print(f"  PLM + {LIMITER_NAMES[lim]:<16s} {mx:10.2e} {mn:10.2e}   ({time.time()-t0:.0f}s)")

    print("\ndone.")


if __name__ == "__main__":
    main()
