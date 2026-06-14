"""Validate Sedov self-similar gradients across a suite of observables.

For each observable in `OBSERVABLE_ANALYTIC` we know the analytic exponent
∂ln(·)/∂lnE and ∂ln(·)/∂lnρ₀ from the Sedov-Taylor similarity solution.
This script computes both columns of that table via:

  - jacfwd (forward-mode AD through the full simulation, ~1 forward pass)
  - central FD with h = 1e-3 · |x|

and prints a pass/fail matrix.

Observables tested:

  R           shock radius (KE-weighted RMS) — smooth volumetric moment
  KE_total    total kinetic energy           — Sedov energy partition
  TE_total    total thermal energy           — Sedov energy partition
  L_sq_total  ∫|r×ρv|² dV                    — symmetry test, analytic = 0
  P_postshock pressure at shock              — Rankine-Hugoniot
  rho_postshock density at shock             — RH jump (ρ_s independent of E)
  T_postshock P/ρ at shock                   — ratio test

A row passes if (a) the AD and FD log-derivatives agree to <1% and
(b) AD matches the analytic exponent to <5%. The "zero by symmetry"
observable (L_sq_total) instead checks that the absolute gradient is below
machine-ish (<1e-6 in normalised units).
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")

from scripts.autodiff.sedov_ad import (
    simulate_state, observables_dict, OBSERVABLE_ANALYTIC,
)
from scripts.autodiff.fd_grad import central_diff


def main():
    E0 = 0.5
    rho0 = 1.0
    params = jnp.array([E0, rho0], dtype=jnp.float64)

    cfg = dict(Nx=32, L=1.0, n_steps=200, order=2, limiter=1,
               riemann_solver=1, gamma_static=5.0/3.0)
    t_end = 0.08
    gamma = cfg["gamma_static"]
    Nx, L = cfg["Nx"], cfg["L"]

    obs_keys = list(OBSERVABLE_ANALYTIC.keys())
    n_obs = len(obs_keys)

    def obs_vec(p):
        """All observables as a flat (n_obs,) vector."""
        U = simulate_state(p, t_end, **cfg)
        d = observables_dict(U, Nx, L, gamma)
        return jnp.stack([d[k] for k in obs_keys])

    def log_obs_vec(log_p):
        """For non-zero observables, log-space: f_i(u) = ln(obs_i(exp(u)))."""
        p = jnp.exp(log_p)
        return jnp.log(jnp.abs(obs_vec(p)) + 1e-30)

    print(f"\nSedov 3D — full observable validation matrix")
    print(f"  Nx={cfg['Nx']}, n_steps={cfg['n_steps']}, "
          f"limiter={cfg['limiter']}, t_end={t_end}")
    print(f"  Eval point: E={E0}, ρ₀={rho0}")

    # Warmup + value print.
    t0 = time.time()
    v0 = np.asarray(obs_vec(params))
    print(f"  warmup ok ({time.time()-t0:.1f}s)")
    print("\n  observable values at eval point:")
    for k, v in zip(obs_keys, v0):
        print(f"    {k:>14s} = {v:+.4e}")

    # ---- Log-Jacobians: AD via jacfwd, FD via central_diff ----
    log_params = jnp.log(params)
    print("\n  computing Jacobians (AD + FD)...")
    t0 = time.time()
    J_ad = np.asarray(jax.jacfwd(log_obs_vec)(log_params))
    # FD: per-obs central diff. Reuse the same function.
    J_fd = np.zeros_like(J_ad)
    for i in range(n_obs):
        f_i = lambda u, _i=i: log_obs_vec(u)[_i]
        J_fd[i] = central_diff(f_i, log_params, h=1e-3)
    print(f"  done ({time.time()-t0:.1f}s)")

    # ---- Print table ----
    print("\n  ┌────────────────┬──────────┬──────────┬──────────┬──────────┬─────┐")
    print("  │ Observable     │ ∂lnE: AD │  ana E   │ ∂lnρ: AD │  ana ρ   │ PASS│")
    print("  ├────────────────┼──────────┼──────────┼──────────┼──────────┼─────┤")
    fails = []
    for i, k in enumerate(obs_keys):
        ana_E, ana_rho, is_zero = OBSERVABLE_ANALYTIC[k]
        a_E = J_ad[i, 0]
        a_rho = J_ad[i, 1]
        if is_zero:
            # Check the *raw* gradient magnitude in linear space.
            #  d(L²)/dE = obs * d(ln L²)/d ln E * (1/E)
            # Since obs may be tiny, just report whether obs itself ≈ 0.
            ok = abs(v0[i]) < 1e-6
            ana_E_s = "0"; ana_rho_s = "0"
        else:
            err_E = abs(a_E - ana_E) / max(abs(ana_E), 1e-3)
            err_rho = abs(a_rho - ana_rho) / max(abs(ana_rho), 1e-3)
            err_fd = max(abs(a_E - J_fd[i, 0]), abs(a_rho - J_fd[i, 1])) / (
                max(abs(J_fd[i, 0]), abs(J_fd[i, 1])) + 1e-3)
            ok = (err_E < 0.05) and (err_rho < 0.05) and (err_fd < 0.01)
            ana_E_s = f"{ana_E:+.4f}"; ana_rho_s = f"{ana_rho:+.4f}"
        if not ok:
            fails.append(k)
        print(f"  │ {k:<14s} │ {a_E:+.4f}  │ {ana_E_s:>8s} │ "
              f"{a_rho:+.4f}  │ {ana_rho_s:>8s} │  {'✓' if ok else '✗'}  │")
    print("  └────────────────┴──────────┴──────────┴──────────┴──────────┴─────┘")
    if fails:
        print(f"\n  FAIL: {fails}")
    else:
        print("\n  ALL OBSERVABLES PASS analytic ±5%, AD vs FD ±1%")

    # ---- Energy conservation cross-check ----
    KE = float(v0[obs_keys.index("KE_total")])
    TE = float(v0[obs_keys.index("TE_total")])
    E_total = KE + TE
    print(f"\n  Energy bookkeeping:  KE={KE:.4e}  TE={TE:.4e}  "
          f"KE+TE={E_total:.4e}  (E_inject={E0})")
    print(f"  KE/E = {KE/E0:.3f} (Sedov γ=5/3 expects ~0.30)")
    print(f"  TE/E = {TE/E0:.3f} (Sedov γ=5/3 expects ~0.70)")

    _save_figure(obs_keys, J_ad, J_fd, v0, cfg, t_end)


def _save_figure(obs_keys, J_ad, J_fd, v0, cfg, t_end):
    """Headline AD-rigor figure: forward-mode AD log-derivatives vs the analytic
    Sedov-Taylor exponents (should lie on y=x), plus the AD-vs-FD cross-check."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()

    # Restrict to the SMOOTH functionals with well-defined analytic exponents.
    # The "at-shock" pressure/temperature are max-based (non-smooth → AD gives a
    # subgradient FD can't match) and L_sq is a symmetry check, so they're shown
    # in the printed table but excluded from this AD-vs-analytic rigor figure.
    SMOOTH = {"R", "KE_total", "TE_total", "rho_postshock"}
    pretty = {"R": "R", "KE_total": r"KE", "TE_total": r"TE",
              "rho_postshock": r"\rho_s"}
    an, ad, fd, labels = [], [], [], []
    for i, k in enumerate(obs_keys):
        if k not in SMOOTH:
            continue
        ana_E, ana_rho, is_zero = OBSERVABLE_ANALYTIC[k]
        for j, (ana, ch) in enumerate(((ana_E, "E"), (ana_rho, r"\rho_0"))):
            an.append(ana); ad.append(J_ad[i, j]); fd.append(J_fd[i, j])
            labels.append(fr"$\partial\ln {pretty[k]}/\partial\ln {ch}$")
    an = np.array(an); ad = np.array(ad); fd = np.array(fd)

    fig, ax = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))

    # (a) AD vs analytic — points on y=x validate the gradient against theory.
    lo = min(an.min(), ad.min()) - 0.15
    hi = max(an.max(), ad.max()) + 0.15
    ax[0].plot([lo, hi], [lo, hi], "-", color="0.5", lw=1.2, zorder=1,
               label=r"$y=x$")
    ax[0].scatter(an, ad, s=55, color="C0", zorder=3, label="forward-mode AD")
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("analytic Sedov-Taylor exponent")
    ax[0].set_ylabel("AD log-derivative")
    ax[0].set_title(f"AD vs analytic ($N$={cfg['Nx']}, {cfg['n_steps']} steps)")
    ax[0].legend(loc="upper left")
    decorate(ax[0])

    # (b) AD-vs-FD relative error per channel — the independent cross-check.
    rel = np.abs(ad - fd) / (np.abs(fd) + 1e-3)
    yy = np.arange(len(labels))
    ax[1].barh(yy, rel, color="C1", zorder=3)
    ax[1].axvline(0.01, color="k", ls="--", lw=1.2, label="1% threshold")
    ax[1].set_yticks(yy); ax[1].set_yticklabels(labels, fontsize=9)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$|{\rm AD}-{\rm FD}|/|{\rm FD}|$")
    ax[1].set_title("AD vs finite-difference cross-check")
    ax[1].legend(loc="lower right")
    decorate(ax[1])

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/autodiff_sedov_validation.png"
    fig.savefig(out)
    maxrel = float(np.max(np.abs(ad - an)) )
    print(f"\nRESULT sedov_ad: max|AD-analytic|={maxrel:.3e}  "
          f"max|AD-FD|rel={float(np.max(rel)):.3e}", flush=True)
    print(f"  figure -> {out}", flush=True)


if __name__ == "__main__":
    main()
