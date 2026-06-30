"""Sedov self-similar gradient regression test.

Moved from scripts/autodiff/validate_sedov.py. Differentiates the self-contained
differentiable 3D Sedov blast (scripts.autodiff.sedov_ad -- the building block,
which uses the production muscl_hancock_divergence kernel) and asserts that the
forward-mode AD log-derivatives match BOTH:

  (a) the analytic Sedov-Taylor self-similar exponents
      (dlnR/dlnE = +1/5, dlnR/dlnrho0 = -1/5, the KE/TE partition, ...), and
  (b) a finite-difference cross-check (tests/fd_grad.central_diff).

This is the assertion form of paper fig18 (autodiff_sedov_validation); run this
file directly to (re)generate that figure.
"""

import pytest
pytestmark = pytest.mark.slow

import numpy as np


def _run_validation(Nx=32, n_steps=200, t_end=0.08):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from scripts.autodiff.sedov_ad import (
        OBSERVABLE_ANALYTIC, observables_dict, simulate_state)
    from tests.fd_grad import central_diff

    cfg = dict(Nx=Nx, L=1.0, n_steps=n_steps, order=2, limiter=1,
               riemann_solver=1, gamma_static=5.0 / 3.0)
    gamma, L = cfg["gamma_static"], cfg["L"]
    params = jnp.array([0.5, 1.0], dtype=jnp.float64)        # E0, rho0
    obs_keys = list(OBSERVABLE_ANALYTIC.keys())

    def obs_vec(p):
        U = simulate_state(p, t_end, **cfg)
        d = observables_dict(U, Nx, L, gamma)
        return jnp.stack([d[k] for k in obs_keys])

    def log_obs_vec(log_p):
        return jnp.log(jnp.abs(obs_vec(jnp.exp(log_p))) + 1e-30)

    v0 = np.asarray(obs_vec(params))
    log_params = jnp.log(params)
    J_ad = np.asarray(jax.jacfwd(log_obs_vec)(log_params))   # forward-mode AD
    J_fd = np.zeros_like(J_ad)
    for i in range(len(obs_keys)):
        J_fd[i] = central_diff(lambda u, _i=i: log_obs_vec(u)[_i], log_params, h=1e-3)
    return obs_keys, J_ad, J_fd, v0, cfg, t_end


def test_sedov_self_similar_gradients():
    from scripts.autodiff.sedov_ad import OBSERVABLE_ANALYTIC
    # Only the SMOOTH functionals have well-defined analytic exponents and a
    # meaningful AD-vs-FD comparison; the max-based "at-shock" observables are
    # subgradients (AD != FD by design) and L^2 is a pure symmetry check.
    SMOOTH = {"R", "KE_total", "TE_total", "rho_postshock"}
    obs_keys, J_ad, J_fd, v0, cfg, t_end = _run_validation()
    ke_scale = abs(v0[obs_keys.index("KE_total")])            # symmetry scale

    def _close(ad_val, ana):
        # 5% relative for genuine exponents; 0.05 absolute when the analytic
        # exponent is ~0 (e.g. dlnKE/dlnrho0 = 0), where a relative error is
        # ill-defined.
        tol = 0.05 * abs(ana) if abs(ana) >= 1e-2 else 0.05
        return abs(ad_val - ana) <= tol

    checked = 0
    for i, k in enumerate(obs_keys):
        ana_E, ana_rho, is_zero = OBSERVABLE_ANALYTIC[k]
        if is_zero:                                           # symmetry: L^2 << KE
            assert abs(v0[i]) < 0.05 * ke_scale, (
                f"{k}: symmetry broken |v0|={v0[i]:.2e} "
                f"(5%*KE={0.05*ke_scale:.2e})")
            continue
        if k not in SMOOTH:
            continue
        assert _close(J_ad[i, 0], ana_E), (
            f"{k}: dlnE AD {J_ad[i,0]:+.4f} vs analytic {ana_E:+.4f}")
        assert _close(J_ad[i, 1], ana_rho), (
            f"{k}: dlnrho AD {J_ad[i,1]:+.4f} vs analytic {ana_rho:+.4f}")
        err_fd = max(abs(J_ad[i, 0] - J_fd[i, 0]),
                     abs(J_ad[i, 1] - J_fd[i, 1])) / (
                     max(abs(J_fd[i, 0]), abs(J_fd[i, 1])) + 1e-3)
        assert err_fd < 0.05, f"{k}: AD vs FD mismatch {err_fd:.2e}"
        checked += 1
    assert checked >= 3, f"only {checked} smooth observables checked"


def _save_figure(obs_keys, J_ad, J_fd, v0, cfg, t_end):
    """Headline AD-rigor figure (paper fig18): forward-mode AD log-derivatives vs
    the analytic Sedov-Taylor exponents (on y=x), plus the AD-vs-FD cross-check."""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.autodiff.sedov_ad import OBSERVABLE_ANALYTIC
    from utils.figstyle import FIGSIZE, apply_style, decorate
    apply_style()

    SMOOTH = {"R", "KE_total", "TE_total", "rho_postshock"}
    pretty = {"R": "R", "KE_total": r"KE", "TE_total": r"TE",
              "rho_postshock": r"\rho_s"}
    an, ad, fd, labels = [], [], [], []
    for i, k in enumerate(obs_keys):
        if k not in SMOOTH:
            continue
        ana_E, ana_rho, _ = OBSERVABLE_ANALYTIC[k]
        for j, (ana, ch) in enumerate(((ana_E, "E"), (ana_rho, r"\rho_0"))):
            an.append(ana); ad.append(J_ad[i, j]); fd.append(J_fd[i, j])
            labels.append(fr"$\partial\ln {pretty[k]}/\partial\ln {ch}$")
    an, ad, fd = np.array(an), np.array(ad), np.array(fd)

    fig, ax = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))
    colors = plt.cm.tab10(np.arange(len(labels)) % 10)
    lo = min(an.min(), ad.min()) - 0.15
    hi = max(an.max(), ad.max()) + 0.15
    ax[0].plot([lo, hi], [lo, hi], "-", color="0.5", lw=1.0, zorder=1, label=r"$y=x$")
    for x, y, lab, c in zip(an, ad, labels, colors):
        ax[0].scatter(x, y, s=70, color=c, zorder=3, edgecolor="k",
                      linewidth=0.4, label=lab)
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].set_xlabel("analytic Sedov-Taylor exponent")
    ax[0].set_ylabel("AD log-derivative")
    ax[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    decorate(ax[0])

    abserr = np.abs(ad - fd)
    yy = np.arange(len(labels))
    ax[1].barh(yy, abserr, color=colors, zorder=3)
    ax[1].axvline(0.01, color="0.5", ls="-", lw=1.0, label=r"$10^{-2}$")
    ax[1].set_yticks(yy); ax[1].set_yticklabels(labels)
    for tick, c in zip(ax[1].get_yticklabels(), colors):
        tick.set_color(c)
    ax[1].set_xscale("log")
    ax[1].set_xlabel(r"$|{\rm AD}-{\rm FD}|$  (finite-difference cross-check)")
    ax[1].legend(loc="lower right")
    decorate(ax[1])

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = "figures/autodiff_sedov_validation.png"
    fig.savefig(out)
    print(f"max|AD-analytic|={float(np.max(np.abs(ad - an))):.3e}  "
          f"max|AD-FD|={float(np.max(abserr)):.3e}  ->  {out}", flush=True)


if __name__ == "__main__":
    _save_figure(*_run_validation())
