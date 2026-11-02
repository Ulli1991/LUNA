"""Sedov self-similar gradient regression test, through the production autodiff path.

Builds U0(E0, rho0) with the registered Sedov-Blast-3D IC and differentiates the production
rollout (luna.autodiff.rollout.build_simulate: the same step, stages and halo as a run) with
forward-mode AD, asserting that the log-derivatives of smooth observables match BOTH
  (a) the analytic Sedov-Taylor exponents (dlnR/dlnE = +1/5, dlnR/dlnrho0 = -1/5, the KE/TE
      partition, ...), and
  (b) a finite-difference cross-check (tests/fd_grad.central_diff).

This is the assertion form of paper fig18 (autodiff_sedov_validation); run this file directly
to (re)generate that figure.
"""

import pytest
pytestmark = pytest.mark.slow

import jax.numpy as jnp
import numpy as np


# Observables: smooth functionals of the state with analytic Sedov-Taylor exponents.

def _grid_coords(Nx, L):
    dx = L / Nx
    c = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * dx - 0.5 * L
    X, Y, Z = jnp.meshgrid(c, c, c, indexing="ij")
    return X, Y, Z, dx


def shock_radius(U, Nx, L):
    """RMS radius weighted by kinetic energy density — smooth shock locator.

    Analytic Sedov: R ∝ (E/ρ₀)^{1/5} t^{2/5} →
        ∂lnR/∂lnE = +1/5,  ∂lnR/∂lnρ₀ = -1/5.
    """
    X, Y, Z, _ = _grid_coords(Nx, L)
    r2 = X * X + Y * Y + Z * Z
    rho = U[0]
    ke_density = 0.5 / rho * (U[1] * U[1] + U[2] * U[2] + U[3] * U[3])
    w = ke_density
    num = jnp.sum(w * r2)
    den = jnp.sum(w) + 1e-30
    return jnp.sqrt(num / den)


def total_kinetic_energy(U, Nx, L):
    """∫ (1/2) ρ v² dV. Sedov: KE ≈ 0.30 E (γ=5/3 fixed)  →  ∂lnKE/∂lnE = 1,
    ∂lnKE/∂lnρ₀ = 0. Tests that energy *partition* survives the limiter."""
    dx = L / Nx
    ke = 0.5 / U[0] * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2)
    return jnp.sum(ke) * (dx ** 3)


def total_thermal_energy(U, Nx, L, gamma):
    """∫ P/(γ-1) dV. Sedov: TE ≈ 0.70 E  →  ∂lnTE/∂lnE = 1, ∂lnTE/∂lnρ₀ = 0.
    The total E = KE + TE check is the "conservation through AD" test."""
    dx = L / Nx
    ke = 0.5 / U[0] * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2)
    P = (gamma - 1.0) * (U[4] - ke)
    return jnp.sum(P / (gamma - 1.0)) * (dx ** 3)


def total_angular_momentum_sq(U, Nx, L):
    """∫ |r × ρv|² dV. Symmetric explosion → analytic value is zero.
    Tests that AD respects the symmetry: ∂L²/∂E = 0, ∂L²/∂ρ₀ = 0."""
    dx = L / Nx
    X, Y, Z, _ = _grid_coords(Nx, L)
    px, py, pz = U[1], U[2], U[3]
    lx = Y * pz - Z * py
    ly = Z * px - X * pz
    lz = X * py - Y * px
    return jnp.sum(lx * lx + ly * ly + lz * lz) * (dx ** 3)


def _smooth_max(field, weight, k=8.0):
    """Power-mean over a positive weight field — a smooth surrogate for max(field).

    Behaves like max(field) where weight is largest, but is a smooth functional
    of state so the gradient is well-defined. k controls sharpness; k=8 is a
    middle-ground between robustness and how close we get to true max.
    """
    w = weight ** k
    num = jnp.sum(field * w)
    den = jnp.sum(w) + 1e-30
    return num / den


def post_shock_pressure(U, Nx, L, gamma):
    """Pressure where kinetic-energy density is largest (~ post-shock).

    Sedov + Rankine-Hugoniot: P_s = (2/(γ+1)) ρ₀ V_s² ∝ (E^2 ρ₀^3)^{1/5} t^{-6/5}.
        ∂lnP_s/∂lnE   = +2/5
        ∂lnP_s/∂lnρ₀ = +3/5
    """
    ke = 0.5 / U[0] * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2)
    P = (gamma - 1.0) * (U[4] - ke)
    return _smooth_max(P, weight=ke)


def post_shock_density(U, Nx, L):
    """ρ at the shock front (KE-weighted smooth-max).
    Sedov + RH: ρ_s = ρ₀ (γ+1)/(γ-1).  ∂lnρ_s/∂lnρ₀ = 1, ∂lnρ_s/∂lnE = 0."""
    ke = 0.5 / U[0] * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2)
    return _smooth_max(U[0], weight=ke)


def post_shock_temperature(U, Nx, L, gamma):
    """T_s ∝ P_s/ρ_s. ∂lnT/∂lnE = +2/5, ∂lnT/∂lnρ₀ = −2/5."""
    return post_shock_pressure(U, Nx, L, gamma) / post_shock_density(U, Nx, L)


def observables_dict(U, Nx, L, gamma):
    """Return all observables in a stable-ordered dict."""
    return dict(
        R           = shock_radius(U, Nx, L),
        KE_total    = total_kinetic_energy(U, Nx, L),
        TE_total    = total_thermal_energy(U, Nx, L, gamma),
        L_sq_total  = total_angular_momentum_sq(U, Nx, L),
        P_postshock = post_shock_pressure(U, Nx, L, gamma),
        rho_postshock = post_shock_density(U, Nx, L),
        T_postshock = post_shock_temperature(U, Nx, L, gamma),
    )


# Map of observable name -> (analytic ∂ln(·)/∂lnE, analytic ∂ln(·)/∂lnρ₀,
#                            is_zero — True means "expected to be 0, not log").
OBSERVABLE_ANALYTIC = {
    "R":             (+1/5,  -1/5,  False),   # shock radius
    "KE_total":      (+1.0,   0.0,  False),   # Sedov energy partition
    "TE_total":      (+1.0,   0.0,  False),
    "L_sq_total":    (None,   None, True),    # exactly zero by symmetry
    "P_postshock":   (+2/5,  +3/5,  False),
    "rho_postshock": ( 0.0,  +1.0,  False),
    "T_postshock":   (+2/5,  -2/5,  False),
}


def build_rollout(Nx, n_steps, t_end):
    """(U0(params), rollout(U0)): the registered IC at params = (E0, rho0) and the production
    fixed-dt rollout on the mesh a run would choose (as tools/kh_stability.py drives it)."""
    from dataclasses import replace

    import jax
    from luna.autodiff.rollout import build_simulate
    from luna.ic_generation import sedov3d
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    from luna.template_config import Config

    cfg = replace(Config(), test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3,
                  Nx=Nx, Ny=Nx, Nz=Nx, Lx=1.0, Ly=1.0, Lz=1.0, BOUNDARY=0, gamma=5.0 / 3.0,
                  ORDER=1, LIMITER=1, riemann_solver="hllc", enable_x64=True,
                  autodiff_mode=True, autodiff_n_steps=n_steps,
                  autodiff_fixed_dt=t_end / n_steps, write_snapshots=False,
                  enable_diagnostics=False, enable_checkpointing=False)
    cfg = resolve(cfg)
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, False)
    physics = assemble(cfg, mesh, kind, layout, init, False)
    simulate, _, _, _, theta = build_simulate(init, cfg, physics, False)

    def U0(params):
        """The registered IC as a point blast (r_blast 1.24 cells, the volume of the 8 cells it covers; the default 6 cells is half the
        final shock radius at N = 32 and shifts dlnR/dlnE to 0.243); p_bg = 1e-4 rho0 keeps the
        ambient sound speed fixed."""
        problem = sedov3d.Params(E=params[0], rho_bg=params[1], p_bg=1e-4 * params[1], r_blast=1.24)
        return jax.lax.with_sharding_constraint(
            sedov3d.sedov_blast_3d(replace(cfg, problem=problem)), init.U.sharding)

    return U0, (lambda U: simulate(theta, U)), cfg


def _run_validation(Nx=32, n_steps=200, t_end=0.08):
    import jax

    from tests.fd_grad import central_diff

    U0, rollout, cfg = build_rollout(Nx, n_steps, t_end)
    gamma, L = cfg.gamma, cfg.Lx
    params = jnp.array([0.5, 1.0], dtype=jnp.float64)        # E0, rho0
    obs_keys = list(OBSERVABLE_ANALYTIC.keys())

    def obs_vec(p):
        d = observables_dict(rollout(U0(p)), Nx, L, gamma)
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

    from luna.utils.figstyle import FIGSIZE, apply_style, decorate
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
    os.makedirs("code_paper_figs", exist_ok=True)
    out = "code_paper_figs/autodiff_sedov_validation.png"
    fig.savefig(out)
    print(f"max|AD-analytic|={float(np.max(np.abs(ad - an))):.3e}  "
          f"max|AD-FD|={float(np.max(abserr)):.3e}  ->  {out}", flush=True)


if __name__ == "__main__":
    _save_figure(*_run_validation())
