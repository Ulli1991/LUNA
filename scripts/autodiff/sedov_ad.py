"""Differentiable 3D Sedov-Taylor blast.

A small, self-contained, jit-friendly Sedov simulation parametrised by
two scalars — the injected energy E and the ambient density ρ₀. It is
written so that `simulate(params, t_end)` is a clean R² → R map suitable
for `jax.jacfwd`, `jax.grad`, or a finite-difference cross-check.

Analytic Sedov scaling (3D, point-blast, γ=5/3):
    R_shock(t) = ξ · (E / ρ₀)^{1/5} · t^{2/5}
giving
    ∂ ln R / ∂ ln E  = +1/5
    ∂ ln R / ∂ ln ρ₀ = -1/5

`shock_radius` is defined as the KE-weighted RMS radius:
    R²_obs = (∫ ρ v² r² dV) / (∫ ρ v² dV)
which is a smooth functional of state (no argmax, no thresholding) so the
gradient through it is well-defined even for limiter-laden hydro.
"""
from __future__ import annotations

import functools
import jax
import jax.numpy as jnp

from hydro.muscl_hancock import muscl_hancock_divergence
from hydro.conversions import cons_to_prim


# ---------------------------------------------------------------------------
# IC: 3D Sedov, central 2×2×2 cube energy injection on a uniform background.
# ---------------------------------------------------------------------------

def build_ic(E_inj, rho_bg, Nx, L, gamma, p_floor_frac=1e-4):
    """Return conservative state U for a parametrised Sedov IC.

    Tracing-friendly: every operation is jax-native, no numpy, no Python
    branches on `E_inj`/`rho_bg`. Treats E_inj and rho_bg as JAX scalars.
    """
    dx = L / Nx
    # Uniform background.
    rho = jnp.full((Nx, Nx, Nx), rho_bg)
    vx = jnp.zeros_like(rho)
    vy = jnp.zeros_like(rho)
    vz = jnp.zeros_like(rho)
    # Set a small reference pressure proportional to rho_bg so the background
    # temperature is fixed under ρ-rescaling (cleanest for FD/AD comparison).
    p_bg = p_floor_frac * rho_bg

    # Central 2×2×2 cube injection (8 cells).
    ix0 = Nx // 2 - 1
    ii = jnp.arange(Nx)[:, None, None]
    jj = jnp.arange(Nx)[None, :, None]
    kk = jnp.arange(Nx)[None, None, :]
    in_cube = ((ii >= ix0) & (ii <= ix0 + 1)
               & (jj >= ix0) & (jj <= ix0 + 1)
               & (kk >= ix0) & (kk <= ix0 + 1))
    V_inject = 8.0 * dx * dx * dx
    p_inject = (gamma - 1.0) * (E_inj / V_inject)
    p = jnp.where(in_cube, p_inject + p_bg, p_bg)

    E = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)


# ---------------------------------------------------------------------------
# Time stepper: forward Euler with the predictor-corrector MUSCL-Hancock
# spatial RHS (MUSCL-Hancock already provides 2nd-order time accuracy).
# ---------------------------------------------------------------------------

def _pad_periodic(W, ghost=1):
    return jnp.pad(W, ((0, 0), (ghost, ghost), (ghost, ghost), (ghost, ghost)),
                   mode="wrap")


def _step(U, dt, *, gamma, limiter, riemann_solver, order, inv_dx):
    W = cons_to_prim(U, gamma)
    W_pad = _pad_periodic(W, ghost=1)
    dUdt = muscl_hancock_divergence(
        W_pad, dt,
        gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
        spatial_dim=3, inv_dx=inv_dx, inv_dy=inv_dx, inv_dz=inv_dx,
        boundary=0,      # periodic
        order=order,
    )
    return U + dt * dUdt


# ---------------------------------------------------------------------------
# Observable: KE-weighted RMS shock radius.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# End-to-end: simulate(params, t_end) → shock radius.
# ---------------------------------------------------------------------------

@functools.partial(
    jax.jit,
    static_argnames=("Nx", "n_steps", "order", "limiter", "riemann_solver",
                     "gamma_static"),
)
def simulate_state(params, t_end, *, Nx=32, L=1.0, n_steps=200,
                   order=2, limiter=1, riemann_solver=1,
                   gamma_static=5.0/3.0):
    """Run Sedov for fixed n_steps with dt = t_end / n_steps. Returns the
    final conservative state. The validator builds the observable matrix
    on top of this single forward pass via `jax.jacfwd`.

    Fixed dt is intentional — adaptive CFL would inject a non-smooth
    dependence on the state into the time grid, contaminating AD vs FD.
    """
    gamma = gamma_static
    E_inj = params[0]
    rho_bg = params[1]
    inv_dx = jnp.asarray(Nx / L, dtype=jnp.float64)

    U0 = build_ic(E_inj, rho_bg, Nx, L, gamma)
    dt = t_end / n_steps

    def body(U, _):
        U = _step(U, dt, gamma=gamma, limiter=limiter,
                  riemann_solver=riemann_solver, order=order, inv_dx=inv_dx)
        return U, None

    U_final, _ = jax.lax.scan(body, U0, None, length=n_steps)
    return U_final


def simulate(params, t_end, **kw):
    """Backward-compat: return shock_radius only."""
    U = simulate_state(params, t_end, **kw)
    Nx = kw.get("Nx", 32)
    L = kw.get("L", 1.0)
    return shock_radius(U, Nx, L)


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
