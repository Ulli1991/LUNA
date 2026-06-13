"""Differentiable CC85 starburst wind: infer feedback loadings from wind
structure (the "useful physics" autodiff example, #43).

Physics inverse problem: given a galactic wind's radial (n, v_r, P) profile,
recover the mass-loading η_M and energy-loading η_E of the central driving.
These are the actual feedback parameters one wants to constrain from observed
CGM / outflow structure (cf. CGOLS, Schneider+ 2018) — not a recover-the-field
demo. The result of interest is the η_M–η_E SENSITIVITY/DEGENERACY: η_M sets
the wind mass flux (density), η_E sets its energy (temperature/velocity), so
the Jacobian of the observables tells us which measurements break the
degeneracy.

The CC85 source deposits ρ̇ = Ṁ/V and ė = Ė/V uniformly in the R_* sphere,
with Ṁ = η_M·SFR and Ė = η_E·(SFR/100)·1e51 — a smooth, linear function of
(η_M, η_E), hence cleanly differentiable.

FULL BACKWARD STACK, all coupled and differentiated together in LUNA's
canonical Arepo unit system (units.UnitSystem; kpc, 1e10 Msun, km/s):
  * hydro (MUSCL-Hancock + HLLC),
  * radiative cooling — explicit forward-Euler sub-cycling on the Sutherland-
    Dopita curve with a 10^4 K floor (Schneider+2018/Cholla method), smooth
    C^2 cubic-spline Lambda + softplus floor so gradients are kink-free,
  * static external potential — Miyamoto-Nagai disk + NFW halo,
  * self-gravity — the differentiable FMG Poisson (custom-vjp adjoint, #42).

Reverse-mode gradients through the whole stack are FD-clean to ~1e-7. Use
jax.grad / jax.jacrev (NOT jacfwd: the self-gravity custom_vjp has no jvp).

Env: LUNA_M82_NX (48), LUNA_M82_NSTEPS (200), LUNA_M82_{COOLING,GRAVITY,SELFGRAV}
(all default ON).
"""
from __future__ import annotations

import os, sys, time
import functools
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")
jax.config.update("jax_enable_x64", True)

from hydro.muscl_hancock import muscl_hancock_divergence
from hydro.conversions import cons_to_prim
from hydro.cooling import (sutherland_dopita_cie,
                           PROTON_MASS_CGS, BOLTZMANN_CGS)
from hydro.external_gravity import combined_acceleration
from gravity.poisson_diff import solve_poisson_fmg_diff
from scripts.autodiff.fd_grad import central_diff
from units import UnitSystem

GAMMA = 5.0 / 3.0

# Use LUNA's canonical Arepo unit system (units.UnitSystem) — the same one the
# M82/CC85 production runs build — instead of hand-rolling constants. All
# derived units + G_CODE come from the 3 base units.
KPC = 3.08568e21
MSUN_G = 1.98892e33
SEC_PER_YR = 3.1557e7
_U = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN_G,
                UnitVelocity_in_cm_per_s=1.0e5)
UNIT_L = _U.UnitLength_in_cm
UNIT_M = _U.UnitMass_in_g
UNIT_V = _U.UnitVelocity_in_cm_per_s
UNIT_T = _U.UnitTime_in_s
UNIT_RHO = _U.UnitDensity_in_cgs
UNIT_P = _U.UnitPressure_in_cgs
UNIT_E = _U.UnitEnergy_in_cgs
G_CODE = _U.GravityConstant            # ~43007, matches external_gravity
MU, X_H, NE_PER_NH = 0.6, 0.7154, 1.189


def _natural_cubic_moments(x, y):
    """Second-derivative moments for a natural cubic spline through (x, y)
    (x ascending). Computed once in numpy; the spline eval in jax is then C^2
    -> smooth gradients (unlike jnp.interp, which is C^0 / kinked at nodes
    and corrupts AD-vs-FD the same way the slope limiters did in #37)."""
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n)); rhs = np.zeros(n)
    A[0, 0] = 1.0; A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]; A[i, i] = 2.0 * (h[i - 1] + h[i]); A[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return np.linalg.solve(A, rhs)


def make_cooling(n_sub=16, T_floor=1.0e4):
    """Forward-Euler sub-cycled radiative cooling, matching Schneider,
    Robertson & Thompson (2018, Cholla/CGOLS): operator-split, explicit
    forward Euler with sub-cycling, a 10^4 K temperature floor, all gas
    ionized (mu=0.6). Rate L = n_e n_H Lambda(T) with the fixed n_e n_H
    normalization.

    Deliberately NOT Townsend exact integration: Townsend's piecewise
    inversion (searchsorted + power-law root + clamps) is a gradient
    minefield. Forward Euler with a smooth log-log Lambda interpolation and
    a maximum() floor is differentiable by construction (no inversion, no
    inf-producing clamp), which is what makes the wind gradients clean."""
    table = sutherland_dopita_cie("solar")
    xt = np.asarray(table.log_T)
    yt = np.asarray(table.log_Lambda)
    mm = _natural_cubic_moments(xt, yt)
    xt_j, yt_j, m_j = jnp.asarray(xt), jnp.asarray(yt), jnp.asarray(mm)

    def Lambda(T):
        # C^2 natural cubic spline of log10(Lambda) vs log10(T) -> smooth grad
        x = jnp.clip(jnp.log10(T), xt_j[0], xt_j[-1])
        k = jnp.clip(jnp.searchsorted(xt_j, x) - 1, 0, xt_j.shape[0] - 2)
        h = xt_j[k + 1] - xt_j[k]
        a = (xt_j[k + 1] - x) / h
        b = (x - xt_j[k]) / h
        y = (a * yt_j[k] + b * yt_j[k + 1]
             + ((a ** 3 - a) * m_j[k] + (b ** 3 - b) * m_j[k + 1]) * (h * h) / 6.0)
        return 10.0 ** y

    def cool(U, dt_code):
        rho = U[0]
        ke = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho
        P0 = (GAMMA - 1.0) * (U[4] - ke)
        rho_cgs = rho * UNIT_RHO
        n_tot = rho_cgs / (MU * PROTON_MASS_CGS)
        n_H = X_H * rho_cgs / PROTON_MASS_CGS
        n_e = NE_PER_NH * n_H
        coef = (GAMMA - 1.0) * n_e * n_H / (n_tot * BOLTZMANN_CGS)   # dT/dt = -coef*Lambda(T)
        dt_sub = (dt_code * UNIT_T) / n_sub
        T0 = MU * PROTON_MASS_CGS * jnp.clip(P0, 1e-30) * UNIT_P / (BOLTZMANN_CGS * rho_cgs)

        # Soft (softplus) temperature floor instead of a hard maximum(): the
        # hard floor is a kink, and as cells cross it under a parameter change
        # the radial-profile loss becomes non-smooth, so reverse-mode AD (a
        # subgradient) disagrees with finite differences (a secant across the
        # kink) — exactly what corrupted the loading recovery. softplus is C^∞,
        # so AD == FD and the inverse problem is well-posed. s<<T_floor keeps
        # the floor sharp (offset s*ln2 ~ 1% of T_floor).
        s = 0.02 * T_floor

        def sub(T, _):
            T_raw = T - coef * Lambda(T) * dt_sub
            return T_floor + s * jax.nn.softplus((T_raw - T_floor) / s), None

        T_final, _ = jax.lax.scan(sub, T0, None, length=n_sub)
        u_cgs = BOLTZMANN_CGS * rho_cgs * T_final / ((GAMMA - 1.0) * MU * PROTON_MASS_CGS)
        P_new = (GAMMA - 1.0) * u_cgs / UNIT_P
        return U.at[4].set(P_new / (GAMMA - 1.0) + ke)

    return cool


# ---------------------------------------------------------------------------
# Compact forward wind model (code units; box centered on the injection)
# ---------------------------------------------------------------------------

def ambient_ic(Nx, L, rho_amb, p_amb):
    """Uniform hot, tenuous CGM ambient at rest."""
    rho = jnp.full((Nx, Nx, Nx), rho_amb, dtype=jnp.float64)
    z = jnp.zeros_like(rho)
    E = jnp.full_like(rho, p_amb / (GAMMA - 1.0))
    return jnp.stack([rho, z, z, z, E])


def _pad_outflow(W):
    # zero-gradient (edge replicate) so the wind leaves the box freely
    return jnp.pad(W, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="edge")


def _hydro_step(U, dt, inv_dx, limiter, order):
    W = cons_to_prim(U, GAMMA)
    W_pad = _pad_outflow(W)
    dUdt = muscl_hancock_divergence(
        W_pad, dt, gamma=GAMMA, limiter=limiter, riemann_solver=1,
        spatial_dim=3, inv_dx=inv_dx, inv_dy=inv_dx, inv_dz=inv_dx,
        boundary=1, order=order)
    return U + dt * dUdt


def _make_simulate(Nx, L, n_steps, dt, sphere_mask, V_sphere,
                   limiter=0, order=1, chunk_len=10, cool_fn=None, grav=None,
                   selfgrav=None):
    """Returns simulate(theta) -> final state, with theta = (Mdot, Edot) in
    code units. Chunked + per-step remat for bounded BPTT memory (same
    pattern as ou_inverse)."""
    inv_dx = jnp.asarray(Nx / L, dtype=jnp.float64)
    assert n_steps % chunk_len == 0
    n_chunks = n_steps // chunk_len

    @jax.jit
    def simulate(theta, U0):
        Mdot, Edot = theta[0], theta[1]
        rho_dot = Mdot / V_sphere
        e_dot = Edot / V_sphere

        @jax.checkpoint
        def step(U, _):
            U = _hydro_step(U, dt, inv_dx, limiter, order)
            U = U.at[0].add(rho_dot * dt * sphere_mask)
            U = U.at[4].add(e_dot * dt * sphere_mask)
            if cool_fn is not None:
                U = cool_fn(U, dt)
            if grav is not None:
                # static external potential: dp = rho g dt, dE = (p.g) dt (work)
                gx, gy, gz = grav
                work = U[1] * gx + U[2] * gy + U[3] * gz
                U = U.at[1].add(U[0] * gx * dt)
                U = U.at[2].add(U[0] * gy * dt)
                U = U.at[3].add(U[0] * gz * dt)
                U = U.at[4].add(work * dt)
            if selfgrav is not None:
                # self-gravity via the differentiable FMG Poisson (#42): the
                # custom-vjp backward is one extra exact FMG solve, so we
                # differentiate THROUGH the gravity solve, not by unrolling it.
                dxg, fourpiG, bc = selfgrav
                phi = solve_poisson_fmg_diff(U[0], dxg, fourpiG, bc)
                sgx = -(jnp.roll(phi, -1, 0) - jnp.roll(phi, 1, 0)) / (2 * dxg)
                sgy = -(jnp.roll(phi, -1, 1) - jnp.roll(phi, 1, 1)) / (2 * dxg)
                sgz = -(jnp.roll(phi, -1, 2) - jnp.roll(phi, 1, 2)) / (2 * dxg)
                work = U[1] * sgx + U[2] * sgy + U[3] * sgz
                U = U.at[1].add(U[0] * sgx * dt)
                U = U.at[2].add(U[0] * sgy * dt)
                U = U.at[3].add(U[0] * sgz * dt)
                U = U.at[4].add(work * dt)
            return U, None

        @jax.checkpoint
        def chunk(U, _):
            U, _ = jax.lax.scan(step, U, None, length=chunk_len)
            return U, None

        U_final, _ = jax.lax.scan(chunk, U0, None, length=n_chunks)
        return U_final

    return simulate


def xray_luminosity(U, Nx, L):
    """Soft-band X-ray luminosity of the hot wind, L_X = ∫ n² Λ_X(T) dV.

    Hot-wind (T ≳ 10⁶ K) emission is free-free dominated, Λ_X ∝ √T, and the
    emission falls out of the X-ray band below ~10⁶ K. We use a smooth proxy
    Λ_X(T) = √T · σ((T−T_X)/ΔT) — differentiable everywhere (no table kink),
    sufficient to demonstrate the ∂L_X/∂loading sensitivity exactly. T ∝ P/ρ
    in code units (μ absorbed into the overall constant, which drops out of
    the logarithmic sensitivity we report)."""
    rho = U[0]
    v2 = (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / (rho ** 2)
    P = (GAMMA - 1.0) * (U[4] - 0.5 * rho * v2)
    # physical temperature [K] (Arepo units)
    T = MU * PROTON_MASS_CGS * jnp.clip(P, 1e-30) * UNIT_P / (BOLTZMANN_CGS * rho * UNIT_RHO)
    T_X, dT = 1.0e6, 3.0e5                    # soft X-ray band threshold [K]
    band = jax.nn.sigmoid((T - T_X) / dT)     # smooth low-T cutoff
    # n_e n_H ∝ rho^2; bremsstrahlung Lambda_X ∝ sqrt(T). Absolute scale drops
    # out of the logarithmic sensitivity (elasticity) we report.
    emiss = rho ** 2 * jnp.sqrt(jnp.clip(T, 1.0)) * band
    dV = (L / Nx) ** 3
    return jnp.sum(emiss) * dV


def radial_profiles(U, Nx, L, nbins=24):
    """Spherically-averaged (n, |v|, P) — the wind 'observation'."""
    c = (jnp.arange(Nx) + 0.5) * (L / Nx) - 0.5 * L
    X, Y, Z = jnp.meshgrid(c, c, c, indexing="ij")
    r = jnp.sqrt(X * X + Y * Y + Z * Z)
    rho = U[0]
    msq = U[1] ** 2 + U[2] ** 2 + U[3] ** 2
    # soften sqrt at zero ambient velocity (sqrt'(0)=inf -> NaN gradient) and
    # floor P>0 (a strong rarefaction can momentarily give E<ke -> log(neg))
    v = jnp.sqrt(msq + 1e-30) / rho
    P = jnp.maximum((GAMMA - 1.0) * (U[4] - 0.5 * msq / rho), 1e-30)
    edges = jnp.linspace(0.0, 0.5 * L, nbins + 1)
    idx = jnp.clip(jnp.digitize(r.ravel(), edges) - 1, 0, nbins - 1)
    def binmean(q):
        s = jnp.zeros(nbins).at[idx].add(q.ravel())
        n = jnp.zeros(nbins).at[idx].add(1.0)
        return s / (n + 1e-30)
    return jnp.stack([binmean(rho), binmean(v), binmean(P)])  # (3, nbins)


# ---------------------------------------------------------------------------


def main():
    Nx = int(os.environ.get("LUNA_M82_NX", "48"))
    n_steps = int(os.environ.get("LUNA_M82_NSTEPS", "200"))
    L = 4.0                                       # 4 kpc box (Arepo units)
    dx = L / Nx

    # Ambient hot CGM: n_H = 1e-3 cm^-3, T = 1e6 K
    nH_amb, T_amb = 1.0e-3, 1.0e6
    rho_amb_cgs = nH_amb * PROTON_MASS_CGS / X_H
    rho_amb = rho_amb_cgs / UNIT_RHO
    P_amb_cgs = (rho_amb_cgs / (MU * PROTON_MASS_CGS)) * BOLTZMANN_CGS * T_amb
    p_amb = P_amb_cgs / UNIT_P
    cs_amb = float(np.sqrt(GAMMA * p_amb / rho_amb))     # km/s (code)

    R_star = 0.3                                  # injection radius [kpc]
    SFR = 20.0                                    # Msun/yr (M82-like)
    dt = 0.2 * dx / (cs_amb + 2500.0)             # fixed dt (wind v_inf ~ 1400 km/s)

    # injection sphere mask + volume (code length = kpc)
    c = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * dx - 0.5 * L
    X, Y, Z = jnp.meshgrid(c, c, c, indexing="ij")
    sphere_mask = ((X * X + Y * Y + Z * Z) < R_star ** 2).astype(jnp.float64)
    V_sphere = float(jnp.sum(sphere_mask)) * dx ** 3
    print(f"M82 wind backward (Arepo units): Nx={Nx}, L={L} kpc, n_steps={n_steps}, "
          f"inj cells={int(jnp.sum(sphere_mask))}, "
          f"dt={dt*UNIT_T/SEC_PER_YR/1e6:.4f} Myr", flush=True)
    print(f"  ambient: n_H={nH_amb:.0e} cm^-3, T={T_amb:.0e} K, cs={cs_amb:.1f} km/s",
          flush=True)

    # feedback loadings (eta_M, eta_E) -> CC85 code injection rates (Schneider+2018)
    def eta_to_loadings(eta):
        Mdot_cgs = eta[0] * SFR * MSUN_G / SEC_PER_YR
        Edot_cgs = eta[1] * (SFR / 100.0) * 1.0e51 / SEC_PER_YR
        return jnp.array([Mdot_cgs / UNIT_M * UNIT_T, Edot_cgs / UNIT_E * UNIT_T])

    cooling_on = os.environ.get("LUNA_M82_COOLING", "1") == "1"
    cool_fn = make_cooling() if cooling_on else None
    print(f"  cooling: {'ON (explicit subcycled S&D93, n_e n_H)' if cooling_on else 'OFF'}",
          flush=True)

    # Static external potential: Miyamoto-Nagai stellar disk + NFW halo, in
    # Arepo units -> combined_acceleration returns code acceleration directly.
    grav_on = os.environ.get("LUNA_M82_GRAVITY", "1") == "1"
    grav = None; gmax = 0.0
    if grav_on:
        gx, gy, gz = combined_acceleration(X, Y, Z, MN_M=1.0, MN_a=0.8, MN_b=0.15,
                                           NFW_M200=50.0, NFW_c=20.0)
        grav = (gx, gy, gz)
        gmax = float(jnp.max(jnp.sqrt(gx ** 2 + gy ** 2 + gz ** 2)))
    print(f"  potential: {'ON (MN M*=1e10 + NFW M200=5e11, |g|max=%.1f code)' % gmax if grav_on else 'OFF'}",
          flush=True)

    selfgrav_on = os.environ.get("LUNA_M82_SELFGRAV", "1") == "1"
    fourpiG = 4.0 * np.pi * G_CODE
    selfgrav = (dx, fourpiG, 0) if selfgrav_on else None   # boundary 0 = periodic
    print(f"  self-gravity: {'ON (differentiable FMG Poisson, 4piG=%.3e)' % fourpiG if selfgrav_on else 'OFF'}",
          flush=True)

    U0 = ambient_ic(Nx, L, rho_amb, p_amb)
    simulate = _make_simulate(Nx, L, n_steps, dt, sphere_mask, V_sphere,
                              cool_fn=cool_fn, grav=grav, selfgrav=selfgrav)

    def sim_eta(eta):
        return simulate(eta_to_loadings(eta), U0)

    eta_true = jnp.array([0.3, 0.6])              # M82-like mass-, energy-loading
    U_obs = sim_eta(eta_true)
    prof_obs = radial_profiles(U_obs, Nx, L)
    assert bool(jnp.all(jnp.isfinite(U_obs))), "forward wind not finite"
    nH_peak = float(prof_obs[0].max()) * UNIT_RHO * X_H / PROTON_MASS_CGS
    print(f"  forward OK; peak n_H = {nH_peak:.3e} cm^-3", flush=True)

    # Loss in log-observable space (n, v, P span orders of magnitude)
    def loss(eta):
        prof = radial_profiles(sim_eta(eta), Nx, L)
        return jnp.mean((jnp.log(prof + 1e-30) - jnp.log(prof_obs + 1e-30)) ** 2)

    # --- Backward (reverse-mode) gradient validation vs central FD ---
    eta0 = jnp.array([0.15, 1.0])                 # wrong loadings
    g_ad = np.asarray(jax.grad(loss)(eta0))
    g_fd = np.asarray(central_diff(loss, eta0, h=1e-4))
    print("\n  reverse-mode gradient validation at eta0=(eta_M=0.15, eta_E=1.0):")
    for i, name in enumerate(("eta_M", "eta_E")):
        rel = abs(g_ad[i] - g_fd[i]) / (abs(g_fd[i]) + 1e-30)
        print(f"    d loss / d {name}: AD={g_ad[i]:+.6e}  FD={g_fd[i]:+.6e}  "
              f"rel={rel:.2e}", flush=True)

    # --- Recover the loadings by gradient descent (log-space Adam) ---
    log_eta = jnp.log(eta0)
    m = jnp.zeros(2); v = jnp.zeros(2)
    print("\n  recovering loadings:", flush=True)
    for it in range(100):
        g = jax.grad(lambda le: loss(jnp.exp(le)))(log_eta)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        log_eta = log_eta - 5e-2 * m / (1 - 0.9**(it+1)) / (
            jnp.sqrt(v / (1 - 0.999**(it+1))) + 1e-8)
        if it % 25 == 0 or it == 99:
            e = jnp.exp(log_eta)
            print(f"    it {it:3d}  L={float(loss(e)):.3e}  "
                  f"eta_M={float(e[0]):.3f} (true 0.3)  "
                  f"eta_E={float(e[1]):.3f} (true 0.6)", flush=True)
    e = np.asarray(jnp.exp(log_eta))
    print(f"\n  recovered: eta_M={e[0]:.4f} (true 0.3, {100*abs(e[0]-0.3)/0.3:.1f}% err)  "
          f"eta_E={e[1]:.4f} (true 0.6, {100*abs(e[1]-0.6)/0.6:.1f}% err)")

    # --- Observable sensitivity (jacrev: self-gravity custom_vjp has no jvp) ---
    def obs_vec(eta):
        return radial_profiles(sim_eta(eta), Nx, L).ravel()
    J = np.asarray(jax.jacrev(obs_vec)(eta_true))
    nb = J.shape[0] // 3
    print("\n  observable sensitivity ‖∂log-prof/∂η‖ per channel:")
    for ci, nm in enumerate(("n", "v", "P")):
        blk = J[ci*nb:(ci+1)*nb]
        sM = np.linalg.norm(blk[:, 0]); sE = np.linalg.norm(blk[:, 1])
        print(f"    {nm}: ∂/∂eta_M={sM:.3e}  ∂/∂eta_E={sE:.3e}  "
              f"ratio M/E={sM/(sE+1e-30):.2f}")

    # --- FLAGSHIP: X-ray luminosity feedback sensitivity (elasticities) ---
    def lnLX(eta):
        return jnp.log(xray_luminosity(sim_eta(eta), Nx, L))
    LX0 = float(jnp.exp(lnLX(eta_true)))
    g_lnLX = np.asarray(jax.grad(lambda le: lnLX(jnp.exp(le)))(jnp.log(eta_true)))
    g_fd = np.asarray(central_diff(lambda le: lnLX(jnp.exp(le)),
                                   jnp.log(eta_true), h=1e-4))
    print(f"\n  === FLAGSHIP: X-ray luminosity feedback sensitivity ===")
    print(f"  L_X(true loadings) = {LX0:.4e} (code emiss units)")
    print(f"  ∂lnL_X/∂lnη_M = {g_lnLX[0]:+.4f}  "
          f"[FD {g_fd[0]:+.4f}, rel {abs(g_lnLX[0]-g_fd[0])/(abs(g_fd[0])+1e-30):.1e}]")
    print(f"  ∂lnL_X/∂lnη_E = {g_lnLX[1]:+.4f}  "
          f"[FD {g_fd[1]:+.4f}, rel {abs(g_lnLX[1]-g_fd[1])/(abs(g_fd[1])+1e-30):.1e}]")
    print(f"  → L_X is {'more' if abs(g_lnLX[1])>abs(g_lnLX[0]) else 'less'} "
          f"sensitive to energy than mass loading "
          f"(|E/M| = {abs(g_lnLX[1])/(abs(g_lnLX[0])+1e-30):.2f})")


if __name__ == "__main__":
    main()
