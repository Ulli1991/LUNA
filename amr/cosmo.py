"""Cosmological (super-comoving) hooks for the block-AMR runner. Gas-only
Phase 1 of the cosmo-AMR port.

Super-comoving variables (Martel & Shapiro 1998) make the gamma=5/3 Euler
equations form-invariant, so the block hydro advance is reused UNCHANGED.
Cosmology enters through exactly three hooks, all ported cell-faithfully from
the dense driver (setup/runner_cosmo_hydro):

  1. conformal time: dtau = dt/a^2, a(tau) from a precomputed lookup table,
     with the expansion step cap  dtau <= dloga_max / (a^2 H(a));
  2. gravity: lap(phi~) = 1.5 Omega_m H0^2 a * delta  -- fed to the AMR
     level-by-level Dirichlet MG as an a-dependent (traced) coefficient;
  3. the floor stack: pressure/temperature floor, Jeans-Truelove floor,
     buffered density floor with momentum zeroing, velocity ceiling and
     temperature ceiling -- applied per level (the Jeans floor uses each
     level's own dx).

Not ported yet (asserted off): Bryan+1995 dual-energy. DM particles are
Phase 2 (RAMSES-style: particles never split; CIC deposit at the finest
covering level, refinement on mass-per-cell).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def make_cosmo_amr(cfg, acfg, rho_bar, *, is_main_process=True):
    """Build the AMR runner's cosmology kit (or raise for unsupported knobs).

    Returns a dict:
      a_tab, tau_tab : jnp lookup arrays, a = interp(tau)
      a_start, a_end : floats
      dtau_cap(a)    : traced expansion cap  dloga_max / (a^2 H(a))
      coeff(a)       : traced Poisson coefficient for a (rho - <rho>) source
      floors(levels, a) -> levels : the full floor stack, per level
    """
    from cosmology import Cosmology

    if getattr(cfg, "DualEnergy", False):
        raise NotImplementedError("cosmo-AMR Phase 1 does not port dual-energy yet")

    Om = float(getattr(cfg, "cosmo_Omega_m", 1.0))
    OL = float(getattr(cfg, "cosmo_Omega_L", 0.0))
    H0 = float(getattr(cfg, "cosmo_H0", 1.0))
    a_start = float(getattr(cfg, "a_init", getattr(cfg, "a_start", 0.02)))
    a_end = float(getattr(cfg, "a_end", 1.0))
    dloga_max = float(getattr(cfg, "dloga_max", 0.01))
    gamma = float(cfg.gamma)
    ndim = acfg.ndim

    cos = Cosmology(Omega_m=Om, Omega_L=OL)
    a_tab_np, tau_tab_np = cos.build_a_of_tau(a_start, a_end + 0.05, H0)
    a_tab = jnp.asarray(np.asarray(a_tab_np))
    tau_tab = jnp.asarray(np.asarray(tau_tab_np))

    Ok = 1.0 - Om - OL

    def E_jnp(a):
        """H(a)/H0, jnp (the Cosmology class version is numpy-only)."""
        return jnp.sqrt(Om * a ** -3 + Ok * a ** -2 + OL)

    def dtau_cap(a):
        return dloga_max / (a * a * H0 * E_jnp(a))

    coeff_fac = 1.5 * Om * H0 * H0                 # lap(phi~) = coeff_fac * a * delta

    def coeff(a):
        # solve_phi_hierarchy multiplies (rho - <rho>); delta = (rho - <rho>)/rho_bar
        return coeff_fac * a / rho_bar

    # ---- floor stack (ported from runner_cosmo_hydro, block/ndim-generic) ----
    p_floor0 = float(getattr(cfg, "cosmo_p_floor", 0.0))
    T_floor = float(getattr(cfg, "cosmo_T_floor", 0.0))
    T_max = float(getattr(cfg, "cosmo_Tmax", 0.0))
    jeans_N = float(getattr(cfg, "cosmo_jeans_ncell", 0.0))
    mu = float(getattr(cfg, "cosmo_mu", 1.22))
    v_max = float(getattr(cfg, "cosmo_vmax", 0.0))
    rho_floor = float(getattr(cfg, "cosmo_rho_floor", 0.0)) * rho_bar
    floor_buffer = float(getattr(cfg, "cosmo_floor_buffer", 1.0))

    def _floor_level(U, aa, dx):
        """One level's floor pass on (n_slots, 2+ndim, *bs) conservative blocks.
        NaN sentinel slots pass through unchanged (NaN comparisons are False and
        max/min with NaN stays NaN)."""
        rho = U[:, 0:1]
        mom = U[:, 1:1 + ndim]
        E = U[:, ndim + 1:ndim + 2]

        # (1) buffered density floor: clamp rho AND zero momentum (v = m/rho
        #     blows up as rho -> floor); reset e_int to the T-floor value.
        if rho_floor > 0.0:
            bad = rho < rho_floor * floor_buffer
            ct0 = aa * aa * 8.2543e-3 * T_floor / mu if T_floor > 0.0 else 0.0
            eint_fl = jnp.maximum(ct0 * rho_floor / (gamma - 1.0), 1e-300)
            rho = jnp.where(bad, rho_floor, rho)
            mom = jnp.where(bad, 0.0, mom)
            E = jnp.where(bad, eint_fl, E)

        ke = 0.5 * jnp.sum(mom * mom, axis=1, keepdims=True) / rho
        eint = E - ke

        # (2) velocity ceiling (preserving e_int): the near-floor dt killer.
        if v_max > 0.0:
            v2 = jnp.sum(mom * mom, axis=1, keepdims=True) / (rho * rho)
            scl = jnp.where(v2 > v_max * v_max, v_max / jnp.sqrt(v2 + 1e-300), 1.0)
            mom = mom * scl
            ke = 0.5 * jnp.sum(mom * mom, axis=1, keepdims=True) / rho

        # (3) temperature ceiling on e_int (spurious hot diffuse cells).
        if T_max > 0.0:
            ctmax = aa * aa * 8.2543e-3 * T_max / mu
            eint = jnp.minimum(eint, ctmax * rho / (gamma - 1.0))

        # (4) pressure floor: P >= max(p_floor, ct*rho [T_floor], cj*rho^2 [Jeans]).
        P = (gamma - 1.0) * eint
        ct = aa * aa * 8.2543e-3 * T_floor / mu if T_floor > 0.0 else 0.0
        cj = (coeff_fac * aa * (jeans_N * dx) ** 2 / (4.0 * np.pi ** 2 * rho_bar)
              if jeans_N > 0.0 else 0.0)
        Pmin = jnp.maximum(jnp.maximum(p_floor0, ct * rho), cj * rho * rho)
        eint = jnp.maximum(P, Pmin) / (gamma - 1.0)

        E = ke + jnp.maximum(eint, 1e-300)
        return jnp.concatenate([rho, mom, E], axis=1)

    def floors(levels, aa):
        out = []
        for L, lvl in enumerate(levels):
            dx = acfg.level_dx(L)[0]
            out.append(lvl._replace(data=_floor_level(lvl.data, aa, dx)))
        return tuple(out)

    if is_main_process:
        z0 = 1.0 / a_start - 1.0
        print(f"  cosmo-AMR (gas-only): Om={Om} OL={OL} H0={H0}  a: {a_start:.4f} "
              f"(z={z0:.1f}) -> {a_end:.4f}; dloga_max={dloga_max}; "
              f"floors: rho={rho_floor:.3e} T={T_floor} Tmax={T_max} "
              f"NJ={jeans_N} vmax={v_max}")

    return dict(a_tab=a_tab, tau_tab=tau_tab, a_start=a_start, a_end=a_end,
                dtau_cap=dtau_cap, coeff=coeff, floors=floors)
