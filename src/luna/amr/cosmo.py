"""Cosmological (super-comoving, Martel & Shapiro 1998) hooks for the
block-AMR runner: conformal time / a(tau) lookup with expansion cap, the
a-dependent Poisson coefficient (lap(phi~) = 1.5 Omega_m H0^2 a delta), and
the per-level floor stack. The block hydro advance is reused unchanged.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def make_cosmo_amr(cfg, acfg, rho_bar, *, is_main_process=True):
    """Build the AMR runner's cosmology kit (or raise for unsupported knobs):
    a dict with ``a_tab``/``tau_tab`` lookup arrays, ``a_start``/``a_end``,
    ``dtau_cap(a)``, ``coeff(a)`` (Poisson coefficient for a (rho - <rho>)
    source), and ``floors(levels, a) -> levels``."""
    from luna.cosmology import Cosmology

    Om = float(cfg.cosmo_Omega_m)
    OL = float(cfg.cosmo_Omega_L)
    H0 = float(cfg.cosmo_H0)
    a_start = float(cfg.a_init)
    a_end = float(cfg.a_end)
    dloga_max = float(cfg.dloga_max)
    gamma = float(cfg.gamma)
    if abs(gamma - 5.0 / 3.0) > 1e-12:
        raise ValueError(f"comoving AMR runs need gamma = 5/3 (got {gamma}): the (3 gamma - 5) energy term "
                         "(Martel & Shapiro 1998 eq. 41) is in the uniform-grid loop only (sources/comoving_energy.py)")
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

    # ---- floor stack: the one in luna/sources/cosmo_floors.py, per level ----------
    from luna.sources.cosmo_floors import floor_core, jeans_coeff, temp_coeff
    knobs = dict(gamma=gamma, rho_floor=float(cfg.cosmo_rho_floor) * rho_bar,
                 floor_buffer=float(cfg.cosmo_floor_buffer), v_max=float(cfg.cosmo_vmax),
                 T_max=float(cfg.cosmo_Tmax), p_floor=float(cfg.cosmo_p_floor))
    T_floor, T_max, jeans_N, mu = (float(cfg.cosmo_T_floor), float(cfg.cosmo_Tmax),
                                   float(cfg.cosmo_jeans_ncell), float(cfg.cosmo_mu))
    rho_floor, v_max = knobs["rho_floor"], knobs["v_max"]

    def _floor_level(U, aa, dx):
        """One level's floor pass on (n_slots, 2+ndim, *bs) conservative blocks.
        NaN sentinel slots pass through unchanged (NaN comparisons are False and
        max/min with NaN stays NaN)."""
        rho, mom, E = floor_core(
            U[:, 0:1], U[:, 1:1 + ndim], U[:, ndim + 1:ndim + 2],
            temp_coeff(aa, T_floor, mu), temp_coeff(aa, T_max, mu),
            jeans_coeff(aa, coeff_fac, jeans_N, dx, rho_bar), sum_axis=1, **knobs)
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

    def floor_level(data, aa, level):
        """Per-level floor stack (the subcycling hook floors one level per
        sub-step, not the whole hierarchy at once)."""
        return _floor_level(data, aa, acfg.level_dx(level)[0])

    return dict(a_tab=a_tab, tau_tab=tau_tab, a_start=a_start, a_end=a_end,
                dtau_cap=dtau_cap, coeff=coeff, floors=floors,
                floor_level=floor_level)
