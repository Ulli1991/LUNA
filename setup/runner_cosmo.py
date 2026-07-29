"""Comoving (super-comoving) integration runner.

Dispatched from main.run when cfg.CosmologicalIntegration is True. RAMSES-style
super-comoving scheme (Martel & Shapiro 1998):

  * the hydro is the UNCHANGED MUSCL-Hancock flux update (step_fn) acting on the
    super-comoving variables rho~ = a^3 rho, v~ = a v_pec, P~ = a^5 P;
  * gravity is the SAME operator-split source term LUNA already uses -- the
    -rho*grad(Phi) update to each fixed cell's momentum (and its work to the
    energy), applied once AFTER the hydro flux update (not a leapfrog) -- but
    with the Poisson coefficient (3/2) Omega_m a in place of 4 pi G;
  * the loop integrates super-comoving time tau (dtau = dt/a^2), capping the
    per-step expansion at dloga_max and advancing the scale factor through the
    cosmology a(tau) table (the RAMSES 'friedman' integration);
  * an optional pressure (temperature) floor cosmo_p_floor keeps the cold,
    highly supersonic gas positive -- the total-energy scheme otherwise loses
    the tiny internal energy to truncation when KE >> e_int (standard in
    cosmological grid codes).

The source-term update mirrors sources/self_gravity.py::_kick_body exactly, so
the comoving path differs from the Newtonian self-gravity ONLY in the Poisson
coefficient, the tau-stepping, and the floor. Nothing moves on the fixed mesh;
only the conserved fields stored in each cell are updated.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from cosmology import Cosmology
from gravity.poisson_bc import make_halo_3d
from gravity.poisson_multigrid import solve_poisson_fmg
from setup.cosmo_common import CosmoClock


def run_simulation_cosmo(U, cfg, step_fn, compute_dt_fn,
                         is_main_process=True):
    """Gas-only super-comoving time loop (see the module docstring for the
    scheme). Returns ``(U, a, step)``; `main.run` wraps it in a RunResult."""
    cosmo = Cosmology(Omega_m=getattr(cfg, "cosmo_Omega_m", 1.0),
                      Omega_L=getattr(cfg, "cosmo_Omega_L", 0.0))
    H0 = float(getattr(cfg, "cosmo_H0", 1.0))
    a = float(getattr(cfg, "a_init", 0.3))
    a_end = float(getattr(cfg, "a_end", 1.0))
    dloga_max = float(getattr(cfg, "dloga_max", 0.01))
    boundary = int(cfg.BOUNDARY)
    gamma = float(cfg.gamma)
    dx = cfg.Lx / cfg.Nx
    inv_2dx = 0.5 / dx
    halo_fn = make_halo_3d(boundary)
    n_extra = int(getattr(cfg, "POISSON_FMG_EXTRA_VCYCLES", 2))
    n_levels = getattr(cfg, "POISSON_N_LEVELS", None)
    p_floor = float(getattr(cfg, "cosmo_p_floor", 0.0))   # pressure/temperature floor

    # a(tau) lookup table (RAMSES 'friedman': integrate da/dtau = a^3 H)
    clock = CosmoClock(cosmo, a, a_end, H0)

    @jax.jit
    def apply_gravity_source(U, fourpiG, dt):
        """Operator-split self-gravity source -rho*grad(Phi) on each fixed cell's
        momentum (+ time-centred energy work), mirroring _kick_body with the
        comoving coefficient fourpiG = (3/2) Omega_m a."""
        rho = U[0]
        rho_src = rho - jnp.mean(rho) if boundary == 0 else rho  # Jeans swindle
        phi = solve_poisson_fmg(rho_src, dx, fourpiG, boundary,
                                n_levels=n_levels, n_extra_vcycles=n_extra)
        phi_pad = halo_fn(phi)
        gx = -(phi_pad[2:, 1:-1, 1:-1] - phi_pad[:-2, 1:-1, 1:-1]) * inv_2dx
        gy = -(phi_pad[1:-1, 2:, 1:-1] - phi_pad[1:-1, :-2, 1:-1]) * inv_2dx
        gz = -(phi_pad[1:-1, 1:-1, 2:] - phi_pad[1:-1, 1:-1, :-2]) * inv_2dx
        mx, my, mz = U[1], U[2], U[3]
        nmx = mx + dt * rho * gx
        nmy = my + dt * rho * gy
        nmz = mz + dt * rho * gz
        E = U[-1] + dt * (0.5 * (mx + nmx) * gx
                          + 0.5 * (my + nmy) * gy
                          + 0.5 * (mz + nmz) * gz)
        return jnp.stack([rho, nmx, nmy, nmz, E])

    @jax.jit
    def floor_pressure(U):
        """Keep P >= p_floor by raising the internal energy (temperature floor)."""
        rho = U[0]
        ke = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho
        P = (gamma - 1.0) * (U[-1] - ke)
        return U.at[-1].set(jnp.maximum(P, p_floor) / (gamma - 1.0) + ke)

    tau, step, t0 = 0.0, 0, time.time()
    if is_main_process:
        print(f"\n[cosmo] super-comoving run  a_init={a:.4f} (z={1/a-1:.3f}) "
              f"-> a_end={a_end:.4f}  Omega_m={cosmo.Omega_m} Omega_L={cosmo.Omega_L} "
              f"p_floor={p_floor:.2e}")
        print(f"[cosmo] {'step':>6} {'a':>9} {'z':>9} {'dloga':>9} {'rho~max':>10}")

    # Gravity-kick time centering (measured on the Zel'dovich pancake,
    # scaling/repro/zel_probe.py): the legacy end-of-step full kick is 1st
    # order in dtau and dominated the velocity error at production dloga;
    # Strang makes the time error negligible (velocity L1 = pure spatial floor).
    #   "strang" -- half-kick @a, hydro, half-kick @a_next (default; 2 FMG/step)
    #   "half"   -- hydro, then one full kick with the coeff at the MIDPOINT
    #   "end"    -- hydro, then one full kick with the coeff at a_next (legacy)
    kick_scheme = str(getattr(cfg, "cosmo_kick_scheme", "strang"))

    while a < a_end - 1e-12:
        dtau = float(compute_dt_fn(U))                          # CFL on (rho~,v~,P~)
        dtau = clock.cap_expansion(dtau, a, dloga_max)
        dtau = clock.cap_to_target(dtau, tau, a_end)            # land exactly on a_end
        a_next = clock.a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        if kick_scheme == "strang":
            U = apply_gravity_source(U, cosmo.poisson_coeff(a), 0.5 * dtau)
            U = step_fn(U, dtau)
            U = apply_gravity_source(U, cosmo.poisson_coeff(a_next), 0.5 * dtau)
        else:
            U = step_fn(U, dtau)
            a_kick = (clock.a_of_tau(tau + 0.5 * dtau)
                      if kick_scheme == "half" else a_next)
            U = apply_gravity_source(U, cosmo.poisson_coeff(a_kick), dtau)
        if p_floor > 0.0:
            U = floor_pressure(U)
        tau += dtau
        a = a_next
        step += 1
        if is_main_process and (step % 25 == 0 or a >= a_end - 1e-12):
            rho = U[0]
            v2 = (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho ** 2
            P = (gamma - 1.0) * (U[-1] - 0.5 * rho * v2)
            print(f"[cosmo] {step:6d} a={a:7.4f} z={1/a-1:7.4f} dloga={dloga:7.5f} "
                  f"rho[{float(jnp.min(rho)):.2e},{float(jnp.max(rho)):.2e}] "
                  f"Pmin={float(jnp.min(P)):.2e}")
            if not np.isfinite(float(jnp.max(rho)) + float(jnp.min(P))):
                print("[cosmo] non-finite state — stopping"); break

    if is_main_process:
        print(f"[cosmo] done: {step} steps, a={a:.4f} (z={1 / a - 1:.4f}), "
              f"{time.time() - t0:.1f}s")
    return U, a, step
