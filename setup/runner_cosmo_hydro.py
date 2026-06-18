"""Coupled gas + dark-matter cosmological runner (adiabatic).

Evolves baryonic gas (Eulerian super-comoving hydro) and collisionless dark
matter (Particle-Mesh N-body) together in an expanding box under their JOINT
self-gravity. Dispatched from main.run when cfg.CosmoHydro is True.

Both species share the RAMSES / Martel-Shapiro super-comoving frame, which is
what makes the coupling clean: the gas super-comoving velocity v~ = a v_pec and
the DM canonical momentum p = a^2 x_dot are the SAME quantity, both kicked by
the same comoving acceleration g = -grad(phi~) over the same super-comoving
time step dtau = dt/a^2.

One Poisson solve per step on the TOTAL matter contrast:

    rho_tot = rho~_gas (grid)  +  CIC-deposit(DM particles)
    delta   = rho_tot / <rho_tot> - 1
    lap(phi~) = (3/2) Omega_m H0^2 a * delta           (one FMG multigrid solve)

The resulting force drives BOTH species:
  * gas  -- the operator-split -rho*grad(phi) kick (+ time-centred energy work)
            applied after the MUSCL-Hancock flux update (as in runner_cosmo);
  * DM   -- a kick-drift-kick leapfrog (as in runner_cosmo_dm),
reusing the end-of-step force as the next step's opening kick so the step costs
a single solve. The global dtau is the min of the gas CFL, the DM acceleration
criterion, and the dloga expansion cap.

Adiabatic: no cooling, no star formation. A pressure (temperature) floor keeps
the cold, highly supersonic gas positive, as in the gas-only comoving runner.
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from cosmology import Cosmology
from gravity.particle_mesh import ParticleMeshGravity
from gravity.poisson_multigrid import solve_poisson_fmg


def run_simulation_cosmo_hydro(U, cfg, grid, step_fn, compute_dt_fn, sharding,
                               dm_pos, dm_p, dm_mass, is_main_process=True):
    """Couple the gas state U (super-comoving conserved grid) with DM particles
    (comoving positions dm_pos, canonical momenta dm_p == v~, masses dm_mass).
    Returns (U, dm_pos, dm_p, a, step)."""
    cosmo = Cosmology(Omega_m=getattr(cfg, "cosmo_Omega_m", 1.0),
                      Omega_L=getattr(cfg, "cosmo_Omega_L", 0.0))
    H0 = float(getattr(cfg, "cosmo_H0", 1.0))
    Om = float(getattr(cfg, "cosmo_Omega_m", 1.0))
    a = float(getattr(cfg, "a_init", 0.3))
    a_end = float(getattr(cfg, "a_end", 1.0))
    dloga_max = float(getattr(cfg, "dloga_max", 0.01))
    gamma = float(cfg.gamma)
    box = float(cfg.Lx)
    p_floor = float(getattr(cfg, "cosmo_p_floor", 0.0))
    eta = float(getattr(cfg, "dm_eta", 0.025))
    n_extra = int(getattr(cfg, "POISSON_FMG_EXTRA_VCYCLES", 2))
    n_levels = getattr(cfg, "POISSON_N_LEVELS", None)
    jeans_N = float(getattr(cfg, "cosmo_jeans_ncell", 0.0))   # Truelove cells (0 = off)
    T_floor = float(getattr(cfg, "cosmo_T_floor", 0.0))       # temperature floor [K] (0 = off)
    mu = float(getattr(cfg, "cosmo_mu", 1.22))
    dual_energy = bool(getattr(cfg, "DualEnergy", False))     # Bryan+1995 dual-energy
    dual_eta = float(getattr(cfg, "cosmo_dual_eta", 1.0e-3))

    pm = ParticleMeshGravity(cfg)
    dx = pm.dx
    eps_soft = float(getattr(cfg, "dm_softening", 0.0)) or dx
    # comoving Poisson coefficient (3/2) Omega_m H0^2 a; for the dimensionless
    # pancake units (H0 = 1) this reduces to (3/2) Omega_m a.
    coeff_fac = 1.5 * Om * H0 * H0
    rho_bar = float(jnp.mean(U[0]))                          # mean comoving gas density

    dm_pos = jnp.asarray(dm_pos)
    dm_p = jnp.asarray(dm_p)
    dm_mass = jnp.asarray(dm_mass)
    active = jnp.ones(dm_pos.shape[0])

    a_tab, tau_tab = cosmo.build_a_of_tau(a, a_end + 0.05, H0)

    def a_of_tau(t):
        return float(np.interp(t, tau_tab, a_tab))

    @jax.jit
    def forces(U_arr, pos, coeff_val):
        """One joint solve -> (gas force grid, DM force at particles, delta_max,
        gas mass, dm mass)."""
        rho_dm = pm.cic_deposit(pos, dm_mass, active)
        rho_tot = U_arr[0] + rho_dm
        delta = rho_tot / jnp.mean(rho_tot) - 1.0
        phi = solve_poisson_fmg(delta, dx, coeff_val, boundary=0,
                                n_levels=n_levels, n_extra_vcycles=n_extra)
        g_gas = pm.grad_phi(phi, order=2)                 # -grad(phi) on the grid
        g_dm = pm.accel_at_points(pm.grad_phi(phi, order=4), pos)
        return g_gas, g_dm, jnp.max(delta)

    @jax.jit
    def gas_kick(U_arr, g_gas, dt):
        """Operator-split gravity source on the gas (momentum + time-centred
        energy work), with the JOINT force g_gas = -grad(phi)."""
        rho = U_arr[0]
        gx, gy, gz = g_gas
        mx, my, mz = U_arr[1], U_arr[2], U_arr[3]
        nmx = mx + dt * rho * gx
        nmy = my + dt * rho * gy
        nmz = mz + dt * rho * gz
        E = U_arr[-1] + dt * (0.5 * (mx + nmx) * gx
                              + 0.5 * (my + nmy) * gy
                              + 0.5 * (mz + nmz) * gz)
        return jnp.stack([rho, nmx, nmy, nmz, E])

    @jax.jit
    def apply_floor(U_arr, cj, ct):
        """Raise P~ to max(p_floor, ct*rho~, cj*rho~^2). The ct term is the
        TEMPERATURE floor (T >= T_floor) -- the guard against the cold-supersonic
        infall instability where the total-energy scheme loses e_int to
        truncation; the cj term is the Jeans-Truelove grid-fragmentation floor."""
        rho = U_arr[0]
        ke = 0.5 * (U_arr[1] ** 2 + U_arr[2] ** 2 + U_arr[3] ** 2) / rho
        P = (gamma - 1.0) * (U_arr[-1] - ke)
        Pmin = jnp.maximum(jnp.maximum(p_floor, ct * rho), cj * rho * rho)
        return U_arr.at[-1].set(jnp.maximum(P, Pmin) / (gamma - 1.0) + ke)

    def jeans_coeff(aa):
        # P~ >= coeff(a) (N_J dx)^2 rho~^2 / (4 pi^2 <rho~>);  4 pi G_eff = coeff/<rho~>
        if jeans_N <= 0.0:
            return 0.0
        return coeff_fac * aa * (jeans_N * dx) ** 2 / (4.0 * np.pi ** 2 * rho_bar)

    def temp_coeff(aa):
        # P~ >= a^2 (kB/m_p) T_floor/mu * rho~   (super-comoving temperature floor)
        if T_floor <= 0.0:
            return 0.0
        return aa * aa * 8.2543e-3 * T_floor / mu

    # --- Bryan+1995 dual-energy (operator-split, super-comoving) -------------
    inv_dx = 1.0 / dx
    inv_2dx = 0.5 / dx

    def _upwind_div_sv(s, vx, vy, vz):
        """div(s v), first-order upwind, periodic (jnp.roll)."""
        out = jnp.zeros_like(s)
        for v, ax in ((vx, 0), (vy, 1), (vz, 2)):
            vf_p = 0.5 * (v + jnp.roll(v, -1, ax))      # face i+1/2
            vf_m = 0.5 * (v + jnp.roll(v, 1, ax))       # face i-1/2
            Fp = vf_p * jnp.where(vf_p > 0, s, jnp.roll(s, -1, ax))
            Fm = vf_m * jnp.where(vf_m > 0, jnp.roll(s, 1, ax), s)
            out = out + (Fp - Fm) * inv_dx
        return out

    @jax.jit
    def dual_energy_update(U_arr, ie_arr, dt, ct, cj):
        """Advect the internal-energy density ie = rho~ e~ (upwind) + the -P~ div(v)
        work, apply the dual-energy switch, and fold in the floors. Where E-KE is
        a reliable fraction of E the internal energy is taken from it (captures
        shock heating); where it is not (cold, KE-dominated) it is taken from the
        advected ie. E is then rebuilt as KE + max(internal, floor), which only
        ever RAISES E -- so total energy is preserved in the shock-resolving
        regions and pressure can never go negative in the cold infall."""
        rho = U_arr[0]
        vx, vy, vz = U_arr[1] / rho, U_arr[2] / rho, U_arr[3] / rho
        ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
        eint_E = U_arr[-1] - ke
        divv = ((jnp.roll(vx, -1, 0) - jnp.roll(vx, 1, 0))
                + (jnp.roll(vy, -1, 1) - jnp.roll(vy, 1, 1))
                + (jnp.roll(vz, -1, 2) - jnp.roll(vz, 1, 2))) * inv_2dx
        P = (gamma - 1.0) * jnp.maximum(eint_E, 0.0)
        ie_adv = ie_arr - dt * _upwind_div_sv(ie_arr, vx, vy, vz) - dt * P * divv
        reliable = eint_E > dual_eta * U_arr[-1]
        ie_sel = jnp.where(reliable, eint_E, ie_adv)
        ie_floor = jnp.maximum(jnp.maximum(p_floor, ct * rho), cj * rho * rho) / (gamma - 1.0)
        ie_out = jnp.maximum(ie_sel, ie_floor)
        return U_arr.at[-1].set(ke + ie_out), ie_out

    if is_main_process:
        m_gas = float(jnp.sum(U[0])) * dx ** 3
        m_dm = float(jnp.sum(dm_mass))
        print(f"\n[cosmo-hydro] coupled gas+DM  a_init={a:.4f} (z={1/a-1:.2f}) "
              f"-> a_end={a_end:.4f}  Om={Om} OL={cfg.cosmo_Omega_L} H0={H0:.4f}")
        print(f"[cosmo-hydro] gas mass={m_gas:.4e}  DM mass={m_dm:.4e}  "
              f"f_b={m_gas/(m_gas+m_dm):.3f}  ({dm_pos.shape[0]} particles, mesh={cfg.Nx}^3)")
        print(f"[cosmo-hydro] {'step':>5} {'a':>8} {'z':>8} {'dloga':>8} "
              f"{'delta_max':>10} {'rho~gas_max':>11}")

    snap_every = int(getattr(cfg, "cosmo_snap_every", 0))
    out_dir = getattr(cfg, "output_dir", "/ptmp/uli/jax-hydro/snapshots/cosmo_hydro")

    def write_snap(a_now, U_arr, pos):
        """Periodic checkpoint: gas + DM (CIC) density grids, so a run that is
        wall-clocked before a_end still leaves a usable cosmic-web snapshot."""
        import os
        import h5py
        os.makedirs(out_dir, exist_ok=True)
        rho_dm = np.asarray(pm.cic_deposit(pos, dm_mass, active))
        fn = os.path.join(out_dir, f"coupled_a{a_now:.4f}.hdf5")
        with h5py.File(fn, "w") as f:
            f.attrs["a"] = a_now
            f.attrs["z"] = 1.0 / a_now - 1.0
            f.attrs["BoxSize"] = float(cfg.Lx)
            f.create_dataset("rho_gas", data=np.asarray(U_arr[0], dtype=np.float32))
            f.create_dataset("rho_dm", data=rho_dm.astype(np.float32))
        print(f"[cosmo-hydro] wrote {fn}", flush=True)

    # dual-energy internal-energy density, advected alongside the conserved state
    ie = (U[-1] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]) if dual_energy else None
    if is_main_process and dual_energy:
        print("[cosmo-hydro] dual-energy formalism ON "
              f"(switch eta={dual_eta:.1e})", flush=True)

    tau, step, t0 = 0.0, 0, time.time()
    g_gas, g_dm, dmax = forces(U, dm_pos, coeff_fac * a)
    while a < a_end - 1e-9:
        gmax = float(jnp.max(jnp.sqrt(jnp.sum(g_dm ** 2, axis=1))))
        dtau = min(float(compute_dt_fn(U)),                       # gas CFL
                   dloga_max / (a * a * cosmo.H(a, H0)),          # expansion cap
                   eta * np.sqrt(eps_soft / (gmax + 1e-30)))      # DM acceleration
        if tau + dtau > tau_tab[-1]:
            dtau = tau_tab[-1] - tau
        a_next = a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        # DM half-kick @a, DM drift, gas hydro flux, JOINT solve @a_next,
        # gas operator-split kick, DM closing half-kick (one solve/step).
        dm_p = dm_p + g_dm * (0.5 * dtau)
        dm_pos = jnp.mod(dm_pos + dm_p * dtau, box)
        U = step_fn(U, dtau)
        g_gas, g_dm, dmax = forces(U, dm_pos, coeff_fac * a_next)
        U = gas_kick(U, g_gas, dtau)
        if dual_energy:
            U, ie = dual_energy_update(U, ie, dtau, temp_coeff(a_next), jeans_coeff(a_next))
        elif p_floor > 0.0 or jeans_N > 0.0 or T_floor > 0.0:
            U = apply_floor(U, jeans_coeff(a_next), temp_coeff(a_next))
        dm_p = dm_p + g_dm * (0.5 * dtau)
        tau += dtau
        a = a_next
        step += 1
        if is_main_process and (step % 10 == 0 or a >= a_end - 1e-9):
            print(f"[cosmo-hydro] {step:5d} {a:8.4f} {1/a-1:8.4f} {dloga:8.5f} "
                  f"{float(dmax):10.3f} {float(jnp.max(U[0])):11.3e}", flush=True)
            if not np.isfinite(float(jnp.max(U[0]))):
                print("[cosmo-hydro] non-finite gas state -- stopping"); break
        if is_main_process and snap_every > 0 and step % snap_every == 0:
            write_snap(a, U, dm_pos)

    if is_main_process:
        m_gas = float(jnp.sum(U[0])) * dx ** 3
        print(f"[cosmo-hydro] done: {step} steps, a={a:.4f} (z={1/a-1:.4f}), "
              f"{time.time()-t0:.1f}s  (gas mass={m_gas:.4e})")
    return U, dm_pos, dm_p, a, step


def mesh_gas_to_grid(gas, pm, a):
    """One-time CIC mesh of MUSIC gas SPH particles -> super-comoving conserved
    grid state (rho~, rho~v~, E~). Gadget velocity u -> v~ = a^{3/2} u, and the
    specific internal energy -> e~_int = a^2 u_int (since P~ = a^5 P)."""
    pos = jnp.asarray(gas.pos)
    mass = jnp.asarray(gas.mass)
    ones = jnp.ones(pos.shape[0])
    vtil = jnp.asarray(gas.vel) * (a ** 1.5)             # super-comoving velocity
    etil = (a ** 2) * jnp.asarray(gas.u_int)             # super-comoving specific u
    rho = pm.cic_deposit(pos, mass, ones)
    mx = pm.cic_deposit(pos, mass * vtil[:, 0], ones)
    my = pm.cic_deposit(pos, mass * vtil[:, 1], ones)
    mz = pm.cic_deposit(pos, mass * vtil[:, 2], ones)
    E = pm.cic_deposit(pos, mass * (0.5 * jnp.sum(vtil ** 2, axis=1) + etil), ones)
    return jnp.stack([rho, mx, my, mz, E])


def run_simulation_cosmo_hydro_music(cfg, is_main_process=True):
    """Assemble a coupled gas+DM run from a MUSIC baryon IC and dispatch the
    coupled runner. Gas (PartType0) is meshed to the grid; DM (PartType1) stays
    as particles. Box + cosmology come from the IC header."""
    from cosmology.music_ic import (music_run_config, read_music_dm_ic,
                                     read_music_gas_ic)
    from setup import create_distributed_mesh, create_solver_distributed
    from utils.grid_params import GridParams

    gas = read_music_gas_ic(cfg.music_ic_path)
    dm = read_music_dm_ic(cfg.music_ic_path, part_type=1)
    cfg = music_run_config(dm, cfg.Nx, base=cfg)             # box + cosmology
    a = float(cfg.a_init)
    # MUSIC may not write InternalEnergy; seed a uniform IGM temperature so the
    # gas has pressure (P=0 gas has no sound speed). Default: CMB at z_init.
    if float(np.max(gas.u_int)) <= 0.0:
        z = 1.0 / a - 1.0
        T_init = float(getattr(cfg, "cosmo_T_init", 0.0)) or 2.725 * (1.0 + z)
        mu = float(getattr(cfg, "cosmo_mu", 1.22))
        gamma = float(cfg.gamma)
        # u = kB T / ((gamma-1) mu m_p), in (km/s)^2 (Gadget velocity units)
        u_init = 8.2543e-3 * T_init / ((gamma - 1.0) * mu)
        gas.u_int = np.full(gas.n_part, u_init, dtype=np.float64)
        if is_main_process:
            print(f"[cosmo-hydro] seeding gas T_init={T_init:.1f} K "
                  f"-> u={u_init:.3e} (km/s)^2 (MUSIC wrote no InternalEnergy)")

    mesh, _ = create_distributed_mesh(cfg)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, _, sharding = create_solver_distributed(cfg, grid, mesh)

    pm = ParticleMeshGravity(cfg)
    U = mesh_gas_to_grid(gas, pm, a)
    dm_pos = jnp.asarray(dm.pos)
    dm_p = jnp.asarray(dm.vel) * (a ** 1.5)                  # canonical momentum p = v~
    dm_mass = jnp.asarray(dm.mass)

    if is_main_process:
        print(f"[cosmo-hydro] MUSIC baryon IC: {gas.n_part} gas + {dm.n_part} DM, "
              f"box={cfg.Lx:.0f} kpc/h, z={1/a-1:.1f}")
    return run_simulation_cosmo_hydro(
        U, cfg, grid, step_fn, compute_dt_fn, sharding,
        dm_pos, dm_p, dm_mass, is_main_process=is_main_process)
