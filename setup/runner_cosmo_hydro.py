"""Coupled gas + DM cosmological runner (adiabatic), cfg.CosmoHydro.

Both species live in the super-comoving frame (gas v~ = a v_pec == DM canonical
momentum p = a^2 x_dot), so ONE Poisson solve per step on the total matter
contrast, lap(phi~) = (3/2) Omega_m H0^2 a delta, drives both: operator-split
gas kick + DM KDK leapfrog (end-of-step force reused as the next opening kick).
dtau = min(gas CFL, DM accel criterion, dloga cap). Pressure floor keeps the
cold supersonic gas positive."""
import time

import jax
import jax.numpy as jnp
import numpy as np

from cosmology import Cosmology
from cosmology.mesh_gas import mesh_gas_to_grid, mesh_gas_to_grid_slab
from gravity.particle_mesh import ParticleMeshGravity
from gravity.poisson_multigrid import solve_poisson_fmg
from setup.cosmo_common import (CosmoClock, dtau_accel, poisson_coeff_fac,
                                softening_from_config)
from setup.run_report import StepLog, require_finite


def run_simulation_cosmo_hydro(U, cfg, step_fn, compute_dt_fn,
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
    n_coarse = int(getattr(cfg, "POISSON_FMG_N_COARSEST", 50))
    n_levels = getattr(cfg, "POISSON_N_LEVELS", None)
    jeans_N = float(getattr(cfg, "cosmo_jeans_ncell", 0.0))   # Truelove cells (0 = off)
    T_floor = float(getattr(cfg, "cosmo_T_floor", 0.0))       # temperature floor [K] (0 = off)
    mu = float(getattr(cfg, "cosmo_mu", 1.22))
    dual_energy = bool(getattr(cfg, "DualEnergy", False))     # Bryan+1995 dual-energy
    dual_eta = float(getattr(cfg, "cosmo_dual_eta", 1.0e-3))
    debug = bool(getattr(cfg, "cosmo_debug", False))          # per-step extrema diagnostics

    pm = ParticleMeshGravity(cfg)
    dx = pm.dx
    eps_soft = softening_from_config(cfg, dx)
    # comoving Poisson coefficient (3/2) Omega_m H0^2 a; for the dimensionless
    # pancake units (H0 = 1) this reduces to (3/2) Omega_m a.
    coeff_fac = poisson_coeff_fac(Om, H0)
    rho_bar = float(jnp.mean(U[0]))                          # mean comoving gas density
    rho_floor = float(getattr(cfg, "cosmo_rho_floor", 0.0)) * rho_bar   # density floor
    # low-density CFL guards (tame the v=p/rho and spurious-hot runaways in diffuse
    # cells -- NOT the dense halos): velocity ceiling, temperature ceiling, and a
    # buffer factor so the momentum reset catches cells *near* the floor too.
    v_max = float(getattr(cfg, "cosmo_vmax", 0.0))           # |v~| ceiling (0 = off)
    T_max = float(getattr(cfg, "cosmo_Tmax", 0.0))           # temperature ceiling [K] (0 = off)
    floor_buffer = float(getattr(cfg, "cosmo_floor_buffer", 1.0))

    dm_pos = jnp.asarray(dm_pos)
    dm_p = jnp.asarray(dm_p)
    dm_mass = jnp.asarray(dm_mass)
    active = jnp.ones(dm_pos.shape[0])

    clock = CosmoClock(cosmo, a, a_end, H0)

    @jax.jit
    def forces(U_arr, pos, coeff_val):
        """One joint solve -> (gas force grid, DM force at particles, delta_max,
        gas mass, dm mass)."""
        rho_dm = pm.cic_deposit(pos, dm_mass, active)
        rho_tot = U_arr[0] + rho_dm
        delta = rho_tot / jnp.mean(rho_tot) - 1.0
        phi = solve_poisson_fmg(delta, dx, coeff_val, boundary=0,
                                n_levels=n_levels, n_extra_vcycles=n_extra,
                                n_coarsest=n_coarse)
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

    def tempmax_coeff(aa):
        # super-comoving e_int ceiling coefficient: e_int <= coeff(a) rho~/(gamma-1)
        # caps T <= T_max, killing the spurious ~1e11 K diffuse cells.
        if T_max <= 0.0:
            return 0.0
        return aa * aa * 8.2543e-3 * T_max / mu

    @jax.jit
    def apply_rho_floor(U_arr, ct, ctmax):
        """Tame the low-density CFL pathology (diffuse cells, NOT dense halos):
          (1) buffered density floor: where rho~ < floor_buffer*rho_floor, clamp
              rho~ and ZERO momentum (v=(rho~v~)/rho~ blows up as rho~->floor);
          (2) velocity ceiling: rescale momentum so |v~| <= v_max, preserving e_int
              -- removes the v=p/rho runaway in near-floor cells (the dt killer);
          (3) temperature ceiling: e_int <= ctmax*rho~/(gamma-1), preventing the
              spurious ~1e11 K diffuse cells the total-energy scheme makes once KE
              runs away. Floors (T_floor) still applied separately by apply_floor."""
        rho = U_arr[0]
        mx, my, mz, E = U_arr[1], U_arr[2], U_arr[3], U_arr[-1]
        bad = rho < rho_floor * floor_buffer               # (1)
        rho_n = jnp.where(bad, rho_floor, rho)
        mx = jnp.where(bad, 0.0, mx)
        my = jnp.where(bad, 0.0, my)
        mz = jnp.where(bad, 0.0, mz)
        eint_fl = jnp.maximum(ct * rho_floor / (gamma - 1.0), 1e-300)
        E = jnp.where(bad, eint_fl, E)
        ke = 0.5 * (mx * mx + my * my + mz * mz) / rho_n
        eint = E - ke
        if v_max > 0.0:                                     # (2)
            v2 = (mx * mx + my * my + mz * mz) / (rho_n * rho_n)
            scl = jnp.where(v2 > v_max * v_max, v_max / jnp.sqrt(v2 + 1e-300), 1.0)
            mx, my, mz = mx * scl, my * scl, mz * scl
            ke = 0.5 * (mx * mx + my * my + mz * mz) / rho_n
        if T_max > 0.0:                                     # (3)
            eint = jnp.minimum(eint, ctmax * rho_n / (gamma - 1.0))
        E_n = ke + jnp.maximum(eint, 1e-300)
        return jnp.stack([rho_n, mx, my, mz, E_n])

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
    out_dir = getattr(cfg, "output_dir", "snapshots/cosmo_hydro")

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

    rep = StepLog(cfg, is_main_process,
                  history_fields=["step", "a", "dloga", "delta_max",
                                  "rho_max"])
    tau, step, t0 = 0.0, 0, time.time()
    g_gas, g_dm, dmax = forces(U, dm_pos, coeff_fac * a)
    while a < a_end - 1e-9:
        gmax = float(jnp.max(jnp.sqrt(jnp.sum(g_dm ** 2, axis=1))))
        dt_cfl = float(compute_dt_fn(U))                          # gas CFL
        dt_dloga = clock.dtau_expansion(a, dloga_max)             # expansion cap
        dt_accel = dtau_accel(eta, eps_soft, gmax)                # DM acceleration
        dtau = min(dt_cfl, dt_dloga, dt_accel)
        if debug and is_main_process and (step % 10 == 0 or dtau < 1e-4
                                          or float(dmax) > 150.0):
            rho = U[0]
            ke = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho
            eint = U[-1] - ke
            P = (gamma - 1.0) * eint
            cs = jnp.sqrt(gamma * jnp.abs(P) / jnp.abs(rho))
            vmag = jnp.sqrt((U[1] ** 2 + U[2] ** 2 + U[3] ** 2)) / rho
            sig = (vmag + cs).reshape(-1)                         # CFL signal speed
            i = jnp.argmax(sig)                                   # worst-CFL cell
            rf, Pf, ef = rho.reshape(-1)[i], P.reshape(-1)[i], eint.reshape(-1)[i]
            vf, cf = vmag.reshape(-1)[i], cs.reshape(-1)[i]
            Tcell = float(Pf) / (float(rf) + 1e-300) * mu / 8.2543e-3 / (a * a)
            ke_eint = float(ke.reshape(-1)[i]) / (abs(float(ef)) + 1e-300)
            lim = ("cfl", "dloga", "accel")[int(np.argmin([dt_cfl, dt_dloga, dt_accel]))]
            print(f"[dbg] s={step} z={1/a-1:.2f} dt[cfl={dt_cfl:.2e} dl={dt_dloga:.2e} "
                  f"ac={dt_accel:.2e}]->{lim}  rho[{float(jnp.min(rho)):.2e},"
                  f"{float(jnp.max(rho)):.2e}] Pmin={float(jnp.min(P)):.2e} "
                  f"csmax={float(jnp.max(cs)):.2e} vmax={float(jnp.max(vmag)):.2e} "
                  f"gmax={gmax:.2e} | worstCFL: rho={float(rf):.2e} P={float(Pf):.2e} "
                  f"T={Tcell:.2e}K v={float(vf):.2e} cs={float(cf):.2e} "
                  f"KE/eint={ke_eint:.2e}", flush=True)
        dtau = clock.cap_to_table(dtau, tau)
        a_next = clock.a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        # DM half-kick @a, DM drift, gas hydro flux, JOINT solve @a_next,
        # gas operator-split kick, DM closing half-kick (one solve/step).
        dm_p = dm_p + g_dm * (0.5 * dtau)
        dm_pos = jnp.mod(dm_pos + dm_p * dtau, box)
        U = step_fn(U, dtau)
        g_gas, g_dm, dmax = forces(U, dm_pos, coeff_fac * a_next)
        U = gas_kick(U, g_gas, dtau)
        if rho_floor > 0.0:
            U = apply_rho_floor(U, temp_coeff(a_next), tempmax_coeff(a_next))
        if dual_energy:
            U, ie = dual_energy_update(U, ie, dtau, temp_coeff(a_next), jeans_coeff(a_next))
        elif p_floor > 0.0 or jeans_N > 0.0 or T_floor > 0.0:
            U = apply_floor(U, jeans_coeff(a_next), temp_coeff(a_next))
        dm_p = dm_p + g_dm * (0.5 * dtau)
        tau += dtau
        a = a_next
        step += 1
        if rep.due(step, a >= a_end - 1e-9):
            rho_max = float(jnp.max(U[0]))
            rep.step(step, f"[cosmo-hydro] {step:5d} {a:8.4f} {1/a-1:8.4f} "
                     f"{dloga:8.5f} {float(dmax):10.3f} {rho_max:11.3e}",
                     row=[step, a, dloga, float(dmax), rho_max], force=True)
            if not np.isfinite(rho_max):
                print("[cosmo-hydro] non-finite gas state -- stopping", flush=True)
                require_finite(rho_max, what="gas density", step=step, t=tau,
                               hint="coupled gas+DM run went NaN.")
        if is_main_process and snap_every > 0 and step % snap_every == 0:
            write_snap(a, U, dm_pos)

    if is_main_process:
        m_gas = float(jnp.sum(U[0])) * dx ** 3
        print(f"[cosmo-hydro] done: {step} steps, a={a:.4f} (z={1/a-1:.4f}), "
              f"{time.time()-t0:.1f}s  (gas mass={m_gas:.4e})")
    return U, dm_pos, dm_p, a, step


def _run_slab_hydro(U, cfg, step_fn, compute_dt_fn, mesh,
                    dm_pos, dm_p, dm_mass, n_total_dm=None, is_main_process=True):
    """Grid-sharded (1-D slab) coupled gas+DM. The gas grid + joint MG solve shard
    along x (the hydro step_fn already carries its own halo exchange); DM particles
    are spatially decomposed into per-slab fixed buffers and exchanged on drift.
    Same super-comoving physics as run_simulation_cosmo_hydro -- only the data
    layout differs, so the single joint solve per step deposits DM + gas density on
    the sharded grid (deposit_slab -> solve_poisson_fmg_shardmap -> grad_phi_slab ->
    gather_slab). Dual-energy is unavailable here (its upwind advection rolls along
    the sharded x-axis); the density + temperature/Jeans floors carry stability."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from gravity.particle_mesh_shardmap import (
        cic_deposit_slab, decompose_to_slabs, exchange_particles_slab,
        gather_accel_slab, grad_phi_slab)
    from gravity.poisson_multigrid import _max_levels, solve_poisson_fmg_shardmap

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
    n_coarse = int(getattr(cfg, "POISSON_FMG_N_COARSEST", 50))
    jeans_N = float(getattr(cfg, "cosmo_jeans_ncell", 0.0))
    T_floor = float(getattr(cfg, "cosmo_T_floor", 0.0))
    mu = float(getattr(cfg, "cosmo_mu", 1.22))
    debug = bool(getattr(cfg, "cosmo_debug", False))

    Nx = int(cfg.Nx)
    Ny = int(getattr(cfg, "Ny", cfg.Nx))
    Nz = int(getattr(cfg, "Nz", cfg.Nx))
    dx = box / Nx
    ndev = int(mesh.shape["x"])
    if Nx % ndev:
        raise ValueError(f"slab decomposition needs Nx ({Nx}) divisible by ndev ({ndev})")
    nx_loc = Nx // ndev
    n_levels = min(_max_levels(Nx), int(np.log2(nx_loc)) + 1)
    coeff_fac = poisson_coeff_fac(Om, H0)
    cap = float(getattr(cfg, "dm_capacity_factor", 1.5))
    eps_soft = softening_from_config(cfg, dx)
    rho_bar = float(jnp.mean(U[0]))
    rho_floor = float(getattr(cfg, "cosmo_rho_floor", 0.0)) * rho_bar
    v_max = float(getattr(cfg, "cosmo_vmax", 0.0))           # |v~| ceiling (0 = off)
    T_max = float(getattr(cfg, "cosmo_Tmax", 0.0))           # temperature ceiling [K] (0 = off)
    floor_buffer = float(getattr(cfg, "cosmo_floor_buffer", 1.0))

    if n_total_dm is not None:
        # multi-process: this rank holds only its index-chunk (~its slab). Assemble
        # the per-slab buffer from the local chunk (no full-N host array, no
        # deadlock-prone full device_put), then settle boundary crossers.
        from gravity.particle_mesh_shardmap import assemble_slab_from_chunk
        (posS, pS, massS), actS, C = assemble_slab_from_chunk(
            [np.asarray(dm_pos), np.asarray(dm_p), np.asarray(dm_mass)],
            [P("x", None), P("x", None), P("x")], n_total_dm, ndev, mesh,
            capacity_factor=cap)
        posS, pS, massS, actS, n_bad0, occ0 = exchange_particles_slab(
            posS, pS, massS, actS, Nx, dx, box, mesh)
        if is_main_process:
            print(f"[cosmo-hydro] DM IC settle: out-of-slab={int(n_bad0)} "
                  f"max_occ={int(occ0)} cap_C={C}", flush=True)
    else:
        # single-process: host-side spatial decomposition into per-slab buffers
        pos_g, mom_g, mass_g, act_g, C = decompose_to_slabs(
            np.asarray(dm_pos), np.asarray(dm_p), np.asarray(dm_mass), Nx, ndev, dx, box, cap)
        posS = jax.device_put(jnp.asarray(pos_g), NamedSharding(mesh, P("x", None)))
        pS = jax.device_put(jnp.asarray(mom_g), NamedSharding(mesh, P("x", None)))
        massS = jax.device_put(jnp.asarray(mass_g), NamedSharding(mesh, P("x")))
        actS = jax.device_put(jnp.asarray(act_g), NamedSharding(mesh, P("x")))
    # optional fp32 particle path: halves every ppermute comm volume and the
    # deposit/exchange buffer-sweep work. Grid rho promotes back to fp64 at
    # (gas + rho_dm) so the Poisson solve is bit-identical; dtau/dx/box are
    # weak-typed python floats so the drift/CIC math stays fp32. Still
    # differentiable (fp32, not integer). fp32 over a 25 Mpc box = ~2.5 pc edge
    # resolution << the 49 kpc PM cell, so PM accuracy is unaffected.
    if bool(getattr(cfg, "dm_fp32", False)):
        posS = posS.astype(jnp.float32)
        pS = pS.astype(jnp.float32)
        massS = massS.astype(jnp.float32)
        actS = actS.astype(jnp.float32)
        if is_main_process:
            print("[cosmo-hydro] DM particle path in fp32 (pos/mom/mass/active)", flush=True)

    clock = CosmoClock(cosmo, a, a_end, H0)

    @jax.jit
    def forces(U_arr, pos, mass, active, a_kick):
        """One joint solve on (gas + DM) density -> (gas force grid, DM accel).
        MUST be jitted: solve_poisson_fmg_shardmap builds a fresh jitted FMG
        closure (baking in fourpiG=coeff*a) on every call, so an unjitted forces
        recompiles the entire multigrid every step (~23 s at 512^3/4-GPU). Jitting
        traces the body once; the FMG then compiles once and every step cache-hits."""
        rho_dm = cic_deposit_slab(pos, mass, active, Nx, Ny, Nz, dx, mesh)
        rho_tot = U_arr[0] + rho_dm
        delta = rho_tot / jnp.mean(rho_tot) - 1.0
        phi = solve_poisson_fmg_shardmap(
            delta, dx, coeff_fac * a_kick, boundary=0, mesh=mesh,
            n_levels=n_levels, n_extra_vcycles=n_extra, n_coarsest=n_coarse)
        g_gas = grad_phi_slab(phi, dx, mesh, order=2)         # gas force grid
        g_dm = gather_accel_slab(grad_phi_slab(phi, dx, mesh, order=4),
                                 pos, Nx, Ny, Nz, dx, mesh)   # DM accel at particles
        # keep DM accel in the particle precision (the fp64 grid gradient would
        # otherwise re-promote pS->posS to fp64 after the first kick, leaking the
        # fp32 path back to fp64); no-op when particles are fp64.
        return g_gas, g_dm.astype(pos.dtype), jnp.max(delta)

    # --- profiling sub-kernels: mirror forces() piece-by-piece so the opaque
    # gravity+PM phase can be split into deposit / FMG / grad / gather. Used only
    # on profiled steps (lazy-jit: zero cost unless cosmo_profile is on). Timing
    # the un-fused pieces over-counts slightly vs the fused forces(), so we report
    # them as a relative sub-split, not absolute ms.
    @jax.jit
    def _prof_deposit(pos, mass, active):
        return cic_deposit_slab(pos, mass, active, Nx, Ny, Nz, dx, mesh)

    @jax.jit
    def _prof_fmg(rho_tot, a_kick):
        delta = rho_tot / jnp.mean(rho_tot) - 1.0
        return solve_poisson_fmg_shardmap(
            delta, dx, coeff_fac * a_kick, boundary=0, mesh=mesh,
            n_levels=n_levels, n_extra_vcycles=n_extra, n_coarsest=n_coarse)

    @jax.jit
    def _prof_ggas(phi):
        return grad_phi_slab(phi, dx, mesh, order=2)

    @jax.jit
    def _prof_gdm(phi, pos):
        return gather_accel_slab(grad_phi_slab(phi, dx, mesh, order=4),
                                 pos, Nx, Ny, Nz, dx, mesh).astype(pos.dtype)

    @jax.jit
    def gas_kick(U_arr, g_gas, dt):
        rho = U_arr[0]
        gx, gy, gz = g_gas
        mx, my, mz = U_arr[1], U_arr[2], U_arr[3]
        nmx = mx + dt * rho * gx
        nmy = my + dt * rho * gy
        nmz = mz + dt * rho * gz
        E = U_arr[-1] + dt * (0.5 * (mx + nmx) * gx + 0.5 * (my + nmy) * gy
                              + 0.5 * (mz + nmz) * gz)
        return jnp.stack([rho, nmx, nmy, nmz, E])

    @jax.jit
    def apply_rho_floor(U_arr, ct, ctmax):
        """Three low-density CFL guards (see single-device path for the rationale):
        (1) buffered density floor + momentum zero, (2) |v~| <= v_max velocity
        ceiling preserving e_int, (3) e_int <= ctmax*rho~/(gamma-1) temp ceiling."""
        rho = U_arr[0]
        mx, my, mz, E = U_arr[1], U_arr[2], U_arr[3], U_arr[-1]
        bad = rho < rho_floor * floor_buffer
        rho_n = jnp.where(bad, rho_floor, rho)
        mx = jnp.where(bad, 0.0, mx)
        my = jnp.where(bad, 0.0, my)
        mz = jnp.where(bad, 0.0, mz)
        eint_fl = jnp.maximum(ct * rho_floor / (gamma - 1.0), 1e-300)
        E = jnp.where(bad, eint_fl, E)
        ke = 0.5 * (mx * mx + my * my + mz * mz) / rho_n
        eint = E - ke
        if v_max > 0.0:
            v2 = (mx * mx + my * my + mz * mz) / (rho_n * rho_n)
            scl = jnp.where(v2 > v_max * v_max, v_max / jnp.sqrt(v2 + 1e-300), 1.0)
            mx, my, mz = mx * scl, my * scl, mz * scl
            ke = 0.5 * (mx * mx + my * my + mz * mz) / rho_n
        if T_max > 0.0:
            eint = jnp.minimum(eint, ctmax * rho_n / (gamma - 1.0))
        E_n = ke + jnp.maximum(eint, 1e-300)
        return jnp.stack([rho_n, mx, my, mz, E_n])

    @jax.jit
    def apply_floor(U_arr, cj, ct):
        rho = U_arr[0]
        ke = 0.5 * (U_arr[1] ** 2 + U_arr[2] ** 2 + U_arr[3] ** 2) / rho
        Pr = (gamma - 1.0) * (U_arr[-1] - ke)
        Pmin = jnp.maximum(jnp.maximum(p_floor, ct * rho), cj * rho * rho)
        return U_arr.at[-1].set(jnp.maximum(Pr, Pmin) / (gamma - 1.0) + ke)

    def jeans_coeff(aa):
        if jeans_N <= 0.0:
            return 0.0
        return coeff_fac * aa * (jeans_N * dx) ** 2 / (4.0 * np.pi ** 2 * rho_bar)

    def temp_coeff(aa):
        if T_floor <= 0.0:
            return 0.0
        return aa * aa * 8.2543e-3 * T_floor / mu

    def tempmax_coeff(aa):
        if T_max <= 0.0:
            return 0.0
        return aa * aa * 8.2543e-3 * T_max / mu

    snap_every = int(getattr(cfg, "cosmo_snap_every", 0))
    out_dir = getattr(cfg, "output_dir", "snapshots/cosmo_hydro")

    def write_snap(a_now, U_arr, pos, mass, active):
        import os

        import h5py
        os.makedirs(out_dir, exist_ok=True)
        # cic_deposit_slab is a shard_map collective -> MUST be entered by every
        # rank together (the caller guarantees the same snapshot step on all ranks).
        rho_dm = cic_deposit_slab(pos, mass, active, Nx, Ny, Nz, dx, mesh)
        if jax.process_count() > 1:
            # multi-node: each rank dumps ONLY its local x-slab to its own file --
            # no all-gather (which would deadlock from a single rank). Reassemble
            # offline by stacking rank files along x via the x_lo/x_hi attrs.
            pid = jax.process_index()
            gas_loc = np.asarray(U_arr[0].addressable_shards[0].data, dtype=np.float32)
            dm_loc = np.asarray(rho_dm.addressable_shards[0].data, dtype=np.float32)
            fn = os.path.join(out_dir, f"coupled_a{a_now:.4f}_rank{pid:03d}.hdf5")
            with h5py.File(fn, "w") as f:
                f.attrs["a"] = a_now
                f.attrs["z"] = 1.0 / a_now - 1.0
                f.attrs["BoxSize"] = float(cfg.Lx)
                f.attrs["rank"] = pid
                f.attrs["nproc"] = jax.process_count()
                f.attrs["Nx"] = Nx
                f.attrs["x_lo"] = pid * nx_loc
                f.attrs["x_hi"] = (pid + 1) * nx_loc
                f.create_dataset("rho_gas", data=gas_loc)
                f.create_dataset("rho_dm", data=dm_loc)
            if pid == 0:
                print(f"[cosmo-hydro] wrote per-rank slab snaps "
                      f"coupled_a{a_now:.4f}_rank*.hdf5 ({jax.process_count()} files)",
                      flush=True)
            return
        fn = os.path.join(out_dir, f"coupled_a{a_now:.4f}.hdf5")
        with h5py.File(fn, "w") as f:
            f.attrs["a"] = a_now
            f.attrs["z"] = 1.0 / a_now - 1.0
            f.attrs["BoxSize"] = float(cfg.Lx)
            f.create_dataset("rho_gas", data=np.asarray(U_arr[0], dtype=np.float32))
            f.create_dataset("rho_dm", data=np.asarray(rho_dm, dtype=np.float32))
        print(f"[cosmo-hydro] wrote {fn}", flush=True)

    if is_main_process:
        m_gas = float(jnp.sum(U[0])) * dx ** 3
        m_dm = float(jnp.sum(massS))
        print(f"\n[cosmo-hydro] grid-sharded slab decomposition across {ndev} devices "
              f"(nx_loc={nx_loc}, capacity {ndev*C} per slab)")
        print(f"[cosmo-hydro] coupled gas+DM  a_init={a:.4f} (z={1/a-1:.2f}) -> "
              f"a_end={a_end:.4f}  Om={Om} OL={cfg.cosmo_Omega_L} H0={H0:.4f}")
        print(f"[cosmo-hydro] gas mass={m_gas:.4e}  DM mass={m_dm:.4e}  "
              f"f_b={m_gas/(m_gas+m_dm):.3f}  (mesh={Nx}^3)")
        print("[cosmo-hydro] compiling first joint force "
              "(deposit -> FMG -> grad -> gather) ...", flush=True)

    # per-phase profiler: every 10 steps (same cadence as the step print) write a
    # percentage breakdown to a SEPARATE csv. Opt-in; when on, the profiled steps
    # carry block_until_ready barriers (the rest of the run stays fully async).
    do_prof = bool(getattr(cfg, "cosmo_profile", False)) and is_main_process
    prof_fh = None
    if do_prof:
        import os
        os.makedirs(out_dir, exist_ok=True)
        prof_fh = open(os.path.join(out_dir, "profile_steps.csv"), "w")
        prof_fh.write("# 512^3-coupled per-phase wallclock breakdown, every 10 steps\n")
        prof_fh.write("step,z,tot_ms,dm_drift_pct,dm_exchange_pct,gas_hydro_pct,"
                      "gravity_pm_pct,gas_kick_floor_pct,dm_kick2_pct,"
                      "pm_deposit_pct,pm_fmg_pct,pm_grad_pct,pm_gather_pct\n")
        prof_fh.flush()

    rep = StepLog(cfg, is_main_process,
                  history_fields=["step", "a", "dloga", "delta_max",
                                  "rho_max"])
    tau, step, t0 = 0.0, 0, time.time()
    tc = time.time()
    g_gas, g_dm, dmax = forces(U, posS, massS, actS, a)
    jax.block_until_ready(g_dm)
    if is_main_process:
        print(f"[cosmo-hydro] first force compiled+ran in {time.time()-tc:.1f}s",
              flush=True)
    if do_prof:   # warm the standalone sub-kernels so the first [prof] line isn't compile-dominated
        _w = _prof_deposit(posS, massS, actS); jax.block_until_ready(_w)
        _wp = _prof_fmg(U[0] + _w, a); jax.block_until_ready(_wp)
        jax.block_until_ready(_prof_ggas(_wp)); jax.block_until_ready(_prof_gdm(_wp, posS))
        print("[cosmo-hydro] profiling sub-kernels compiled", flush=True)
    prof_this = False
    def _bar(x):
        if prof_this:
            jax.block_until_ready(x)
        return x
    n_bad_tot = 0
    while a < a_end - 1e-9:
        gmag = jnp.sqrt(jnp.sum(g_dm ** 2, axis=1))
        gmax = float(jnp.max(jnp.where(actS > 0, gmag, 0.0)))
        dt_cfl = float(compute_dt_fn(U))
        dt_dloga = clock.dtau_expansion(a, dloga_max)
        dt_accel = dtau_accel(eta, eps_soft, gmax)
        dtau = min(dt_cfl, dt_dloga, dt_accel)
        if debug and is_main_process and (step % 10 == 0 or float(dmax) > 150.0):
            rho = U[0]
            ke = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho   # KE density
            eint = U[-1] - ke                                      # internal-energy density
            Pr = (gamma - 1.0) * eint
            vmag = jnp.sqrt(U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / rho
            cs = jnp.sqrt(gamma * jnp.abs(Pr) / rho)
            sig = (vmag + cs).reshape(-1)                          # CFL signal speed
            i = jnp.argmax(sig)                                    # worst-CFL cell
            rf, Pf, ef = rho.reshape(-1)[i], Pr.reshape(-1)[i], eint.reshape(-1)[i]
            kf, vf, cf = ke.reshape(-1)[i], vmag.reshape(-1)[i], cs.reshape(-1)[i]
            # T_phys ~ (P/rho)*mu/(kB/m_p)/a^2 ; ke/eint flags cold KE-dominated cells
            Tcell = float(Pf) / (float(rf) + 1e-300) * mu / 8.2543e-3 / (a * a)
            ke_eint = float(kf) / (abs(float(ef)) + 1e-300)
            lim = ("cfl", "dloga", "accel")[int(np.argmin([dt_cfl, dt_dloga, dt_accel]))]
            print(f"[dbg] s={step} z={1/a-1:.2f} dt[cfl={dt_cfl:.2e} dl={dt_dloga:.2e} "
                  f"ac={dt_accel:.2e}]->{lim}  rho[{float(jnp.min(rho)):.2e},"
                  f"{float(jnp.max(rho)):.2e}] Pmin={float(jnp.min(Pr)):.2e} "
                  f"csmax={float(jnp.max(cs)):.2e} vmax={float(jnp.max(vmag)):.2e} "
                  f"gmax={gmax:.2e} | worstCFL: rho={float(rf):.2e} P={float(Pf):.2e} "
                  f"T={Tcell:.2e}K v={float(vf):.2e} cs={float(cf):.2e} "
                  f"KE/eint={ke_eint:.2e}", flush=True)
        dtau = clock.cap_to_table(dtau, tau)
        a_next = clock.a_of_tau(tau + dtau)
        dloga = np.log(a_next / a)
        # DM half-kick @a, drift + exchange to owning slab, gas hydro flux,
        # JOINT solve @a_next, gas kick + floors, DM closing half-kick.
        prof_this = do_prof and (step % 10 == 0)
        _pt = {}
        _t = time.time()
        pS = pS + g_dm * (0.5 * dtau)
        posS = jnp.mod(posS + pS * dtau, box); _bar(posS)
        if prof_this: _pt["dm_drift"] = time.time() - _t; _t = time.time()
        posS, pS, massS, actS, n_bad, max_occ = exchange_particles_slab(
            posS, pS, massS, actS, Nx, dx, box, mesh); _bar(posS)
        if prof_this: _pt["dm_exchange"] = time.time() - _t; _t = time.time()
        U = step_fn(U, dtau); _bar(U)
        if prof_this: _pt["gas_hydro"] = time.time() - _t; _t = time.time()
        g_gas, g_dm, dmax = forces(U, posS, massS, actS, a_next); _bar(g_dm)
        if prof_this: _pt["gravity_pm"] = time.time() - _t; _t = time.time()
        U = gas_kick(U, g_gas, dtau)
        if rho_floor > 0.0:
            U = apply_rho_floor(U, temp_coeff(a_next), tempmax_coeff(a_next))
        if p_floor > 0.0 or jeans_N > 0.0 or T_floor > 0.0:
            U = apply_floor(U, jeans_coeff(a_next), temp_coeff(a_next))
        _bar(U)
        if prof_this: _pt["gas_kick_floor"] = time.time() - _t; _t = time.time()
        pS = pS + g_dm * (0.5 * dtau); _bar(pS)
        if prof_this:
            _pt["dm_kick2"] = time.time() - _t
            # split the fused gravity+PM phase via the standalone sub-kernels
            # (relative only -- un-fused, so it over-counts vs the fused phase time)
            _t = time.time(); _rdm = _prof_deposit(posS, massS, actS); jax.block_until_ready(_rdm); _dep = time.time() - _t
            _t = time.time(); _phi = _prof_fmg(U[0] + _rdm, a_next); jax.block_until_ready(_phi); _fmg = time.time() - _t
            _t = time.time(); _gg = _prof_ggas(_phi); jax.block_until_ready(_gg); _grd = time.time() - _t
            _t = time.time(); _gd = _prof_gdm(_phi, posS); jax.block_until_ready(_gd); _gat = time.time() - _t
            _tot = sum(_pt.values()) or 1e-30; _sub = (_dep + _fmg + _grd + _gat) or 1e-30
            prof_fh.write(
                f"{step},{1/a_next-1:.3f},{_tot*1e3:.1f},"
                f"{_pt['dm_drift']/_tot*100:.1f},{_pt['dm_exchange']/_tot*100:.1f},"
                f"{_pt['gas_hydro']/_tot*100:.1f},{_pt['gravity_pm']/_tot*100:.1f},"
                f"{_pt['gas_kick_floor']/_tot*100:.1f},{_pt['dm_kick2']/_tot*100:.1f},"
                f"{_dep/_sub*100:.1f},{_fmg/_sub*100:.1f},{_grd/_sub*100:.1f},{_gat/_sub*100:.1f}\n")
            prof_fh.flush()
        tau += dtau
        a = a_next
        step += 1
        n_bad_tot += int(n_bad)
        if rep.due(step, a >= a_end - 1e-9):
            rho_max = float(jnp.max(U[0]))
            rep.step(step, f"[cosmo-hydro] {step:5d} {a:8.4f} {1/a-1:8.4f} "
                     f"{dloga:8.5f} dmax={float(dmax):9.3f} rhomax={rho_max:10.3e} "
                     f"occ={int(max_occ)} nbad={int(n_bad)} "
                     f"{(time.time()-t0)/step:5.2f}s/st",
                     row=[step, a, dloga, float(dmax), int(max_occ), int(n_bad)],
                     force=True)
            if not np.isfinite(rho_max):
                print("[cosmo-hydro] non-finite gas state -- stopping", flush=True)
                require_finite(rho_max, what="gas density", step=step, t=tau,
                               hint="coupled gas+DM slab run went NaN.")
        # write_snap enters a shard_map collective (cic_deposit_slab) -> every
        # rank must call it on the SAME step. step % snap_every is identical on all
        # ranks, so gate only on that (not is_main_process) when multi-process.
        if snap_every > 0 and step % snap_every == 0 and (
                jax.process_count() > 1 or is_main_process):
            write_snap(a, U, posS, massS, actS)

    if is_main_process:
        m_gas = float(jnp.sum(U[0])) * dx ** 3
        print(f"[cosmo-hydro] done: {step} steps, a={a:.4f} (z={1/a-1:.4f}), "
              f"{time.time()-t0:.1f}s  (gas mass={m_gas:.4e}, "
              f"cumulative out-of-slab: {n_bad_tot})")
    if jax.process_count() > 1 or is_main_process:    # final snap (collective: all ranks)
        write_snap(a, U, posS, massS, actS)
    if prof_fh is not None:
        prof_fh.close()
        print(f"[cosmo-hydro] per-phase profile -> {out_dir}/profile_steps.csv", flush=True)
    return U, posS, pS, a, step


def run_simulation_cosmo_hydro_registered_ic(cfg, dm_pos, dm_p, dm_mass,
                                             is_main_process=True):
    """Coupled gas+DM run where the GAS comes from cfg's registered test
    problem (`initialize_simulation_distributed`) and the DM arrays are
    caller-provided (synthetic ICs, e.g. the coupled Zel'dovich pancake).
    Builds mesh/solver exactly like `main.run`'s dense path, then dispatches
    the coupled loop. Returns (U, dm_pos, dm_p, a, step)."""
    from setup import (create_distributed_mesh, create_solver_distributed,
                       initialize_simulation_distributed)
    from utils.grid_params import GridParams

    mesh, _ = create_distributed_mesh(cfg)
    U, _, _ = initialize_simulation_distributed(cfg, mesh)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, _ = create_solver_distributed(cfg, grid, mesh)
    return run_simulation_cosmo_hydro(
        U, cfg, step_fn, compute_dt_fn,
        jnp.asarray(dm_pos), jnp.asarray(dm_p), jnp.asarray(dm_mass),
        is_main_process=is_main_process)


def run_simulation_cosmo_hydro_music(cfg, is_main_process=True):
    """Assemble a coupled gas+DM run from a MUSIC baryon IC and dispatch the
    coupled runner. Gas (PartType0) is meshed to the grid; DM (PartType1) stays
    as particles. Box + cosmology come from the IC header."""
    from cosmology.music_ic import (music_part_count, music_run_config,
                                     read_music_dm_ic, read_music_gas_ic)
    from setup import create_distributed_mesh, create_solver_distributed
    from utils.grid_params import GridParams

    # multi-node: each process reads ONLY its contiguous index-chunk of particles
    # (~its x-slab, by MUSIC x-major ordering) so we never hold the whole IC on
    # any rank. Single-process (1-node) reads the whole file as before. The global
    # counts Ng/Nd are kept so every rank computes an identical per-slab capacity.
    multiproc = jax.process_count() > 1 and bool(getattr(cfg, "dm_domain_decomp", False))
    Ng = Nd = None
    if multiproc:
        nproc, pid = jax.process_count(), jax.process_index()
        Ng = music_part_count(cfg.music_ic_path, 0)
        Nd = music_part_count(cfg.music_ic_path, 1)
        glo, ghi = pid * Ng // nproc, (Ng if pid == nproc - 1 else (pid + 1) * Ng // nproc)
        dlo, dhi = pid * Nd // nproc, (Nd if pid == nproc - 1 else (pid + 1) * Nd // nproc)
        gas = read_music_gas_ic(cfg.music_ic_path, glo, ghi)
        dm = read_music_dm_ic(cfg.music_ic_path, 1, dlo, dhi)
        print(f"[cosmo-hydro] rank {pid}/{nproc}: gas[{glo}:{ghi}] dm[{dlo}:{dhi}] "
              f"(chunked IC read)", flush=True)
    else:
        gas = read_music_gas_ic(cfg.music_ic_path)
        dm = read_music_dm_ic(cfg.music_ic_path, part_type=1)
    cfg = music_run_config(dm, cfg.Nx, base=cfg)             # box + cosmology (header)
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

    grid = GridParams.from_config(cfg)
    # >512^3: grid-sharded slab path -- gas grid + joint MG solve shard along x,
    # DM decomposed into per-slab buffers. The hydro step_fn carries its own halos.
    slab = bool(getattr(cfg, "dm_domain_decomp", False)) and jax.device_count() > 1
    if slab:
        from gravity.particle_mesh_shardmap import make_slab_mesh
        mesh = make_slab_mesh(jax.devices())
    else:
        mesh, _ = create_distributed_mesh(cfg)
    compute_dt_fn, step_fn, sharding = create_solver_distributed(cfg, grid, mesh)

    if slab:
        Nx = int(cfg.Nx)
        dx = float(cfg.Lx) / Nx
        U = mesh_gas_to_grid_slab(gas, mesh, Nx, int(getattr(cfg, "Ny", cfg.Nx)),
                                  int(getattr(cfg, "Nz", cfg.Nx)), dx, float(cfg.Lx), a,
                                  n_total_gas=Ng)
        if is_main_process:
            tag = "chunked" if multiproc else "whole-file"
            print(f"[cosmo-hydro] MUSIC baryon IC: "
                  f"{Ng if multiproc else gas.n_part} gas + "
                  f"{Nd if multiproc else dm.n_part} DM, "
                  f"box={cfg.Lx:.0f} kpc/h, z={1/a-1:.1f} (grid-sharded slab, {tag} IC)")
        return _run_slab_hydro(
            U, cfg, step_fn, compute_dt_fn, mesh,
            np.asarray(dm.pos), np.asarray(dm.vel) * (a ** 1.5), np.asarray(dm.mass),
            n_total_dm=Nd, is_main_process=is_main_process)

    pm = ParticleMeshGravity(cfg)
    U = mesh_gas_to_grid(gas, pm, a)
    dm_pos = jnp.asarray(dm.pos)
    dm_p = jnp.asarray(dm.vel) * (a ** 1.5)                  # canonical momentum p = v~
    dm_mass = jnp.asarray(dm.mass)

    if is_main_process:
        print(f"[cosmo-hydro] MUSIC baryon IC: {gas.n_part} gas + {dm.n_part} DM, "
              f"box={cfg.Lx:.0f} kpc/h, z={1/a-1:.1f}")
    return run_simulation_cosmo_hydro(
        U, cfg, step_fn, compute_dt_fn,
        dm_pos, dm_p, dm_mass, is_main_process=is_main_process)
