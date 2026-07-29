"""On-the-fly sink particles with self-gravity coupling (3D only).

Hybrid gravity: FMG-multigrid Poisson for the gas; gas→sink via CIC-interpolated
gas-only φ (no self-force); sink→gas via analytic softened point-mass fields;
sink→sink direct N-body. JAX needs static shapes: `sink_max_count` slots
(pos[N,3], vel[N,3], mass[N], active[N]) gated by the `active` mask, threaded
through the runner loop as state.
"""
import jax
import jax.numpy as jnp
import numpy as np

from gravity.poisson_multigrid import _max_levels, solve_poisson_fmg
from gravity.poisson_bc import make_halo_3d


def setup_sinks(cfg, sharding, is_main_process: bool = True):
    """Return a sink-state dict, or None when disabled.

    The returned dict carries the per-step closure `apply(U, state, dt)` and the
    initial carry `state0 = (pos, vel, mass, active)`.
    """
    if not getattr(cfg, "sinks_enabled", False):
        return None
    if cfg.SPATIAL_DIMENSION != 3:
        raise ValueError("sinks require SPATIAL_DIMENSION == 3")

    dx = cfg.Lx / cfg.Nx
    G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))
    fourpiG = 4.0 * jnp.pi * G

    pb = getattr(cfg, "POISSON_BOUNDARY", None)
    boundary = int(cfg.BOUNDARY if pb is None else pb)
    halo_fn = make_halo_3d(boundary)
    n_levels = int(getattr(cfg, "POISSON_N_LEVELS", None) or _max_levels(cfg.Nx))
    n_vpl = getattr(cfg, "POISSON_FMG_VCYCLES_PER_LEVEL", 1)
    n_extra = getattr(cfg, "POISSON_FMG_EXTRA_VCYCLES", 0)
    n_pre = getattr(cfg, "POISSON_N_PRE", 2)
    n_post = getattr(cfg, "POISSON_N_POST", 2)
    n_coarsest = getattr(cfg, "POISSON_N_COARSEST", 50)
    inv_2dx = 0.5 / dx
    dV = dx ** 3

    N = int(getattr(cfg, "sink_max_count", 4))
    eps = float(getattr(cfg, "sink_softening", 0.0)) or 2.0 * dx
    eps2 = eps * eps
    r_acc = float(getattr(cfg, "sink_accretion_radius", 0.0)) or 2.0 * dx
    rho_create = float(getattr(cfg, "sink_creation_density", 0.0)) or float("inf")
    r_sep = float(getattr(cfg, "sink_creation_separation", 0.0)) or 4.0 * dx
    # fraction of the gas removed into a sink on creation/accretion; the cells
    # keep (1 - frac) of their density so the hydro never sees a zero-density cell.
    sink_create_frac = float(getattr(cfg, "sink_create_frac", 0.0)) or 0.9

    # cell-centre coordinate axes (code units)
    xc = (jnp.arange(cfg.Nx) + 0.5) * dx
    yc = (jnp.arange(cfg.Ny) + 0.5) * (cfg.Ly / cfg.Ny)
    zc = (jnp.arange(cfg.Nz) + 0.5) * (cfg.Lz / cfg.Nz)
    X, Y, Z = jnp.meshgrid(xc, yc, zc, indexing="ij")

    # ---- gravity helpers -------------------------------------------------
    def _phi_gas(rho):
        rho_src = rho - jnp.mean(rho) if boundary == 0 else rho
        return solve_poisson_fmg(
            rho_src, dx, fourpiG, boundary=boundary, n_levels=n_levels,
            n_vcycles_per_level=n_vpl, n_extra_vcycles=n_extra,
            n_pre=n_pre, n_post=n_post, n_coarsest=n_coarsest)

    def _grad_phi(phi):
        pp = halo_fn(phi)
        gx = -(pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) * inv_2dx
        gy = -(pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) * inv_2dx
        gz = -(pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) * inv_2dx
        return gx, gy, gz

    def _div_v(U):
        """∇·v on the grid — the converging-flow indicator for sink formation."""
        rho = jnp.maximum(U[0], 1e-30)
        vx = U[1] / rho; vy = U[2] / rho; vz = U[3] / rho
        px = halo_fn(vx); py = halo_fn(vy); pz = halo_fn(vz)
        dvx = (px[2:, 1:-1, 1:-1] - px[:-2, 1:-1, 1:-1]) * inv_2dx
        dvy = (py[1:-1, 2:, 1:-1] - py[1:-1, :-2, 1:-1]) * inv_2dx
        dvz = (pz[1:-1, 1:-1, 2:] - pz[1:-1, 1:-1, :-2]) * inv_2dx
        return dvx + dvy + dvz

    def _sink_field_on_grid(pos, mass, active):
        """Σ_i analytic softened point-mass field of the sinks, on the grid."""
        gx = jnp.zeros_like(X); gy = jnp.zeros_like(X); gz = jnp.zeros_like(X)
        for i in range(N):
            ddx = X - pos[i, 0]; ddy = Y - pos[i, 1]; ddz = Z - pos[i, 2]
            inv_r3 = active[i] * G * mass[i] * (ddx*ddx + ddy*ddy + ddz*ddz + eps2) ** -1.5
            gx = gx - ddx * inv_r3; gy = gy - ddy * inv_r3; gz = gz - ddz * inv_r3
        return gx, gy, gz

    def _interp(field, p):
        """Trilinear (CIC) interpolation of a grid field at a position p."""
        fx = p[0] / dx - 0.5; fy = p[1] / dx - 0.5; fz = p[2] / dx - 0.5
        i0 = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, cfg.Nx - 2)
        j0 = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, cfg.Ny - 2)
        k0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, cfg.Nz - 2)
        tx = jnp.clip(fx - i0, 0.0, 1.0); ty = jnp.clip(fy - j0, 0.0, 1.0)
        tz = jnp.clip(fz - k0, 0.0, 1.0)
        out = 0.0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = ((tx if di else 1-tx) * (ty if dj else 1-ty)
                         * (tz if dk else 1-tz))
                    out = out + w * field[i0+di, j0+dj, k0+dk]
        return out

    def _sink_accel(pos, vel, mass, active, gx_gas, gy_gas, gz_gas):
        """Acceleration on each sink: gas→sink (interp gas-only field) +
        sink→sink (direct, self-excluded, softened)."""
        ax = jnp.zeros(N); ay = jnp.zeros(N); az = jnp.zeros(N)
        for i in range(N):
            # gas → sink i (interpolate the gas-only field at the sink position)
            gxi = _interp(gx_gas, pos[i]); gyi = _interp(gy_gas, pos[i]); gzi = _interp(gz_gas, pos[i])
            # sink j → sink i (direct, exclude self)
            ddx = pos[:, 0] - pos[i, 0]; ddy = pos[:, 1] - pos[i, 1]; ddz = pos[:, 2] - pos[i, 2]
            inv_r3 = active * G * mass * (ddx*ddx + ddy*ddy + ddz*ddz + eps2) ** -1.5
            inv_r3 = inv_r3.at[i].set(0.0)          # no self-force
            gxi = gxi + jnp.sum(inv_r3 * ddx)
            gyi = gyi + jnp.sum(inv_r3 * ddy)
            gzi = gzi + jnp.sum(inv_r3 * ddz)
            ax = ax.at[i].set(active[i] * gxi); ay = ay.at[i].set(active[i] * gyi)
            az = az.at[i].set(active[i] * gzi)
        return jnp.stack([ax, ay, az], axis=1)

    # ---- gas kick (momentum + energy) ------------------------------------
    def _kick_gas(U, gx, gy, gz, dt):
        rho = U[0]
        nmx = U[1] + dt * rho * gx
        nmy = U[2] + dt * rho * gy
        nmz = U[3] + dt * rho * gz
        mxm = 0.5 * (U[1] + nmx); mym = 0.5 * (U[2] + nmy); mzm = 0.5 * (U[3] + nmz)
        nE = U[-1] + dt * (mxm * gx + mym * gy + mzm * gz)
        return U.at[1].set(nmx).at[2].set(nmy).at[3].set(nmz).at[-1].set(nE)

    # ---- accretion: gas inside r_acc → sink (mass + momentum) -------------
    def _accrete(U, pos, vel, mass, active):
        rho = U[0]
        for i in range(N):
            r2 = (X - pos[i, 0])**2 + (Y - pos[i, 1])**2 + (Z - pos[i, 2])**2
            inside = active[i] * (r2 < r_acc * r_acc)
            # SEREN-style threshold accretion: take only the gas ABOVE ρ_SINK,
            # leaving the cell at ρ_SINK (not a fixed 50%/step of everything —
            # that swallowed most of the cloud).
            dm_cell = inside * jnp.maximum(rho - rho_create, 0.0)
            dM = jnp.sum(dm_cell) * dV
            # accreted momentum (cell momentum fraction)
            frac = jnp.where(rho > 0, dm_cell / jnp.maximum(rho, 1e-30), 0.0)
            dpx = jnp.sum(frac * U[1]) * dV; dpy = jnp.sum(frac * U[2]) * dV
            dpz = jnp.sum(frac * U[3]) * dV
            # conserve: new sink momentum = old + accreted; new vel = p/m
            m_old = mass[i]; m_new = m_old + dM
            px = m_old * vel[i, 0] + dpx; py = m_old * vel[i, 1] + dpy; pz = m_old * vel[i, 2] + dpz
            vx = jnp.where(m_new > 0, px / m_new, 0.0)
            vy = jnp.where(m_new > 0, py / m_new, 0.0)
            vz = jnp.where(m_new > 0, pz / m_new, 0.0)
            mass = mass.at[i].set(m_new)
            vel = vel.at[i].set(jnp.array([vx, vy, vz]))
            # remove the accreted gas (all conserved vars, incl. energy) from grid
            kf = jnp.where(inside > 0, 1.0 - jnp.minimum(frac, 1.0), 1.0)
            U = U * kf[None]
        return U, pos, vel, mass, active

    # ---- creation: a bound density peak above threshold → new sink -------
    def _maybe_create(U, pos, vel, mass, active, phi, divv):
        rho = U[0]
        # SEREN / Bate-Bonnell-Price candidate: the gravitational POTENTIAL
        # MINIMUM (deepest well) — not merely the density max.
        idx = jnp.argmin(phi.reshape(-1))
        ix = idx // (cfg.Ny * cfg.Nz)
        iy = (idx // cfg.Nz) % cfg.Ny
        iz = idx % cfg.Nz
        rho_pk = rho[ix, iy, iz]
        divv_pk = divv[ix, iy, iz]
        pk = jnp.array([X[ix, iy, iz], Y[ix, iy, iz], Z[ix, iy, iz]])
        # distance to nearest active sink
        d2 = (pos[:, 0]-pk[0])**2 + (pos[:, 1]-pk[1])**2 + (pos[:, 2]-pk[2])**2
        near = jnp.any(active * (d2 < r_sep * r_sep))
        free_slot = jnp.argmin(active.astype(jnp.int32))   # first inactive slot
        has_free = jnp.any(active < 0.5)
        # gas inside r_acc of the candidate → the prospective sink. Remove only a
        # FRACTION (sink_create_frac), leaving a positive density floor — zeroing
        # the cells makes the next hydro step do v = mom/rho = 0/0 = NaN.
        r2 = (X - pk[0])**2 + (Y - pk[1])**2 + (Z - pk[2])**2
        rem = sink_create_frac * (r2 < r_acc * r_acc).astype(rho.dtype)
        M0 = jnp.sum(rem * rho) * dV
        p0 = jnp.array([jnp.sum(rem * U[1]), jnp.sum(rem * U[2]),
                        jnp.sum(rem * U[3])]) * dV
        v0 = jnp.where(M0 > 0, p0 / M0, 0.0)
        # SEREN bound criterion (vi): the seed region's total mechanical energy —
        # kinetic (in its COM frame) + gravitational — must be negative.
        vx = U[1] / rho; vy = U[2] / rho; vz = U[3] / rho
        v_rel2 = (vx - v0[0])**2 + (vy - v0[1])**2 + (vz - v0[2])**2
        E_kin = jnp.sum(rem * 0.5 * rho * v_rel2) * dV
        E_grav = jnp.sum(rem * rho * phi) * dV          # phi < 0 in the well → binding
        bound = (E_kin + E_grav) < 0.0
        # form a sink iff (SEREN i, ii, iv, vi): density > ρ_SINK, potential-min
        # candidate, converging flow ∇·v<0, BOUND (E_mech<0), isolated, free slot.
        do_create = (rho_pk > rho_create) & (divv_pk < 0.0) & bound & (~near) & has_free

        s = free_slot
        pos = pos.at[s].set(jnp.where(do_create, pk, pos[s]))
        vel = vel.at[s].set(jnp.where(do_create, v0, vel[s]))
        mass = mass.at[s].set(jnp.where(do_create, M0, mass[s]))
        active = active.at[s].set(jnp.where(do_create, 1.0, active[s]))
        # remove the accreted fraction (all conserved vars, incl. energy), keeping
        # a positive floor in the cells (1 - sink_create_frac of the gas remains).
        keep = jnp.where(do_create, 1.0 - rem, 1.0)
        U = U * keep[None]
        return U, pos, vel, mass, active

    # ---- one full sink+gravity step --------------------------------------
    @jax.jit
    def _apply(U, pos, vel, mass, active, dt):
        # 1) gas self-gravity potential (gas only)
        phi = _phi_gas(U[0])
        gxg, gyg, gzg = _grad_phi(phi)
        divv = _div_v(U)                     # converging-flow field (pre-kick infall)
        # 2) sink field on the grid (sink → gas)
        sgx, sgy, sgz = _sink_field_on_grid(pos, mass, active)
        # 3) kick the gas with gas self-grav + sink field
        U = _kick_gas(U, gxg + sgx, gyg + sgy, gzg + sgz, dt)
        # 4) sink acceleration (gas→sink via gas-only field, sink→sink direct)
        acc = _sink_accel(pos, vel, mass, active, gxg, gyg, gzg)
        # 5) leapfrog (KDK simplified to a drift-kick here; dt small)
        vel = vel + dt * acc * active[:, None]
        pos = pos + dt * vel * active[:, None]
        # 6) accrete gas onto sinks, then maybe create a new sink
        U, pos, vel, mass, active = _accrete(U, pos, vel, mass, active)
        U, pos, vel, mass, active = _maybe_create(U, pos, vel, mass, active, phi, divv)
        return U, pos, vel, mass, active

    def apply(U, state, dt):
        pos, vel, mass, active = state
        U, pos, vel, mass, active = _apply(U, pos, vel, mass, active, dt)
        return U, (pos, vel, mass, active)

    state0 = (jnp.zeros((N, 3)), jnp.zeros((N, 3)), jnp.zeros(N),
              jnp.zeros(N))   # all slots inactive

    if is_main_process:
        print(f"✓ Sinks (on-the-fly, PM+direct N-body): N_max={N}, "
              f"ρ_create={rho_create:.3e}, r_acc={r_acc:.3e}, ε={eps:.3e}, "
              f"BC={boundary}")

    return {"apply": apply, "state0": state0, "n_max": N}
