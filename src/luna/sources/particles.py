"""Prescribed star + planet point masses with Plummer-softened gravity, sinks and damping zones (2D).

Used by the disc-planet test (de Val-Borro+2006); options in docs/CONFIG.md.
"""
import jax
import jax.numpy as jnp
import numpy as np


def setup_particles(cfg, sharding, is_main_process: bool = True):
    """Return {"half": fn(U, pos, h) -> U, "advance": fn(pos, dt) -> pos, "pos", "omega"} or None."""
    if not cfg.particles_enabled:
        return None
    if cfg.SPATIAL_DIMENSION != 2:
        raise ValueError("particle module currently supports SPATIAL_DIMENSION == 2")

    G = float(cfg.GRAVITATIONAL_G)
    soft2 = float(cfg.particle_softening) ** 2

    sink_enabled = bool(cfg.particle_sink_enabled)
    r_sink = float(cfg.particle_sink_radius)
    sink_rate = float(cfg.particle_sink_rate)
    sink_on_star = bool(cfg.particle_sink_on_star)
    # The star sink is the inner cavity: without it a central density spike blows up the cold disc.
    r_sink_star = float(cfg.particle_star_sink_radius)
    sink_vdamp = float(cfg.particle_sink_vdamp)
    # (particle index, accretion radius) of each sink.
    sink_specs = []
    if sink_enabled:
        if sink_on_star:
            sink_specs.append((0, r_sink_star))
        sink_specs.append((1, r_sink))

    cx, cy = 0.5 * cfg.Lx, 0.5 * cfg.Ly
    dx, dy = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny
    x = (jnp.arange(cfg.Nx, dtype=jnp.float64) + 0.5) * dx
    y = (jnp.arange(cfg.Ny, dtype=jnp.float64) + 0.5) * dy
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    M_star = float(cfg.particle_star_mass)
    q = float(cfg.particle_planet_q)
    a = float(cfg.particle_planet_a)
    M_planet = q * M_star
    omega = float(np.sqrt(G * M_star / a ** 3))

    pos0 = jnp.array([[cx, cy], [cx + a, cy]], dtype=jnp.float64)
    mass = jnp.array([M_star, M_planet], dtype=jnp.float64)

    # Damping zones relax U toward the planet-free IC near the inner and outer radii (de Val-Borro+2006).
    damp_enabled = bool(cfg.disc_damping_enabled)
    if damp_enabled:
        from luna.ic_generation.ic import initial_conditions
        damp_target = initial_conditions(cfg)
        Rc = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        Rd_in = float(cfg.disc_damp_R_in)
        Rd_out = float(cfg.disc_damp_R_out)
        R_half = 0.5 * min(cfg.Lx, cfg.Ly)
        w_in = jnp.clip((Rd_in - Rc) / max(Rd_in, 1e-6), 0.0, 1.0) ** 2
        w_out = jnp.clip((Rc - Rd_out) / max(R_half - Rd_out, 1e-6), 0.0, 1.0) ** 2
        damp_w = jnp.maximum(w_in, w_out)
        damp_rate = float(cfg.disc_damp_rate)

        @jax.jit
        def _damp(U, dt):
            f = damp_w * jnp.clip(damp_rate * dt, 0.0, 1.0)
            return U + (damp_target - U) * f[None]
    else:
        _damp = None

    @jax.jit
    def _accel(pos):
        gx = jnp.zeros_like(X)
        gy = jnp.zeros_like(Y)
        for i in range(int(pos.shape[0])):
            ddx = X - pos[i, 0]
            ddy = Y - pos[i, 1]
            inv_r3 = (ddx * ddx + ddy * ddy + soft2) ** (-1.5)
            gx = gx - G * mass[i] * ddx * inv_r3
            gy = gy - G * mass[i] * ddy * inv_r3
        return gx, gy

    @jax.jit
    def _kick(U, gx, gy, dt):
        rho = U[0]
        mx, my = U[1], U[2]
        nmx = mx + dt * rho * gx
        nmy = my + dt * rho * gy
        mxm = 0.5 * (mx + nmx)
        mym = 0.5 * (my + nmy)
        nE = U[-1] + dt * (mxm * gx + mym * gy)
        return U.at[1].set(nmx).at[2].set(nmy).at[-1].set(nE)

    @jax.jit
    def _advance(pos, dt):
        # Star (index 0) fixed; planet (index 1) rotates.
        ang = omega * dt
        rx = pos[1, 0] - cx
        ry = pos[1, 1] - cy
        c, s = jnp.cos(ang), jnp.sin(ang)
        nx = cx + rx * c - ry * s
        ny = cy + rx * s + ry * c
        return pos.at[1, 0].set(nx).at[1, 1].set(ny)

    sink_edge = 0.5 * max(dx, dy)

    @jax.jit
    def _accrete(U, pos, dt):
        # Accreted gas is discarded, not added to the point masses (fixed-q benchmark).
        frac = jnp.zeros_like(X)
        for i, ri in sink_specs:
            r = jnp.sqrt((X - pos[i, 0]) ** 2 + (Y - pos[i, 1]) ** 2)
            w = 0.5 * (1.0 - jnp.tanh((r - ri) / sink_edge))
            frac = jnp.maximum(frac, w)
        keep = 1.0 - jnp.clip(sink_rate * dt, 0.0, 0.99) * frac
        U = U * keep[None]               # rho, momenta, E together: v and T unchanged
        if sink_vdamp > 0.0:
            vkeep = 1.0 - jnp.clip(sink_vdamp * dt, 0.0, 0.99) * frac
            U = U.at[1:-1].multiply(vkeep[None])
        return U

    def _half(U, pos, h):
        """Apply gravity, accretion and damping to the gas over time h at positions pos."""
        gx, gy = _accel(pos)
        U = _kick(U, gx, gy, h)
        if sink_specs:
            U = _accrete(U, pos, h)
        if _damp is not None:
            U = _damp(U, h)
        return U

    if is_main_process:
        print(f"✓ Particles: star M={M_star} + planet q={q} "
              f"(a={a}, Ω={omega:.4f}, P_orb={2*np.pi/omega:.3f}), soft²={soft2:.2e}")
        if sink_specs:
            spec = ", ".join(f"{'star' if i == 0 else 'planet'} r={ri:.3f}"
                             for i, ri in sink_specs)
            print(f"  sink: {spec}, rate={sink_rate:.3f}")

    return {"half": _half, "advance": _advance, "pos": pos0, "omega": omega}
