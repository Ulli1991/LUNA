"""Basic particle (point-mass) module — gravity coupling to the gas.

A small set of point masses — a central star, an orbiting planet, or (later)
sink particles — that exert a Plummer-softened gravitational acceleration on the
gas grid each step:

    g(x) = -G Σ_i m_i (x - r_i) / (|x - r_i|² + ε²)^{3/2}

applied operator-split to the gas momentum and energy exactly like the
external-gravity source (kick momentum, midpoint-momentum energy work). Built
for the disc-planet interaction test (de Val-Borro et al. 2006 / GANDALF,
Hubber+ 2018 §6.11) and reusable as the basis for Boss-Bodenheimer sink
particles.

Motion: the central star is held fixed at the box centre; the planet is held on
a **prescribed circular Keplerian orbit** of radius a and rate Ω = √(GM⋆/a³)
(it does not migrate, per the test). Particle positions are threaded through the
runner loop as state (like the OU-forcing state). Currently 2D
(SPATIAL_DIMENSION == 2), the form the de Val-Borro benchmark uses.

Enable with cfg.particles_enabled = True; configure the star/planet via
particle_star_mass, particle_planet_q, particle_planet_a, particle_softening.
"""
import jax
import jax.numpy as jnp
import numpy as np


def setup_particles(cfg, sharding, is_main_process: bool = True):
    """Return {"apply": fn(U, pos, dt) -> (U, pos), "pos": pos0} or None."""
    if not getattr(cfg, "particles_enabled", False):
        return None
    if cfg.SPATIAL_DIMENSION != 2:
        raise ValueError("particle module currently supports SPATIAL_DIMENSION == 2")

    G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))
    soft2 = float(getattr(cfg, "particle_softening", 0.0)) ** 2

    # Optional sink particles: drain gas within an accretion radius of a sink
    # (the planet always; the central star too when particle_sink_on_star). The
    # removal multiplies ALL conserved variables by (1 - f) inside the radius, so
    # mass, momentum and energy leave together — the remaining gas keeps its
    # velocity and temperature. Mass-conserving in the sense that accreted gas
    # simply leaves the grid (the point-mass orbits are prescribed, so we do not
    # feed the accreted mass back into the particle mass — fixed-mass planet, per
    # the de Val-Borro benchmark).
    sink_enabled = bool(getattr(cfg, "particle_sink_enabled", False))
    r_sink = float(getattr(cfg, "particle_sink_radius", 0.0))
    sink_rate = float(getattr(cfg, "particle_sink_rate", 0.0))
    sink_on_star = bool(getattr(cfg, "particle_sink_on_star", False))
    # The central star acts as an inner CAVITY (the Cartesian analog of a
    # cylindrical inner boundary) with its own, larger accretion radius — it must
    # swallow the cold floor gas that free-falls into the deep potential, or a
    # central density spike forms and the cold disc blows up.
    r_sink_star = float(getattr(cfg, "particle_star_sink_radius", r_sink))
    # Velocity damping inside the sink regions (de Val-Borro 2006-style boundary
    # damping): relax the gas momentum toward zero so the strong central gravity
    # cannot fling cavity gas to high speed and pollute the disc. Energy is left
    # to the isothermal projection that follows. 0 → no damping (pure removal).
    sink_vdamp = float(getattr(cfg, "particle_sink_vdamp", 0.0))
    # (particle index, accretion radius) pairs that act as sinks.
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

    # star at the box centre + planet on a circular orbit at radius a
    M_star = float(getattr(cfg, "particle_star_mass", 1.0))
    q = float(getattr(cfg, "particle_planet_q", 1.0e-3))
    a = float(getattr(cfg, "particle_planet_a", 1.0))
    M_planet = q * M_star
    omega = float(np.sqrt(G * M_star / a ** 3))   # planet Keplerian rate

    pos0 = jnp.array([[cx, cy], [cx + a, cy]], dtype=jnp.float64)
    mass = jnp.array([M_star, M_planet], dtype=jnp.float64)

    # de Val-Borro (2006) boundary damping zones: near the inner/outer radii,
    # relax U toward the INITIAL axisymmetric disc (the IC has no planet, so it
    # IS the axisymmetric target) with a rate ramping smoothly to the edge. This
    # — not a vacuum cavity drain — is what keeps the cold disc stable.
    damp_enabled = bool(getattr(cfg, "disc_damping_enabled", False))
    if damp_enabled:
        from ic_generation.ic import initial_conditions
        damp_target = initial_conditions(cfg)            # axisymmetric disc state
        Rc = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        Rd_in = float(getattr(cfg, "disc_damp_R_in", 0.5))
        Rd_out = float(getattr(cfg, "disc_damp_R_out", 2.2))
        R_half = 0.5 * min(cfg.Lx, cfg.Ly)
        w_in = jnp.clip((Rd_in - Rc) / max(Rd_in, 1e-6), 0.0, 1.0) ** 2
        w_out = jnp.clip((Rc - Rd_out) / max(R_half - Rd_out, 1e-6), 0.0, 1.0) ** 2
        damp_w = jnp.maximum(w_in, w_out)
        damp_rate = float(getattr(cfg, "disc_damp_rate", 5.0))

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
        # star (index 0) fixed; planet (index 1) rotates on its circular orbit
        ang = omega * dt
        rx = pos[1, 0] - cx
        ry = pos[1, 1] - cy
        c, s = jnp.cos(ang), jnp.sin(ang)
        nx = cx + rx * c - ry * s
        ny = cy + rx * s + ry * c
        return pos.at[1, 0].set(nx).at[1, 1].set(ny)

    sink_edge = 0.5 * max(dx, dy)        # tanh smoothing width ~ half a cell

    @jax.jit
    def _accrete(U, pos, dt):
        # fraction of conserved quantities to remove this step inside each sink's
        # own accretion radius (smooth tanh boundary; combined by max over sinks).
        # Gravitational masses are held FIXED — accreted gas is discarded, not fed
        # back into the point mass (correct for the fixed-q disc-planet benchmark;
        # BB79 growing sinks will instead thread particle mass as state).
        frac = jnp.zeros_like(X)
        for i, ri in sink_specs:
            r = jnp.sqrt((X - pos[i, 0]) ** 2 + (Y - pos[i, 1]) ** 2)
            w = 0.5 * (1.0 - jnp.tanh((r - ri) / sink_edge))
            frac = jnp.maximum(frac, w)
        keep = 1.0 - jnp.clip(sink_rate * dt, 0.0, 0.99) * frac
        U = U * keep[None]               # remove rho, momenta, E together
        if sink_vdamp > 0.0:             # extra velocity damping on the momentum
            vkeep = 1.0 - jnp.clip(sink_vdamp * dt, 0.0, 0.99) * frac
            U = U.at[1:-1].multiply(vkeep[None])
        return U

    def _apply(U, pos, dt):
        gx, gy = _accel(pos)
        U = _kick(U, gx, gy, dt)
        if sink_specs:
            U = _accrete(U, pos, dt)
        if _damp is not None:
            U = _damp(U, dt)
        pos = _advance(pos, dt)
        return U, pos

    if is_main_process:
        print(f"✓ Particles: star M={M_star} + planet q={q} "
              f"(a={a}, Ω={omega:.4f}, P_orb={2*np.pi/omega:.3f}), soft²={soft2:.2e}")
        if sink_specs:
            spec = ", ".join(f"{'star' if i == 0 else 'planet'} r={ri:.3f}"
                             for i, ri in sink_specs)
            print(f"  sink: {spec}, rate={sink_rate:.3f}")

    return {"apply": _apply, "pos": pos0, "omega": omega}
