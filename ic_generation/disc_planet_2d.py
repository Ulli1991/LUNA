"""2D disc-planet interaction IC. Ref: de Val-Borro et al. 2006.

Gas disc in rotational balance with the star's softened potential; star and
planet live in the particle module (sources/particles.py) — this IC only lays
down the gas. Requires isothermal_cs2 > 0 (globally isothermal, not the
locally-isothermal benchmark). State is [Σ, Σ v_x, Σ v_y, E].
"""
import jax.numpy as jnp

from ._grid import local_range


def disc_planet_2d(cfg, x_slice=None, y_slice=None):
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))
    M_star = float(getattr(cfg, "particle_star_mass", 1.0))
    soft2 = float(getattr(cfg, "particle_softening", 0.0)) ** 2

    cs2 = float(getattr(cfg, "isothermal_cs2", 0.0))
    if cs2 <= 0.0:
        raise ValueError(
            "Disc-Planet-2D requires isothermal_cs2 > 0 (globally isothermal disc)."
        )

    Sigma0 = float(getattr(cfg, "disc_sigma0", 1.0))
    p_idx = float(getattr(cfg, "disc_sigma_index", 0.5))
    R_in = float(getattr(cfg, "disc_R_in", 0.4))
    R_out = float(getattr(cfg, "disc_R_out", 2.5))
    floor = float(getattr(cfg, "disc_floor", 1.0e-4))

    xs, xe = local_range(Nx, x_slice)
    ys, ye = local_range(Ny, y_slice)

    dx, dy = Lx / Nx, Ly / Ny
    # Radial smoothing length of the disc edges. The floor→disc transition is a
    # huge density jump in a cold, supersonic flow; smoothing it over several
    # cells (not ~2) keeps the HLLC solver from forming negative states there.
    w_edge = float(getattr(cfg, "disc_edge_width", 0.0)) or max(4.0 * dx, 0.08)
    cx, cy = 0.5 * Lx, 0.5 * Ly
    x_full = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * dx - cx
    y_full = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dy - cy
    X, Y = jnp.meshgrid(x_full[xs:xe], y_full[ys:ye], indexing="ij")
    R = jnp.sqrt(X * X + Y * Y)
    Rs = jnp.maximum(R, 0.5 * dx)

    # Surface density: power law tapered to the floor between R_in and R_out.
    taper = (0.5 * (1.0 + jnp.tanh((Rs - R_in) / w_edge))
             * 0.5 * (1.0 + jnp.tanh((R_out - Rs) / w_edge)))
    Sigma = Sigma0 * Rs ** (-p_idx) * taper + floor

    # Rotational balance with the SOFTENED star potential:
    #   v_phi^2 = R |dΦ/dR| + (R/Σ) dP/dR
    # gravity term R|dΦ/dR| = G M⋆ R^2 / (R^2+ε^2)^{3/2};
    # pressure term for P = cs2·Σ, Σ∝R^-p:  (R/Σ) dP/dR ≈ -p·cs2 (sub-Keplerian).
    g_R = G * M_star * Rs / (Rs * Rs + soft2) ** 1.5
    vphi2 = jnp.clip(Rs * g_R - p_idx * cs2, 0.0)
    vphi = jnp.sqrt(vphi2) * taper          # spin only the disc body, not the floor

    vx = vphi * (-Y / Rs)
    vy = vphi * (X / Rs)

    rho = Sigma
    P = cs2 * rho
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return jnp.stack([rho, rho * vx, rho * vy, E], axis=0)
