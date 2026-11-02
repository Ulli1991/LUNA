"""2D gas disc for disc-planet interaction (de Val-Borro+ 2006); star and planet are particles.

Globally isothermal (isothermal_cs2 > 0). State is [Σ, Σ v_x, Σ v_y, E].
"""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Disc-Planet-2D's parameters, cfg.problem."""
    sigma0: float = 1.0
    sigma_index: float = 0.5
    R_in: float = 0.4
    R_out: float = 2.5
    floor: float = 1.0e-4
    edge_width: float = 0.0


def disc_planet_2d(cfg, x_slice=None, y_slice=None):
    """U for a disc in rotational balance with the softened star potential."""
    Nx, Ny = cfg.Nx, cfg.Ny
    Lx, Ly = cfg.Lx, cfg.Ly
    gamma = cfg.gamma

    G = float(cfg.GRAVITATIONAL_G)
    M_star = float(cfg.particle_star_mass)
    soft2 = float(cfg.particle_softening) ** 2

    cs2 = float(cfg.isothermal_cs2)
    if cs2 <= 0.0:
        raise ValueError(
            "Disc-Planet-2D requires isothermal_cs2 > 0 (globally isothermal disc)."
        )

    Sigma0 = float(cfg.problem.sigma0)
    p_idx = float(cfg.problem.sigma_index)
    R_in = float(cfg.problem.R_in)
    R_out = float(cfg.problem.R_out)
    floor = float(cfg.problem.floor)

    xs, xe = local_range(Nx, x_slice)
    ys, ye = local_range(Ny, y_slice)

    dx, dy = Lx / Nx, Ly / Ny
    # edges smoothed over several cells: a sharper floor-disc jump gives HLLC negative states
    w_edge = float(cfg.problem.edge_width) or max(4.0 * dx, 0.08)
    cx, cy = 0.5 * Lx, 0.5 * Ly
    x_full = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * dx - cx
    y_full = (jnp.arange(Ny, dtype=jnp.float64) + 0.5) * dy - cy
    X, Y = jnp.meshgrid(x_full[xs:xe], y_full[ys:ye], indexing="ij")
    R = jnp.sqrt(X * X + Y * Y)
    Rs = jnp.maximum(R, 0.5 * dx)

    taper = (0.5 * (1.0 + jnp.tanh((Rs - R_in) / w_edge))
             * 0.5 * (1.0 + jnp.tanh((R_out - Rs) / w_edge)))
    Sigma = Sigma0 * Rs ** (-p_idx) * taper + floor

    # v_phi^2 = R |dΦ/dR| + (R/Σ) dP/dR, the pressure term ≈ -p cs2 for Σ ∝ R^-p
    g_R = G * M_star * Rs / (Rs * Rs + soft2) ** 1.5
    vphi2 = jnp.clip(Rs * g_R - p_idx * cs2, 0.0)
    vphi = jnp.sqrt(vphi2) * taper          # the floor does not rotate

    vx = vphi * (-Y / Rs)
    vy = vphi * (X / Rs)

    rho = Sigma
    P = cs2 * rho
    E = P / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return jnp.stack([rho, rho * vx, rho * vy, E], axis=0)
