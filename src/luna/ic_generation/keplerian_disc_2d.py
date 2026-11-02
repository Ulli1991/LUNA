"""2D pressure-supported Keplerian disc in the static point-mass potential (external_potential='point_mass',
Phi = -GM / sqrt(R^2 + eps^2)): an exact steady state of the Euler equations.

rho = rho_d + rho_floor, rho_d = S(R) sqrt((1 + eps^2) / (R^2 + eps^2)),
S = [1 + (R_in/R)^k_in]^-1 [1 + (R/R_out)^k_out]^-1, p = cs0^2 rho + P0 and
v_phi^2 = R dPhi/dR + cs0^2 R rho_d' / rho. All gas rotates (no shear layer at the edges); the power-law edges
keep v_phi^2 >= 0 out to the corners for cs0^2 (k + 1) < R dPhi/dR and, for cs0 k <~ 2, R v_phi rising
(Rayleigh stable; tests/test_hydro_hard_keplerian_disc.py checks both for the defaults). State [rho, rho v_x, rho v_y, E], exact cell averages (Gauss-Legendre, q^2 points per cell).
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Keplerian-Disc-2D's parameters, cfg.problem."""
    R_in: float = 0.5           # inner edge of the disc
    R_out: float = 1.6          # outer edge of the disc
    k_in: float = 12.0          # inner edge power
    k_out: float = 20.0         # outer edge power
    cs0: float = 0.05           # isothermal sound speed, p = cs0^2 rho + P0
    rho_floor: float = 1.0e-8   # density added everywhere (the gas off the disc)
    P0: float = 0.0             # uniform pressure added everywhere
    q: int = 3                  # Gauss points per axis for the cell averages


def exact(R, GM, eps, P):
    """(rho, p, v_phi) of the steady state at radius R > 0 (numpy); refuses v_phi^2 < 0."""
    R = np.asarray(R, np.float64)
    s_in = 1.0 / (1.0 + (P.R_in / R) ** P.k_in)
    s_out = 1.0 / (1.0 + (R / P.R_out) ** P.k_out)
    r2 = R * R + eps * eps
    rho_d = s_in * s_out * np.sqrt((1.0 + eps * eps) / r2)
    rho = rho_d + P.rho_floor
    dln = (P.k_in * (1.0 - s_in) - P.k_out * (1.0 - s_out)) / R - R / r2      # dln(rho_d)/dR
    vphi2 = GM * R * R / r2 ** 1.5 + P.cs0 ** 2 * R * dln * rho_d / rho
    if np.min(vphi2) < 0.0:
        raise ValueError(f"Keplerian-Disc-2D: v_phi^2 = {np.min(vphi2):.3g} < 0; lower problem.cs0 or k_out")
    return rho, P.cs0 ** 2 * rho + P.P0, np.sqrt(vphi2)


def cell_averages(n, L, GM, eps, gamma, P, x_slice=None, y_slice=None):
    """Exact cell averages of [rho, rho v_x, rho v_y, E] on the n x n grid of side L centred on the mass (numpy)."""
    gx, gw = np.polynomial.legendre.leggauss(int(P.q))
    xs, xe = local_range(n, x_slice)
    ys, ye = local_range(n, y_slice)
    h = L / n
    U = 0.0
    for a, wa in zip(gx, gw):
        for b, wb in zip(gx, gw):
            x = (np.arange(xs, xe) + 0.5 + 0.5 * a) * h - 0.5 * L
            y = (np.arange(ys, ye) + 0.5 + 0.5 * b) * h - 0.5 * L
            X, Y = np.meshgrid(x, y, indexing="ij")
            R = np.hypot(X, Y)
            rho, p, v = exact(R, GM, eps, P)
            vx, vy = -v * Y / R, v * X / R
            U = U + 0.25 * wa * wb * np.stack([rho, rho * vx, rho * vy, p / (gamma - 1.0) + 0.5 * rho * v * v])
    return U


def keplerian_disc_2d(cfg, x_slice=None, y_slice=None):
    """U of the steady disc; refuses unless the point-mass potential is on and centred on a square box."""
    if cfg.external_potential != "point_mass":
        raise ValueError("Keplerian-Disc-2D needs external_potential='point_mass' (the disc's gravity)")
    if cfg.Nx != cfg.Ny or cfg.Lx != cfg.Ly:
        raise ValueError("Keplerian-Disc-2D: square grid and box only (Nx == Ny, Lx == Ly)")
    if cfg.external_center is not None and tuple(cfg.external_center) != (0.5 * cfg.Lx, 0.5 * cfg.Ly):
        raise ValueError("Keplerian-Disc-2D: the point mass sits at the box centre (external_center=None)")
    if cfg.Nx % 2:
        raise ValueError("Keplerian-Disc-2D: even Nx, so the mass sits on a cell corner, off every Gauss point")
    U = cell_averages(cfg.Nx, cfg.Lx, float(cfg.external_GM), float(cfg.external_softening), float(cfg.gamma),
                      cfg.problem, x_slice, y_slice)
    return jnp.asarray(U)
