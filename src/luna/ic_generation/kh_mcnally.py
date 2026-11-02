"""Smooth Kelvin-Helmholtz of McNally, Lyra & Passy 2012 (ApJS 201, 18, eqs. 1-5): two exponential shear layers."""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """KH-McNally-2D's parameters, cfg.problem; the defaults are the paper's."""
    L: float = 0.025                 # layer thickness, eq. 1
    rho1: float = 1.0                # outer stripe
    rho2: float = 2.0                # middle stripe
    U1: float = 0.5                  # V_x of the outer stripe
    U2: float = -0.5                 # V_x of the middle stripe
    P0: float = 2.5                  # uniform pressure
    amp: float = 0.01                # V_y = amp sin(4 pi x), eq. 5
    upper_seed_sign: float = 1.0     # seed sign for y >= Ly/2; -1 is a broken setup (test teeth)


def _lower_half(yh, f1, f2, L):
    """Eq. 1 (and 3) for 0 <= y < 1/2 in units of the box height."""
    fm = 0.5 * (f1 - f2)
    return np.where(yh < 0.25, f1 - fm * np.exp((yh - 0.25) / L), f2 + fm * np.exp((0.25 - yh) / L))


def profile(n, f1, f2, L):
    """Eq. 1 at the n cell centres of [0, 1), mirrored about 1/2 so the IC is exactly symmetric for even n."""
    y = (np.arange(n) + 0.5) / n
    if n % 2:
        return np.where(y < 0.5, _lower_half(y, f1, f2, L), _lower_half(1.0 - y, f1, f2, L))
    lo = _lower_half(y[:n // 2], f1, f2, L)
    return np.concatenate([lo, lo[::-1]])


def seed(n, amp):
    """amp sin(4 pi x) at the n cell centres of [0, 1); for 4 | n a shift by n/4 negates it exactly."""
    if n % 4:
        return amp * np.sin(4 * np.pi * (np.arange(n) + 0.5) / n)
    q = amp * np.sin(4 * np.pi * (np.arange(n // 4) + 0.5) / n)
    return np.concatenate([q, -q, q, -q])


def kh_mcnally(cfg, x_slice=None, y_slice=None):
    """U from point values at cell centres (M(0) = amp exactly); profiles in units of Ly, the seed of Lx."""
    p = cfg.problem
    x0, x1 = local_range(cfg.Nx, x_slice)
    y0, y1 = local_range(cfg.Ny, y_slice)
    L = p.L                                   # in units of the box height, as the paper's unit box
    rho = profile(cfg.Ny, p.rho1, p.rho2, L)[y0:y1]
    vx = profile(cfg.Ny, p.U1, p.U2, L)[y0:y1]
    upper = (np.arange(y0, y1) + 0.5) / cfg.Ny >= 0.5
    sign = np.where(upper, p.upper_seed_sign, 1.0)
    vy = seed(cfg.Nx, p.amp)[x0:x1, None] * sign[None, :]
    rho, vx = np.broadcast_to(rho, vy.shape), np.broadcast_to(vx, vy.shape)
    E = p.P0 / (cfg.gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return jnp.asarray(np.stack([rho, rho * vx, rho * vy, E]))
