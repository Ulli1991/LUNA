"""Uniform neutral gas at rest for the STARBENCH D-type expansion tests.

The source and ionization belong to the radiation stage. See docs/RADIATION.md.
"""
from dataclasses import dataclass

import jax.numpy as jnp

from luna.ic_generation._grid import local_range


@dataclass(frozen=True)
class Params:
    """HII-Region-3D's parameters, cfg.problem."""
    rho0: float = 1.0


def hii_region_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U with rho = cfg.problem.rho0, v = 0, P = rho rad_cs2_neutral."""
    rho0 = float(cfg.problem.rho0)
    cs2 = float(cfg.rad_cs2_neutral)
    if cs2 <= 0.0:
        raise ValueError("HII-Region-3D needs rad_cs2_neutral > 0: the "
                         "neutral sound speed sets the initial pressure "
                         "(luna.radiation.ionization.starbench_units)")
    x0, x1 = local_range(int(cfg.Nx), x_slice)
    y0, y1 = local_range(int(cfg.Ny), y_slice)
    z0, z1 = local_range(int(cfg.Nz), z_slice)
    shape = (x1 - x0, y1 - y0, z1 - z0)

    if x_slice is None and y_slice is None and z_slice is None:
        print("HII region (STARBENCH D-type expansion):")
        print(f"  Domain: [{cfg.Lx} x {cfg.Ly} x {cfg.Lz}], "
              f"grid [{cfg.Nx} x {cfg.Ny} x {cfg.Nz}]")
        print(f"  rho_0 = {rho0:.6e}, cs_neutral = {cs2 ** 0.5:.6e} "
              f"(code units), P_0 = {rho0 * cs2:.6e}")

    rho = jnp.full(shape, rho0)
    zero = jnp.zeros(shape)
    E = rho * cs2 / (float(cfg.gamma) - 1.0)
    return jnp.stack([rho, zero, zero, zero, E])
