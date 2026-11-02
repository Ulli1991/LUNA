"""3D Sedov-Taylor blast."""
import numbers
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range

# Physical parameters used when cfg.units is enabled: warm-ISM background.
N_AMBIENT_CGS = 1.0  # cm^-3
MU = 1.0
T_AMBIENT_K = 1.0e4
E_INJECT_ERG = 1.0e51

PROTON_MASS_CGS = 1.6726219e-24
BOLTZMANN_CGS = 1.380649e-16


@dataclass(frozen=True)
class Params:
    """Sedov-Blast-3D's parameters, cfg.problem (code units, units off; may be JAX tracers)."""
    E: float = 0.5           # injected energy
    rho_bg: float = 1.0      # ambient density
    p_bg: float = 1e-2       # ambient pressure
    r_blast: float = 6.0     # top-hat radius in cells (1.24: the volume of 2x2x2 cells)


def sedov_blast_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Sedov U: units on, 1e51 erg in the central 2x2x2 cells; off, cfg.problem.E in a top-hat of r_blast cells."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    gamma = cfg.gamma

    dx = Lx / Nx
    dy = Ly / Ny
    dz = Lz / Nz

    is_global_call = x_slice is None and y_slice is None and z_slice is None
    units_on = cfg.units is not None and cfg.units.is_enabled

    if units_on:
        rho_phys = N_AMBIENT_CGS * MU * PROTON_MASS_CGS
        p_phys = N_AMBIENT_CGS * BOLTZMANN_CGS * T_AMBIENT_K
        rho_bg = rho_phys / cfg.units.UnitDensity_in_cgs
        p_bg = p_phys / cfg.units.UnitPressure_in_cgs
        E_inject = E_INJECT_ERG / cfg.units.UnitEnergy_in_cgs
    else:
        rho_bg, p_bg, E_inject = cfg.problem.rho_bg, cfg.problem.p_bg, cfg.problem.E

    if units_on and cfg.problem != Params():
        raise ValueError("Sedov-Blast-3D: problem parameters are code units and apply with units off; "
                         "with units on the blast is the fixed 1e51 erg warm-ISM setup")
    # differentiated through (E, rho_bg, p_bg as tracers): no host prints
    verbose = is_global_call and all(isinstance(v, numbers.Real) for v in (rho_bg, p_bg, E_inject))
    if verbose:
        print("3D Sedov Blast Configuration:")
        print(f"  Domain: [{Lx} x {Ly} x {Lz}]")
        print(f"  Grid: [{Nx} x {Ny} x {Nz}]")
        print(f"  dx={dx:.6e}, dy={dy:.6e}, dz={dz:.6e}")
        print(f"  Total cells: {Nx*Ny*Nz:,}")
        print(f"  Units: {'physical (Arepo)' if units_on else 'dimensionless'}")
        print(f"  rho_bg = {rho_bg:.6e} (code units)")
        print(f"  p_bg   = {p_bg:.6e} (code units)")
        print(f"  E_inject = {E_inject:.6e} (code units)")

    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    z_start, z_end = local_range(Nz, z_slice)
    Nx_local = x_end - x_start
    Ny_local = y_end - y_start
    Nz_local = z_end - z_start

    rho = jnp.full((Nx_local, Ny_local, Nz_local), rho_bg)
    vx = jnp.zeros_like(rho)
    vy = jnp.zeros_like(rho)
    vz = jnp.zeros_like(rho)
    p = jnp.full_like(rho, p_bg)

    if units_on:
        # the 8 cells around the centre: indices {N/2-1, N/2} on each axis
        ix0, iy0, iz0 = Nx // 2 - 1, Ny // 2 - 1, Nz // 2 - 1

        ii = jnp.arange(x_start, x_end)[:, None, None]
        jj = jnp.arange(y_start, y_end)[None, :, None]
        kk = jnp.arange(z_start, z_end)[None, None, :]
        in_cube = (
            (ii >= ix0)
            & (ii <= ix0 + 1)
            & (jj >= iy0)
            & (jj <= iy0 + 1)
            & (kk >= iz0)
            & (kk <= iz0 + 1)
        )

        n_inject = 8
        V_inject = n_inject * dx * dy * dz
        p_inject = (gamma - 1.0) * (E_inject / V_inject)

        if verbose:
            print(
                f"  Sedov inject: central 2x2x2 cube ({n_inject} cells), "
                f"V_inject = {V_inject:.4e}"
            )
            print(f"  p_inject (code) = {p_inject:.4e}")

        p = jnp.where(in_cube, p_inject + p_bg, p_bg)
    else:
        x_full = jnp.linspace(0, Lx, Nx, endpoint=False) + 0.5 * dx
        y_full = jnp.linspace(0, Ly, Ny, endpoint=False) + 0.5 * dy
        z_full = jnp.linspace(0, Lz, Nz, endpoint=False) + 0.5 * dz
        X, Y, Z = jnp.meshgrid(
            x_full[x_start:x_end],
            y_full[y_start:y_end],
            z_full[z_start:z_end],
            indexing="ij",
        )
        r = jnp.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2 + (Z - 0.5 * Lz) ** 2)
        r_blast = cfg.problem.r_blast * max(dx, dy, dz)
        V_blast = (4.0 / 3.0) * jnp.pi * r_blast**3
        e_blast = E_inject / V_blast
        p_blast = (gamma - 1.0) * e_blast
        p = jnp.where(r < r_blast, p_blast + p_bg, p_bg)

    E = p / (gamma - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
    U = jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)
    return U
