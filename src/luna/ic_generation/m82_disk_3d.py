"""M82-like rotating isothermal disk in vertical HSE in an MN + NFW potential, in a hot CGM.

Schneider+ 2018. Units 1e10 Msun / kpc / km/s; the CC85 injection is a runner source.
"""

from dataclasses import dataclass

import jax.numpy as jnp

from luna.sources.external_gravity import (
    vcirc_at_midplane,
)

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """M82-Disk-3D's parameters, cfg.problem."""
    M_gas: float = 0.25
    R_scale_kpc: float = 1.6
    R_trunc_kpc: float = 4.5
    T_K: float = 1.0e4
    n_ambient_cgs: float = 1.0e-4
    T_ambient_K: float = 1.0e6


_PROTON_MASS = 1.67262e-24
_BOLTZMANN = 1.38065e-16
_MSUN = 1.98892e33
_KPC = 3.08568e21
_KM = 1.0e5
_KB_over_mu_mp = _BOLTZMANN / (0.6 * _PROTON_MASS)


def _isothermal_cs_code(T_K, mu=0.6):
    """Isothermal sound speed sqrt(kT/(mu m_p)) in km/s."""
    cs_cgs = (_BOLTZMANN * T_K / (mu * _PROTON_MASS)) ** 0.5
    return cs_cgs / _KM


def _miyamoto_nagai_potential(R, z, M, a, b):
    """Miyamoto-Nagai potential at (R, z) in (km/s)^2."""
    from luna.sources.external_gravity import G_CODE

    return -G_CODE * M / jnp.sqrt(R * R + (a + jnp.sqrt(z * z + b * b)) ** 2)


def _nfw_potential(R, z, M_200, c):
    """NFW potential at (R, z) in (km/s)^2."""
    from luna.sources.external_gravity import G_CODE, _nfw_params_from_M200_c

    r_s, rho_s = _nfw_params_from_M200_c(M_200, c)
    r = jnp.sqrt(R * R + z * z + 1e-10)
    x = r / r_s
    return -4.0 * jnp.pi * G_CODE * rho_s * r_s**3 * jnp.log(1.0 + x) / r


def m82_disk_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """U for the M82-like disk from cfg.problem and the m82_* potential."""
    if not cfg.units.is_enabled:
        raise ValueError("m82_disk_3d requires cfg.units enabled (Arepo system).")

    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    gamma = cfg.gamma

    # x, y in the disk plane, z perpendicular
    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    z_start, z_end = local_range(Nz, z_slice)

    dx = cfg.Lx / Nx
    dy = cfg.Ly / Ny
    dz = cfg.Lz / Nz
    cx = 0.5 * cfg.Lx
    cy = 0.5 * cfg.Ly
    cz = 0.5 * cfg.Lz
    x = (jnp.arange(x_start, x_end) + 0.5) * dx - cx
    y = (jnp.arange(y_start, y_end) + 0.5) * dy - cy
    z = (jnp.arange(z_start, z_end) + 0.5) * dz - cz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    R = jnp.sqrt(X * X + Y * Y)

    M_disk_MN = cfg.m82_stellar_MN_M
    a_MN = cfg.m82_stellar_MN_a_kpc
    b_MN = cfg.m82_stellar_MN_b_kpc
    M_200_NFW = cfg.m82_halo_NFW_M_vir
    c_NFW = cfg.m82_halo_NFW_c

    T_disk = cfg.problem.T_K
    cs2 = _isothermal_cs_code(T_disk) ** 2

    M_disk_gas = cfg.problem.M_gas
    R_d = cfg.problem.R_scale_kpc
    R_trunc = cfg.problem.R_trunc_kpc

    # Sigma_0 so the exponential disk truncated at R_trunc holds M_disk_gas
    rt_over_rd = R_trunc / R_d
    norm_factor = 2.0 * jnp.pi * R_d**2 * (1.0 - (1.0 + rt_over_rd) * jnp.exp(-rt_over_rd))
    Sigma_0 = M_disk_gas / norm_factor

    # isothermal HSE: rho(R, z) = rho_0(R) exp(-(Phi(R,z) - Phi(R,0)) / cs2)
    Phi_xy = _miyamoto_nagai_potential(R, Z, M_disk_MN, a_MN, b_MN) + _nfw_potential(
        R, Z, M_200_NFW, c_NFW
    )
    Phi_R0 = _miyamoto_nagai_potential(
        R, jnp.zeros_like(Z), M_disk_MN, a_MN, b_MN
    ) + _nfw_potential(R, jnp.zeros_like(Z), M_200_NFW, c_NFW)
    z_profile = jnp.exp(-(Phi_xy - Phi_R0) / cs2)

    # rho_0 = Sigma / (sqrt(2 pi) H), H = cs / omega_z(MN, centre): approximate, not exactly M_disk_gas
    from luna.sources.external_gravity import G_CODE

    omega_z2 = G_CODE * M_disk_MN / (b_MN * (a_MN + b_MN) ** 2) + 1e-30
    H_iso = jnp.sqrt(cs2 / omega_z2)

    Sigma_R = Sigma_0 * jnp.exp(-R / R_d) * (R < R_trunc).astype(R.dtype)
    rho_0_R = Sigma_R / (jnp.sqrt(2.0 * jnp.pi) * H_iso)

    rho_disk = rho_0_R * z_profile

    n_amb = cfg.problem.n_ambient_cgs
    T_amb = cfg.problem.T_ambient_K
    mu = cfg.cooling_mu
    rho_amb_cgs = n_amb * mu * _PROTON_MASS
    p_amb_cgs = n_amb * _BOLTZMANN * T_amb
    rho_amb = rho_amb_cgs / cfg.units.UnitDensity_in_cgs
    p_amb = p_amb_cgs / cfg.units.UnitPressure_in_cgs

    rho = jnp.maximum(rho_disk, rho_amb)
    p_disk = rho_disk * cs2
    p = jnp.where(rho_disk > rho_amb, p_disk, p_amb)

    # disk rotates at the midplane circular velocity; the CGM is at rest
    v_circ = vcirc_at_midplane(R, M_disk_MN, a_MN, b_MN, M_200_NFW, c_NFW)
    in_disk = (rho_disk > rho_amb).astype(rho.dtype)
    R_safe = jnp.where(R > 0, R, 1.0)
    vx = -v_circ * Y / R_safe * in_disk
    vy = v_circ * X / R_safe * in_disk
    vz = jnp.zeros_like(rho)

    mom_x = rho * vx
    mom_y = rho * vy
    mom_z = rho * vz
    E = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, mom_x, mom_y, mom_z, E], axis=0)
