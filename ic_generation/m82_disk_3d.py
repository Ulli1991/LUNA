"""M82-like disk galaxy IC: rotating isothermal exponential disk in vertical HSE
in a static MN + NFW potential, embedded in a hot CGM. Ref: Schneider+ 2018.

Units: mass in 1e10 Msun, length in kpc, velocity in km/s (requires cfg.units).
The central CC85 injection is NOT added here — the runner applies it as an
operator-split source each step.
"""

import jax.numpy as jnp

from hydro.external_gravity import (
    vcirc_at_midplane,
)

from ._grid import local_range

# Same physical constants as the rest of the CC85 setup.
_PROTON_MASS = 1.67262e-24
_BOLTZMANN = 1.38065e-16
_MSUN = 1.98892e33
_KPC = 3.08568e21
_KM = 1.0e5
_KB_over_mu_mp = _BOLTZMANN / (0.6 * _PROTON_MASS)  # (km/s)^2 per K (after unit-fixing)


def _isothermal_cs_code(T_K, mu=0.6):
    """Return isothermal sound speed c_s = sqrt(kT/(mu m_p)) in km/s code units."""
    cs_cgs = (_BOLTZMANN * T_K / (mu * _PROTON_MASS)) ** 0.5  # cm/s
    return cs_cgs / _KM  # km/s code


def _miyamoto_nagai_potential(R, z, M, a, b):
    """MN potential at (R,z), code units (km/s)^2."""
    from hydro.external_gravity import G_CODE

    return -G_CODE * M / jnp.sqrt(R * R + (a + jnp.sqrt(z * z + b * b)) ** 2)


def _nfw_potential(R, z, M_200, c):
    """NFW spherical potential at (R,z), code units (km/s)^2."""
    from hydro.external_gravity import G_CODE, _nfw_params_from_M200_c

    r_s, rho_s = _nfw_params_from_M200_c(M_200, c)
    r = jnp.sqrt(R * R + z * z + 1e-10)
    x = r / r_s
    return -4.0 * jnp.pi * G_CODE * rho_s * r_s**3 * jnp.log(1.0 + x) / r


def m82_disk_3d(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Construct U for a rotating M82-like disk in vertical HSE.

    Uses cfg.m82_* parameters (see template_config.py).
    """
    if not cfg.units.is_enabled:
        raise ValueError("m82_disk_3d requires cfg.units enabled (Arepo system).")

    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    gamma = cfg.gamma

    # Slices (Cartesian box, x,y in-plane, z perpendicular).
    x_start, x_end = local_range(Nx, x_slice)
    y_start, y_end = local_range(Ny, y_slice)
    z_start, z_end = local_range(Nz, z_slice)
    nx_l = x_end - x_start
    ny_l = y_end - y_start
    nz_l = z_end - z_start

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

    # ----- Potential parameters in code units -----
    M_disk_MN = cfg.m82_stellar_MN_M
    a_MN = cfg.m82_stellar_MN_a_kpc
    b_MN = cfg.m82_stellar_MN_b_kpc
    M_200_NFW = cfg.m82_halo_NFW_M_vir
    c_NFW = cfg.m82_halo_NFW_c

    # ----- Disk thermodynamics -----
    T_disk = cfg.m82_disk_T_K
    cs2 = _isothermal_cs_code(T_disk) ** 2  # (km/s)^2 code

    # ----- Disk profile parameters -----
    M_disk_gas = cfg.m82_disk_M_gas
    R_d = cfg.m82_disk_R_scale_kpc
    R_trunc = cfg.m82_disk_R_trunc_kpc

    # Central surface density Sigma_0 from integrating exponential * Heaviside
    # int_0^R_trunc Sigma_0 exp(-R/R_d) * 2 pi R dR = M_disk_gas
    # = Sigma_0 * 2*pi*R_d^2 * (1 - (1 + R_trunc/R_d) * exp(-R_trunc/R_d))
    rt_over_rd = R_trunc / R_d
    norm_factor = 2.0 * jnp.pi * R_d**2 * (1.0 - (1.0 + rt_over_rd) * jnp.exp(-rt_over_rd))
    Sigma_0 = M_disk_gas / norm_factor

    # ----- Compute rho_disk(R, z) via isothermal HSE -----
    # rho(R, z) = rho_0(R) * exp(-(Phi(R,z) - Phi(R,0)) / cs2)
    Phi_xy = _miyamoto_nagai_potential(R, Z, M_disk_MN, a_MN, b_MN) + _nfw_potential(
        R, Z, M_200_NFW, c_NFW
    )
    Phi_R0 = _miyamoto_nagai_potential(
        R, jnp.zeros_like(Z), M_disk_MN, a_MN, b_MN
    ) + _nfw_potential(R, jnp.zeros_like(Z), M_200_NFW, c_NFW)
    z_profile = jnp.exp(-(Phi_xy - Phi_R0) / cs2)

    # Midplane density: Sigma(R) = rho_0(R) * 2 * H_eff, where
    #   H_eff = int_-inf^inf exp(-(Phi-Phi_0)/cs2) dz / 2
    # We approximate using the local cell column for simplicity.
    # Build sigma_R first, then convert to rho_0(R) by dividing by H_eff.
    # Use a simple scale-height approximation H_iso = cs / Omega_z, where
    # Omega_z^2 ~ -d2Phi/dz2 at z=0 ~ G M / (a+b)^3 (MN at z=0 small-z limit).
    # MN vertical-oscillation frequency at (R=0, z=0):
    #   omega_z^2 = d2Phi/dz2 = G*M / (b * (a+b)^2)
    from hydro.external_gravity import G_CODE

    omega_z2 = G_CODE * M_disk_MN / (b_MN * (a_MN + b_MN) ** 2) + 1e-30
    H_iso = jnp.sqrt(cs2 / omega_z2)  # scale height in kpc

    # Surface density with truncation
    Sigma_R = Sigma_0 * jnp.exp(-R / R_d) * (R < R_trunc).astype(R.dtype)
    rho_0_R = Sigma_R / (jnp.sqrt(2.0 * jnp.pi) * H_iso)

    rho_disk = rho_0_R * z_profile

    # ----- Ambient hot CGM -----
    n_amb = cfg.cc85_n_ambient_cgs
    T_amb = cfg.cc85_T_ambient_K
    mu = cfg.cc85_mu
    rho_amb_cgs = n_amb * mu * _PROTON_MASS
    p_amb_cgs = n_amb * _BOLTZMANN * T_amb
    rho_amb = rho_amb_cgs / cfg.units.UnitDensity_in_cgs
    p_amb = p_amb_cgs / cfg.units.UnitPressure_in_cgs

    # Final density: max(disk, ambient) -- disk wherever > ambient, CGM otherwise
    rho = jnp.maximum(rho_disk, rho_amb)
    p_disk = rho_disk * cs2  # isothermal: p = cs2 * rho
    p = jnp.where(rho_disk > rho_amb, p_disk, p_amb)

    # ----- Rotation: v_phi^2 = R * |g_R| at z=0 (use midplane circular vel) -----
    v_circ = vcirc_at_midplane(R, M_disk_MN, a_MN, b_MN, M_200_NFW, c_NFW)
    in_disk = (rho_disk > rho_amb).astype(rho.dtype)
    # phi-hat = (-Y/R, X/R, 0) ; vx = -v_circ * Y / R, vy = v_circ * X / R
    R_safe = jnp.where(R > 0, R, 1.0)
    vx = -v_circ * Y / R_safe * in_disk
    vy = v_circ * X / R_safe * in_disk
    vz = jnp.zeros_like(rho)

    mom_x = rho * vx
    mom_y = rho * vy
    mom_z = rho * vz
    E = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)

    return jnp.stack([rho, mom_x, mom_y, mom_z, E], axis=0)
