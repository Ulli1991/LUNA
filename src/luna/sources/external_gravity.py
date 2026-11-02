"""External static potentials as an operator-split source: the M82 Miyamoto-Nagai disk + NFW halo
(Arepo units: kpc, 1e10 Msun, km/s; NFW from (M_200, c)), and analytic ones (sin, uniform, point mass)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as _np
from jax.sharding import NamedSharding, PartitionSpec

from luna.units import arepo_unit_system

# Derived from units.py, never a literal, so all modules share one G.
G_CODE = arepo_unit_system().GravityConstant

# Critical density at z=0 in Arepo code units (1e10 Msun / kpc^3).
RHO_CRIT_CODE = 1.4e-8


def _miyamoto_nagai_accel(X, Y, Z, M, a, b):
    """Return (g_x, g_y, g_z) of a Miyamoto-Nagai disk centred at the origin."""
    R2 = X * X + Y * Y
    zb = jnp.sqrt(Z * Z + b * b)
    a_zb = a + zb
    D = R2 + a_zb * a_zb
    D32 = D * jnp.sqrt(D)  # D^(3/2)
    inv_D32 = 1.0 / D32
    g_x = -G_CODE * M * X * inv_D32
    g_y = -G_CODE * M * Y * inv_D32
    g_z = -G_CODE * M * a_zb * Z * inv_D32 / zb
    return g_x, g_y, g_z


def _nfw_params_from_M200_c(M_200, c):
    """Return (r_s, rho_s) given M_200 in 1e10 Msun and concentration c."""
    r_200_cubed = M_200 / (4.0 / 3.0 * jnp.pi * 200.0 * RHO_CRIT_CODE)
    r_200 = r_200_cubed ** (1.0 / 3.0)
    r_s = r_200 / c
    f_c = jnp.log(1.0 + c) - c / (1.0 + c)
    rho_s = M_200 / (4.0 * jnp.pi * r_s**3 * f_c)
    return r_s, rho_s


def _nfw_accel(X, Y, Z, M_200, c):
    """Return (g_x, g_y, g_z) of a spherical NFW halo."""
    r_s, rho_s = _nfw_params_from_M200_c(M_200, c)
    r2 = X * X + Y * Y + Z * Z
    r = jnp.sqrt(r2 + 1e-10)  # softening to avoid r=0
    x = r / r_s
    M_enc = 4.0 * jnp.pi * rho_s * r_s**3 * (jnp.log(1.0 + x) - x / (1.0 + x))
    g_mag = -G_CODE * M_enc / (r2 + 1e-10)
    g_x = g_mag * X / r
    g_y = g_mag * Y / r
    g_z = g_mag * Z / r
    return g_x, g_y, g_z


def combined_acceleration(
    X, Y, Z, MN_M, MN_a, MN_b, NFW_M200, NFW_c, center_x=0.0, center_y=0.0, center_z=0.0
):
    """Return the Miyamoto-Nagai disk + NFW halo acceleration centred on (center_*)."""
    Xc = X - center_x
    Yc = Y - center_y
    Zc = Z - center_z
    gmn = _miyamoto_nagai_accel(Xc, Yc, Zc, MN_M, MN_a, MN_b)
    gnfw = _nfw_accel(Xc, Yc, Zc, NFW_M200, NFW_c)
    return tuple(a + b for a, b in zip(gmn, gnfw))


def vcirc_at_midplane(R, MN_M, MN_a, MN_b, NFW_M200, NFW_c):
    """Return the midplane circular velocity sqrt(R |g_R(R, 0)|) of MN + NFW."""
    a_b = MN_a + MN_b
    D = R * R + a_b * a_b
    g_R_mn = G_CODE * MN_M * R / (D * jnp.sqrt(D))
    r_s, rho_s = _nfw_params_from_M200_c(NFW_M200, NFW_c)
    x = R / r_s
    M_enc = 4.0 * jnp.pi * rho_s * r_s**3 * (jnp.log(1.0 + x) - x / (1.0 + x))
    g_R_nfw = G_CODE * M_enc / (R * R + 1e-10)
    g_R_tot = g_R_mn + g_R_nfw
    return jnp.sqrt(R * g_R_tot)


@jax.jit
def _kick(U, dt, g):
    """Kick momenta U[1:1+d] by rho g dt; energy with the mid-step momentum (g: (d, *grid))."""
    d = g.shape[0]
    m = U[1:1 + d]
    m_new = m + dt * U[0] * g
    return U.at[1:1 + d].set(m_new).at[-1].add(dt * jnp.sum(0.5 * (m + m_new) * g, axis=0))


def analytic_acceleration(cfg, dtype):
    """g = -grad phi at the cell centres, (d, *grid), for cfg.external_potential."""
    d = cfg.SPATIAL_DIMENSION
    shape, box = (cfg.Nx, cfg.Ny, cfg.Nz)[:d], (cfg.Lx, cfg.Ly, cfg.Lz)[:d]
    X = _np.meshgrid(*[(_np.arange(n, dtype=_np.float64) + 0.5) * (b / n) for n, b in zip(shape, box)], indexing="ij")
    g = _np.zeros((d,) + shape)
    kind, ax = cfg.external_potential, int(cfg.external_axis)
    if kind == "sin":                       # phi = phi0 sin(2 pi x_ax / lambda)
        k = 2.0 * _np.pi / cfg.external_wavelength
        g[ax] = -cfg.external_phi0 * k * _np.cos(k * X[ax])
    elif kind == "uniform":                 # g = external_g along +axis
        g[ax] = cfg.external_g
    else:                                   # point mass: phi = -GM / sqrt(r^2 + eps^2)
        c = cfg.external_center or tuple(0.5 * b for b in box)
        r = _np.stack([x - ci for x, ci in zip(X, c)])
        g = -cfg.external_GM * r / (_np.sum(r * r, axis=0) + cfg.external_softening ** 2) ** 1.5
    return g.astype(dtype)


def setup_external_gravity(cfg, sharding, is_main_process: bool = True):
    """Return {"apply": fn(U, dt) -> U} or None when disabled."""
    fdt = _np.float64 if cfg.enable_x64 else _np.float32
    if cfg.external_potential:
        if jax.process_count() > 1:
            raise NotImplementedError("external_potential: analytic potentials are single-process for now")
        g = jax.device_put(jnp.asarray(analytic_acceleration(cfg, fdt)), NamedSharding(sharding.mesh, sharding.spec))
        if is_main_process:
            print(f"✓ External potential: {cfg.external_potential}")
        return {"apply": lambda U, dt: _kick(U, dt, g)}
    if not cfg.m82_use_external_gravity:
        return None
    if cfg.SPATIAL_DIMENSION != 3:
        raise ValueError("m82_use_external_gravity requires 3D.")

    dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
    cx, cy, cz = 0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz
    global_shape = (cfg.Nx, cfg.Ny, cfg.Nz)
    coord_sharding = NamedSharding(sharding.mesh, PartitionSpec("x", "y", "z"))

    if jax.process_count() > 1:
        from luna.utils.distributed import compute_local_slice

        xs, ys, zs = compute_local_slice(cfg.Nx, cfg.Ny, sharding.mesh, jax.process_index(), cfg.Nz)
        x_loc = (_np.arange(xs[0], xs[1], dtype=fdt) + 0.5) * dx - cx
        y_loc = (_np.arange(ys[0], ys[1], dtype=fdt) + 0.5) * dy - cy
        z_loc = (_np.arange(zs[0], zs[1], dtype=fdt) + 0.5) * dz - cz
        Xl, Yl, Zl = _np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
        gx_loc, gy_loc, gz_loc = combined_acceleration(
            Xl,
            Yl,
            Zl,
            cfg.m82_stellar_MN_M,
            cfg.m82_stellar_MN_a_kpc,
            cfg.m82_stellar_MN_b_kpc,
            cfg.m82_halo_NFW_M_vir,
            cfg.m82_halo_NFW_c,
        )
        local_dev = jax.local_devices()[0]

        def _assemble(local_arr):
            d = jax.device_put(_np.asarray(local_arr), local_dev)
            return jax.make_array_from_single_device_arrays(
                shape=global_shape, sharding=coord_sharding, arrays=[d]
            )

        gx = _assemble(gx_loc)
        gy = _assemble(gy_loc)
        gz = _assemble(gz_loc)
    else:
        x_full = (jnp.arange(cfg.Nx, dtype=fdt) + 0.5) * dx - cx
        y_full = (jnp.arange(cfg.Ny, dtype=fdt) + 0.5) * dy - cy
        z_full = (jnp.arange(cfg.Nz, dtype=fdt) + 0.5) * dz - cz
        X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
        gx, gy, gz = combined_acceleration(
            X,
            Y,
            Z,
            cfg.m82_stellar_MN_M,
            cfg.m82_stellar_MN_a_kpc,
            cfg.m82_stellar_MN_b_kpc,
            cfg.m82_halo_NFW_M_vir,
            cfg.m82_halo_NFW_c,
        )
        coord_spec = PartitionSpec(*sharding.spec[1:])
        coord_sharding = NamedSharding(sharding.mesh, coord_spec)
        gx = jax.device_put(gx, coord_sharding)
        gy = jax.device_put(gy, coord_sharding)
        gz = jax.device_put(gz, coord_sharding)

    # g is a jit argument, not a closure: XLA would inline a sharded array as an MLIR constant.
    @jax.jit
    def _apply_inner(U, dt, gx_in, gy_in, gz_in):
        rho = U[0]
        mx, my, mz = U[1], U[2], U[3]
        new_mx = mx + dt * rho * gx_in
        new_my = my + dt * rho * gy_in
        new_mz = mz + dt * rho * gz_in
        mx_mid = 0.5 * (mx + new_mx)
        my_mid = 0.5 * (my + new_my)
        mz_mid = 0.5 * (mz + new_mz)
        new_E = U[-1] + dt * (mx_mid * gx_in + my_mid * gy_in + mz_mid * gz_in)
        U_new = U.at[1].set(new_mx)
        U_new = U_new.at[2].set(new_my)
        U_new = U_new.at[3].set(new_mz)
        U_new = U_new.at[-1].set(new_E)
        return U_new

    def _apply(U, dt):
        return _apply_inner(U, dt, gx, gy, gz)

    if is_main_process:
        print(
            f"✓ External M82 gravity: "
            f"MN(M={cfg.m82_stellar_MN_M}, a={cfg.m82_stellar_MN_a_kpc}, "
            f"b={cfg.m82_stellar_MN_b_kpc}) + "
            f"NFW(M200={cfg.m82_halo_NFW_M_vir}, c={cfg.m82_halo_NFW_c})"
        )

    return {"apply": _apply}
