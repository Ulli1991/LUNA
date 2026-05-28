"""Runner-side wiring for the external static-potential gravity source.

Position-dependent (gx, gy, gz) from Miyamoto-Nagai (stellar disk) + NFW
(dark halo), applied as an operator-split per-step source on U.

Multi-process aware (KH-IC sharding pattern: each rank builds only its local
slice and assembles via make_array_from_single_device_arrays). The JIT'd
update function receives (gx, gy, gz) as ARGUMENTS, not closures, so XLA
does not try to inline the sharded multi-process arrays as MLIR constants.
"""
import jax
import jax.numpy as jnp
import numpy as _np
from jax.sharding import NamedSharding, PartitionSpec

from hydro.external_gravity import combined_acceleration


def setup_external_gravity(cfg, sharding, is_main_process: bool = True):
    """Return {"apply": fn(U, dt) -> U} or None when disabled."""
    if not getattr(cfg, "m82_use_external_gravity", False):
        return None
    if cfg.SPATIAL_DIMENSION != 3:
        raise ValueError("m82_use_external_gravity requires 3D.")

    dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
    cx, cy, cz = 0.5 * cfg.Lx, 0.5 * cfg.Ly, 0.5 * cfg.Lz
    global_shape = (cfg.Nx, cfg.Ny, cfg.Nz)
    coord_sharding = NamedSharding(sharding.mesh, PartitionSpec("x", "y", "z"))

    if jax.process_count() > 1:
        from utils.distributed import compute_local_slice
        xs, ys, zs = compute_local_slice(
            cfg.Nx, cfg.Ny, sharding.mesh, jax.process_index(), cfg.Nz)
        x_loc = (_np.arange(xs[0], xs[1], dtype=_np.float32) + 0.5) * dx - cx
        y_loc = (_np.arange(ys[0], ys[1], dtype=_np.float32) + 0.5) * dy - cy
        z_loc = (_np.arange(zs[0], zs[1], dtype=_np.float32) + 0.5) * dz - cz
        Xl, Yl, Zl = _np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
        gx_loc, gy_loc, gz_loc = combined_acceleration(
            Xl, Yl, Zl,
            cfg.m82_stellar_MN_M, cfg.m82_stellar_MN_a_kpc,
            cfg.m82_stellar_MN_b_kpc,
            cfg.m82_halo_NFW_M_vir, cfg.m82_halo_NFW_c,
        )
        local_dev = jax.local_devices()[0]
        def _assemble(local_arr):
            d = jax.device_put(_np.asarray(local_arr), local_dev)
            return jax.make_array_from_single_device_arrays(
                shape=global_shape, sharding=coord_sharding, arrays=[d])
        gx = _assemble(gx_loc); gy = _assemble(gy_loc); gz = _assemble(gz_loc)
    else:
        x_full = (jnp.arange(cfg.Nx, dtype=jnp.float32) + 0.5) * dx - cx
        y_full = (jnp.arange(cfg.Ny, dtype=jnp.float32) + 0.5) * dy - cy
        z_full = (jnp.arange(cfg.Nz, dtype=jnp.float32) + 0.5) * dz - cz
        X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
        gx, gy, gz = combined_acceleration(
            X, Y, Z,
            cfg.m82_stellar_MN_M, cfg.m82_stellar_MN_a_kpc, cfg.m82_stellar_MN_b_kpc,
            cfg.m82_halo_NFW_M_vir, cfg.m82_halo_NFW_c,
        )
        coord_spec = PartitionSpec(*sharding.spec[1:])
        coord_sharding = NamedSharding(sharding.mesh, coord_spec)
        gx = jax.device_put(gx, coord_sharding)
        gy = jax.device_put(gy, coord_sharding)
        gz = jax.device_put(gz, coord_sharding)

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
        print(f"✓ External M82 gravity: "
              f"MN(M={cfg.m82_stellar_MN_M}, a={cfg.m82_stellar_MN_a_kpc}, "
              f"b={cfg.m82_stellar_MN_b_kpc}) + "
              f"NFW(M200={cfg.m82_halo_NFW_M_vir}, c={cfg.m82_halo_NFW_c})")

    return {"apply": _apply}
