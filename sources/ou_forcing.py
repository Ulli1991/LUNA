"""Runner-side wiring for the Ornstein-Uhlenbeck driven-turbulence source.

Builds the OU driver and the per-step apply function. The (X, Y, Z)
coordinate arrays are constructed multi-process-aware (local slice + KH-IC
assembly) and passed as JIT ARGUMENTS to the apply (not closures) so XLA
cannot inline them as MLIR constants for multi-process sharded arrays.
"""

import jax
import jax.numpy as jnp
import numpy as _np
from jax.sharding import NamedSharding, PartitionSpec

from hydro.turbulence import OUDriver


def setup_ou_forcing(cfg, sharding, is_main_process: bool = True):
    """Return dict {"a": a_state, "key": key, "apply": fn(U, a, key, dt)} or None."""
    if cfg.ou_config is None:
        return None

    driver = OUDriver(cfg.ou_config)
    dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
    global_shape = (cfg.Nx, cfg.Ny, cfg.Nz) if cfg.SPATIAL_DIMENSION == 3 else (cfg.Nx, cfg.Ny)
    coord_spec = PartitionSpec(*sharding.spec[1:])
    coord_sharding = NamedSharding(sharding.mesh, coord_spec)

    if jax.process_count() > 1:
        from utils.distributed import compute_local_slice

        xs, ys, zs = compute_local_slice(
            cfg.Nx,
            cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 1,
            sharding.mesh,
            jax.process_index(),
            cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else 1,
        )
        x_loc = (_np.arange(xs[0], xs[1], dtype=_np.float32) + 0.5) * dx
        y_loc = (_np.arange(ys[0], ys[1], dtype=_np.float32) + 0.5) * dy
        z_loc = (_np.arange(zs[0], zs[1], dtype=_np.float32) + 0.5) * dz
        if cfg.SPATIAL_DIMENSION == 3:
            Xl, Yl, Zl = _np.meshgrid(x_loc, y_loc, z_loc, indexing="ij")
        else:
            Xl, Yl = _np.meshgrid(x_loc, y_loc, indexing="ij")
            Zl = _np.zeros_like(Xl)
        local_dev = jax.local_devices()[0]

        def _assemble(local_arr):
            d = jax.device_put(_np.asarray(local_arr), local_dev)
            return jax.make_array_from_single_device_arrays(
                shape=global_shape, sharding=coord_sharding, arrays=[d]
            )

        X = _assemble(Xl)
        Y = _assemble(Yl)
        Z = _assemble(Zl)
    else:
        x_full = (jnp.arange(cfg.Nx, dtype=jnp.float32) + 0.5) * dx
        y_full = (jnp.arange(cfg.Ny, dtype=jnp.float32) + 0.5) * dy
        z_full = (jnp.arange(cfg.Nz, dtype=jnp.float32) + 0.5) * dz
        if cfg.SPATIAL_DIMENSION == 3:
            X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
        else:
            X, Y = jnp.meshgrid(x_full, y_full, indexing="ij")
            Z = jnp.zeros_like(X)
        X = jax.device_put(X, coord_sharding)
        Y = jax.device_put(Y, coord_sharding)
        Z = jax.device_put(Z, coord_sharding)

    key = jax.random.PRNGKey(int(cfg.ou_seed))
    key, sub = jax.random.split(key)
    a_state = driver.init_state(sub)

    @jax.jit
    def _apply_inner(U, a, key, dt, X_in, Y_in, Z_in):
        new_key, sub = jax.random.split(key)
        a_new = driver.evolve(a, dt, sub)
        U_new = driver.apply(U, a_new, dt, X_in, Y_in, Z_in)
        return U_new, a_new, new_key

    def _apply(U, a, key, dt):
        return _apply_inner(U, a, key, dt, X, Y, Z)

    if is_main_process:
        print(
            f"✓ OU driver active: {driver.n_modes} modes "
            f"(k in [{cfg.ou_config.k_min}, {cfg.ou_config.k_max}]), "
            f"zeta={cfg.ou_config.solenoidal_weight}, "
            f"T_ac={cfg.ou_config.autocorrelation_time}, "
            f"rms_a={cfg.ou_config.rms_acceleration}"
        )

    return {"a": a_state, "key": key, "apply": _apply}
