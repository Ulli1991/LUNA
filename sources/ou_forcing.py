"""Runner-side wiring for the Ornstein-Uhlenbeck driven-turbulence source.

Builds the OU driver and the per-step apply function. The (X, Y, Z)
coordinate arrays are constructed multi-process-aware (local slice + KH-IC
assembly) and passed as JIT ARGUMENTS to the apply (not closures) so XLA
cannot inline them as MLIR constants for multi-process sharded arrays.
"""

import jax
import jax.numpy as jnp
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

    def _coords():
        # Rebuild the coordinate meshgrids as TRACED values (cheap arange +
        # meshgrid, constant-folded by XLA), sharded via with_sharding_constraint.
        # Recomputing — rather than closing over precomputed sharded X/Y/Z — is
        # REQUIRED for reverse-mode AD under multi-process: a captured concrete
        # sharded array becomes a non-addressable MLIR constant when the loss is
        # jitted ("Closing over jax.Array that spans non-addressable devices").
        x_full = (jnp.arange(cfg.Nx, dtype=jnp.float32) + 0.5) * dx
        y_full = (jnp.arange(cfg.Ny, dtype=jnp.float32) + 0.5) * dy
        z_full = (jnp.arange(cfg.Nz, dtype=jnp.float32) + 0.5) * dz
        if cfg.SPATIAL_DIMENSION == 3:
            X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
        else:
            X, Y = jnp.meshgrid(x_full, y_full, indexing="ij")
            Z = jnp.zeros_like(X)
        X = jax.lax.with_sharding_constraint(X, coord_sharding)
        Y = jax.lax.with_sharding_constraint(Y, coord_sharding)
        Z = jax.lax.with_sharding_constraint(Z, coord_sharding)
        return X, Y, Z

    key = jax.random.PRNGKey(int(cfg.ou_seed))
    key, sub = jax.random.split(key)
    a_state = driver.init_state(sub)

    @jax.jit
    def _apply_inner(U, a, key, dt):
        new_key, sub = jax.random.split(key)
        a_new = driver.evolve(a, dt, sub)
        X, Y, Z = _coords()
        U_new = driver.apply(U, a_new, dt, X, Y, Z)
        return U_new, a_new, new_key

    def _apply(U, a, key, dt):
        return _apply_inner(U, a, key, dt)

    if is_main_process:
        print(
            f"✓ OU driver active: {driver.n_modes} modes "
            f"(k in [{cfg.ou_config.k_min}, {cfg.ou_config.k_max}]), "
            f"zeta={cfg.ou_config.solenoidal_weight}, "
            f"T_ac={cfg.ou_config.autocorrelation_time}, "
            f"rms_a={cfg.ou_config.rms_acceleration}"
        )

    return {"a": a_state, "key": key, "apply": _apply}
