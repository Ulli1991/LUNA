"""Ornstein-Uhlenbeck driven-turbulence source: the OU driver and its evolve/kick pair."""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from luna.hydro.turbulence import OUDriver


def setup_ou_forcing(cfg, sharding, is_main_process: bool = True):
    """Return {"a", "key", "evolve": fn(a, key, dt) -> (a, key), "kick": fn(U, a, h) -> U, and the unit-state
    form for forcing_theta: "driver", "b", "unit_evolve"} or None."""
    if cfg.ou_config is None:
        return None

    # FP32 forcing, but state precision under autodiff: FP32 rounds per shard layout (docs/AUTODIFF.md).
    dtype = float if getattr(cfg, "autodiff_mode", False) else jnp.float32
    driver = OUDriver(cfg.ou_config, dtype=dtype)
    dx, dy, dz = cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz
    ax_x, ax_y, ax_z = (tuple(sharding.spec[1:]) + (None,))[:3]   # 2D: no z axis
    specs = {"x": PartitionSpec(None, None, ax_x), "yz": PartitionSpec(None, ax_y, ax_z),
             "xyz": PartitionSpec(None, ax_x, ax_y, ax_z)}

    def _pin(arr, kind):
        return jax.lax.with_sharding_constraint(arr, NamedSharding(sharding.mesh, specs[kind]))

    def _coords():
        # Traced, not closed over: a captured multi-process sharded array breaks reverse-mode AD.
        x = (jnp.arange(cfg.Nx, dtype=dtype) + 0.5) * dx
        y = (jnp.arange(cfg.Ny, dtype=dtype) + 0.5) * dy
        z = ((jnp.arange(cfg.Nz, dtype=dtype) + 0.5) * dz if cfg.SPATIAL_DIMENSION == 3
             else jnp.zeros(1, dtype=dtype))
        return x, y, z

    key = jax.random.PRNGKey(int(cfg.ou_seed))
    key, sub = jax.random.split(key)
    a_state = driver.init_state(sub)
    b_state = driver.unit_state(sub)

    # Separate so the runner can Strang-split the kick around hydro with ONE draw per step.
    @jax.jit
    def _evolve(a, key, dt):
        """Advance the OU state one step (one random draw)."""
        new_key, sub = jax.random.split(key)
        return driver.evolve(a, dt, sub), new_key

    @jax.jit
    def _unit_evolve(b, key, dt):
        """_evolve on the unit state (forcing_theta): the same split, the same draw."""
        new_key, sub = jax.random.split(key)
        return driver.unit_evolve(b, dt, sub), new_key

    @jax.jit
    def _kick(U, a, h):
        """Apply the force from mode amplitudes `a` over a time `h`."""
        x, y, z = _coords()
        return driver.apply(U, a, h, x, y, z, _pin)

    # The kick follows hydro_kernel: Pallas builds the force per tile and never stores it (docs/history/CODE_NOTES.md).
    from luna.hydro import muscl_hancock_pallas as mhp
    why = (None if cfg.SPATIAL_DIMENSION == 3 and not getattr(cfg, "autodiff_mode", False)
           else "the OU kick outside 3-D or under autodiff_mode (the XLA kick runs there)")
    kernel = "XLA"
    if mhp.select(getattr(cfg, "hydro_kernel", "xla"), why):
        from luna.hydro.turbulence_pallas import make_kick
        kernel = "Pallas"
        _kick = make_kick(driver, sharding.mesh, sharding.spec, (cfg.Nx, cfg.Ny, cfg.Nz), (dx, dy, dz))

    if is_main_process:
        print(
            f"✓ OU driver active ({kernel} kick): "
            f"{driver.n_modes} modes "
            f"(k in [{cfg.ou_config.k_min}, {cfg.ou_config.k_max}]), "
            f"zeta={cfg.ou_config.solenoidal_weight}, "
            f"T_ac={cfg.ou_config.autocorrelation_time}, "
            f"rms_a={cfg.ou_config.rms_acceleration}"
        )

    return {"a": a_state, "key": key, "evolve": _evolve, "kick": _kick,
            "driver": driver, "b": b_state, "unit_evolve": _unit_evolve}
