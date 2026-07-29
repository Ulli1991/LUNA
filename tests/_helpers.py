"""Shared builders for the LUNA test suite: AMR configs, blob/uniform ICs,
and the short-run production-path driver used by the regression tests."""

import numpy as np
import jax.numpy as jnp

GAMMA = 5.0 / 3.0


def cfg3d(n, bs=16, **kw):
    """3D AMRConfig on the unit box: (n,)*3 base grid, (bs,)*3 blocks."""
    from amr.config import AMRConfig
    return AMRConfig(ndim=3, box=(1.0,) * 3, base_grid=(n,) * 3,
                     block_shape=(bs,) * 3, **kw)


def blob_U(n, amp=0.5, sigma=0.09, p_amp=0.3, vx_amp=0.0, sigma2=None, E0=None):
    """Gaussian-blob conserved IC (5, n, n, n) on the unit box.

    rho = 1 + amp*g, p = 1 + p_amp*g with g = exp(-r^2 / (2 sigma^2))
    (or exp(-r^2 / sigma2) when sigma2 is given), vx = vx_amp*sin(2 pi x).
    E0, if given, overrides the total energy with that uniform value.
    """
    x = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    g = np.exp(-r2 / (2 * sigma ** 2 if sigma2 is None else sigma2))
    rho = 1.0 + amp * g
    p = 1.0 + p_amp * g
    u = vx_amp * np.sin(2 * np.pi * X)
    U = np.zeros((5, n, n, n))
    U[0] = rho
    U[1] = rho * u
    U[4] = E0 if E0 is not None else p / (GAMMA - 1.0) + 0.5 * rho * u * u
    return jnp.asarray(U)


def uniform_U(rho):
    """3D conserved state from a density field: uniform pressure, v = 0."""
    E = np.ones_like(rho) / (GAMMA - 1.0)
    z = np.zeros_like(rho)
    return jnp.asarray(np.stack([rho, z, z, z, E], axis=0))


def _build_cfg_for_test(**overrides):
    """Construct a Config with quiet/fast test defaults; allow overrides."""
    from template_config import Config

    cfg = Config(**overrides)
    cfg.production_mode = True
    cfg.PLOT_AFTER_SIM = False
    cfg.enable_diagnostics = True
    cfg.enable_checkpointing = False
    return cfg


def _make_step_fns(cfg):
    """Return (compute_dt_fn, step_fn) for a Config — same shard_map path as
    production, on a single-device mesh."""
    import jax
    from jax.sharding import Mesh

    from time_integration.integrators import create_muscl_hancock_modular_step
    from utils.grid_params import GridParams
    from utils.halo_exchange_shardmap import make_halo_shardmap

    grid = GridParams.from_config(cfg)
    sdim = cfg.SPATIAL_DIMENSION
    names = ("x", "y", "z")[:sdim]
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * sdim), names)
    pad = make_halo_shardmap(mesh, sdim)

    return create_muscl_hancock_modular_step(
        pad,
        cfg.gamma,
        cfg.ORDER,
        cfg.LIMITER,
        grid.dx,
        grid.dy,
        grid.dz,
        grid.inv_dx,
        grid.inv_dy,
        grid.inv_dz,
        cfg.BOUNDARY,
        cfg.SPATIAL_DIMENSION,
        cfg.CFL,
        sharding=None,
        riemann_solver=getattr(cfg, "riemann_solver", "hllc"),
        use_shardmap=True,
        mesh=mesh,
    )


def _run_short(cfg):
    """Drive a short simulation (no main.py machinery), return final state."""
    from ic_generation.ic import initial_conditions

    U = initial_conditions(cfg)
    compute_dt_fn, step_fn = _make_step_fns(cfg)
    t = 0.0
    while t < cfg.t_end:
        dt = float(compute_dt_fn(U))
        dt = min(dt, cfg.t_end - t)
        U = step_fn(U, dt)
        t += dt
        if dt < 1e-15:
            break
    return np.asarray(U)
