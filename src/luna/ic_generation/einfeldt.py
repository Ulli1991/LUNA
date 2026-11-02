"""Einfeldt rarefaction tests from the Athena suite, named rho-|m|-|n|-E (density, normal and
transverse momentum, total energy per volume), states split at Lx/2, gamma = 1.4:

    1-2-0-3  Einfeldt-Strong      u = -2 | +2, p = 0.4   (Toro's test 2): near vacuum
    1-1-0-5  Einfeldt-Shear       u = -1 | +1, p = 1.8
    1-1-2-5  Einfeldt-Transverse  u = -1 | +1, v = +2 | -2, p = 1.0: needs the transverse velocity,
                                  so it runs in 2D (a strip uniform in y)

Exact solution: luna.hydro.exact_riemann; v rides the contact.
"""
import jax.numpy as jnp

from ._grid import local_range


def _states(m, n, E):
    """(rho, u, v, p) left and right of Athena's rho = 1, m = -+m, n = +-n, E."""
    g1 = 0.4                                     # gamma - 1; the problems are defined at 1.4
    p = g1 * (E - 0.5 * (m * m + n * n))
    return (1.0, -m, n, p), (1.0, m, -n, p)


STRONG = _states(2.0, 0.0, 3.0)
SHEAR = _states(1.0, 0.0, 5.0)
TRANSVERSE = _states(1.0, 2.0, 5.0)


def _check(cfg, name):
    if abs(cfg.gamma - 1.4) > 1e-12:
        raise ValueError(f"{name} is defined with gamma = 1.4; got gamma={cfg.gamma}")


def _one_d(cfg, x_slice, states):
    """U = [rho, rho u, E] with the jump at Lx/2."""
    (rl, ul, _, pl), (rr, ur, _, pr) = states
    x_start, x_end = local_range(cfg.Nx, x_slice)
    left = (jnp.arange(x_start, x_end) + 0.5) * (cfg.Lx / cfg.Nx) < 0.5 * cfg.Lx
    rho, u, p = jnp.where(left, rl, rr), jnp.where(left, ul, ur), jnp.where(left, pl, pr)
    return jnp.stack([rho, rho * u, p / (cfg.gamma - 1.0) + 0.5 * rho * u * u], axis=0)


def einfeldt_strong_rarefaction(cfg, x_slice=None):
    """Athena 1-2-0-3: strong opposing velocities, a near-vacuum centre."""
    _check(cfg, "Einfeldt-Strong")
    return _one_d(cfg, x_slice, STRONG)


def einfeldt_shear_rarefaction(cfg, x_slice=None):
    """Athena 1-1-0-5: moderate opposing velocities."""
    _check(cfg, "Einfeldt-Shear")
    return _one_d(cfg, x_slice, SHEAR)


def einfeldt_transverse_rarefaction(cfg, x_slice=None, y_slice=None):
    """Athena 1-1-2-5 on a 2D strip: U = [rho, rho u, rho v, E], uniform in y."""
    _check(cfg, "Einfeldt-Transverse")
    (rl, ul, vl, pl), (rr, ur, vr, pr) = TRANSVERSE
    x_start, x_end = local_range(cfg.Nx, x_slice)
    y_start, y_end = local_range(cfg.Ny, y_slice)
    left = (jnp.arange(x_start, x_end) + 0.5) * (cfg.Lx / cfg.Nx) < 0.5 * cfg.Lx
    shape = (x_end - x_start, y_end - y_start)
    pick = lambda a, b: jnp.broadcast_to(jnp.where(left, a, b)[:, None], shape)  # noqa: E731
    rho, u, v, p = pick(rl, rr), pick(ul, ur), pick(vl, vr), pick(pl, pr)
    E = p / (cfg.gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return jnp.stack([rho, rho * u, rho * v, E], axis=0)
