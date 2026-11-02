"""Per-block helpers inside shard_map over P(None, 'x', 'y', 'z'): periodic shifts, derived fields."""
import jax
import jax.numpy as jnp

AXES = ("x", "y", "z")


def _block_ahead(x, name, n_dev, q):
    """The block of the device q places ahead along mesh axis `name` (periodic)."""
    q %= n_dev
    if q == 0:
        return x
    return jax.lax.ppermute(x, name, [(j, (j - q) % n_dev) for j in range(n_dev)])


def roll_ahead(x, d, lag, mesh_shape, lead=0):
    """x at global index i + lag along spatial axis d (array axis lead + d), periodic, any lag >= 0."""
    ax = lead + d
    n = x.shape[ax]
    n_dev = mesh_shape[d]
    q, r = divmod(int(lag) % (n * n_dev), n)
    a = _block_ahead(x, AXES[d], n_dev, q)
    if r == 0:
        return a
    b = _block_ahead(x, AXES[d], n_dev, q + 1)
    return jnp.concatenate([jax.lax.slice_in_dim(a, r, n, axis=ax),
                            jax.lax.slice_in_dim(b, 0, r, axis=ax)], axis=ax)


def ddx(f, d, dx, mesh_shape, n_global, lead=0):
    """Second-order central difference of f along spatial axis d."""
    return (roll_ahead(f, d, 1, mesh_shape, lead)
            - roll_ahead(f, d, n_global - 1, mesh_shape, lead)) / (2.0 * dx)


def primitives(U, gamma, isothermal=False):
    """(rho, v (3, ...), sound speed) of a local block of U = (rho, rho v, E); isothermal: sqrt(p / rho)."""
    rho = U[0]
    v = U[1:4] / rho
    e = U[-1] - 0.5 * rho * jnp.sum(v * v, axis=0)
    cs = jnp.sqrt(jnp.maximum((1.0 if isothermal else gamma) * (gamma - 1.0) * e / rho, 0.0))
    return rho, v, cs


def vorticity_divergence(v, dx, mesh_shape, n_global):
    """(omega (3, ...), div v) by central differences."""
    d = [[ddx(v[i], j, dx, mesh_shape, n_global) for j in range(3)] for i in range(3)]
    omega = jnp.stack([d[2][1] - d[1][2], d[0][2] - d[2][0], d[1][0] - d[0][1]])
    return omega, d[0][0] + d[1][1] + d[2][2]
