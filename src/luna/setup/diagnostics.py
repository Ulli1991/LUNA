"""The periodic diagnostic reductions of a gas run."""
from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("mesh",))
def diag_reductions(U, gamma, mesh=None):
    """(min_rho, min_P, max_speed, max_mach, sum_rho, sum_E) in one pass: the
    derived fields are elementwise (shard-local); only the six reductions
    are collectives, explicit shard_map ones on a device mesh."""
    rho = U[0]
    Nvar = U.shape[0]
    vx = U[1] / rho
    vy = U[2] / rho if Nvar >= 4 else 0.0
    vz = U[3] / rho if Nvar >= 5 else 0.0
    speed2 = vx * vx + vy * vy + vz * vz
    KE = 0.5 * rho * speed2
    P = (gamma - 1.0) * (U[-1] - KE)
    cs2 = gamma * P / rho
    mach2 = speed2 / cs2
    if mesh is not None:
        from luna.utils.mesh_reduce import grid_reduce
        return (grid_reduce(mesh, "min", rho), grid_reduce(mesh, "min", P),
                jnp.sqrt(grid_reduce(mesh, "max", speed2)),
                jnp.sqrt(grid_reduce(mesh, "max", mach2)),
                grid_reduce(mesh, "sum", rho), grid_reduce(mesh, "sum", U[-1]))
    return (jnp.min(rho), jnp.min(P), jnp.sqrt(jnp.max(speed2)),
            jnp.sqrt(jnp.max(mach2)), jnp.sum(rho), jnp.sum(U[-1]))
