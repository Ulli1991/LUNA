"""Conservative U = [rho, rho*v..., E] <-> primitive W = [rho, v..., P] conversion."""

from functools import partial

import jax.numpy as jnp
from jax import jit


def _kinetic_energy(rho, vs):
    """0.5·ρ·|v|², summed x, y, z in order."""
    return 0.5 * rho * sum(v * v for v in vs)


@partial(jit, static_argnums=(1,))
def cons_to_prim(U, gamma):
    """Conservative [rho, rho*v..., E] -> primitive [rho, v..., P]; dimension from ``U.ndim``."""
    ndim = U.ndim - 1
    rho = U[0]
    inv_rho = 1.0 / rho
    gm1 = gamma - 1.0
    vs = [U[i] * inv_rho for i in range(1, ndim + 1)]
    E = U[ndim + 1]
    ke = _kinetic_energy(rho, vs)
    P = gm1 * (E - ke)
    return jnp.stack([rho, *vs, P])


def cons_to_prim_floored(U, gamma, smallr, smallc):
    """cons_to_prim with RAMSES floors rho >= smallr, e_int >= smalle rho (umuscl.f90:1813, 1847); U is not changed."""
    ndim = U.ndim - 1
    rho = jnp.maximum(U[0], smallr)
    inv_rho = 1.0 / rho
    vs = [U[i] * inv_rho for i in range(1, ndim + 1)]
    smalle = smallc ** 2 / gamma / (gamma - 1.0)
    e = jnp.maximum(U[ndim + 1] - _kinetic_energy(rho, vs), smalle * rho)
    return jnp.stack([rho, *vs, (gamma - 1.0) * e])


@partial(jit, static_argnums=(1,))
def prim_to_cons(W, gamma):
    """Primitive [rho, v..., P] -> conservative [rho, rho*v..., E]; dimension from ``W.ndim``."""
    ndim = W.ndim - 1
    rho = W[0]
    gm1 = gamma - 1.0
    vs = [W[i] for i in range(1, ndim + 1)]
    P = W[ndim + 1]
    momenta = [rho * v for v in vs]
    ke = _kinetic_energy(rho, vs)
    E = P / gm1 + ke
    return jnp.stack([rho, *momenta, E])
