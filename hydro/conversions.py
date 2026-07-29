"""Conversion between conservative U = [rho, rho*v..., E] and primitive
W = [rho, v..., P] variables (E = total energy density, P = pressure)."""

from functools import partial

import jax.numpy as jnp
from jax import jit


def _kinetic_energy(rho, vs):
    """0.5·ρ·|v|² with the velocity-squared terms accumulated as a
    sequential sum (v0·v0 + v1·v1 + ...) so the FP evaluation order is
    fixed. The 1D case keeps its historical association 0.5·ρ·v·v."""
    if len(vs) == 1:
        return 0.5 * rho * vs[0] * vs[0]
    v2 = vs[0] * vs[0]
    for v in vs[1:]:
        v2 = v2 + v * v
    return 0.5 * rho * v2


@partial(jit, static_argnums=(1,))
def cons_to_prim(U, gamma):
    """Conservative [rho, rho*v..., E] -> primitive [rho, v..., P];
    same shape, dimensionality from ``U.ndim``. ``gamma`` is static."""
    ndim = U.ndim - 1
    rho = U[0]
    inv_rho = 1.0 / rho
    gm1 = gamma - 1.0
    vs = [U[i] * inv_rho for i in range(1, ndim + 1)]
    E = U[ndim + 1]
    ke = _kinetic_energy(rho, vs)
    P = gm1 * (E - ke)
    return jnp.stack([rho, *vs, P])


@partial(jit, static_argnums=(1,))
def prim_to_cons(W, gamma):
    """Primitive [rho, v..., P] -> conservative [rho, rho*v..., E];
    same shape, dimensionality from ``W.ndim``. ``gamma`` is static."""
    ndim = W.ndim - 1
    rho = W[0]
    gm1 = gamma - 1.0
    vs = [W[i] for i in range(1, ndim + 1)]
    P = W[ndim + 1]
    momenta = [rho * v for v in vs]
    ke = _kinetic_energy(rho, vs)
    E = P / gm1 + ke
    return jnp.stack([rho, *momenta, E])
