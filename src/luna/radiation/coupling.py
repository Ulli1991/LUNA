"""Radiation-gas coupling: transport substeps, radiation force, absorption/emission.

Each piece is exactly conservative cell by cell; lab frame, no O(v/c) terms.
See docs/RADIATION.md.
"""
import jax
import jax.numpy as jnp

from luna.radiation.m1 import E_FLOOR, _as_group_field, m1_damp, m1_exchange, m1_step
from luna.radiation.m2_stage import add_isotropic, is_m2


def substeps(R, dt, dx, *, chat, kappa, n_sub, boundary=0, order=2, pad=None,
             scheme="rk2", closure="m1", params=None, kernel="auto"):
    """Run `n_sub` transport + flux-damping substeps over dt; return (R, dF).

    dF: flux each group lost to the opacity, summed over substeps, (G, ndim, N...).
    """
    h = dt / n_sub
    if closure != "m1":
        from luna.radiation import m2_stage

        def body2(carry, _):
            Rc, acc = carry
            Rc = m2_stage.transport(Rc, h, dx, chat=chat, boundary=boundary, pad=pad,
                                    closure=closure, params=params, kernel=kernel)
            Rd = m2_stage.damp(Rc, h, chat=chat, kappa=kappa)
            return (Rd, acc + chat * (Rc[:, 1:4] - Rd[:, 1:4])), None

        (R, dF), _ = jax.lax.scan(body2, (R, jnp.zeros_like(R[:, 1:4])), None, length=n_sub)
        return R, dF

    def body(carry, _):
        Rc, acc = carry
        Rc = m1_step(Rc, h, dx, chat=chat, boundary=boundary, order=order, pad=pad,
                     scheme=scheme)
        Rd = m1_damp(Rc, h, chat=chat, kappa=kappa)
        return (Rd, acc + (Rc[:, 1:] - Rd[:, 1:])), None

    (R, dF), _ = jax.lax.scan(body, (R, jnp.zeros_like(R[:, 1:])), None,
                              length=n_sub)
    return R, dF


def apply_force(U, R, dF, *, chat, c_true=None):
    """Give the gas the momentum the opacity removed from the flux; return (U, R).

    dp = dF / (c_hat c_true), true c as in RAMSES-RT; the work moves from the
    radiation energy into E_tot.
    """
    c_true = float(chat if not c_true else c_true)
    ndim = dF.shape[1]
    dp = jnp.sum(dF, axis=0) / (chat * c_true)
    rho = U[0]
    p_old = U[1:1 + ndim]
    p_new = p_old + dp
    dkin = jnp.sum(p_new ** 2 - p_old ** 2, axis=0) / (2.0 * rho)
    w = jnp.sum(jnp.abs(dF), axis=1)                      # (G, N...)
    frac = w / jnp.maximum(jnp.sum(w, axis=0), E_FLOOR)   # 0 where nothing moved
    if is_m2(R):                                          # M2: the work is isotropic
        R = add_isotropic(R, -frac * dkin)
    else:
        R = R.at[:, 0].add(-frac * dkin)
    U = U.at[1:1 + ndim].set(p_new).at[-1].add(dkin)
    return U, R


def absorb_emit(U, R, dt, *, chat, kappa_a, source):
    """Exchange energy between radiation and gas internal energy (exact exponential)."""
    if is_m2(R):                                          # M2: exchange on m0
        E = R[:, 0]
        S = _as_group_field(source, R)
        dE = S + (E - S) * jnp.exp(-chat * _as_group_field(kappa_a, R) * dt) - E
        return U.at[-1].add(-jnp.sum(dE, axis=0)), add_isotropic(R, dE)
    R, e = m1_exchange(R, U[-1], dt, chat=chat, kappa_a=kappa_a, source=source)
    return U.at[-1].set(e), R


def total_energy(U, R):
    """Return gas total energy plus radiation energy."""
    return jnp.sum(U[-1]) + jnp.sum(R[:, 0])
