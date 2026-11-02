"""One-dimensional HLL fluxes using pressure and sound speed supplied by an EOS."""

from typing import NamedTuple

import jax.numpy as jnp


class FaceFlux(NamedTuple):
    flux: object
    signal_speed: object
    valid: object


def hll_eos_1d(left, right, pressure_left, pressure_right, sound_left, sound_right):
    """States are [rho, rho*u, optional rho*X..., total energy], with variables on axis zero."""
    left, right = jnp.asarray(left), jnp.asarray(right)
    if left.shape != right.shape or left.ndim < 1 or left.shape[0] < 3:
        raise ValueError("matching 1D conservative states need at least three variables")
    thermodynamics = [
        jnp.broadcast_to(jnp.asarray(x), left.shape[1:])
        for x in (pressure_left, pressure_right, sound_left, sound_right)
    ]
    valid = (
        jnp.all(jnp.isfinite(left) & jnp.isfinite(right), axis=0) & (left[0] > 0) & (right[0] > 0)
    )
    for value in thermodynamics:
        valid = valid & jnp.isfinite(value) & (value > 0)
    safe = jnp.zeros_like(left).at[0].set(1).at[-1].set(1)
    left, right = [jnp.where(valid[None], state, safe) for state in (left, right)]
    pressure_left, pressure_right, sound_left, sound_right = [
        jnp.where(valid, x, 1.0) for x in thermodynamics
    ]
    velocity_left, velocity_right = left[1] / left[0], right[1] / right[0]
    slow = jnp.minimum(velocity_left - sound_left, velocity_right - sound_right)
    fast = jnp.maximum(velocity_left + sound_left, velocity_right + sound_right)
    flux_left = (
        (left * velocity_left).at[1].add(pressure_left).at[-1].add(pressure_left * velocity_left)
    )
    flux_right = (
        (right * velocity_right)
        .at[1]
        .add(pressure_right)
        .at[-1]
        .add(pressure_right * velocity_right)
    )
    middle = (fast * flux_left - slow * flux_right + slow * fast * (right - left)) / (fast - slow)
    flux = jnp.where(slow[None] >= 0, flux_left, jnp.where(fast[None] <= 0, flux_right, middle))
    valid = valid & jnp.all(jnp.isfinite(flux), axis=0) & jnp.isfinite(slow) & jnp.isfinite(fast)
    return FaceFlux(
        jnp.where(valid[None], flux, jnp.nan),
        jnp.where(valid, jnp.maximum(abs(slow), abs(fast)), jnp.nan),
        valid,
    )
