"""Safeguarded temperature recovery from specific internal energy, on a fixed composition branch."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from luna.eos.helmholtz import evaluate, _require_fp64


class TemperatureResult(NamedTuple):
    temperature: jax.Array
    converged: jax.Array
    iterations: jax.Array
    energy_residual: jax.Array  # e(T) - target, erg/g; diagnostic even after a stalled solve
    relative_correction: jax.Array  # |e(T) - target| / (T cv)
    relative_energy_resolution: jax.Array  # eps |e| / (T cv), a conditioning diagnostic


def temperature_from_energy(
    table,
    rho,
    energy,
    mean_mass,
    mean_charge,
    *,
    temperature_guess=None,
    temperature_bounds=None,
    coulomb=True,
    temperature_rtol=1e-8,
    energy_rtol=1e-12,
    max_iterations=80,
):
    """Batched Newton in log T with bisection; failed temperatures are NaN, never a floored answer."""
    _require_fp64()
    if (
        not np.isfinite(temperature_rtol)
        or not np.isfinite(energy_rtol)
        or temperature_rtol <= 0
        or energy_rtol <= 0
        or not isinstance(max_iterations, (int, np.integer))
        or max_iterations < 0
    ):
        raise ValueError("positive tolerances and nonnegative max_iterations required")
    lower, upper = table.temperature[0], table.temperature[-1]
    if temperature_bounds is not None:
        lower, upper = temperature_bounds
    if temperature_guess is None:
        temperature_guess = jnp.sqrt(lower * upper)
    rho, energy, mean_mass, mean_charge, guess, lower, upper = jnp.broadcast_arrays(
        *[
            jnp.asarray(v, jnp.float64)
            for v in (rho, energy, mean_mass, mean_charge, temperature_guess, lower, upper)
        ]
    )

    def thermo(temperature):
        return evaluate(table, rho, temperature, mean_mass, mean_charge, coulomb=coulomb)

    cold, hot = thermo(lower), thermo(upper)
    eligible = (
        cold.valid
        & hot.valid
        & jnp.isfinite(energy)
        & jnp.isfinite(guess)
        & (guess > 0)
        & (lower < upper)
        & (cold.energy <= hot.energy)
        & (energy >= cold.energy)
        & (energy <= hot.energy)
    )
    temperature = jnp.clip(guess, lower, upper)
    temperature = jnp.where(energy == cold.energy, lower, temperature)
    temperature = jnp.where(energy == hot.energy, upper, temperature)
    state = thermo(temperature)

    def errors(temperature, state):
        residual = state.energy - energy
        correction = jnp.abs(residual) / (temperature * state.cv)
        energy_scale = jnp.maximum(jnp.abs(energy), jnp.abs(state.energy))
        converged = (
            eligible
            & state.valid
            & (correction <= temperature_rtol)
            & (jnp.abs(residual) <= energy_rtol * energy_scale)
        )
        return residual, correction, converged

    residual, correction, converged = errors(temperature, state)
    count = jnp.zeros_like(energy, jnp.int32)
    carry = (temperature, jnp.log(lower), jnp.log(upper), state, converged, count)

    def pending(carry):
        _, _, _, state, converged, count = carry
        return jnp.any(eligible & state.valid & ~converged & (count < max_iterations))

    def advance(carry):
        temperature, log_lower, log_upper, state, converged, count = carry
        active = eligible & state.valid & ~converged & (count < max_iterations)
        residual = state.energy - energy
        log_temperature = jnp.log(temperature)
        log_lower = jnp.where(active & (residual < 0), log_temperature, log_lower)
        log_upper = jnp.where(active & (residual > 0), log_temperature, log_upper)
        proposal = log_temperature - residual / (temperature * state.cv)
        midpoint = 0.5 * (log_lower + log_upper)
        use_newton = (
            jnp.isfinite(proposal)
            & (proposal > log_lower)
            & (proposal < log_upper)
            & (jnp.abs(proposal - log_temperature) < 0.5 * (log_upper - log_lower))
        )
        next_temperature = jnp.clip(
            jnp.exp(jnp.where(use_newton, proposal, midpoint)), lower, upper
        )
        temperature = jnp.where(active, next_temperature, temperature)
        state = thermo(temperature)
        _, _, converged = errors(temperature, state)
        return temperature, log_lower, log_upper, state, converged, count + active.astype(jnp.int32)

    temperature, _, _, state, converged, count = jax.lax.while_loop(pending, advance, carry)
    residual, correction, converged = errors(temperature, state)
    resolution = jnp.finfo(jnp.float64).eps * jnp.abs(energy) / (temperature * state.cv)
    return TemperatureResult(
        jnp.where(converged, temperature, jnp.nan),
        converged,
        count,
        residual,
        correction,
        resolution,
    )


def temperature_from_energy_checked(table, rho, energy, mean_mass, mean_charge, **options):
    """Host boundary: raise if any cell is invalid, unbracketed or unconverged."""
    result = temperature_from_energy(table, rho, energy, mean_mass, mean_charge, **options)
    if not np.asarray(result.converged).all():
        raise ValueError(
            "Helmholtz temperature inversion failed: invalid state, energy bracket or convergence"
        )
    return result
