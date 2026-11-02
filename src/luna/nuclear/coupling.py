"""Standalone Strang coupling of aprox13 burns to the Helmholtz MUSCL–Hancock prototype."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from luna.ode.rodas3 import Options, integrate
from luna.hydro.stellar_1d import make_solver
from luna.nuclear.burn import BurnParams, make_network


class ReactiveStep(NamedTuple):
    state: object
    temperature: object
    signal_speed: object
    valid: object
    burn_steps: object
    burn_rejected: object
    slope_reduced: object
    flux_fraction: object
    burn_suppressed: object
    boundary_flux: object
    burn_skipped: object


def shock_mask(state, pressure, *, boundary="outflow", threshold=2 / 3, acceleration=0.0, dx=1.0):
    """Flag compression with a cell-scale pressure jump exceeding the hydrostatic gradient."""
    if boundary not in ("outflow", "periodic", "reflecting"):
        raise ValueError("unsupported shock boundary")
    mode = "wrap" if boundary == "periodic" else "edge"
    velocity = jnp.pad(state[1] / state[0], (1, 1), mode=mode)
    if boundary == "reflecting":
        velocity = velocity.at[0].set(-velocity[1])
    padded = jnp.pad(pressure, (1, 1), mode=mode)
    jump = 0.5 * (padded[2:] - padded[:-2]) - state[0] * acceleration * dx
    return (velocity[2:] < velocity[:-2]) & (abs(jump) > threshold * pressure)


def make_step(
    table,
    *,
    screening,
    neutrinos,
    boundary="outflow",
    coulomb=True,
    rtol=1e-6,
    burn_limit=0.1,
    order=2,
    suppress_shock_burning=True,
    shock_threshold=2 / 3,
    geometry="planar",
    self_gravity=False,
    equilibrium=None,
    burn_min_temperature=0.0,
    burn_min_density=0.0,
):
    """Build a CPU-testable split step; excessive composition or temperature changes refuse."""
    if not np.isfinite(burn_limit) or not 0 < burn_limit <= 1:
        raise ValueError("burn_limit must be in (0,1]")
    if not np.isfinite(shock_threshold) or shock_threshold <= 0:
        raise ValueError("shock_threshold must be positive and finite")
    if not np.isfinite([burn_min_temperature, burn_min_density]).all() or min(
        burn_min_temperature, burn_min_density
    ) < 0:
        raise ValueError("burn thresholds must be nonnegative and finite")
    if self_gravity:
        if geometry != "spherical" or boundary != "outflow":
            raise ValueError("self-gravity requires spherical geometry and an outflow exterior")
        from luna.gravity.spherical_1d import make_solver as make_radial

        hydro = make_radial(table, coulomb=coulomb, order=order, equilibrium=equilibrium)
    else:
        if equilibrium is not None:
            raise ValueError("equilibrium balancing requires spherical self-gravity")
        hydro = make_solver(
            table, boundary=boundary, coulomb=coulomb, order=order, geometry=geometry
        )
    network = make_network(
        table, screening=screening, neutrinos=neutrinos, coulomb=coulomb, rtol=rtol
    )
    options = Options(max_steps=20000, tiers=((None, 1),), diff_fields=())

    def burn(state, guess, duration, previous_shock, dx):
        thermo = hydro.recover(state, guess)
        acceleration = 0.0
        if self_gravity:
            from luna.gravity.spherical_1d import shell_gravity

            acceleration = shell_gravity(state[0], dx).acceleration
        blocked = (
            previous_shock
            | shock_mask(
                state,
                thermo.pressure,
                boundary=boundary,
                threshold=shock_threshold,
                acceleration=acceleration,
                dx=dx,
            )
            if suppress_shock_burning
            else jnp.zeros_like(thermo.valid)
        )
        density = state[0]
        cold = ~((thermo.temperature > burn_min_temperature) & (density > burn_min_density))
        energy = state[-1] / density - 0.5 * (state[1] / density) ** 2
        initial = jnp.concatenate(
            [jnp.moveaxis(state[2:-1] / density, 0, -1), energy[:, None]], axis=-1
        )
        duration = jnp.broadcast_to(jnp.asarray(duration), density.shape)
        valid_duration = jnp.isfinite(duration) & (duration >= 0)
        result = integrate(
            initial,
            BurnParams(density, thermo.temperature),
            jnp.where(thermo.valid & valid_duration & ~blocked & ~cold, duration, 0.0),
            net=network,
            opt=options,
        )
        updated = state.at[2:-1].set(density * jnp.moveaxis(result.y[:, :13], -1, 0))
        updated = updated.at[-1].add(density * (result.y[:, 13] - energy))
        after = hydro.recover(updated, result.params.temperature_guess)
        composition_change = jnp.max(abs(result.y[:, :13] - initial[:, :13]), axis=-1)
        temperature_change = abs(after.temperature / thermo.temperature - 1)
        valid = (
            thermo.valid
            & valid_duration
            & after.valid
            & ~result.failed
            & (composition_change <= burn_limit)
            & (temperature_change <= burn_limit)
        )
        return updated, after.temperature, valid, result.n_steps, result.n_rejected, blocked, cold

    def step(state, temperature_guess, dt, dx):
        first, guess, valid_first, steps_first, rejected_first, blocked_first, cold_first = burn(
            state, temperature_guess, 0.5 * dt, jnp.zeros(state.shape[1], dtype=bool), dx
        )
        advected = hydro.step(first, guess, dt, dx)
        proceed = valid_first & advected.valid
        final, guess, valid_last, steps_last, rejected_last, blocked_last, cold_last = burn(
            jnp.where(proceed[None], advected.state, first),
            jnp.where(proceed, advected.temperature, guess),
            jnp.where(proceed, 0.5 * dt, 0.0),
            blocked_first,
            dx,
        )
        valid = valid_first & advected.valid & valid_last
        return ReactiveStep(
            jnp.where(valid[None], final, jnp.nan),
            guess,
            advected.signal_speed,
            valid,
            steps_first + steps_last,
            rejected_first + rejected_last,
            advected.slope_reduced,
            advected.flux_fraction,
            jnp.stack([blocked_first, blocked_last]),
            advected.boundary_flux,
            jnp.stack([cold_first, cold_last]),
        )

    return jax.jit(step)
