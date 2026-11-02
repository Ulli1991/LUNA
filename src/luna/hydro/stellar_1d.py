"""Standalone 1D Helmholtz MUSCL-Hancock hydro with conservative species transport."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from luna.eos.helmholtz import evaluate
from luna.eos.inversion import temperature_from_energy
from luna.hydro.eos_fluxes import hll_eos_1d
from luna.hydro.slopes import moncen
from luna.nuclear.aprox13 import mean_composition


class FluidState(NamedTuple):
    temperature: object
    pressure: object
    sound_speed: object
    valid: object


class StepResult(NamedTuple):
    state: object
    temperature: object
    signal_speed: object
    valid: object
    boundary_flux: object
    slope_reduced: object
    flux_fraction: object
    face_flux: object


class Solver(NamedTuple):
    recover: object
    step: object


def admissible_flux(state, guess, high, low, dt, dx, recover, update=None, periodic=False):
    """Conservatively reduce a shared high-order flux until the complete update is EOS-admissible."""

    def trial(theta):
        flux = jnp.where(theta == 1, high, jnp.where(theta == 0, low, low + theta * (high - low)))
        if periodic:   # faces 0 and N are one face: one value, else round-off leaks mass through it
            flux = flux.at[:, -1].set(flux[:, 0])
        updated = (
            state - dt / dx * (flux[:, 1:] - flux[:, :-1])
            if update is None
            else update(flux, theta)
        )
        return flux, updated, recover(updated, guess)

    flux, updated, after = trial(1.0)

    def reduce(_):
        def pending(carry):
            theta, valid = carry
            return ~valid & (theta > 2.0**-20)

        def halve(carry):
            theta = 0.5 * carry[0]
            return theta, jnp.all(trial(theta)[2].valid)

        theta, valid = jax.lax.while_loop(pending, halve, (jnp.asarray(1.0), jnp.asarray(False)))
        theta = jax.lax.stop_gradient(jnp.where(valid, theta, 0.0))
        return (*trial(theta), theta)

    return jax.lax.cond(
        jnp.all(after.valid), lambda _: (flux, updated, after, jnp.asarray(1.0)), reduce, None
    )


def make_solver(table, *, boundary="outflow", coulomb=True, order=2, geometry="planar"):
    """Build a local 1D solver; state layout is rho, momentum, thirteen rho*X, total energy."""
    if boundary not in ("periodic", "outflow", "reflecting"):
        raise ValueError("boundary must be periodic, outflow, or reflecting (left wall)")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if geometry not in ("planar", "spherical"):
        raise ValueError("geometry must be planar or spherical")
    radial = geometry == "spherical"
    if radial and boundary != "outflow":
        raise ValueError("spherical geometry uses a reflecting origin and outflow exterior")
    mode = "wrap" if boundary == "periodic" else "edge"

    def pad_state(state, ghosts):
        padded = jnp.pad(state, ((0, 0), (ghosts, ghosts)), mode=mode)
        if radial or boundary == "reflecting":
            reflection = state[:, :ghosts][:, ::-1].at[1].multiply(-1)
            padded = padded.at[:, :ghosts].set(reflection)
        return padded

    def recover(table, state, temperature_guess):
        if state.ndim != 2 or state.shape[0] != 16:
            raise ValueError("stellar 1D state must have shape (16, cells)")
        density = state[0]
        safe_density = jnp.where(density > 0, density, 1.0)
        velocity = state[1] / safe_density
        energy = state[-1] / safe_density - 0.5 * velocity**2
        fractions = jnp.moveaxis(state[2:-1] / safe_density, 0, -1)
        mass, charge = mean_composition(fractions)
        recovered = temperature_from_energy(
            table,
            density,
            energy,
            mass,
            charge,
            temperature_guess=temperature_guess,
            coulomb=coulomb,
        )
        root = jax.lax.stop_gradient(recovered.temperature)
        thermo = evaluate(table, density, root, mass, charge, coulomb=coulomb)
        # The implicit first derivative avoids differentiating the root iteration.
        temperature = root + (energy - thermo.energy) / jax.lax.stop_gradient(thermo.cv)
        thermo = evaluate(table, density, temperature, mass, charge, coulomb=coulomb)
        valid = (
            recovered.converged
            & thermo.valid
            & jnp.all(jnp.isfinite(state), axis=0)
            & (density > 0)
        )
        return FluidState(temperature, thermo.pressure, thermo.sound_speed, valid)

    def step(table, state, temperature_guess, dt, dx):
        before = recover(table, state, temperature_guess)
        source_pressure = before.pressure
        if radial:
            radius = dx * jnp.arange(state.shape[1] + 1)
            area = radius**2
            volume = (radius[1:] ** 3 - radius[:-1] ** 3) / 3

        if order == 1:
            padded = pad_state(state, 1)
            pressure = jnp.pad(before.pressure, (1, 1), mode=mode)
            sound = jnp.pad(before.sound_speed, (1, 1), mode=mode)
            face = hll_eos_1d(
                padded[:, :-1], padded[:, 1:], pressure[:-1], pressure[1:], sound[:-1], sound[1:]
            )
            predictor_valid = before.valid
            slope_reduced = jnp.zeros_like(before.valid)
        else:
            padded = pad_state(state, 2)
            center = padded[:, 1:-1]
            dl, dr = center - padded[:, :-2], padded[:, 2:] - center
            slope = moncen(dl, dr)
            fractions = padded[2:-1] / padded[0]
            fraction_center = fractions[:, 1:-1]
            fraction_slope = moncen(
                fraction_center - fractions[:, :-2], fractions[:, 2:] - fraction_center
            )
            # Project only composition slopes onto sum(dX)=0; do not flatten Euler fields.
            fraction_slope -= fraction_center * jnp.sum(fraction_slope, axis=0)
            slope = slope.at[2:-1].set(fraction_center * slope[0] + center[0] * fraction_slope)
            guess = jnp.pad(before.temperature, (1, 1), mode=mode)
            center_pressure = jnp.pad(before.pressure, (1, 1), mode=mode)

            def predict(slope):
                cells = center.shape[1]
                edges = jnp.concatenate([center - 0.5 * slope, center + 0.5 * slope], axis=1)
                edge_eos = recover(table, edges, jnp.tile(guess, 2))
                velocity = edges[1] / edges[0]
                flux = (
                    (edges * velocity)
                    .at[1]
                    .add(edge_eos.pressure)
                    .at[-1]
                    .add(edge_eos.pressure * velocity)
                )
                correction = -0.5 * dt / dx * (flux[:, cells:] - flux[:, :cells])
                if radial:
                    rcenter = dx * (jnp.arange(cells) - 0.5)
                    velocity_center = center[1] / center[0]
                    geometric = -2 * velocity_center / rcenter * center
                    geometric = geometric.at[-1].add(
                        -2 * center_pressure * velocity_center / rcenter
                    )
                    correction += 0.5 * dt * geometric
                half = edges + jnp.tile(correction, (1, 2))
                half_eos = recover(table, half, edge_eos.temperature)
                low_half = FluidState(*(field[:cells] for field in half_eos))
                high_half = FluidState(*(field[cells:] for field in half_eos))
                valid = (
                    edge_eos.valid[:cells]
                    & edge_eos.valid[cells:]
                    & low_half.valid
                    & high_half.valid
                )
                return half[:, :cells], half[:, cells:], low_half, high_half, valid

            candidate = predict(slope)
            reduced = ~candidate[-1]
            # Flatten only EOS-inadmissible predictors; never repair their thermodynamics.
            low, high, low_half, high_half, admissible = jax.lax.cond(
                jnp.any(reduced),
                lambda _: predict(jnp.where(reduced[None], 0.0, slope)),
                lambda _: candidate,
                operand=None,
            )
            slope_reduced = reduced[1:-1]
            face = hll_eos_1d(
                high[:, :-1],
                low[:, 1:],
                high_half.pressure[:-1],
                low_half.pressure[1:],
                high_half.sound_speed[:-1],
                low_half.sound_speed[1:],
            )
            predictor_valid = admissible[1:-1]
            source_pressure = 0.5 * (low_half.pressure[1:-1] + high_half.pressure[1:-1])
        base = pad_state(state, 1)
        pressure = jnp.pad(before.pressure, (1, 1), mode=mode)
        sound = jnp.pad(before.sound_speed, (1, 1), mode=mode)
        low_face = hll_eos_1d(
            base[:, :-1], base[:, 1:], pressure[:-1], pressure[1:], sound[:-1], sound[1:]
        )
        high_valid = jnp.all(face.valid) & jnp.all(predictor_valid)
        high_flux = jnp.where(high_valid, face.flux, low_face.flux)
        source_pressure = jnp.where(high_valid, source_pressure, before.pressure)

        def radial_update(flux, theta):
            pressure = before.pressure + theta * (source_pressure - before.pressure)
            rate = -(area[1:] * flux[:, 1:] - area[:-1] * flux[:, :-1]) / volume
            rate = rate.at[1].add(pressure * (area[1:] - area[:-1]) / volume)
            return state + dt * rate

        flux, updated, after, fraction = admissible_flux(
            state,
            before.temperature,
            high_flux,
            low_face.flux,
            dt,
            dx,
            lambda state, guess: recover(table, state, guess),
            update=radial_update if radial else None,
            periodic=mode == "wrap",
        )
        fraction = jnp.where(high_valid, fraction, 0.0)
        speed = jnp.maximum(
            jnp.max(low_face.signal_speed), jnp.max(jnp.where(high_valid, face.signal_speed, 0.0))
        )
        valid_step = (
            jnp.isfinite(dt)
            & jnp.isfinite(dx)
            & (dt >= 0)
            & (dx > 0)
            & (dt * speed <= (dx / 6 if radial else 0.5 * dx))
        )
        valid = before.valid & after.valid & low_face.valid[:-1] & low_face.valid[1:] & valid_step
        return StepResult(
            jnp.where(valid[None], updated, jnp.nan),
            after.temperature,
            speed,
            valid,
            (area[0] * flux[:, 0] - area[-1] * flux[:, -1]) if radial else flux[:, 0] - flux[:, -1],
            slope_reduced,
            fraction,
            flux,
        )

    recover_kernel, step_kernel = jax.jit(recover), jax.jit(step)
    return Solver(
        lambda state, guess: recover_kernel(table, state, guess),
        lambda state, guess, dt, dx: step_kernel(table, state, guess, dt, dx),
    )
