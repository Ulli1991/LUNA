"""Newtonian fields and binding energy for piecewise-constant spherical density shells."""

from typing import NamedTuple

import jax.numpy as jnp

from luna.eos.helmholtz import _require_fp64

G = 6.67430e-8


class ShellGravity(NamedTuple):
    acceleration: object
    potential: object
    face_potential: object
    mass: object
    binding_energy: object


def shell_gravity(density, dx):
    """Return volume-averaged fields, exact shell self-energy, and potential with zero at infinity."""
    _require_fp64()
    density = jnp.asarray(density, jnp.float64)
    if density.ndim != 1 or density.size < 1:
        raise ValueError("density must be a nonempty vector")
    radius = dx * jnp.arange(density.size + 1)
    a, b = radius[:-1], radius[1:]
    volume = 4 * jnp.pi / 3 * (b**3 - a**3)
    mass = density * volume
    enclosed = jnp.concatenate([jnp.zeros(1), jnp.cumsum(mass)])
    invr = 1.5 * (b**2 - a**2) / (b**3 - a**3)
    invr2 = 3 * (b - a) / (b**3 - a**3)
    meanr = 0.75 * (b**4 - a**4) / (b**3 - a**3)
    meanr2 = 0.6 * (b**5 - a**5) / (b**3 - a**3)
    exterior = jnp.cumsum((mass * invr)[::-1])[::-1]
    exterior_face = jnp.concatenate([exterior, jnp.zeros(1)])
    potential = -G * (
        enclosed[:-1] * invr
        + exterior_face[1:]
        + 4 * jnp.pi * density * (0.5 * b**2 - meanr2 / 6 - a**3 * invr / 3)
    )
    face_potential = -G * (enclosed / jnp.where(radius > 0, radius, 1) + exterior_face)
    acceleration = -G * (enclosed[:-1] * invr2 + 4 * jnp.pi * density / 3 * (meanr - a**3 * invr2))
    valid = jnp.all(jnp.isfinite(density) & (density > 0)) & jnp.isfinite(dx) & (dx > 0)
    return ShellGravity(
        *[
            jnp.where(valid, value, jnp.nan)
            for value in (
                acceleration,
                potential,
                face_potential,
                enclosed[-1],
                0.5 * jnp.sum(mass * potential),
            )
        ]
    )


def conservative_work(old_density, new_density, mass_flux, dx, dt):
    """Gas-energy density change whose global budget closes against shell binding energy and escape."""
    old, new = shell_gravity(old_density, dx), shell_gravity(new_density, dx)
    radius = dx * jnp.arange(old_density.size + 1)
    volume = (radius[1:] ** 3 - radius[:-1] ** 3) / 3
    potential = 0.5 * (old.potential + new.potential)
    potential_flux = mass_flux * 0.5 * (old.face_potential + new.face_potential)
    transport = -(radius[1:] ** 2 * potential_flux[1:] - radius[:-1] ** 2 * potential_flux[:-1])
    return dt * transport / volume - potential * (new_density - old_density)


def make_solver(table, *, coulomb=True, order=2, equilibrium=None):
    """Compose radial hydro with symmetric momentum kicks and conservative gravitational work."""
    import jax
    from luna.hydro.stellar_1d import Solver, make_solver as make_hydro

    hydro = make_hydro(table, coulomb=coulomb, order=order, geometry="spherical")
    fields = jax.jit(shell_gravity)
    work = jax.jit(conservative_work)

    def kick(state, acceleration, duration):
        momentum = state[1] + duration * state[0] * acceleration
        change = 0.5 * (momentum**2 - state[1] ** 2) / state[0]
        return state.at[1].set(momentum).at[-1].add(change), change

    kick = jax.jit(kick)

    def step(state, guess, dt, dx):
        old = fields(state[0], dx)
        kicked, first_work = kick(state, old.acceleration, 0.5 * dt)
        advected = hydro.step(kicked, guess, dt, dx)
        new = fields(advected.state[0], dx)
        final, _ = kick(advected.state, new.acceleration, 0.5 * dt)
        change = work(state[0], final[0], advected.face_flux[0], dx, dt)
        final = final.at[-1].set(advected.state[-1] + change - first_work)
        after = hydro.recover(final, advected.temperature)
        valid = (
            advected.valid & after.valid & jnp.isfinite(old.potential) & jnp.isfinite(new.potential)
        )
        potential_flux = advected.face_flux[0] * 0.5 * (old.face_potential + new.face_potential)
        radius = dx * state.shape[1]
        boundary = advected.boundary_flux.at[-1].add(-(radius**2) * potential_flux[-1])
        return advected._replace(
            state=jnp.where(valid[None], final, jnp.nan),
            temperature=after.temperature,
            valid=valid,
            boundary_flux=boundary,
        )

    if equilibrium is None:
        return Solver(hydro.recover, step)
    reference = jnp.asarray(equilibrium, jnp.float64)
    if reference.ndim != 2 or reference.shape[0] != 16 or not bool(jnp.all(reference[1] == 0)):
        raise ValueError("equilibrium must be a stationary (16, cells) state")
    raw_step = step

    def balanced(state, guess, dt, dx):
        if state.shape != reference.shape:
            raise ValueError("equilibrium and evolved state shapes differ")
        current = raw_step(state, guess, dt, dx)
        background = raw_step(reference, guess, dt, dx)
        flux = current.face_flux - background.face_flux
        radius = dx * jnp.arange(state.shape[1] + 1)
        volume = (radius[1:] ** 3 - radius[:-1] ** 3) / 3
        updated = (
            state - dt * (radius[1:] ** 2 * flux[:, 1:] - radius[:-1] ** 2 * flux[:, :-1]) / volume
        )
        updated = updated.at[1].set(current.state[1] - background.state[1] + reference[1])
        updated = updated.at[-1].add(work(state[0], updated[0], flux[0], dx, dt))
        after = hydro.recover(updated, current.temperature)
        valid = current.valid & background.valid & after.valid
        old, new = fields(state[0], dx), fields(updated[0], dx)
        potential_flux = flux[0] * 0.5 * (old.face_potential + new.face_potential)
        boundary = radius[0] ** 2 * flux[:, 0] - radius[-1] ** 2 * flux[:, -1]
        boundary = boundary.at[-1].add(-(radius[-1] ** 2) * potential_flux[-1])
        return current._replace(
            state=jnp.where(valid[None], updated, jnp.nan),
            temperature=after.temperature,
            valid=valid,
            boundary_flux=boundary,
            face_flux=flux,
            slope_reduced=current.slope_reduced | background.slope_reduced,
            flux_fraction=jnp.minimum(current.flux_fraction, background.flux_fraction),
        )

    return Solver(hydro.recover, balanced)
