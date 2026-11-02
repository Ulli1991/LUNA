"""Steady planar detonation jump conditions; prescribed ash CJ is not an NSE eigenvalue solve."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from luna.eos.helmholtz import evaluate
from luna.nuclear.aprox13 import (
    binding_energy,
    mean_composition,
    BINDING_MEV,
    MASS_NUMBERS,
    NA,
    MEV_TO_ERG,
)


class Jump(NamedTuple):
    density: float
    temperature: float
    pressure: float
    speed: float
    heat: float
    residual: float


def prescribed_ash_cj(table, density, temperature, fuel, ash, *, coulomb=True):
    """Solve reactive Hugoniot and frozen-sonic tangency for explicitly prescribed composition."""
    fuel, ash = jnp.asarray(fuel), jnp.asarray(ash)
    mf, zf = mean_composition(fuel)
    ma, za = mean_composition(ash)
    eos = jax.jit(
        lambda table, rho, temp, mass, charge: evaluate(
            table, rho, temp, mass, charge, coulomb=coulomb
        )
    )
    initial = eos(table, density, temperature, mf, zf)
    heat = float(binding_energy(ash) - binding_energy(fuel))
    if not bool(initial.valid) or not np.isfinite(heat) or heat <= 0:
        raise ValueError("CJ inputs require valid fuel/ash and positive heat release")
    scale = float(initial.pressure) + density * heat

    def residual(log_state):
        rho, temp = np.exp(log_state)
        final = eos(table, rho, temp, ma, za)
        if not bool(final.valid):
            raise ValueError("CJ iteration left the EOS domain")
        volume_jump = 1 / density - 1 / rho
        mass_flux = rho * float(final.sound_speed)
        return np.array(
            [
                (float(final.pressure - initial.pressure) - mass_flux**2 * volume_jump) / scale,
                (
                    float(final.energy - initial.energy)
                    - heat
                    - 0.5 * float(final.pressure + initial.pressure) * volume_jump
                )
                / heat,
            ]
        )

    result = least_squares(
        residual,
        np.log([1.7 * density, max(temperature * 1.1, 5e9)]),
        bounds=(
            np.log([density * 1.0001, temperature]),
            np.log([density * 8, float(table.temperature[-1]) * 0.99]),
        ),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=200,
    )
    rho, temp = np.exp(result.x)
    error = float(np.max(abs(residual(result.x))))
    if not result.success or error > 1e-10:
        raise RuntimeError(f"CJ solve failed: {result.message}, residual={error}")
    final = eos(table, rho, temp, ma, za)
    speed = float(rho * final.sound_speed / density)
    if speed <= float(initial.sound_speed):
        raise RuntimeError("CJ branch is not supersonic into fuel")
    return Jump(float(rho), float(temp), float(final.pressure), speed, heat, error)


def frozen_shock(table, density, temperature, fuel, speed, *, coulomb=True):
    """Find the nontrivial frozen-composition Rankine-Hugoniot shock at a prescribed speed."""
    mass, charge = mean_composition(jnp.asarray(fuel))
    eos = jax.jit(
        lambda table, rho, temp: evaluate(table, rho, temp, mass, charge, coulomb=coulomb)
    )
    initial = eos(table, density, temperature)
    if not bool(initial.valid) or not np.isfinite(speed) or speed <= float(initial.sound_speed):
        raise ValueError("frozen shock requires valid fuel and a supersonic speed")
    pressure_scale, energy_scale = density * speed**2, speed**2

    def residual(log_state):
        rho, temp = np.exp(log_state)
        final = eos(table, rho, temp)
        if not bool(final.valid):
            raise ValueError("frozen shock left EOS domain")
        return np.array(
            [
                float(
                    final.pressure
                    - initial.pressure
                    + (density * speed) ** 2 * (1 / rho - 1 / density)
                )
                / pressure_scale,
                float(
                    final.energy
                    - initial.energy
                    - 0.5 * (final.pressure + initial.pressure) * (1 / density - 1 / rho)
                )
                / energy_scale,
            ]
        )

    result = least_squares(
        residual,
        np.log([3 * density, max(temperature * 1.1, 3e9)]),
        bounds=(
            np.log([density * 1.01, temperature]),
            np.log([density * 10, float(table.temperature[-1]) * 0.99]),
        ),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=200,
    )
    rho, temp = np.exp(result.x)
    error = float(np.max(abs(residual(result.x))))
    if not result.success or error > 1e-10:
        raise RuntimeError(f"frozen shock solve failed: {error}")
    final = eos(table, rho, temp)
    return Jump(float(rho), float(temp), float(final.pressure), float(speed), 0.0, error)


class ZND(NamedTuple):
    rhs: object
    jacobian: object
    recover: object
    shock: Jump


def make_znd(table, density, temperature, fuel, speed, *, screening="screen5", coulomb=True):
    """Steady reactive flow: enforce mass, momentum and total enthalpy before evaluating rates."""
    from luna.nuclear.aprox13 import fixed_temperature_rhs

    fuel = jnp.asarray(fuel)
    mass, charge = mean_composition(fuel)
    initial = evaluate(table, density, temperature, mass, charge, coulomb=coulomb)
    shock = frozen_shock(table, density, temperature, fuel, speed, coulomb=coulomb)
    mass_flux = density * speed
    momentum = initial.pressure + density * speed**2
    enthalpy = initial.energy + initial.pressure / density + 0.5 * speed**2
    binding_weights = jnp.asarray(BINDING_MEV) / jnp.asarray(MASS_NUMBERS) * NA * MEV_TO_ERG
    pressure_scale, energy_scale = density * speed**2, speed**2
    start = jnp.log(jnp.array([shock.density, shock.temperature]))

    def recover(table, fractions):
        # Signed Newton/Radau trial abundances are an algebraic extension, not accepted profiles.
        mass, charge = mean_composition(fractions, negative_tolerance=1e-6)
        heat = jnp.dot(fractions - fuel, binding_weights)

        def equations(log_state):
            rho, temp = jnp.exp(log_state)
            thermo = evaluate(table, rho, temp, mass, charge, coulomb=coulomb)
            residual = jnp.array(
                [
                    (thermo.pressure + mass_flux**2 / rho - momentum) / pressure_scale,
                    (
                        thermo.energy
                        + thermo.pressure / rho
                        + 0.5 * (mass_flux / rho) ** 2
                        - enthalpy
                        - heat
                    )
                    / energy_scale,
                ]
            )
            jacobian = jnp.array(
                [
                    [
                        (thermo.dpdr - mass_flux**2 / rho**2) * rho / pressure_scale,
                        thermo.dpdt * temp / pressure_scale,
                    ],
                    [
                        (
                            thermo.dedr
                            + thermo.dpdr / rho
                            - thermo.pressure / rho**2
                            - mass_flux**2 / rho**3
                        )
                        * rho
                        / energy_scale,
                        (thermo.cv + thermo.dpdt / rho) * temp / energy_scale,
                    ],
                ]
            )
            return residual, jacobian, thermo

        def iterate(_, log_state):
            residual, jacobian, _ = equations(log_state)
            correction = jnp.linalg.solve(jacobian, residual)
            return log_state - jnp.clip(correction, -0.3, 0.3)

        root = jax.lax.stop_gradient(jax.lax.fori_loop(0, 40, iterate, start))
        residual, jacobian, _ = equations(root)
        log_state = root - jnp.linalg.solve(jax.lax.stop_gradient(jacobian), residual)
        rho, temp = jnp.exp(log_state)
        residual, _, thermo = equations(log_state)
        relative_speed = mass_flux / rho
        valid = (
            thermo.valid & (jnp.max(abs(residual)) < 1e-9) & (relative_speed < thermo.sound_speed)
        )
        return (
            jnp.where(valid, rho, jnp.nan),
            jnp.where(valid, temp, jnp.nan),
            jnp.max(abs(residual)),
            relative_speed / thermo.sound_speed,
        )

    def rhs(table, state):
        rho, temp, _, _ = recover(table, state[:13])
        derivative, _ = fixed_temperature_rhs(
            rho, temp, state[:13], screening=screening, negative_tolerance=1e-6
        )
        return jnp.concatenate([derivative, (mass_flux / rho)[None]])

    rhs_kernel = jax.jit(rhs)
    jacobian_kernel = jax.jit(jax.jacfwd(rhs, argnums=1))
    recover_kernel = jax.jit(recover)
    return ZND(
        lambda state: rhs_kernel(table, state),
        lambda state: jacobian_kernel(table, state),
        lambda fractions: recover_kernel(table, fractions),
        shock,
    )


def shock_cell_quadrature(cells, shock_coordinate):
    """Return cell-unit nodes, left-state weights and right-state weights, split exactly at a jump."""
    if not isinstance(cells, int) or cells < 1 or not np.isfinite(shock_coordinate):
        raise ValueError("positive cell count and finite shock coordinate required")
    nodes, weights = np.polynomial.legendre.leggauss(3)
    fraction = np.clip(shock_coordinate - np.arange(cells), 0.0, 1.0)
    points = np.arange(cells)[:, None] + fraction[:, None] * (nodes + 1) / 2
    return points, fraction[:, None] * weights / 2, 1 - fraction
