"""Hold a balanced white dwarf and evolve a radial velocity perturbation for two dynamical times."""

import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from luna.eos.helmholtz import load_table
from luna.gravity.spherical_1d import G, make_solver, shell_gravity
from luna.ic_generation.white_dwarf_1d import isothermal_model


def check(table_path, output, dynamical_times=2.0):
    table = load_table(table_path)
    cells, dx = 128, 1e7
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    model = isothermal_model(table, 1e7, 1e7, fractions, cells, dx)
    reference = model.state
    solver = make_solver(table, equilibrium=reference)
    gravity = shell_gravity(reference[0], dx)
    dynamical_time = float(jnp.sqrt(model.surface_radius**3 / (G * gravity.mass)))
    radius = dx * jnp.arange(cells + 1)
    volume = 4 * jnp.pi / 3 * (radius[1:] ** 3 - radius[:-1] ** 3)
    report = dict(
        jax=jax.__version__,
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        cells=cells,
        dynamical_time=dynamical_time,
        requested_dynamical_times=dynamical_times,
        method="explicit discrete reference-residual balancing",
        passed=False,
    )
    print("white dwarf: exact reference hold", flush=True)
    state, guess = reference, model.temperature
    for _ in range(3):
        result = solver.step(state, guess, 1e-4, dx)
        if not np.asarray(result.valid).all():
            raise AssertionError("balanced reference refused")
        np.testing.assert_array_equal(result.state, reference)
        state, guess = result.state, result.temperature
    report["reference_held_bitwise"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    center = dx * (jnp.arange(cells) + 0.5)
    velocity = jnp.where(
        center < model.surface_radius, 1e4 * jnp.sin(jnp.pi * center / model.surface_radius), 0.0
    )
    state = (
        reference.at[1].set(reference[0] * velocity).at[-1].add(0.5 * reference[0] * velocity**2)
    )
    initial_density = state[0]
    initial_energy = float(jnp.dot(state[-1], volume) + shell_gravity(state[0], dx).binding_energy)
    initial_momentum = np.asarray(state[1])
    time, steps, boundary = 0.0, 0, 0.0
    stop = dynamical_times * dynamical_time
    max_mach = 0.0
    while time < stop:
        primitive = solver.recover(state, guess)
        speed = float(jnp.max(abs(state[1] / state[0]) + primitive.sound_speed))
        dt = min(0.1 * dx / speed, stop - time)
        result = solver.step(state, guess, dt, dx)
        if not np.asarray(result.valid).all():
            raise AssertionError(f"perturbed white dwarf refused at {time}")
        state, guess = result.state, result.temperature
        time, steps = time + dt, steps + 1
        boundary += 4 * np.pi * dt * float(result.boundary_flux[-1])
        core = reference[0] > 1e4
        max_mach = max(
            max_mach,
            float(jnp.max(jnp.where(core, abs(state[1] / state[0]) / primitive.sound_speed, 0.0))),
        )
        if steps % 100 == 0:
            report.update(
                steps=steps, duration=time, completed_dynamical_times=time / dynamical_time
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{steps} steps, {time / dynamical_time:.3f} dynamical times", flush=True)
        if steps > 20000:
            raise AssertionError("white dwarf step budget exhausted")
    final_energy = float(jnp.dot(state[-1], volume) + shell_gravity(state[0], dx).binding_energy)
    error = abs(final_energy - initial_energy - boundary) / abs(initial_energy)
    density_change = float(
        jnp.dot(abs(state[0] - initial_density), volume) / jnp.dot(initial_density, volume)
    )
    momentum_change = float(np.max(abs(np.asarray(state[1]) - initial_momentum)))
    report.update(
        steps=steps,
        duration=time,
        completed_dynamical_times=time / dynamical_time,
        energy_budget_error=error,
        mass_weighted_density_change=density_change,
        maximum_core_mach=max_mach,
        maximum_momentum_change=momentum_change,
    )
    report["passed"] = bool(
        error < 2e-12 and density_change < 0.01 and max_mach < 0.05 and momentum_change > 1e6
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(f"perturbed reference stability failed: {report}")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("stellar-equilibrium.json"))
    parser.add_argument("--dynamical-times", type=float, default=2.0)
    args = parser.parse_args()
    check(args.table, args.report, args.dynamical_times)
