"""Check a small-amplitude Helmholtz acoustic wave and grid convergence on CPU."""

import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from luna.eos.helmholtz import load_table, evaluate_checked
from luna.hydro.stellar_1d import make_solver
from luna.nuclear.aprox13 import mean_composition


def check(table_path, output, order=2):
    table = load_table(table_path)
    solver = make_solver(table, boundary="periodic", order=order)
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    mass, charge = mean_composition(fractions)
    density, temperature, length, amplitude = 1e6, 1e9, 1e7, 1e-5
    base = evaluate_checked(table, density, temperature, mass, charge)
    sound = float(base.sound_speed)
    final_time = 0.2 * length / sound
    report = dict(
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        density=density,
        temperature=temperature,
        length=length,
        amplitude=amplitude,
        final_time=final_time,
        boundary="periodic",
        order=order,
        burning=False,
        runs=[],
        passed=False,
    )
    for cells in (32, 64, 128):
        print(f"{cells} cells: Helmholtz acoustic wave", flush=True)
        dx = length / cells
        phase = 2 * np.pi * (np.arange(cells) + 0.5) / cells
        rho = density * (1 + amplitude * np.sin(phase))
        thermo = evaluate_checked(table, rho, temperature, mass, charge)
        state = jnp.concatenate(
            [
                jnp.asarray(rho)[None],
                jnp.zeros((1, cells)),
                fractions[:, None] * rho,
                (rho * thermo.energy)[None],
            ]
        )
        initial = np.asarray(state)
        guess = jnp.full(cells, temperature)
        time, steps, reductions = 0.0, 0, 0
        while time < final_time:
            primitive = solver.recover(state, guess)
            if not np.asarray(primitive.valid).all():
                raise AssertionError("invalid recovered state")
            speed = float(jnp.max(abs(state[1] / state[0]) + primitive.sound_speed))
            dt = min(0.3 * dx / speed, final_time - time)
            result = solver.step(state, guess, dt, dx)
            jax.block_until_ready(result.state)
            if not np.asarray(result.valid).all():
                raise AssertionError(f"{cells}: invalid step {steps} at {time}")
            reductions += int(np.count_nonzero(result.slope_reduced))
            state, guess = result.state, result.temperature
            time += dt
            steps += 1
            if steps > 2000:
                raise AssertionError("acoustic check exceeded step cap")
        final = np.asarray(state)
        pressure_amplitude = float(base.dpdr) * density * amplitude
        angle = 2 * np.pi * sound * final_time / length
        expected_density = (
            density
            + (density * amplitude - pressure_amplitude / sound**2) * np.sin(phase)
            + pressure_amplitude / sound**2 * np.sin(phase) * np.cos(angle)
        )
        expected_velocity = -pressure_amplitude / (density * sound) * np.cos(phase) * np.sin(angle)
        rho_error = float(np.mean(abs(final[0] - expected_density)) / (density * amplitude))
        velocity_error = float(
            np.mean(abs(final[1] / final[0] - expected_velocity))
            / (pressure_amplitude / (density * sound))
        )
        scale = np.maximum(np.sum(abs(initial), axis=1), density * cells)
        scale[1] = density * sound * cells
        conservation = float(np.max(abs(final.sum(axis=1) - initial.sum(axis=1)) / scale))
        species_error = float(np.max(abs(final[2:-1].sum(axis=0) / final[0] - 1)))
        if conservation > 2e-13 or species_error > 1e-12 or final[2:-1].min() < 0:
            raise AssertionError("hydro conservation or abundance bounds failed")
        entry = dict(
            cells=cells,
            steps=steps,
            slope_reductions=reductions,
            density_l1_normalized=rho_error,
            velocity_l1_normalized=velocity_error,
            conservation_scaled_error=conservation,
            species_sum_error=species_error,
        )
        report["runs"].append(entry)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    for previous, current in zip(report["runs"], report["runs"][1:]):
        for key in ("density_l1_normalized", "velocity_l1_normalized"):
            if current[key] > (0.4 if order == 2 else 0.75) * previous[key]:
                raise AssertionError(f"grid refinement failed to reduce {key}")
    if (
        report["runs"][-1]["density_l1_normalized"] > 0.1
        or report["runs"][-1]["velocity_l1_normalized"] > 0.1
    ):
        raise AssertionError("acoustic error too large")
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("stellar-hydro.json"))
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    check(args.table, args.report, args.order)
