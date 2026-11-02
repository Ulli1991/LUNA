"""Check uniform reactive hydro against the equivalent one-zone burn on the external EOS table."""

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from luna.ode.rodas3 import Options, integrate
from luna.eos.helmholtz import load_table
from luna.nuclear.burn import BurnParams, initial_state, make_network
from luna.nuclear.coupling import make_step
from luna.nuclear.aprox13 import BINDING_MEV, MASS_NUMBERS, NA, MEV_TO_ERG


def check_nonuniform(table_path, output):
    table = load_table(table_path)
    density, cells, temperature = 1e6, 4, 1e9
    phase = 2 * jnp.pi * (jnp.arange(cells) + 0.5) / cells
    fractions = (
        jnp.zeros((cells, 13))
        .at[:, 0]
        .set(0.8 + 0.05 * jnp.sin(phase))
        .at[:, 1]
        .set(0.2 - 0.05 * jnp.sin(phase))
    )
    guess = temperature * (1 + 0.02 * jnp.cos(phase))
    initial = initial_state(table, density, guess, fractions)
    state = jnp.concatenate(
        [
            jnp.full((1, cells), density),
            jnp.zeros((1, cells)),
            density * initial[:, :13].T,
            (density * initial[:, 13])[None],
        ]
    )
    step = make_step(table, screening="screen5", neutrinos="none", boundary="periodic")
    binding = np.asarray(BINDING_MEV) / np.asarray(MASS_NUMBERS) * NA * MEV_TO_ERG
    report = dict(
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        hydro_order=2,
        cells=cells,
        duration=0.001,
        neutrinos="none",
        passed=False,
        runs=[],
    )
    for count in (10, 20, 40):
        print(f"nonuniform reactive hydro: {count} split steps", flush=True)
        current, warm = state, guess
        for _ in range(count):
            result = step(current, warm, 0.001 / count, 1e6)
            if not np.asarray(result.valid).all():
                raise AssertionError("nonuniform reactive step refused")
            current, warm = result.state, result.temperature
        final = np.asarray(current)
        release = float(np.dot((final[2:-1] - np.asarray(state[2:-1])).sum(axis=1), binding))
        heating = float((final[-1] - np.asarray(state[-1])).sum())
        budget = abs(heating - release) / abs(release)
        np.testing.assert_allclose(final[0].sum(), float(state[0].sum()), rtol=1e-13)
        np.testing.assert_allclose(final[2:-1].sum(axis=0), final[0], rtol=1e-12)
        if budget > 1e-7 or final[2:-1].min() < 0:
            raise AssertionError(f"nonuniform mass/energy budget failed: {budget}")
        report["runs"].append(
            dict(
                steps=count,
                energy_budget_error=budget,
                final_state=final.tolist(),
                final_temperature=np.asarray(warm).tolist(),
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
    return finish_nonuniform(report, output)


def finish_nonuniform(report, output):
    """Check recorded fixed-grid timestep convergence without assuming second-order MOL accuracy."""
    solutions = [np.asarray(run["final_state"]) for run in report["runs"]]
    scale = np.maximum(abs(solutions[-1]), 1.0)
    first = float(np.max(abs(solutions[0] - solutions[1]) / scale))
    second = float(np.max(abs(solutions[1] - solutions[2]) / scale))
    if second >= 0.8 * first:
        raise AssertionError(f"timestep refinement did not converge: {first}, {second}")
    report.update(
        passed=True,
        coarse_timestep_difference=first,
        fine_timestep_difference=second,
        observed_time_order=float(np.log2(first / second)),
        convergence_scope="fixed four-cell grid; not a joint space-time order test",
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}), flush=True)
    return report


def check(table_path, output):
    table = load_table(table_path)
    density, temperature, duration, cells = 1e6, 1e9, 1e-4, 4
    start = initial_state(table, density, temperature, jnp.eye(13)[0])
    state = jnp.concatenate(
        [
            jnp.full((1, cells), density),
            jnp.zeros((1, cells)),
            jnp.broadcast_to((density * start[:13])[:, None], (13, cells)),
            jnp.full((1, cells), density * start[-1]),
        ]
    )
    step = make_step(table, screening="screen5", neutrinos="sneut5", boundary="periodic")
    print("uniform reactive hydro: two half-burns and one hydro step", flush=True)
    result = step(state, jnp.full(cells, temperature), duration, 1e6)
    jax.block_until_ready(result.state)
    if not np.asarray(result.valid).all():
        raise AssertionError(f"reactive hydro refused: {result}")
    network = make_network(table, screening="screen5", neutrinos="sneut5")
    reference = integrate(
        jnp.broadcast_to(start, (cells, 14)),
        BurnParams(jnp.full(cells, density), jnp.full(cells, temperature)),
        duration,
        net=network,
        opt=Options(max_steps=20000, tiers=((None, 1),), diff_fields=()),
    )
    if np.asarray(reference.failed).any():
        raise AssertionError("one-zone comparison failed")
    final = np.asarray(result.state)
    fractions = (final[2:-1] / final[0]).T
    np.testing.assert_allclose(fractions, reference.y[:, :13], rtol=5e-5, atol=1e-12)
    heat = final[-1] / final[0] - float(start[-1])
    reference_heat = np.asarray(reference.y[:, 13]) - float(start[-1])
    np.testing.assert_allclose(heat, reference_heat, rtol=5e-5, atol=1e5)
    np.testing.assert_allclose(final[0], density, rtol=1e-14)
    np.testing.assert_allclose(fractions.sum(axis=1), 1.0, rtol=1e-12)
    if np.max(abs(final[1] / final[0])) > 1e-5:
        raise AssertionError("uniform burning generated significant bulk motion")
    report = dict(
        passed=True,
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        hydro_order=2,
        cells=cells,
        duration=duration,
        temperature=np.asarray(result.temperature).tolist(),
        helium=fractions[:, 0].tolist(),
        relative_heating_difference=float(np.max(abs(heat - reference_heat) / abs(reference_heat))),
        burn_steps=np.asarray(result.burn_steps).tolist(),
        burn_rejected=np.asarray(result.burn_rejected).tolist(),
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reactive-hydro.json"))
    parser.add_argument("--case", choices=("uniform", "nonuniform"), default="uniform")
    args = parser.parse_args()
    (check if args.case == "uniform" else check_nonuniform)(args.table, args.report)
