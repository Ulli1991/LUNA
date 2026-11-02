"""Joint space/time convergence of a smooth, periodic, self-heating He/C flow."""

import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from luna.eos.helmholtz import load_table
from luna.hydro.stellar_1d import make_solver
from luna.nuclear.aprox13 import binding_energy
from luna.nuclear.burn import initial_state
from luna.nuclear.coupling import make_step


def check(table_path, output, grids=(16, 32, 64)):
    table = load_table(table_path)
    hydro = make_solver(table, boundary="periodic")
    advance = make_step(
        table, screening="screen5", neutrinos="none", boundary="periodic", rtol=1e-8
    )
    length, stop = 1e7, 0.001
    report = dict(
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        length=length,
        duration=stop,
        cfl=0.1,
        burning=True,
        passed=False,
        runs=[],
    )
    solutions = []
    for cells in grids:
        print(f"{cells} cells: joint reactive refinement", flush=True)
        phase = 2 * jnp.pi * (jnp.arange(cells) + 0.5) / cells
        rho = 1e6 * (1 + 0.01 * jnp.sin(phase))
        temperature = 1e9 * (1 + 0.01 * jnp.cos(phase))
        fractions = jnp.zeros((cells, 13)).at[:, 0].set(0.8 + 0.01 * jnp.sin(phase))
        fractions = fractions.at[:, 1].set(1 - fractions[:, 0])
        initial = initial_state(table, rho, temperature, fractions)
        velocity = 1e7 * jnp.cos(phase)
        state = jnp.concatenate(
            [
                rho[None],
                (rho * velocity)[None],
                rho * fractions.T,
                (rho * (initial[:, -1] + 0.5 * velocity**2))[None],
            ]
        )
        conserved = np.asarray(state).sum(axis=1)
        initial_budget = float(jnp.sum(state[-1] - rho * binding_energy(fractions)))
        time, steps = 0.0, 0
        dx = length / cells
        while time < stop:
            primitive = hydro.recover(state, temperature)
            speed = float(jnp.max(abs(state[1] / state[0]) + primitive.sound_speed))
            dt = min(0.1 * dx / speed, stop - time)
            result = advance(state, temperature, dt, dx)
            if not np.asarray(result.valid).all():
                raise AssertionError(f"reactive refinement refused at {cells} cells, step {steps}")
            if np.asarray(result.burn_suppressed).any() or float(result.flux_fraction) != 1:
                raise AssertionError("smooth refinement unexpectedly activated shock/flux limiting")
            state, temperature = result.state, result.temperature
            time += dt
            steps += 1
        final = np.asarray(state)
        final_budget = float(
            jnp.sum(state[-1] - state[0] * binding_energy((state[2:-1] / state[0]).T))
        )
        budget_error = abs(final_budget - initial_budget) / abs(initial_budget)
        np.testing.assert_allclose(final[2:-1].sum(axis=0), final[0], rtol=2e-12)
        np.testing.assert_allclose(final[0].sum(), conserved[0], rtol=2e-13)
        if final[2:-1].min() < 0 or budget_error > 2e-12:
            raise AssertionError("joint refinement lost physical bounds or energy")
        solutions.append(final)
        report["runs"].append(
            dict(
                cells=cells,
                steps=steps,
                energy_budget_error=budget_error,
                final_state=final.tolist(),
                temperature=np.asarray(temperature).tolist(),
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
    scale = np.maximum(np.mean(abs(solutions[-1]), axis=1), 1e4)
    scale[1] = 1e6 * 1e7
    differences = []
    for coarse, fine in zip(solutions, solutions[1:]):
        restricted = fine.reshape(16, coarse.shape[1], -1).mean(axis=2)
        differences.append(np.mean(abs(coarse - restricted), axis=1) / scale)
    # Trace products do not set a relative order gate near their roundoff/error floors.
    fields = (0, 1, 2, 3, 15)
    ratios = [float(differences[1][i] / differences[0][i]) for i in fields]
    report.update(
        fields=list(fields), refinement_ratios=ratios, observed_orders=(-np.log2(ratios)).tolist()
    )
    report["passed"] = bool(max(ratios) < 0.4)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise AssertionError(f"reactive joint convergence below second-order gate: {ratios}")
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reactive-refinement.json"))
    args = parser.parse_args()
    check(args.table, args.report)
