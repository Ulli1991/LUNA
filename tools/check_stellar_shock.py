"""Check a stationary Mach-3 Helmholtz shock constructed from Rankine-Hugoniot conditions."""

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from luna.eos.helmholtz import load_table, evaluate_checked, evaluate
from luna.hydro.stellar_1d import make_solver
from luna.nuclear.aprox13 import mean_composition


def check(table_path, output, order=2):
    table = load_table(table_path)
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    mass, charge = mean_composition(fractions)
    eos = jax.jit(lambda rho, temp: evaluate(table, rho, temp, mass, charge))
    rho, temp = 1e6, 1e8
    upstream = evaluate_checked(table, rho, temp, mass, charge)
    velocity = 3 * float(upstream.sound_speed)
    ram = rho * velocity**2

    def residual(log_state):
        compression, temperature = np.exp(log_state)
        state = eos(rho * compression, temperature)
        target_pressure = upstream.pressure + ram * (1 - 1 / compression)
        target_energy = (
            upstream.energy
            + 0.5 * (target_pressure + upstream.pressure) * (1 - 1 / compression) / rho
        )
        return np.array(
            [
                (state.pressure - target_pressure) / ram,
                (state.energy - target_energy) / velocity**2,
            ],
            float,
        )

    root = least_squares(
        residual,
        np.log([4.0, 2e9]),
        bounds=(np.log([1.1, 1e8]), np.log([7.0, 1e10])),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=200,
    )
    if not root.success or max(abs(residual(root.x))) > 1e-10:
        raise AssertionError(f"Hugoniot solve failed: {root.message}, {residual(root.x)}")
    compression, temp_right = np.exp(root.x)
    right = evaluate_checked(table, rho * compression, temp_right, mass, charge)
    if right.entropy <= upstream.entropy:
        raise AssertionError("shock did not increase entropy")

    def conservative(density, speed, thermo):
        return np.r_[
            density,
            density * speed,
            density * np.asarray(fractions),
            density * (float(thermo.energy) + 0.5 * speed**2),
        ]

    left_state = conservative(rho, velocity, upstream)
    right_state = conservative(rho * compression, velocity / compression, right)
    signal = max(
        abs(velocity) + float(upstream.sound_speed),
        abs(velocity / compression) + float(right.sound_speed),
    )
    length = 1e7
    stop = 0.2 * length / signal
    report = dict(
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        mach=3.0,
        order=order,
        compression=float(compression),
        upstream_temperature=temp,
        downstream_temperature=float(temp_right),
        hugoniot_residual=residual(root.x).tolist(),
        duration=stop,
        runs=[],
        passed=False,
    )
    solver = make_solver(table, boundary="outflow", order=order)
    for cells in (32, 64, 128):
        print(f"{cells} cells: stationary stellar shock", flush=True)
        mask = np.arange(cells) < cells // 2
        exact = np.where(mask[None], left_state[:, None], right_state[:, None])
        state = jnp.asarray(exact)
        guess = jnp.asarray(np.where(mask, temp, temp_right))
        dx = length / cells
        time, steps, reductions = 0.0, 0, 0
        boundary_budget = np.zeros(16)
        while time < stop:
            primitive = solver.recover(state, guess)
            speed = float(jnp.max(abs(state[1] / state[0]) + primitive.sound_speed))
            dt = min(0.3 * dx / speed, stop - time)
            result = solver.step(state, guess, dt, dx)
            if not np.asarray(result.valid).all():
                raise AssertionError(f"invalid shock step {steps} at {time}")
            reductions += int(np.count_nonzero(result.slope_reduced))
            state, guess = result.state, result.temperature
            boundary_budget += dt / dx * np.asarray(result.boundary_flux)
            time += dt
            steps += 1
            if steps > 2000:
                raise AssertionError("shock exceeded step cap")
        final = np.asarray(state)
        scales = np.maximum(abs(left_state) + abs(right_state), rho)
        error = float(np.max(np.mean(abs(final - exact), axis=1) / scales))
        conservation = float(
            np.max(abs(final.sum(axis=1) - exact.sum(axis=1) - boundary_budget) / (cells * scales))
        )
        if conservation > 1e-13 or final[2:-1].min() < 0:
            raise AssertionError(f"shock conservation failed: {conservation}")
        np.testing.assert_allclose(final[2:-1].sum(axis=0), final[0], rtol=1e-12)
        report["runs"].append(
            dict(
                cells=cells,
                steps=steps,
                slope_reductions=reductions,
                l1_scaled_error=error,
                conservation_scaled_error=conservation,
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["runs"][-1]), flush=True)
    if any(
        b["l1_scaled_error"] > 0.8 * a["l1_scaled_error"]
        for a, b in zip(report["runs"], report["runs"][1:])
    ):
        raise AssertionError("stationary shock did not converge under refinement")
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("stellar-shock.json"))
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    check(args.table, args.report, args.order)
