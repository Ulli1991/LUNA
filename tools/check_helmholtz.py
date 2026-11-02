"""Compare the JAX forward EOS against compiled Microphysics reference states using an external table."""

import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from luna.eos.helmholtz import evaluate, load_table
from luna.eos.inversion import temperature_from_energy
from luna.reference_data import reference_path


def check(table_path):
    reference = json.loads(
        Path(reference_path("microphysics/helmholtz_eos/helmholtz_reference.json")).read_text()
    )
    if hashlib.sha256(Path(table_path).read_bytes()).hexdigest() != reference["table_sha256"]:
        raise ValueError("table checksum differs from the Microphysics reference table")
    inputs = np.asarray(reference["inputs"])
    table = load_table(table_path)
    got = []
    temperature_errors = []
    maximum_iterations = 0
    for coulomb in (False, True):
        rows = inputs[:, -1] == coulomb
        state = jax.jit(
            lambda rho, temp, abar, zbar: evaluate(table, rho, temp, abar, zbar, coulomb=coulomb)
        )(*inputs[rows, :4].T)
        if not np.asarray(state.valid).all():
            raise AssertionError("invalid reference state")
        actual = np.stack(
            [np.asarray(getattr(state, name)) for name in reference["columns"]], axis=-1
        )
        expected = np.asarray(reference["values"])[rows]
        # Cold degenerate entropy/cv lose digits through cancellation of free-energy terms.
        tolerances = dict(
            pressure=3e-12,
            energy=3e-12,
            entropy=1e-7,
            dpdr=3e-14,
            dpdt=3e-8,
            dedr=1e-10,
            cv=4e-7,
            gamma1=3e-12,
            sound_speed=3e-12,
        )
        for column, name in enumerate(reference["columns"]):
            np.testing.assert_allclose(
                actual[:, column], expected[:, column], rtol=tolerances[name], atol=0, err_msg=name
            )
        got.append(np.max(np.abs((actual - expected) / expected), axis=0))
        density, temperature, mean_mass, mean_charge = inputs[rows, :4].T
        recovered = jax.jit(
            lambda density, energy, mean_mass, mean_charge: temperature_from_energy(
                table, density, energy, mean_mass, mean_charge, coulomb=coulomb
            )
        )(density, state.energy, mean_mass, mean_charge)
        if not np.asarray(recovered.converged).all():
            raise AssertionError("temperature round trip did not converge")
        np.testing.assert_allclose(recovered.temperature, temperature, rtol=2e-8)
        temperature_errors.append(
            np.max(np.abs(np.asarray(recovered.temperature) / temperature - 1))
        )
        maximum_iterations = max(maximum_iterations, int(np.max(recovered.iterations)))
    for name, error in zip(reference["columns"], np.max(got, axis=0)):
        print(f"{name}: max relative error {error:.3g}")
    print(f"Passed {len(inputs)} reference states (Coulomb on/off, batched JIT, FP64).")
    print(
        f"Temperature round trips: max relative error {max(temperature_errors):.3g}, "
        f"max iterations {maximum_iterations}."
    )


def stability_sweep(table_path):
    table = load_table(table_path)
    density, temperature = np.meshgrid(np.logspace(4, 10, 9), np.logspace(5, 10, 9))
    density, temperature = density.ravel(), temperature.ravel()
    results = []
    for mean_mass in (4.0, 12.0, 1 / (0.5 / 12 + 0.5 / 16)):
        for coulomb in (False, True):

            def solve(density, temperature):
                state = evaluate(
                    table, density, temperature, mean_mass, mean_mass / 2, coulomb=coulomb
                )
                recovered = temperature_from_energy(
                    table, density, state.energy, mean_mass, mean_mass / 2, coulomb=coulomb
                )
                return state, recovered

            state, recovered = jax.jit(solve)(density, temperature)
            if not np.asarray(state.valid & recovered.converged).all():
                raise AssertionError("invalid or unconverged WD sweep state")
            if not np.asarray(state.dpdr > 0).all():
                raise AssertionError("negative isothermal compressibility")
            residual = density**2 * state.dedr - state.pressure + temperature * state.dpdt
            scale = np.maximum(np.abs(state.pressure), np.abs(temperature * state.dpdt))
            maxwell = float(np.max(np.abs(residual) / scale))
            if maxwell > 1e-12:
                raise AssertionError("Maxwell identity failed")
            error = np.abs(np.asarray(recovered.temperature) / temperature - 1)
            # A numerical allowance for the 36-term interpolant, not a rigorous error bound.
            bound = 2e-8 + 64 * np.asarray(recovered.relative_energy_resolution)
            if not (error <= bound).all():
                raise AssertionError(
                    f"round-trip error for A={mean_mass}, coulomb={coulomb}: "
                    f"rho={density[error > bound]}, T={temperature[error > bound]}, "
                    f"error={error[error > bound]}, bound={bound[error > bound]}"
                )
            results.append(
                dict(
                    mean_mass=mean_mass,
                    coulomb=coulomb,
                    cells=len(density),
                    maxwell=maxwell,
                    temperature_error=float(error.max()),
                    precision_limited=int(np.sum(error > 2e-8)),
                )
            )
    print(f"Passed stability/round-trip sweep: {sum(r['cells'] for r in results)} WD states.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    devices = jax.devices()
    if args.require_gpu and any(d.platform != "gpu" for d in devices):
        raise RuntimeError("GPU validation requested but JAX did not select GPUs")
    print(f"JAX {jax.__version__}: {devices}")
    check(args.table)
    sweep = stability_sweep(args.table) if args.sweep else []
    if args.report:
        args.report.write_text(
            json.dumps(
                dict(
                    jax=jax.__version__,
                    devices=list(map(str, devices)),
                    reference_passed=True,
                    sweep=sweep,
                ),
                indent=2,
            )
            + "\n"
        )
