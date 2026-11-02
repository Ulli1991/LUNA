"""Check a one-zone helium burn with explicit screening; optional independent Radau integration."""

import argparse
import json
import os
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from luna.ode.rodas3 import Options, integrate
from luna.eos.helmholtz import load_table
from luna.eos.inversion import temperature_from_energy_checked
from luna.nuclear.aprox13 import MASS_NUMBERS, BINDING_MEV, MEV_TO_ERG, NA, mean_composition
from luna.nuclear.burn import BurnParams, initial_state, make_network


def check(table_path, duration, compare_scipy=False, screening="none", neutrinos="none"):
    table = load_table(table_path)
    network = make_network(table, screening=screening, neutrinos=neutrinos, rtol=1e-6)
    density, temperature = 1e6, 1e9
    params = BurnParams(density, temperature)
    start = initial_state(table, density, temperature, jnp.eye(13)[0])
    started = time.perf_counter()
    result = integrate(
        start[None],
        params,
        duration,
        net=network,
        opt=Options(max_steps=4000, tiers=((None, 1),), diff_fields=()),
    )
    jax.block_until_ready(result.y)
    integration_seconds = time.perf_counter() - started
    if np.asarray(result.failed).any():
        raise AssertionError(
            f"Rodas3 burn failed at {result.t}: steps={result.n_steps}, rejected={result.n_rejected}"
        )
    final = np.asarray(result.y[0])
    if final[:13].min() < 0 or abs(final[:13].sum() - 1) > 1e-12:
        raise AssertionError("burn violated abundance bounds or total mass")
    mean_mass, mean_charge = mean_composition(final[:13])
    recovered = temperature_from_energy_checked(
        table,
        density,
        final[-1],
        mean_mass,
        mean_charge,
        temperature_guess=result.params.temperature_guess[0],
    )
    thermal = final[-1] - float(start[-1])
    release = float(
        np.dot((final[:13] - np.asarray(start[:13])) / np.asarray(MASS_NUMBERS), BINDING_MEV)
        * NA
        * MEV_TO_ERG
    )
    budget = abs(thermal - release) / abs(release)
    if not (
        final[0] < 1
        and recovered.temperature > temperature
        and thermal > 0
        and (neutrinos != "none" or budget < 1e-7)
    ):
        raise AssertionError(f"failed burning/heating/energy budget: {budget}")
    report = dict(
        physics=dict(screening=screening, neutrinos=neutrinos),
        duration=duration,
        temperature=float(recovered.temperature),
        helium=float(final[0]),
        nuclear_thermal_difference=budget,
        steps=int(result.n_steps[0]),
        rejected=int(result.n_rejected[0]),
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        integration_seconds=integration_seconds,
    )
    print("Rodas3 check passed: " + json.dumps(report), flush=True)
    if compare_scipy or neutrinos != "none":
        from scipy.integrate import solve_ivp

        def rhs_function(state):
            derivative, aux = network.rhs(state[:14], params)
            return jnp.concatenate([derivative, aux["neutrino_loss"][None]])

        rhs = jax.jit(rhs_function)
        jacobian = jax.jit(jax.jacfwd(rhs_function))
        reference = solve_ivp(
            lambda time, state: np.asarray(rhs(state)),
            (0, duration),
            np.r_[np.asarray(start), 0.0],
            method="Radau",
            jac=lambda time, state: np.asarray(jacobian(state)),
            rtol=1e-9,
            atol=np.r_[np.full(13, 1e-14), 1e5, 1e2],
        )
        if not reference.success:
            raise AssertionError(reference.message)
        np.testing.assert_allclose(final[:13], reference.y[:13, -1], rtol=5e-5, atol=1e-12)
        reference_heat = reference.y[13, -1] - float(start[-1])
        np.testing.assert_allclose(thermal, reference_heat, rtol=5e-5, atol=1e5)
        report["radau_heating_relative_difference"] = abs(thermal - reference_heat) / abs(
            reference_heat
        )
        report["radau_comparison_passed"] = True
        escaped = float(reference.y[14, -1])
        budget = abs(thermal + escaped - release) / abs(release)
        report["escaped_neutrino_energy"] = escaped
        if budget > 1e-7:
            raise AssertionError(f"thermal + escaped neutrino energy budget: {budget}")
    report["energy_budget_error"] = budget
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--duration", type=float, default=1e-4)
    parser.add_argument("--scipy", action="store_true")
    parser.add_argument("--screening", choices=("none", "screen5"), default="none")
    parser.add_argument("--neutrinos", choices=("none", "sneut5"), default="none")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    report = check(args.table, args.duration, args.scipy, args.screening, args.neutrinos)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
