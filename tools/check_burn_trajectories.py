"""Compare screened, cooled He and C/O burn trajectories and a Rodas3 tolerance ladder."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from luna.ode.rodas3 import Options, integrate
from luna.reference_data import reference_path
from luna.eos.helmholtz import load_table
from luna.eos.inversion import temperature_from_energy_checked
from luna.nuclear.aprox13 import MASS_NUMBERS, BINDING_MEV, MEV_TO_ERG, NA, mean_composition
from luna.nuclear.burn import BurnParams, initial_state, make_network


def energy_budget(states, start, escaped, binding):
    """Energy residual, release scale and a floating-point allowance for the linear invariant."""
    release = (states[:, :13] - start[:13]) @ binding
    error = abs(states[:, 13] - start[13] + escaped - release)
    scale = np.maximum(abs(release), 1e8)
    roundoff = (
        64
        * np.finfo(float).eps
        * (
            abs(states[:, 13])
            + abs(start[13])
            + (abs(states[:, :13]) + abs(start[:13])) @ abs(binding)
        )
    )
    return error, scale, roundoff


def validate(table_path, output, selected_case="all", extended=False):
    table = load_table(table_path)
    network = make_network(table, screening="screen5", neutrinos="sneut5")
    cases = [
        ("helium", 1e6, 1e9, np.eye(13)[0], np.geomspace(1e-4, 1e-2, 9)),
        (
            "carbon_oxygen",
            1e9,
            1e9,
            (np.eye(13)[1] + np.eye(13)[2]) / 2,
            np.geomspace(1e-4, 6e-2, 13),
        ),
    ]
    cases.append(
        (
            "hot_carbon_oxygen",
            1e7,
            2e9,
            (np.eye(13)[1] + np.eye(13)[2]) / 2,
            np.geomspace(1e-7, 1e-2, 16),
        )
    )
    if extended:
        carbon_rich = np.zeros(13)
        carbon_rich[1:3] = (0.7, 0.3)
        oxygen_rich = np.zeros(13)
        oxygen_rich[1:3] = (0.3, 0.7)
        cases = [
            ("dilute_helium", 1e5, 2e9, np.eye(13)[0], np.geomspace(1e-6, 0.1, 13)),
            ("hot_helium", 1e6, 3e9, np.eye(13)[0], np.geomspace(1e-7, 0.1, 16)),
            ("helium_exhaustion", 1e6, 3e9, np.eye(13)[0], np.geomspace(1e-5, 10.0, 16)),
            ("carbon_rich", 1e7, 2e9, carbon_rich, np.geomspace(1e-7, 0.1, 16)),
            ("oxygen_rich", 1e8, 2e9, oxygen_rich, np.geomspace(1e-8, 0.01, 16)),
            ("nickel_relaxation", 1e7, 5e9, np.eye(13)[12], np.geomspace(1e-10, 0.01, 17)),
        ]
    tolerances = (1e-5, 1e-6, 1e-7)
    cases = [case for case in cases if selected_case in ("all", case[0])]
    if not cases:
        raise ValueError("unknown case for the selected suite")
    report = dict(
        passed=False,
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        physics=dict(screening="screen5", neutrinos="sneut5"),
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        reference_solver=dict(
            method="Radau",
            rtol=1e-10,
            abundance_atol=1e-15,
            energy_atol=1e3,
            escaped_energy_atol=1e-2,
            first_step=1e-12,
        ),
        cases=[],
    )
    root = Path(__file__).parents[1]
    sources = [
        *root.glob("src/luna/nuclear/*.py"),
        root / "src/luna/eos/helmholtz.py",
        root / "src/luna/eos/inversion.py",
        root / "src/luna/ode/rodas3.py",
    ]
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(sources)
    }
    binding = np.asarray(BINDING_MEV) / np.asarray(MASS_NUMBERS) * NA * MEV_TO_ERG

    def reference_rhs(state, params):
        derivative, aux = network.rhs(state[:14], params)
        return jnp.concatenate([derivative, aux["neutrino_loss"][None]])

    rhs = jax.jit(reference_rhs)
    jacobian = jax.jit(jax.jacfwd(reference_rhs))
    prepared = []
    for name, density, temperature, fractions, times in cases:
        params = BurnParams(density, temperature)
        start = np.asarray(initial_state(table, density, temperature, fractions))
        print(f"{name}: Radau reference to {times[-1]:g} s", flush=True)

        def checked_rhs(t, y):
            value = np.asarray(rhs(y, params))
            if not np.isfinite(value).all():
                _, aux = network.rhs(jnp.asarray(y[:14]), params)
                raise ValueError(
                    f"{name}: nonfinite Radau RHS at t={t}; min X={y[:13].min()}, sum X={y[:13].sum()}, energy={y[13]}, aux={aux}"
                )
            return value

        reference = solve_ivp(
            checked_rhs,
            (0, times[-1]),
            np.r_[start, 0.0],
            method="Radau",
            first_step=1e-12,
            jac=lambda t, y: np.asarray(jacobian(y, params)),
            t_eval=times,
            rtol=1e-10,
            atol=np.r_[np.full(13, 1e-15), 1e3, 1e-2],
        )
        if not reference.success or not np.isfinite(reference.y).all():
            raise AssertionError(f"{name}: Radau failed: {reference.message}")
        if reference.y[:13].min() < -1e-14 or np.max(abs(reference.y[:13].sum(axis=0) - 1)) > 1e-10:
            raise AssertionError(f"{name}: reference abundance bounds or mass conservation failed")
        mass, charge = mean_composition(reference.y[:13].T, negative_tolerance=1e-14)
        reference_temperature = temperature_from_energy_checked(
            table, density, reference.y[13], mass, charge, temperature_guess=temperature
        ).temperature
        residual, scale, roundoff = energy_budget(reference.y.T, start, reference.y[14], binding)
        reference_budget = np.max(residual / scale)
        if np.any(residual > 1e-8 * scale + roundoff):
            raise AssertionError(f"{name}: reference energy budget failed: {reference_budget}")
        entry = dict(
            name=name,
            density=density,
            initial_temperature=temperature,
            initial_fractions=fractions.tolist(),
            times=times.tolist(),
            reference=reference.y.T.tolist(),
            reference_temperature=np.asarray(reference_temperature).tolist(),
            reference_energy_budget_error=float(reference_budget),
            reference_energy_budget_excess_over_roundoff=float(
                np.max(np.maximum(residual - roundoff, 0) / scale)
            ),
            reference_nfev=reference.nfev,
            runs=[],
        )
        report["cases"].append(entry)
        prepared.append((entry, start, params, reference.y.T))
        print(
            f"{name}: reference complete, Tfinal={float(reference_temperature[-1]):.9g} K; {reference.nfev} RHS evaluations",
            flush=True,
        )
        output.write_text(json.dumps(report, indent=2) + "\n")

    for tolerance in tolerances:
        options = Options(max_steps=20000, tiers=((None, 1),), diff_fields=(), rtol=tolerance)
        for entry, start, params, reference in prepared:
            state, previous = jnp.asarray(start[None]), 0.0
            warm_params = BurnParams(
                jnp.asarray([params.density]), jnp.asarray([params.temperature_guess])
            )
            trajectory, steps, rejected = [], 0, 0
            started = time.perf_counter()
            for target in entry["times"]:
                result = integrate(state, warm_params, target - previous, net=network, opt=options)
                jax.block_until_ready(result.y)
                if bool(result.failed[0]):
                    raise AssertionError(
                        f"{entry['name']} rtol={tolerance}: failed at {previous + float(result.t[0])} s; accepted={result.n_steps}, rejected={result.n_rejected}"
                    )
                state, warm_params, previous = result.y, result.params, target
                abundances = np.asarray(state[0, :13])
                if abundances.min() < 0 or abs(abundances.sum() - 1) > 1e-10:
                    raise AssertionError("abundance bounds or mass conservation failed")
                trajectory.append(np.asarray(state[0]))
                steps += int(result.n_steps[0])
                rejected += int(result.n_rejected[0])
            trajectory = np.asarray(trajectory)
            mass, charge = mean_composition(trajectory[:, :13])
            temperatures = np.asarray(
                temperature_from_energy_checked(
                    table,
                    params.density,
                    trajectory[:, 13],
                    mass,
                    charge,
                    temperature_guess=params.temperature_guess,
                ).temperature
            )
            thermal = trajectory[:, 13] - start[13]
            reference_heat = reference[:, 13] - start[13]
            heat_error = np.max(
                abs(thermal - reference_heat) / np.maximum(abs(reference_heat), 1e8)
            )
            abundance_error = np.max(np.sum(abs(trajectory[:, :13] - reference[:, :13]), axis=1))
            temperature_error = float(
                np.max(abs(temperatures / np.asarray(entry["reference_temperature"]) - 1))
            )
            residual, scale, roundoff = energy_budget(trajectory, start, reference[:, 14], binding)
            budget = np.max(residual / scale)
            budget_excess = float(np.max(np.maximum(residual - roundoff, 0) / scale))
            run = dict(
                rtol=tolerance,
                states=trajectory.tolist(),
                temperature=temperatures.tolist(),
                steps=steps,
                rejected=rejected,
                seconds=time.perf_counter() - started,
                max_abundance_l1_error=float(abundance_error),
                max_heating_relative_error=float(heat_error),
                max_temperature_relative_error=temperature_error,
                max_energy_budget_relative_error=float(budget),
                max_energy_budget_excess_over_roundoff=budget_excess,
            )
            if entry["name"] == "helium":
                golden = json.loads(
                    (
                        Path(reference_path("microphysics/helium_burn/helium_burn_reference.json"))
                    ).read_text()
                )
                bounds = golden["comparison_tolerances"]
                np.testing.assert_allclose(
                    temperatures[-1], golden["final_temperature"], rtol=bounds["temperature_rtol"]
                )
                np.testing.assert_allclose(
                    thermal[-1], golden["added_energy"], rtol=bounds["energy_rtol"]
                )
                np.testing.assert_allclose(
                    trajectory[-1, :13],
                    golden["final_mass_fractions"],
                    rtol=bounds["abundance_rtol"],
                    atol=bounds["abundance_atol"],
                )
                run["microphysics_benchmark_passed"] = True
            entry["runs"].append(run)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"{entry['name']} rtol={tolerance:g}: {steps} steps, {rejected} rejected; abundance L1={abundance_error:.3g}, heat={heat_error:.3g}, budget={budget:.3g}, Tfinal={temperatures[-1]:.9g}",
                flush=True,
            )
            if (
                max(abundance_error, heat_error, temperature_error) > 100 * tolerance
                or budget_excess > 1e-7
            ):
                raise AssertionError(f"{entry['name']}: trajectory comparison failed")
    for entry in report["cases"]:
        first, last = entry["runs"][0], entry["runs"][-1]
        for metric in ("max_abundance_l1_error", "max_heating_relative_error"):
            if last[metric] > first[metric] + 1e-9:
                raise AssertionError(
                    f"{entry['name']}: tightening tolerance did not improve {metric}"
                )
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("burn-trajectories.json"))
    parser.add_argument("--case", default="all")
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    validate(args.table, args.report, args.case, args.extended)
