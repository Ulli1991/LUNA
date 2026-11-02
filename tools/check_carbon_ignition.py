"""Measure dense C/O ignition and isolate screening, neutrino and EOS Coulomb contributions."""

import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

from luna.eos.helmholtz import load_table
from luna.eos.inversion import temperature_from_energy_checked
from luna.nuclear.aprox13 import MASS_NUMBERS, BINDING_MEV, NA, MEV_TO_ERG, mean_composition
from luna.nuclear.burn import BurnParams, initial_state, make_network


def measure(table_path, output, selected_case="all", rtol=1e-9):
    table = load_table(table_path)
    params = BurnParams(1e9, 1e9)
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    report = dict(
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        solver=dict(
            method="Radau",
            rtol=rtol,
            abundance_atol=1e-15,
            energy_atol=1e3,
            escaped_energy_atol=1e-2,
            first_step=1e-12,
        ),
        benchmark=dict(
            source="https://arxiv.org/pdf/astro-ph/0510367",
            table=1,
            density=1e9,
            temperature=1e9,
            carbon=0.5,
            oxygen=0.5,
            endpoint_carbon=0.05,
            ignition_time=0.0596,
            printed_time_half_width=5e-5,
        ),
        cases=[],
    )
    settings = [
        ("baseline", "screen5", "sneut5", True),
        ("no_neutrinos", "screen5", "none", True),
        ("no_eos_coulomb", "screen5", "sneut5", False),
        ("no_screening", "none", "sneut5", True),
    ]
    for name, screening, neutrinos, coulomb in settings:
        if selected_case not in ("all", name):
            continue
        print(f"{name}: integrating to 90% carbon consumption", flush=True)
        network = make_network(table, screening=screening, neutrinos=neutrinos, coulomb=coulomb)
        start = np.asarray(
            initial_state(
                table, params.density, params.temperature_guess, fractions, coulomb=coulomb
            )
        )

        def rhs_function(state):
            derivative, aux = network.rhs(state[:14], params)
            return jnp.concatenate([derivative, aux["neutrino_loss"][None]])

        rhs = jax.jit(rhs_function)
        jacobian = jax.jit(jax.jacfwd(rhs_function))

        def checked_rhs(t, state):
            derivative = np.asarray(rhs(state))
            if not np.isfinite(derivative).all():
                raise ValueError(f"{name}: invalid trial state at {t}: {state}")
            return derivative

        def ignition(t, state):
            return state[1] - 0.05

        ignition.terminal = True
        ignition.direction = -1
        result = solve_ivp(
            checked_rhs,
            (0, 10.0),
            np.r_[start, 0.0],
            method="Radau",
            jac=lambda t, y: np.asarray(jacobian(y)),
            first_step=1e-12,
            rtol=rtol,
            atol=np.r_[np.full(13, 1e-15), 1e3, 1e-2],
            events=ignition,
        )
        if not result.success:
            raise AssertionError(result.message)
        final = result.y[:, -1]
        mass, charge = mean_composition(final[:13], negative_tolerance=1e-14)
        temperature = temperature_from_energy_checked(
            table,
            params.density,
            final[13],
            mass,
            charge,
            temperature_guess=params.temperature_guess,
            coulomb=coulomb,
        ).temperature
        release = float(
            np.dot((final[:13] - start[:13]) / np.asarray(MASS_NUMBERS), BINDING_MEV)
            * NA
            * MEV_TO_ERG
        )
        budget = abs(final[13] - start[13] + final[14] - release) / max(abs(release), 1e8)
        if abs(final[:13].sum() - 1) > 1e-10 or final[:13].min() < -1e-14 or budget > 1e-7:
            raise AssertionError(f"{name}: abundance or energy budget failed: {budget}")
        time = float(result.t_events[0][0]) if len(result.t_events[0]) else None
        entry = dict(
            name=name,
            screening=screening,
            neutrinos=neutrinos,
            eos_coulomb=coulomb,
            ignition_time=time,
            benchmark_ratio=time / 0.0596 if time is not None else None,
            final_time=float(result.t[-1]),
            final_temperature=float(temperature),
            final_state=final.tolist(),
            energy_budget_error=float(budget),
            nfev=result.nfev,
        )
        if name == "no_eos_coulomb":
            if time is None:
                raise AssertionError("carbon ignition endpoint was not reached")
            np.testing.assert_allclose(
                time,
                report["benchmark"]["ignition_time"],
                rtol=0,
                atol=report["benchmark"]["printed_time_half_width"],
            )
            entry["published_time_match"] = True
        report["cases"].append(entry)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("carbon-ignition.json"))
    parser.add_argument(
        "--case",
        choices=("all", "baseline", "no_neutrinos", "no_eos_coulomb", "no_screening"),
        default="all",
    )
    parser.add_argument("--rtol", type=float, default=1e-9)
    args = parser.parse_args()
    if not np.isfinite(args.rtol) or args.rtol <= 0:
        parser.error("rtol must be positive and finite")
    measure(args.table, args.report, args.case, args.rtol)
