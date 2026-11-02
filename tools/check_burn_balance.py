"""Measure final reaction balance and cooling time scales in a saved one-zone report."""

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from luna.eos.helmholtz import evaluate_checked, load_table
from luna.nuclear.aprox13 import (
    effective_rates,
    reaction_flows,
    fixed_temperature_rhs,
    mean_composition,
)
from luna.nuclear.neutrinos import sneut5


@jax.jit
def balance(density, temperature, fractions):
    rates = effective_rates(
        temperature, density=density, mass_fractions=fractions, screening="screen5"
    )
    forward = reaction_flows(density, fractions, rates.forward, jnp.zeros_like(rates.reverse))
    reverse = -reaction_flows(density, fractions, jnp.zeros_like(rates.forward), rates.reverse)
    gross = jnp.sum(forward + reverse)
    net = jnp.sum(abs(forward - reverse))
    derivative, heating = fixed_temperature_rhs(
        density, temperature, fractions, screening="screen5"
    )
    return net / jnp.maximum(gross, 1e-300), jnp.sum(abs(derivative)), heating


def diagnose(table_path, input_path, output):
    table = load_table(table_path)
    report = json.loads(input_path.read_text())
    if not report.get("passed"):
        raise ValueError("trajectory report has not passed its comparisons")
    results = []
    for case in report["cases"]:
        run = case["runs"][-1]
        temperature = run["temperature"][-1]
        fractions = np.asarray(run["states"][-1][:13])
        ratio, change_rate, heating = balance(case["density"], temperature, fractions)
        mass, charge = mean_composition(fractions)
        thermo = evaluate_checked(table, case["density"], temperature, mass, charge)
        cooling = float(sneut5(case["density"], temperature, mass, charge).total)
        entry = dict(
            name=case["name"],
            duration=case["times"][-1],
            temperature=temperature,
            helium=float(fractions[0]),
            carbon_oxygen=float(fractions[1:3].sum()),
            nickel=float(fractions[12]),
            net_to_gross_reaction_flow=float(ratio),
            abundance_change_per_second=float(change_rate),
            nuclear_heating=float(heating),
            neutrino_cooling=cooling,
            cooling_time=float(thermo.cv * temperature / cooling) if cooling > 0 else None,
        )
        results.append(entry)
        print(json.dumps(entry), flush=True)
    output.write_text(
        json.dumps(dict(input_report=str(input_path), cases=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("trajectories", type=Path)
    parser.add_argument("--report", type=Path, default=Path("burn-balance.json"))
    args = parser.parse_args()
    diagnose(args.table, args.trajectories, args.report)
