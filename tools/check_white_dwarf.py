"""Independent global virial and grid-refinement checks of the isothermal white-dwarf builder."""

import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from luna.eos.helmholtz import load_table
from luna.gravity.spherical_1d import shell_gravity
from luna.hydro.stellar_1d import make_solver
from luna.ic_generation.white_dwarf_1d import isothermal_model


def check(table_path, output):
    table = load_table(table_path)
    fractions = jnp.zeros(13).at[1:3].set(0.5)
    report = dict(
        jax=jax.__version__,
        devices=list(map(str, jax.devices())),
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
        central_density=1e7,
        temperature=1e7,
        ambient_density=1e-4,
        passed=False,
        runs=[],
    )
    for cells in (64, 128, 256):
        print(f"{cells} cells: white dwarf virial balance", flush=True)
        dx = 1.28e9 / cells
        model = isothermal_model(table, 1e7, 1e7, fractions, cells, dx)
        thermo = make_solver(table).recover(model.state, model.temperature)
        gravity = shell_gravity(model.state[0], dx)
        radii = dx * jnp.arange(cells + 1)
        volumes = 4 * jnp.pi / 3 * (radii[1:] ** 3 - radii[:-1] ** 3)
        pressure = (
            3 * jnp.dot(thermo.pressure, volumes)
            - 4 * jnp.pi * radii[-1] ** 3 * thermo.pressure[-1]
        )
        error = abs(float(pressure + gravity.binding_energy)) / abs(float(gravity.binding_energy))
        report["runs"].append(
            dict(
                cells=cells,
                surface_radius=model.surface_radius,
                mass_solar=float(gravity.mass) / 1.98847e33,
                virial_relative_error=error,
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["runs"][-1]), flush=True)
    errors = [r["virial_relative_error"] for r in report["runs"]]
    if errors[-1] > 1e-3 or max(b / a for a, b in zip(errors, errors[1:])) > 0.4:
        raise AssertionError(
            f"white dwarf virial balance did not converge at second order: {errors}"
        )
    report["passed"] = True
    report["scope"] = "initial model virial balance; evolution stability is a separate gate"
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("white-dwarf.json"))
    args = parser.parse_args()
    check(args.table, args.report)
