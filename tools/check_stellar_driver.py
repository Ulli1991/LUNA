"""Hotspot, radial-gravity, ignition and restart checks through the stellar runner."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import jax

jax.config.update("jax_enable_x64", True)
import h5py
import numpy as np

from luna.stellar import Config, run
from luna.gravity.spherical_1d import shell_gravity
from luna.nuclear.aprox13 import MASS_NUMBERS, BINDING_MEV, NA, MEV_TO_ERG


def read(path):
    with h5py.File(path) as source:
        return np.asarray(source["state"]), np.asarray(source["temperature"])


def check(table_path, output, case="all"):
    report = dict(
        jax=jax.__version__, devices=list(map(str, jax.devices())), passed=False, runs=[],
        table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
    )
    cases = ("he", "co", "white-dwarf", "white-dwarf-ignition") if case == "all" else (case,)
    binding = np.asarray(BINDING_MEV) / np.asarray(MASS_NUMBERS) * NA * MEV_TO_ERG
    with tempfile.TemporaryDirectory(prefix="luna-stellar-") as temporary:
        root = Path(temporary)
        for name in cases:
            print(f"stellar driver: {name}", flush=True)
            if name == "white-dwarf-ignition":
                cfg = Config(
                    case="white-dwarf",
                    cells=32,
                    length=1.28e9,
                    density=1e7,
                    temperature=1e7,
                    hot_temperature=3e9,
                    ignition_radius=1.2e8,
                    geometry="spherical",
                    gravity=True,
                    well_balanced=True,
                )
                stop, dt_max = 4e-8, 1e-8
            elif name == "white-dwarf":
                cfg = Config(
                    case=name,
                    cells=128,
                    length=1.28e9,
                    density=1e7,
                    temperature=1e7,
                    burn=False,
                    geometry="spherical",
                    gravity=True,
                )
                stop, dt_max = 0.001, 1e-4
            else:
                cfg = Config(
                    case=name,
                    cells=16,
                    length=1e7,
                    density=1e6 if name == "he" else 1e7,
                    temperature=1e8,
                    hot_temperature=2e9 if name == "he" else 3e9,
                )
                stop, dt_max = 1e-5, 2e-6
            result = run(table_path, cfg, stop, root / name, dt_max=dt_max)
            print(f"{name}: full run complete ({result['steps']} steps)", flush=True)
            initial, initial_temperature = read(root / name / "state_00000000.h5")
            final, temperature = read(result["checkpoint"])
            np.testing.assert_allclose(final[2:-1].sum(axis=0), final[0], rtol=3e-12)
            dx = cfg.length / cfg.cells
            if cfg.gravity:
                radii = dx * np.arange(cfg.cells + 1)
                volumes = 4 * np.pi / 3 * (radii[1:] ** 3 - radii[:-1] ** 3)
                gravity_change = float(
                    shell_gravity(final[0], dx).binding_energy
                    - shell_gravity(initial[0], dx).binding_energy
                )
                boundary = 4 * np.pi * np.asarray(result["diagnostics"]["boundary_budget"])
            else:
                volumes = np.full(cfg.cells, dx)
                gravity_change = 0.0
                boundary = np.asarray(result["diagnostics"]["boundary_budget"])
            initial_energy = float(np.dot(initial[-1] - binding @ initial[2:-1], volumes))
            final_energy = float(np.dot(final[-1] - binding @ final[2:-1], volumes))
            escape = boundary[-1] - np.dot(binding, boundary[2:-1])
            budget = abs(final_energy - initial_energy + gravity_change - escape) / abs(
                initial_energy
            )
            mass_error = abs(np.dot(final[0] - initial[0], volumes) - boundary[0]) / np.dot(
                initial[0], volumes
            )
            if budget > 2e-12 or mass_error > 2e-13:
                raise AssertionError(f"{name}: driver budget error {budget}, {mass_error}")
            species = 0 if name == "he" else 1
            burned = float(
                np.dot(initial[2 + species] - final[2 + species], volumes) + boundary[2 + species]
            )
            if cfg.burn and burned <= 0:
                raise AssertionError("hotspot did not burn fuel")
            entry = dict(
                case=name,
                config=asdict(cfg),
                steps=result["steps"],
                time=result["time"],
                energy_budget_error=budget,
                mass_budget_error=float(mass_error),
                burned_fuel_mass=burned,
                maximum_velocity=float(np.max(abs(final[1] / final[0]))),
                maximum_temperature=float(temperature.max()),
                diagnostics=result["diagnostics"],
            )
            if name in ("he", "white-dwarf-ignition"):
                split_time = 4e-6 if name == "he" else 2e-8
                half = run(table_path, cfg, split_time, root / "split", dt_max=dt_max)
                resumed = run(
                    table_path,
                    cfg,
                    stop,
                    root / "resumed",
                    restart=half["checkpoint"],
                    dt_max=dt_max,
                )
                restored, restored_temperature = read(resumed["checkpoint"])
                np.testing.assert_array_equal(restored, final)
                np.testing.assert_array_equal(restored_temperature, temperature)
                entry["restart_bitwise_equal"] = True
                if name == "white-dwarf-ignition":
                    with h5py.File(root / name / "state_00000000.h5") as source:
                        reference = np.asarray(source["equilibrium"])
                    with h5py.File(resumed["checkpoint"]) as source:
                        np.testing.assert_array_equal(source["equilibrium"], reference)
                    np.testing.assert_array_equal(reference[:-1], initial[:-1])
                    injected = float(np.dot(initial[-1] - reference[-1], volumes))
                    np.testing.assert_allclose(
                        injected, resumed["diagnostics"]["ignition_energy"], rtol=2e-14
                    )
                    assert (
                        resumed["diagnostics"]["ignition_energy"]
                        == result["diagnostics"]["ignition_energy"]
                    )
                    assert entry["maximum_velocity"] > 0
                    # Sum changes before integrating to avoid subtracting bulk binding energies.
                    energy_residual = float(
                        np.dot(
                            final[-1] - initial[-1] - binding @ (final[2:-1] - initial[2:-1]),
                            volumes,
                        )
                    ) + gravity_change - escape
                    thermal_gravity_scale = abs(float(np.dot(initial[-1], volumes))) + abs(
                        float(shell_gravity(initial[0], dx).binding_energy)
                    )
                    entry["thermal_gravity_budget_error"] = (
                        abs(energy_residual) / thermal_gravity_scale
                    )
                    if entry["thermal_gravity_budget_error"] > 2e-12:
                        raise AssertionError("ignited WD thermal/gravity budget failed")
                    entry["cold_reference_preserved"] = True
            if name == "white-dwarf":
                entry["maximum_density_relative_change"] = float(
                    np.max(abs(final[0] / initial[0] - 1))
                )
                entry["equilibrium_validated"] = False
            report["runs"].append(entry)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
    report["passed"] = True
    report["scope"] = (
        "hotspot/central-ignition smoke tests and short radial evolution; "
        "no detonation-speed, sustained propagation or explosion gate"
    )
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("stellar-driver.json"))
    parser.add_argument(
        "--case", choices=("all", "he", "co", "white-dwarf", "white-dwarf-ignition"), default="all"
    )
    args = parser.parse_args()
    check(args.table, args.report, args.case)
