"""Write stellar budget histories and radial profiles from compatible spherical checkpoints."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import h5py
import numpy as np

from luna.nuclear.aprox13 import SPECIES
from luna.stellar import Config, read_checkpoint
from luna.stellar_diagnostics import measure, budgets


def analyze(initial_path, paths, report_path, profiles_path):
    with h5py.File(initial_path) as source:
        cfg = Config(**json.loads(source.attrs["config"]))
        table_hash = source.attrs["table_sha256"]
    cfg.validate()
    if cfg.geometry != "spherical" or not cfg.gravity or cfg.neutrinos != "none":
        raise ValueError("budgets require spherical self-gravity without neutrino losses")
    state, temperature, time, steps, initial_diagnostics = read_checkpoint(initial_path, cfg, table_hash)
    if time != 0 or steps != 0:
        raise ValueError("initial checkpoint must precede all evolution")
    initial, _ = measure(state, temperature, cfg.length / cfg.cells)
    candidates = {Path(initial_path).resolve(), *(Path(path).resolve() for path in paths)}
    ordered = []
    for path in candidates:
        with h5py.File(path) as source:
            ordered.append((int(source.attrs["steps"]), path))
    ordered.sort()
    if len({step for step, _ in ordered}) != len(ordered):
        raise ValueError("ambiguous duplicate checkpoint steps")
    report = dict(
        scope="exploratory spherical evolution; budget checks do not validate a detonation or yields",
        table_sha256=table_hash,
        config=asdict(cfg),
        ignition_energy_erg=initial_diagnostics.get("ignition_energy", 0.0),
        profiles=str(profiles_path),
        runs=[],
        budget_checks_passed=True,
    )
    previous = -np.inf
    with h5py.File(profiles_path, "x") as profiles:
        profiles.attrs["schema"] = "luna-stellar-profiles-v1"
        profiles.attrs["species"] = json.dumps(SPECIES)
        profiles.attrs["table_sha256"] = table_hash
        profiles.attrs["config"] = json.dumps(asdict(cfg))
        for _, path in ordered:
            state, temperature, time, steps, diagnostics = read_checkpoint(path, cfg, table_hash)
            if time <= previous:
                raise ValueError("checkpoint times must increase with steps")
            previous = time
            if diagnostics.get("ignition_energy", 0.0) != report["ignition_energy_erg"]:
                raise ValueError("checkpoint ignition energy changed")
            current, radial = measure(state, temperature, cfg.length / cfg.cells)
            budget = budgets(initial, current, diagnostics["boundary_budget"])
            report["budget_checks_passed"] &= (
                budget["mass_budget_error"] < 2e-12 and budget["energy_budget_error"] < 1e-9
            )
            report["runs"].append(dict(
                checkpoint=str(path), time_s=time, steps=steps, **current, **budget,
                controller=diagnostics,
            ))
            group = profiles.create_group(f"step_{steps:08d}")
            group.attrs["time_s"] = time
            group.attrs["checkpoint"] = str(path)
            for name, data in radial.items():
                group.create_dataset(name, data=data, compression="gzip")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["budget_checks_passed"]:
        raise AssertionError("stellar mass/energy budget checks failed; inspect report")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.initial, args.checkpoints, args.report, args.profiles)
    print(json.dumps(dict(checkpoints=len(result["runs"]),
                          final_time_s=result["runs"][-1]["time_s"],
                          budget_checks_passed=result["budget_checks_passed"])))
