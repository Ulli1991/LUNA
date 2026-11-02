"""Check prescribed-ash CJ conditions against independently compiled external routines."""

import argparse
import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

from luna.eos.helmholtz import load_table
from luna.nuclear.detonation import prescribed_ash_cj
from luna.reference_data import reference_path


def check(table_path, output, reference=None):
    reference = Path(reference or reference_path("microphysics/detonation_cj/detonation_cj_reference.json"))
    fixture = json.loads(reference.read_text())
    table_hash = hashlib.sha256(table_path.read_bytes()).hexdigest()
    if table_hash != fixture["table_sha256"]:
        raise ValueError("CJ reference EOS table differs")
    table = load_table(table_path)
    report = dict(
        source=fixture["source"],
        table_sha256=table_hash,
        jax=jax.__version__,
        scope=fixture["scope"],
        passed=False,
        runs=[],
    )
    for row, expected in zip(fixture["inputs"], fixture["values"], strict=True):
        actual = prescribed_ash_cj(
            table,
            row["density"],
            row["temperature"],
            row["fuel"],
            row["ash"],
            coulomb=row["coulomb"],
        )
        values = np.array(actual[:4])
        error = float(np.max(abs(values / np.array(expected) - 1)))
        report["runs"].append(
            dict(
                case=row["case"],
                fuel_density=row["density"],
                coulomb=row["coulomb"],
                ash_density=actual.density,
                ash_temperature=actual.temperature,
                speed=actual.speed,
                jump_residual=actual.residual,
                upstream_relative_error=error,
            )
        )
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["runs"][-1]), flush=True)
        if error > 2e-7:
            raise AssertionError(f"prescribed-ash CJ reference mismatch: {error}")
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", type=Path)
    parser.add_argument("--report", type=Path, default=Path("detonation-reference.json"))
    args = parser.parse_args()
    check(args.table, args.report)
