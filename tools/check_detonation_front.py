"""Check front-displacement sensitivity using saved detonation snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def check(report_path, output):
    report = json.loads(report_path.read_text())
    runs = []
    for run in report["runs"]:
        cells = run["cells"]
        path = report_path.with_suffix(f".{cells}.npz")
        with np.load(path) as snapshot:
            for key in ("reference_sha256", "table_sha256"):
                if str(snapshot[key]) != report[key]:
                    raise ValueError(f"snapshot {key} mismatch: {path}")
            dx, duration = float(snapshot["dx"]), float(snapshot["time"])

            def position(state, width):
                gradient = np.maximum(-np.diff(state[0]), 0)
                peak = gradient.argmax()
                window = np.arange(max(peak - width, 0), min(peak + width + 1, cells - 1))
                if not np.isfinite(gradient).all() or gradient[window].sum() <= 0:
                    raise ValueError(f"no finite shock gradient: {path}")
                return np.dot(window + 1, gradient[window]) / gradient[window].sum() * dx

            errors = {}
            for width in (1, 2, 3, 4, 8):
                measured = (
                    position(snapshot["state"], width) - position(snapshot["initial"], width)
                ) / duration
                errors[str(width)] = float(abs(measured / report["reference_speed"] - 1))
            runs.append(dict(cells=cells, translation_speed_errors_by_gradient_window=errors))
    result = dict(
        scope="snapshot displacement diagnostic; not a replacement for time-history speed fits",
        report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
        runs=runs,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.report, args.output)
