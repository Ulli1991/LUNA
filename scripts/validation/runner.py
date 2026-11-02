"""Programmatic sweeps for the plot scripts: a Config per scan value through
main.run, the same entry path as `python main.py`. JAX is initialised once per
process, before the loop."""
from dataclasses import replace

from luna.template_config import Config
from luna.setup.jax_init import initialize_jax_distributed
from luna.main import run


def sweep(base_overrides, scan_field, scan_values, output_dir_template):
    """Run a sequence of simulations varying a single field.

    base_overrides : dict
        Overrides applied to every run.
    scan_field : str
        Config field name to vary (e.g. "Nx").
    scan_values : iterable
        Values of scan_field to run.
    output_dir_template : str
        Format string with one {} placeholder for the scan value.
        e.g. "snapshots/sod_{}"

    Returns a list of (value, output_dir) pairs.
    """
    results = []
    initialize_jax_distributed(enable_x64=bool(base_overrides.get("enable_x64", False)))
    for val in scan_values:
        overrides = dict(base_overrides)
        overrides[scan_field] = val
        if scan_field == "Nx":                     # 2D/3D scans: a cube unless set
            overrides.setdefault("Ny", val)
            overrides.setdefault("Nz", val)
        overrides["output_dir"] = output_dir_template.format(val)
        res = run(replace(Config(), **overrides))
        results.append((val, res.output_dir))
    return results
