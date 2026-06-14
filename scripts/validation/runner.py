"""
Programmatic LUNA runner for validation sweeps.

Builds a Config object in Python (via dataclasses.replace on
template_config.Config) and delegates to main.run, so per-test plot scripts
can sweep over Nx / t_end / test_problem / etc. without shell scripting —
going through the SAME entry path as `python main.py`.

JAX-distributed init is one-shot per process — sweep() calls
initialize_jax_distributed() exactly once before its loop.
"""
from dataclasses import replace

from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


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
    for i, val in enumerate(scan_values):
        overrides = dict(base_overrides)
        overrides[scan_field] = val
        # For 2D/3D scans we usually want Nx=Ny=Nz=val unless explicitly set.
        if scan_field == "Nx":
            overrides.setdefault("Ny", val)
            overrides.setdefault("Nz", val)
        overrides["output_dir"] = output_dir_template.format(val)
        # Drop x64 sentinel before constructing Config (not a Config field).
        overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
        _, _, out = run(replace(Config(), **overrides))
        results.append((val, out))
    return results
