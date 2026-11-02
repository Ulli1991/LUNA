"""Per-device FLOP/byte estimates of the compiled step from XLA's cost model (LUNA_FLOP_COUNT=1).

Estimates, not hardware counters.
"""

import jax


def _cost_dict(compiled):
    """Return Compiled.cost_analysis() as a dict across JAX versions, or None."""
    ca = compiled.cost_analysis()
    if ca is None:
        return None
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else None
    return ca if isinstance(ca, dict) else None


def cost_analysis(jitted_fn, *args):
    """Return flops/bytes/transcendentals of `jitted_fn(*args)`, or None if unavailable (not zero)."""
    try:
        ca = _cost_dict(jitted_fn.lower(*args).compile())
    except Exception:
        return None
    if ca is None:
        return None
    return {
        "flops": float(ca.get("flops", 0.0)),
        "bytes": float(ca.get("bytes accessed", 0.0)),
        "transcendentals": float(ca.get("transcendentals", 0.0)),
    }


def report_step_cost(step_fn, compute_dt_fn, U, dt, total_cells, is_main_process):
    """Print the per-cell-update cost of one step + dt; return FLOP per cell-update or None."""
    step_cost = cost_analysis(step_fn, U, dt)
    dt_cost = cost_analysis(compute_dt_fn, U)
    if step_cost is None:
        if is_main_process:
            print("(LUNA_FLOP_COUNT: cost_analysis unavailable on this backend)")
        return None
    if dt_cost is None:
        dt_cost = {"flops": 0.0, "bytes": 0.0, "transcendentals": 0.0}

    local_cells = total_cells / jax.device_count()
    flops = step_cost["flops"] + dt_cost["flops"]
    bytes_ = step_cost["bytes"] + dt_cost["bytes"]
    flops_per_cell = flops / local_cells
    if is_main_process:
        ai = flops / bytes_ if bytes_ > 0 else float("nan")
        print(
            f"XLA cost model (per device, per step): "
            f"{step_cost['flops']:.3e} FLOP step + {dt_cost['flops']:.3e} FLOP dt, "
            f"{bytes_:.3e} B accessed\n"
            f"  -> {flops_per_cell:.0f} FLOP/cell-update, "
            f"{bytes_ / local_cells:.0f} B/cell-update, "
            f"arithmetic intensity {ai:.2f} FLOP/B"
            + (f", transcendentals {(step_cost['transcendentals'] + dt_cost['transcendentals']) / local_cells:.0f}/cell"
               if step_cost["transcendentals"] or dt_cost["transcendentals"] else "")
        )
    return flops_per_cell
