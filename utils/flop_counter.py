"""XLA cost-model FLOP/byte accounting for the compiled step functions.

Numbers come from `Compiled.cost_analysis()` on the post-fusion HLO module,
i.e. the same executable the time loop runs — an XLA cost-model *estimate*,
not a hardware counter (Nsight is the ground truth if the two disagree).
In SPMD mode every process compiles the same per-device module, so the
returned figures are per-device; per-cell values therefore divide by the
LOCAL (per-device) cell count and hold at any scale.

Opt-in via LUNA_FLOP_COUNT=1 (runner_distributed). The extra .lower().compile()
is served from the jit/persistent-compilation cache when available; at worst
it repeats one compile at startup and never touches the step loop.
"""

import jax


def _cost_dict(compiled):
    """Normalize Compiled.cost_analysis() across JAX versions (dict vs [dict])."""
    ca = compiled.cost_analysis()
    if ca is None:
        return None
    if isinstance(ca, (list, tuple)):
        ca = ca[0] if ca else None
    return ca if isinstance(ca, dict) else None


def cost_analysis(jitted_fn, *args):
    """flops / bytes / transcendentals of `jitted_fn(*args)`, or None.

    None means the backend does not implement cost analysis (or `jitted_fn`
    is not an AOT-lowerable jit object) — callers must treat the feature as
    unavailable, not as zero cost.
    """
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
    """Print the per-cell-update cost of one time-loop iteration (step + dt).

    Returns flops per cell-update (float) for live TFLOP/s reporting in the
    perf line, or None if cost analysis is unavailable.
    """
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
