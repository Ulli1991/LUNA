"""Where a step's time goes: run a config with every step timed part by part, table + JSON.

    python tools/step_breakdown.py <config, run script or step_timing.jsonl> [name=value ...] [steps=10] [warmup=3] [json=path]"""
import json
import os
import statistics
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPTIONS = {"steps": 10, "warmup": 3, "json": ""}
QUIET = {"write_snapshots": False, "PLOT_AFTER_SIM": False, "enable_checkpointing": False,
         "cosmo_checkpoint_every": 0}


def options(argv):
    """(the tool's options, the rest of argv)."""
    opts, rest = dict(OPTIONS), []
    for tok in argv:
        k, _, v = tok.partition("=")
        if k in OPTIONS and v:
            opts[k] = type(OPTIONS[k])(v)
        else:
            rest.append(tok)
    return opts, rest


def load(path, argv):
    """The Config from a Config file or a run script with `config(...)`, quiet unless set."""
    import runpy

    from luna.setup.cli import apply, load_config, overrides, script_args, split_args
    mod = runpy.run_path(path)
    if callable(mod.get("config")):
        params, over = script_args(mod["config"], argv)
        cfg = apply(mod["config"](**params), over)
    else:
        over = overrides(split_args(argv)[1])
        cfg = apply(load_config(path), over)
    quiet = {k: v for k, v in QUIET.items() if k not in over and hasattr(cfg, k)}
    return replace(cfg, **quiet)


def summarize(records, warmup):
    """Median over the timed steps after `warmup` of each part's ms (0 where a step skips it)."""
    from luna.setup.step_timing import tree_order
    keep = [r for r in records if r["step"] > warmup] or records[-1:]
    names = tree_order(list(dict.fromkeys(k for r in keep for k in r["parts_ms"])))
    parts, spikes = {}, {}
    for k in names:
        lo = [r["parts_ms"].get(k, [0.0, 0.0])[0] for r in keep]
        hi = [r["parts_ms"].get(k, [0.0, 0.0])[1] for r in keep]
        parts[k] = [statistics.median(lo), statistics.median(hi)]
        cut = 2.0 * parts[k][1] + 1.0
        steps = [r["step"] for r, v in zip(keep, hi) if v > cut]
        if steps:
            spikes[k] = {"steps": steps, "max_ms": max(hi)}
    total = [statistics.median(r["total_ms"][0] for r in keep),
             statistics.median(r["total_ms"][1] for r in keep)]
    between = statistics.median(
        max(r["total_ms"][1] - sum(v[1] for k, v in r["parts_ms"].items() if "/" not in k), 0.0)
        for r in keep)
    return {"step": f"median of steps {keep[0]['step']}-{keep[-1]['step']}",
            "steps": [r["step"] for r in keep], "ranks": keep[-1]["ranks"],
            "total_ms": total, "between_ms": between, "parts_ms": parts, "spikes": spikes}


def report(summary):
    """The table, the parts far above their median on some steps, and those slower than the hydro."""
    from luna.setup.step_timing import format_table, over_hydro
    summary["over_hydro"] = over_hydro(summary["parts_ms"])
    lines = [format_table(summary).replace("⏱ step", "⏱")]
    for k, v in summary["spikes"].items():
        lines.append(f"  {k}: {v['max_ms']:.2f} ms at most, above twice its median on steps {v['steps']}")
    lines.append("parts slower than the hydro step: " + (", ".join(summary["over_hydro"]) or "none"))
    return "\n".join(lines)


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return None
    path, (opts, argv) = argv[0], options(argv[1:])
    if path.endswith(".jsonl"):
        recs, extra, out_dir = [json.loads(line) for line in open(path)], {}, os.path.dirname(path)
    else:
        cfg = replace(load(path, argv), timing_every=1, max_steps=opts["warmup"] + opts["steps"])
        from luna.setup.cli import launch
        launch(cfg)
        import jax
        if jax.process_index() != 0:
            return None
        from luna.setup.step_timing import JSONL
        recs = [json.loads(line) for line in open(os.path.join(cfg.output_dir, JSONL))]
        extra = dict(config=os.path.abspath(path), argv=argv, Nx=cfg.Nx, Ny=cfg.Ny, Nz=cfg.Nz,
                     devices=jax.device_count())
        out_dir = cfg.output_dir
    summary = summarize(recs, opts["warmup"])
    summary.update(extra)
    print("\n" + report(summary), flush=True)
    out = opts["json"] or os.path.join(out_dir, "step_breakdown.json")
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"wrote {out}")
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
