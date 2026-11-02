"""What a gradient of LUNA costs, measured.

    python tools/bench_autodiff.py --out bench_out            # the matrix
    python tools/bench_autodiff.py --case hydro --nx 64 --steps 8 --json -
    python tools/bench_autodiff.py --out bench_out --max-rollout hydro:64

The numbers the differentiable-simulation papers report for a solver are the
same four, and they are what this writes out:

    forward_s     one rollout, compiled, blocked on
    grad_s        value-and-gradient of a scalar loss of that rollout
    ratio         grad_s / forward_s -- the price of the adjoint
    peak_bytes    the device high-water mark of the gradient

against the two axes that actually bind: the rollout LENGTH (the tape) and
the RESOLUTION (the state). Plus the one number a user hits first -- the
longest rollout that still fits -- and the same gradient on one device and
on several.

Every case runs in its OWN PROCESS. Three reasons, all of them learned the
hard way: the device high-water mark is not resettable, so a shared process
reports the largest case for all of them; an out-of-memory rollout takes the
runtime with it and would end the sweep; and a compile cache would hide the
compile cost of everything after the first case.

The rollout is the production step (luna/autodiff/rollout.py's
`build_simulate` scans luna/setup/evolve.py's palindrome), so what is timed
here is what a run costs, not a model of it.

On an APU (MI300A) the job must export XLA_PYTHON_CLIENT_PREALLOCATE=false,
as the production submit scripts do: "device memory" there IS the node's
RAM, and the default preallocation of a fraction of it walks straight
through a job's cgroup limit as soon as a rollout gets long enough. The
symptom is a SIGSEGV in the longer rollouts and nothing in the log, which
reads exactly like an autodiff bug and is not one. Every row records the
allocator settings it was measured under.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The physics each case turns on, over the same box and the same step.
CASES = {
    "hydro": {},
    "gravity": {"SELF_GRAVITY": True},
    "radiation": {"radiation_enabled": True, "rad_chat": 20.0, "rad_E0": 1.0,
                  "rad_kappa_a": 1.0, "rad_kappa_s": 0.5, "rad_substeps": 2},
    "coupled": {"SELF_GRAVITY": True,
                "radiation_enabled": True, "rad_chat": 20.0, "rad_E0": 1.0,
                "rad_kappa_a": 1.0, "rad_kappa_s": 0.5, "rad_substeps": 2},
}


def _config(case, nx, steps, mesh, ckpt="step", snaps=32):
    from dataclasses import replace
    from luna.template_config import Config
    if case not in CASES:
        raise SystemExit(f"unknown case {case!r}: {', '.join(CASES)}")
    return replace(Config(), test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3,
                   Nx=nx, Ny=nx, Nz=nx, Lx=1.0, Ly=1.0, Lz=1.0,
                   BOUNDARY=0, LIMITER=0, enable_x64=True,
                   MESH_SHAPE=mesh, autodiff_mode=True, autodiff_n_steps=steps,
                   autodiff_dt_mode="stopgrad", write_snapshots=False,
                   autodiff_checkpoint=ckpt, autodiff_snapshots=snaps,
                   enable_diagnostics=False, enable_checkpointing=False,
                   **CASES[case])


def _build(case, nx, steps, mesh, ckpt="step", snaps=32):
    """(simulate, theta, U0): the differentiable rollout of that case."""
    from dataclasses import replace
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind
    from luna.autodiff.rollout import build_simulate

    cfg = resolve(_config(case, nx, steps, mesh, ckpt, snaps))
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    jmesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, jmesh, kind, layout, False)
    physics = assemble(cfg, jmesh, kind, layout, init, False)
    # A FIXED dt, half the opening CFL estimate: the rollout has to be the
    # same amount of work at every length, and an adaptive dt would also put
    # a reduction in the tape that the resolution sweep would then confound.
    import jax
    dt0 = float(physics.hydro.compute_dt(init.U))
    cfg = replace(cfg, autodiff_fixed_dt=0.5 * dt0)
    simulate, step_once, carry0, dt_info, theta = build_simulate(
        init, cfg, physics, False)
    return simulate, theta, init.U, {"dt": 0.5 * dt0, "dt_info": dt_info,
                                     "devices": int(jax.device_count()),
                                     "step_once": step_once, "carry0": carry0}


def _peak_bytes():
    """The device high-water mark, or None when the backend does not say.

    Reported as null rather than 0: a missing measurement and a measurement
    of nothing are not the same number.
    """
    import jax
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return None
    v = stats.get("peak_bytes_in_use")
    return int(v) if v is not None else None


def _time(fn, arg, reps):
    """(median seconds, compile seconds): the first call pays the compile."""
    import jax
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn(arg))
    compile_s = time.perf_counter() - t0
    del out
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(arg))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), compile_s


def run_case(case, nx, steps, mesh, reps=3, ckpt="step", snaps=32, mode="vjp"):
    """One measured row. Runs in this process; the driver gives it its own.
    mode "vjp": value-and-gradient (reverse); "jvp": value and directional
    derivative along a random tangent (forward, no tape). Either way the
    derivative's time is `grad_s` and the row says which."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    simulate, theta, U0, info = _build(case, nx, steps, mesh, ckpt, snaps)

    def loss(U0):
        return jnp.sum(simulate(theta, U0) ** 2)

    fwd = jax.jit(loss)
    fwd_s, fwd_compile = _time(fwd, U0, reps)
    if mode == "jvp":
        from luna.autodiff.stability import random_like
        v = random_like(U0, 1)
        jv = jax.jit(lambda U, v: jax.jvp(loss, (U,), (v,)))
        grad_s, grad_compile = _time(lambda U: jv(U, v), U0, reps)
    else:
        grad_s, grad_compile = _time(jax.jit(jax.value_and_grad(loss)), U0, reps)
    cells = nx ** 3
    return {"case": case, "mode": mode, "nx": nx, "cells": cells, "steps": steps,
            "mesh": ",".join(str(m) for m in mesh), "devices": info["devices"],
            "dt": info["dt"], "reps": reps, "ckpt": ckpt,
            "snapshots": snaps if ckpt == "revolve" else None,
            "advances": _advances(ckpt, steps, snaps),
            "forward_s": fwd_s, "grad_s": grad_s, "ratio": grad_s / fwd_s,
            "forward_compile_s": fwd_compile, "grad_compile_s": grad_compile,
            "peak_bytes": _peak_bytes(),
            "cell_steps_per_s": cells * steps / fwd_s,
            "x64": bool(jax.config.jax_enable_x64),
            "jax": jax.__version__,
            "env": {k: os.environ.get(k) for k in
                    ("XLA_PYTHON_CLIENT_PREALLOCATE",
                     "XLA_PYTHON_CLIENT_MEM_FRACTION",
                     "HIP_LAUNCH_BLOCKING", "AMD_SERIALIZE_KERNEL",
                     "XLA_FLAGS")}}


# --------------------------------------------------------------------------
# the driver: one subprocess per case
# --------------------------------------------------------------------------

_VIS_VARS = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
             "CUDA_VISIBLE_DEVICES")


def _visibility(want):
    """Environment that shows a child exactly `want` of THIS job's devices.

    It narrows the visibility list that is already set rather than writing a
    new one, because those indices are absolute. Setting HIP_VISIBLE_DEVICES=0
    inside a Slurm allocation that was given GPUs 2 and 3 does not select the
    job's first GPU -- it selects the machine's, which belongs to somebody
    else's job. That is how the first run of this benchmark produced four
    SIGSEGVs and a set of timings measured on a contended device (2026-09-21).
    """
    for var in _VIS_VARS:
        cur = os.environ.get(var, "")
        ids = [t for t in cur.split(",") if t.strip()]
        if ids:
            if want > len(ids):
                raise SystemExit(f"{var}={cur} gives {len(ids)} devices, "
                                 f"the mesh asks for {want}")
            keep = ",".join(ids[:want])
            # every list that is set has to agree, or ROCm warns about a
            # conflict and silently prefers one of them
            return {v: keep for v in _VIS_VARS if os.environ.get(v)}
    return {"HIP_VISIBLE_DEVICES": ",".join(str(i) for i in range(want)),
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(want))}


def _advances(ckpt, steps, snaps):
    """Steps the forward sweep plus the recomputation advance, from the
    schedule (per-step checkpointing: the forward sweep only)."""
    if ckpt != "revolve":
        return steps
    from luna.time_integration.revolve import binomial_advances
    return binomial_advances(steps, min(snaps, steps)) + 1


def _child(case, nx, steps, mesh, reps, timeout_s, fail_dir=None,
           ckpt="step", snaps=32, mode="vjp"):
    """Run one case in a fresh interpreter; return its row or why it failed."""
    # -u: a child that dies on a signal loses a block-buffered pipe, and the
    # crash report then has nothing in it at all (measured: rc=-11, empty
    # stdout and stderr, not even the banner the run had already printed).
    cmd = [sys.executable, "-u", os.path.abspath(__file__), "--case", case,
           "--nx", str(nx), "--steps", str(steps), "--reps", str(reps),
           "--mesh", ",".join(str(m) for m in mesh), "--mode", mode, "--json", "-",
           "--ckpt", ckpt, "--snapshots", str(snaps)]
    want = 1
    for m in mesh:
        want *= int(m)
    env = dict(os.environ)
    env.update(_visibility(want))
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                           timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired:
        return {"case": case, "nx": nx, "steps": steps, "status": "timeout",
                "ckpt": ckpt, "snapshots": snaps if ckpt == "revolve" else None,
                "mesh": ",".join(str(m) for m in mesh),
                "wall_s": time.perf_counter() - t0}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("{"):
            row = json.loads(line)
            row["status"] = "ok"
            row["wall_s"] = time.perf_counter() - t0
            return row
    tail = (p.stderr or p.stdout).strip().splitlines()[-6:]
    row = {"case": case, "nx": nx, "steps": steps, "status": "fail",
           "ckpt": ckpt, "snapshots": snaps if ckpt == "revolve" else None,
           "mesh": ",".join(str(m) for m in mesh),
           # a crash has no row of its own to record the conditions in
           "env": {k: env.get(k) for k in
                   ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
                    "HIP_LAUNCH_BLOCKING", "AMD_SERIALIZE_KERNEL", "XLA_FLAGS")},
           "returncode": p.returncode, "wall_s": time.perf_counter() - t0,
           "tail": "\n".join(tail)}
    if fail_dir:                      # the whole output, for a crash with none
        os.makedirs(fail_dir, exist_ok=True)
        name = (f"{case}_nx{nx}_n{steps}_m{'x'.join(str(m) for m in mesh)}"
                f"_{ckpt}{snaps if ckpt == 'revolve' else ''}.log")
        with open(os.path.join(fail_dir, name), "w") as f:
            f.write(f"$ {' '.join(cmd)}\nrc={p.returncode}\n"
                    f"visibility={ {k: env.get(k) for k in _VIS_VARS} }\n"
                    f"--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}\n")
        row["log"] = name
    return row


def max_rollout(case, nx, mesh, reps, timeout_s, cap=4096):
    """The longest rollout that still runs: double until it does not, then
    bisect. Each attempt is its own process, so an out-of-memory one is a
    non-zero exit and not the end of the sweep."""
    lo, n, last = 0, 1, None
    while n <= cap:
        row = _child(case, nx, n, mesh, 1, timeout_s)
        print(f"  rollout {n:5d}: {row['status']}", flush=True)
        if row["status"] != "ok":
            break
        lo, last, n = n, row, n * 2
    hi = min(n, cap + 1)
    while hi - lo > 1:                      # bisect the gap
        mid = (lo + hi) // 2
        row = _child(case, nx, mid, mesh, 1, timeout_s)
        print(f"  rollout {mid:5d}: {row['status']}", flush=True)
        if row["status"] == "ok":
            lo, last = mid, row
        else:
            hi = mid
    return {"case": case, "nx": nx, "mesh": ",".join(str(m) for m in mesh),
            "longest_rollout": lo, "at_peak_bytes": (last or {}).get("peak_bytes"),
            "grad_s": (last or {}).get("grad_s")}


def write_outputs(out_dir, rows, limits, meta):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "autodiff_bench.json")
    with open(path, "w") as f:
        json.dump({"meta": meta, "rows": rows, "limits": limits}, f, indent=1)
    def _launch(r):
        e = r.get("env")
        if e is None:
            return "?"
        return "serial" if (e.get("HIP_LAUNCH_BLOCKING")
                            or e.get("AMD_SERIALIZE_KERNEL")) else "async"

    stacks = sorted({r["jax"] for r in rows if r.get("jax")})
    if any(not r.get("jax") for r in rows):
        stacks.append("unrecorded (rows measured before the stack was logged)")
    md = ["# Autodiff benchmark\n", f"{meta['when']} - {meta['platform']}\n",
          ("jax " + ", ".join(stacks) + "\n") if stacks else "",
          "`launch`: async is the default; serial is HIP_LAUNCH_BLOCKING=1 or "
          "AMD_SERIALIZE_KERNEL=3, which the radiation gradient needs on ROCm "
          "and which costs 3x on the multigrid.\n"]
    md += [
          "| case | nx | steps | mesh | checkpoints | launch | forward s | grad s | ratio | peak GiB |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ck = (f"revolve {r['snapshots']}" if r.get("ckpt") == "revolve"
              else "every step")
        head = (f"| {r['case']} | {r['nx']} | {r['steps']} | {r.get('mesh', '?')} "
                f"| {ck} | {_launch(r)} |")
        if r.get("status") != "ok":
            md.append(f"{head} **{r['status']}** (rc {r.get('returncode', '?')}) | | | |")
            continue
        pk = r["peak_bytes"]
        md.append(f"{head} {r['forward_s']:.4f} | {r['grad_s']:.4f} | "
                  f"{r['ratio']:.2f} | "
                  + (f"{pk / 2**30:.3f} |" if pk else "n/a |"))
    if limits:
        md += ["", "| case | nx | longest rollout | peak GiB there |",
               "|---|---|---|---|"]
        for c in limits:
            pk = c["at_peak_bytes"]
            md.append(f"| {c['case']} | {c['nx']} | {c['longest_rollout']} | "
                      + (f"{pk / 2**30:.3f} |" if pk else "n/a |"))
    with open(os.path.join(out_dir, "autodiff_bench.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    return path


def figure(out_dir, rows):
    """One figure: the adjoint's price and its memory, against rollout length."""
    # Only the rows measured with the launches left ASYNC: serializing them
    # is free for the hydro and the radiation and 3x slower for the
    # multigrid, so one figure cannot hold both without lying about one of
    # them. The table keeps every row, with its environment.
    def _async(r):
        e = r.get("env") or {}
        return not (e.get("HIP_LAUNCH_BLOCKING") or e.get("AMD_SERIALIZE_KERNEL"))

    rows = [r for r in rows if _async(r)]
    ok = [r for r in rows if r.get("status") == "ok"]
    if not ok:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    def _series(r):
        return (r["case"], r["nx"], r.get("snapshots") or 0)

    colors = {}

    for case, nx, snaps in sorted({_series(r) for r in ok}):
            pts = sorted(((r["steps"], r) for r in ok
                          if _series(r) == (case, nx, snaps)
                          and r.get("devices", 1) == 1),
                         key=lambda sr: sr[0])
            if not pts:
                continue
            lbl = f"{case} {nx}$^3$" + (f", revolve {snaps}" if snaps else "")
            sty = dict(ls="-" if case == "hydro" else "--",
                       marker="o" if nx <= 128 else "s",
                       color=colors.setdefault((case, nx, snaps),
                                               f"C{len(colors) % 10}"))
            a.plot([s for s, _ in pts], [r["ratio"] for _, r in pts], label=lbl, **sty)
            mem = [(s, r["peak_bytes"]) for s, r in pts if r["peak_bytes"]]
            if mem:
                b.plot([s for s, _ in mem], [m / 2 ** 30 for _, m in mem],
                       label=lbl, **sty)
    a.set_xlabel("rollout length (steps)")
    a.set_ylabel("grad time / forward time")
    a.set_xscale("log", base=2)
    a.axhline(1.0, color="0.7", lw=0.8)
    # crosses where the next longer rollout did not survive: the gradient
    # crashes are part of the picture, not a gap in it
    for ser in sorted({_series(r) for r in rows if r.get("status") != "ok"}):
        bad = [r["steps"] for r in rows
               if _series(r) == ser and r.get("status") != "ok"]
        for steps in sorted(set(bad)):
            good = [r for r in ok if _series(r) == ser
                    and r["steps"] < steps and r.get("devices", 1) == 1]
            if good:
                last = max(good, key=lambda r: r["steps"])
                a.plot([last["steps"]], [last["ratio"]], "kx", ms=9, mew=1.6)
    a.set_title("launches async; x: the next longer rollout did not survive",
                fontsize=8)
    b.set_xlabel("rollout length (steps)")
    b.set_ylabel("peak device memory (GiB)")
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    path = os.path.join(out_dir, "autodiff_bench.png")
    fig.savefig(path, dpi=140)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", help="run ONE case in this process and print it")
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mesh", default="1,1,1")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ckpt", default="step", help="'step' or 'revolve'")
    ap.add_argument("--snapshots", default="32",
                    help="revolve snapshot counts (comma-separated in the matrix)")
    ap.add_argument("--mode", default="vjp", choices=("vjp", "jvp"),
                    help="reverse (value_and_grad) or forward (jvp) derivative")
    ap.add_argument("--json", default=None, help="'-' for stdout (child mode)")
    ap.add_argument("--out", default="bench_out")
    ap.add_argument("--cases", default="hydro,gravity,radiation,coupled")
    ap.add_argument("--sizes", default="32,64")
    ap.add_argument("--rollouts", default="1,2,4,8,16")
    ap.add_argument("--meshes", default="1,1,1")
    ap.add_argument("--max-rollout", default="",
                    help="case:nx pairs to find the longest rollout for")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    ap.add_argument("--merge", default="",
                    help="comma-separated result files to combine into one "
                         "table and figure instead of measuring")
    a = ap.parse_args(argv)
    mesh = tuple(int(v) for v in a.mesh.split(","))

    if a.merge:
        rows, limits, meta = [], [], {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
                                      "platform": "merged", "sources": []}
        for f in a.merge.split(","):
            d = json.load(open(f))
            rows += d.get("rows", [])
            limits += d.get("limits", [])
            meta["sources"].append({"file": f, "meta": d.get("meta", {})})
        path = write_outputs(a.out, rows, limits, meta)
        fig = figure(a.out, rows)
        print(f"{len(rows)} rows -> {path}" + (f"\nfigure -> {fig}" if fig else ""))
        return 0

    if a.case:
        row = run_case(a.case, a.nx, a.steps, mesh, a.reps, a.ckpt,
                       int(a.snapshots), a.mode)
        text = json.dumps(row)
        if a.json in (None, "-"):
            print(text, flush=True)
        else:
            open(a.json, "w").write(text + "\n")
        return 0

    import platform
    meta = {"when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.node(), "argv": vars(a)}
    meshes = [tuple(int(v) for v in m.split(",")) for m in a.meshes.split(";")]
    rows = []
    snap_list = ([int(v) for v in a.snapshots.split(",")] if a.ckpt == "revolve"
                 else [32])
    for case in a.cases.split(","):
        for m in meshes:
            for nx in (int(v) for v in a.sizes.split(",")):
                for steps in (int(v) for v in a.rollouts.split(",")):
                  for snaps in snap_list:
                    r = _child(case, nx, steps, m, a.reps, a.timeout_s,
                               os.path.join(a.out, 'failures'), a.ckpt, snaps, a.mode)
                    rows.append(r)
                    tag = (f"{r['grad_s']:.4f}s grad, ratio {r['ratio']:.2f}"
                           if r["status"] == "ok" else r["status"])
                    print(f"{case:9s} nx={nx:4d} steps={steps:4d} mesh={m} "
                          f"{a.ckpt}{snaps if a.ckpt == 'revolve' else ''}: {tag}",
                          flush=True)
    limits = []
    for spec in (s for s in a.max_rollout.split(",") if s):
        case, _, nx = spec.partition(":")
        print(f"longest rollout, {case} at nx={nx}:", flush=True)
        limits.append(max_rollout(case, int(nx), meshes[0], 1, a.timeout_s))
    path = write_outputs(a.out, rows, limits, meta)
    fig = figure(a.out, rows)
    print(f"\n{len(rows)} rows -> {path}" + (f"\nfigure -> {fig}" if fig else ""))
    return 0 if all(r.get("status") == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
