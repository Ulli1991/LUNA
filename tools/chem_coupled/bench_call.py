"""Time the production chemistry stage call on a saved 512^3 state, and compare its result to a reference.

    python tools/chem_coupled/bench_call.py --run DIR --state ckpt_step1273 --label NAME [--save-ref] [--ref DIR]"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh  # noqa: E402

from luna.sources import chemistry as ch  # noqa: E402
from tools.chem_coupled import speedups as sp  # noqa: E402


def stage_inputs(run, state):
    """(consts, U, carry, dt [code]) of a saved state on one GPU."""
    c, U, arr, meta = sp.load(run, state)
    U = jnp.asarray(U)
    carry = {"S": jnp.asarray(arr["S"]), "tdust": jnp.asarray(arr["tdust"]),
             "h": jnp.asarray(arr["h"])}
    return c, U, carry, meta["dt"]


@jax.jit
def summarize(U0, U, S):
    """Per-cell (x_H2, x_H+, x_CO, e_int) after the call, in code units."""
    ke = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
    return jnp.stack([S[0] / U[0], S[1] / U[0], S[2] / U[0], U[4] - ke], -1).reshape(-1, 4)


def compare(y, td, ref_dir, c):
    """Max / p99 relative differences and the count of cells outside rtol + atol, per component."""
    yr = np.load(os.path.join(ref_dir, "y.npy"), mmap_mode="r")
    tdr = np.load(os.path.join(ref_dir, "tdust.npy"), mmap_mode="r")
    atol = np.array([1e-7, 1e-12, 1e-14, 0.0])
    out = {}
    names = ("x_H2", "x_H+", "x_CO", "e")
    n = y.shape[0]
    step = 2 ** 24
    acc = {k: [] for k in names}
    cnt = {k: [0, 0] for k in names}
    for s in range(0, n, step):
        a, b = np.asarray(y[s:s + step]), np.asarray(yr[s:s + step])
        for j, k in enumerate(names):
            d = np.abs(a[:, j] - b[:, j])
            rel = d / np.maximum(np.abs(b[:, j]), 1e-300)
            acc[k].append(rel[np.abs(b[:, j]) > atol[j]])
            sc = np.abs(b[:, j]) if k != "e" else np.abs(b[:, j])
            cnt[k][0] += int(np.sum(d > 1e-4 * sc + atol[j]))
            cnt[k][1] += int(np.sum(d > 1e-2 * sc + atol[j]))
    for k in names:
        r = np.concatenate(acc[k])
        out[k] = dict(max_rel=float(r.max()), p99_rel=float(np.percentile(r, 99)),
                      p999_rel=float(np.percentile(r, 99.9)),
                      n_out_1e4=cnt[k][0], n_out_1pct=cnt[k][1])
    tdn = np.asarray(td).reshape(-1)
    rel = np.abs(tdn - tdr) / tdr
    out["tdust"] = dict(max_rel=float(rel.max()), p99_rel=float(np.percentile(rel, 99)),
                        n_gt_1pct=int(np.sum(rel > 1e-2)))
    return out


def conservation(U0, carry0, U, carry):
    """Relative change of total H (rho), C (rho x_C+ + CO implied) and of the mass, over the call."""
    m0, m1 = float(jnp.sum(U0[0])), float(jnp.sum(U[0]))
    s0 = [float(jnp.sum(v)) for v in carry0["S"]]
    s1 = [float(jnp.sum(v)) for v in carry["S"]]
    return dict(mass_rel=(m1 - m0) / m0, S_sums_before=s0, S_sums_after=s1,
                h_nuclei_rel=0.0 if m0 == m1 else (m1 - m0) / m0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--save-ref", action="store_true")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--opt", default="{}", help="json: Chemistry stage options (if supported)")
    a = ap.parse_args()
    c, U0, carry0, dt = stage_inputs(a.run, a.state)
    opts = json.loads(a.opt)
    for k, v in opts.items():
        setattr(c, k, tuple(v) if isinstance(v, list) else v)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    apply = ch.make_apply(c, mesh)
    f = jax.jit(apply)
    t0 = time.perf_counter()
    U, carry = jax.block_until_ready(f(U0, carry0, dt))
    t_first = time.perf_counter() - t0
    w = []
    for _ in range(a.reps):
        del U, carry
        time.sleep(0.2)
        win = [time.clock_gettime_ns(k) for k in (time.CLOCK_MONOTONIC, time.CLOCK_BOOTTIME)]
        t0 = time.perf_counter()
        U, carry = jax.block_until_ready(f(U0, carry0, dt))
        w.append(time.perf_counter() - t0)
        win += [time.clock_gettime_ns(k) for k in (time.CLOCK_MONOTONIC, time.CLOCK_BOOTTIME)]
    n = U0[0].size
    att = carry["n_steps"] + carry["n_rejected"]
    res = dict(label=a.label, state=a.state, opts=opts, cells=n, ms_median=1e3 * float(np.median(w)),
               ms_min=1e3 * float(np.min(w)), ms_all=[1e3 * x for x in w], first_s=t_first, last_window_mono_boot=win,
               ns_per_cell=1e9 * float(np.median(w)) / n,
               mean_attempts=float(jnp.mean(att.astype(jnp.float64))), max_attempts=int(jnp.max(att)),
               failed=int(jnp.sum(carry["failed"])), floored=int(jnp.sum(carry["floored"])),
               conservation=conservation(U0, carry0, U, carry))
    for k in ("n_explicit",):
        if k in carry:
            res[k] = int(jnp.sum(carry[k]))
    print(json.dumps(res, indent=1), flush=True)
    y = summarize(U0, U, carry["S"])
    del U
    ref_dir = os.path.join(a.out, "ref", a.state)
    if a.save_ref:
        os.makedirs(ref_dir, exist_ok=True)
        np.save(os.path.join(ref_dir, "y.npy"), np.asarray(y))
        np.save(os.path.join(ref_dir, "tdust.npy"), np.asarray(carry["tdust"]).reshape(-1))
    elif os.path.exists(os.path.join(a.ref or ref_dir, "y.npy")):
        res["accuracy"] = compare(y, carry["tdust"], a.ref or ref_dir, c)
        print(json.dumps(res["accuracy"], indent=1), flush=True)
    with open(os.path.join(a.out, f"bench_{a.label}_{a.state}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
