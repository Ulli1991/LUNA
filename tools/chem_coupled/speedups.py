"""Offline cost of the chemistry stage on a saved state of run_turb_chem.py, and of candidate speed-ups.
    python tools/chem_coupled/speedups.py --run DIR --state state_04.0Myr [--reps 2]
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.ode import rodas3 as ig  # noqa: E402

from luna.chemistry import networks  # noqa: E402
from luna.chemistry import rates_net5  # noqa: E402
from luna.sources import chemistry as ch  # noqa: E402

from tools.chem_coupled import run_turb_chem as rt  # noqa: E402


SMALL = ((2, 1), (6, 64), (16, 256), (48, 1024), (None, 4096))
CAP1 = ((1, 1), (7, 64), (16, 256), (48, 1024), (None, 4096))


def load(run, name):
    """Config constants, U, the chemistry carry and the step's dt [code] of a saved state."""
    eq = json.load(open(os.path.join(run, "equilibrium.json")))
    setup = json.load(open(os.path.join(run, "setup.json")))
    cfg, _ = rt.make_config(setup["n"], eq["y"])
    d = os.path.join(run, name)
    meta = json.load(open(os.path.join(d, "meta.json")))
    arr = {k: np.load(os.path.join(d, f"chem_{k}.npy")) for k in ("S", "tdust", "h")}
    return ch.ChemConsts(cfg), np.load(os.path.join(d, "U.npy")), arr, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ks", default="2,4,8")
    ap.add_argument("--groups", default="tiers,skip,every")
    a = ap.parse_args()
    c, U, arr, meta = load(a.run, a.state)
    dt_code = meta["dt"]
    dt_s = dt_code * c.t_s
    print(f"{a.state}: t = {meta.get('t_myr', float('nan')):.3f} Myr, dt = {dt_s / rt.YR:.1f} yr, "
          f"{U[0].size} cells, {jax.devices()[0].platform}", flush=True)

    @jax.jit
    def inputs(U, S):
        y0, n_h, cols, _, _ = ch.cell_inputs(U, S, c)
        temp = rt.temperature(y0[..., :3], y0[..., 3], n_h)
        f = lambda a: a.reshape(-1)  # noqa: E731
        return y0.reshape(-1, 4), f(n_h), tuple(f(x) for x in cols), f(temp)

    y0, n_h, cols, temp = inputs(jnp.asarray(U), jnp.asarray(arr["S"]))
    del U
    td0, h0 = jnp.asarray(arr["tdust"]).reshape(-1), jnp.asarray(arr["h"]).reshape(-1)
    data = (y0, n_h, cols, td0, h0)

    def opt_run(opt=ig.Options()):
        f = jax.jit(lambda d, dts: ch.integrate_cells(c, *d, dts, opt=opt))
        return lambda dts: f(data, dts)
    run = opt_run()

    @jax.jit
    def rhs_all(y0, n_h, cols, td0):
        p = ch.params(c, n_h, cols, td0)
        f = jax.vmap(lambda y, pp: rates_net5.rhs(y, pp)[0])
        return jax.lax.map(lambda b: f(*b), (y0.reshape(-1, 2 ** 20, 4), jax.tree_util.tree_map(
            lambda v: v.reshape((-1, 2 ** 20) + v.shape[1:]), ig.broadcast_params(p, y0.shape[0])))
        ).reshape(-1, 4)

    def timed(f, *args):
        out = jax.block_until_ready(f(*args))
        w = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            out = jax.block_until_ready(f(*args))
            w.append(time.perf_counter() - t0)
        return out, min(w)

    n = y0.shape[0]
    out_f = os.path.join(a.run, f"speedups_{a.state}.json")
    res = json.load(open(out_f)) if os.path.exists(out_f) else {}
    res.update(t_myr=meta.get("t_myr", float("nan")), dt_yr=dt_s / rt.YR, cells=n)

    def record(name, d):
        res[name] = d
        print(name, d, flush=True)
        json.dump(res, open(out_f, "w"), indent=1)

    full, t_full = timed(run, jnp.full((n,), dt_s))
    att = np.asarray(full[3]) + np.asarray(full[4])
    record("full", dict(ms=t_full * 1e3, ns_per_cell=t_full / n * 1e9,
                        mean_attempts=float(att.mean()), multi_frac=float((att > 1).mean()),
                        failed=int(np.asarray(full[5]).sum())))
    groups = a.groups.split(",")
    if "tiers" in groups:
        for name, opt in (("tiers_always_run", ig.Options(skip_empty_tiers=False)),
                          ("tiers_small_chunks", ig.Options(tiers=SMALL)),
                          ("tiers_cap1_small_chunks", ig.Options(tiers=CAP1)),
                          ("one_pass_no_tiers", ig.Options(tiers=((None, 1),)))):
            _, t = timed(opt_run(opt), jnp.full((n,), dt_s))
            record(name, dict(ms=t * 1e3, tiers=str(opt.tiers)))
    if "skip" in groups:
        f0, t_rhs = timed(rhs_all, y0, n_h, cols, td0)
        record("rhs_eval_all_cells", dict(ms=t_rhs * 1e3))
        yv, fv = np.asarray(y0), np.asarray(f0)
        floor = np.asarray(networks.NET5.ctrl_floor(rates_net5.make_params(
            1.0, c.g0, abundc=ch.ABUNDC, abundo=ch.ABUNDO)))
        use = (fv != 0) & (yv > floor[None])
        tau = np.min(np.where(use, np.abs(yv) / np.where(use, np.abs(fv), 1.0), np.inf), axis=1)
        yfull = np.asarray(full[0])
        tnp = np.asarray(temp)
        for name, skip in (("skip_hot_T_gt_5e5", tnp > ch.SKIP_T),
                           ("skip_tau_gt_10dt", tau > 10 * dt_s),
                           ("skip_tau_gt_100dt", tau > 100 * dt_s)):
            d = dict(frac_cells=float(skip.mean()), frac_attempts=float(att[skip].sum() / att.sum()))
            if skip.any() and not skip.all():
                _, t = timed(compacted(c, data, dt_s, np.nonzero(~skip)[0]))
                d["ms_integrate_rest_compacted"] = t * 1e3
            if skip.any() and name.startswith("skip_tau"):
                err = np.abs(yv[skip, 3] + dt_s * fv[skip, 3] - yfull[skip, 3]) / yfull[skip, 3]
                d.update(euler_max_rel_err_e=float(err.max()),
                         euler_p99_rel_err_e=float(np.percentile(err, 99)))
            record(name, d)
    if "every" in groups:
        for k in (int(v) for v in a.ks.split(",")):
            out, t = timed(run, jnp.full((n,), k * dt_s))
            att_k = np.asarray(out[3]) + np.asarray(out[4])
            record(f"every_{k}", dict(ms_per_call=t * 1e3, ms_per_step=t * 1e3 / k,
                                      mean_attempts=float(att_k.mean()),
                                      failed=int(np.asarray(out[5]).sum())))


def compacted(c, data, dt_s, keep):
    """Jitted: gather the kept cells (padded to a whole chunk), integrate them, scatter back."""
    m = 2 ** c.chunk
    pad = np.concatenate([keep, np.full((-len(keep)) % m, keep[-1])])
    idx = jnp.asarray(pad)

    @jax.jit
    def f(d, idx):
        y0, n_h, cols, td0, h0 = d
        g = lambda v: v[idx]  # noqa: E731
        r = ch.integrate_cells(c, g(y0), g(n_h), tuple(g(x) for x in cols), g(td0), g(h0),
                               jnp.full((idx.shape[0],), dt_s))
        return y0.at[idx].set(r[0])
    return lambda: f(data, idx)


if __name__ == "__main__":
    main()
