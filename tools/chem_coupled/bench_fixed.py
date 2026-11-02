"""integrate_fixed (plain JAX or the fused kernel) on a saved 512^3 state: time, flagged cells, outputs for the gate.

    python tools/chem_coupled/bench_fixed.py --run DIR --state ckpt_step1273 --what plain|fused|pass2 --out DIR"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.ode import rodas3 as ig  # noqa: E402
from luna.sources import chemistry as ch  # noqa: E402
from tools.chem_coupled import fixed_k as fk  # noqa: E402


def timed(f, *args, reps=5):
    out = jax.block_until_ready(f(*args))
    w = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        w.append(time.perf_counter() - t0)
    return out, float(np.median(w)), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--what", default="plain")
    ap.add_argument("--chunk2", type=int, default=2 ** 16)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--save", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = os.path.join(a.out, a.state)
    os.makedirs(d, exist_ok=True)
    c, dt, y0, n_h, cols, td0, _ = fk.load_cells(a.run, a.state)
    n = y0.shape[0]
    c.dust_rtol = 1e-2
    net = ch.fixed_net(1e-2)
    opt = ig.Options(chunk=2 ** 20, chunk2=a.chunk2)
    res = dict(state=a.state, cells=n, what=a.what, chunk2=a.chunk2)
    res_f = os.path.join(d, f"result_hybrid_{a.what}{a.tag}.json")

    if a.what in ("plain", "fused"):
        solve = None
        if a.what == "fused":
            from luna.chemistry import fused
            solve = fused.solve
        p = ch.params(c, n_h, cols, td0, per_cell=False)
        f = jax.jit(lambda y0, p: ig.integrate_fixed(y0, p, dt, net=net, opt=opt, solve=solve))
        t0 = time.perf_counter()
        r = jax.block_until_ready(f(y0, p))
        res["first_call_s"] = time.perf_counter() - t0
        r, t_med, w = timed(f, y0, p, reps=a.reps)
        res.update(ms=1e3 * t_med, ms_all=[1e3 * x for x in w], ns_per_cell=1e9 * t_med / n,
                   flagged=int(r.flagged.sum()), nonfinite=int((~jnp.isfinite(r.y)).any(1).sum()))
        if a.save:
            np.save(os.path.join(d, f"hybrid_{a.what}{a.tag}.npy"), np.asarray(r.y))
            np.save(os.path.join(d, f"hybrid_{a.what}{a.tag}_flag.npy"), np.asarray(r.flagged))
            np.save(os.path.join(d, f"hybrid_{a.what}{a.tag}_td.npy"), np.asarray(r.params.tdust))
        print(json.dumps(res), flush=True)
        json.dump(res, open(res_f, "w"), indent=1)
        return

    if a.what == "pass2":
        # fixed_k.py's second pass alone (jacfwd, k = 4), unused slots on cell n-1 or on distinct cells
        flag = jnp.asarray(np.load(os.path.join(d, "fixed_k1_err.npy")) > 0.1)
        nflag = int(flag.sum())
        for cap in (2 ** 20, -(-nflag // 2 ** 14) * 2 ** 14):
            for fill in ("last", "spread"):
                def second(y0, n_h, cols, td0, flag, cap=cap, fill=fill):
                    idx = jnp.nonzero(flag, size=cap, fill_value=n)[0]
                    gi = (jnp.minimum(idx, n - 1) if fill == "last"
                          else jnp.where(idx < n, idx, jnp.arange(cap)))
                    g = lambda v: v[gi]  # noqa: E731
                    pg = ch.params(c, g(n_h), tuple(g(v) for v in cols), g(td0))
                    return jax.vmap(lambda y, pp: fk.fixed_cell(4, y, pp, dt))(g(y0), pg)[0]
                with fk.fixed_dust():
                    _, t, _ = timed(jax.jit(second), y0, n_h, cols, td0, flag, reps=3)
                res[f"cap{cap}_{fill}_ms"] = 1e3 * t
                print(cap, fill, 1e3 * t, flush=True)
        res["flagged"] = nflag
        json.dump(res, open(res_f, "w"), indent=1)


if __name__ == "__main__":
    main()
