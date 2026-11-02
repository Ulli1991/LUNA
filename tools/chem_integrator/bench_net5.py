"""Cost of the network-5 integrator per cell on states sampled from the DVODE trajectories.

    python tools/chem_integrator/bench_net5.py --n 64 --dt-yr 1e3 [--chunk 20] [--sharded]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.ode import rodas3 as ig  # noqa: E402

from luna.chemistry import networks  # noqa: E402
from tools.chem_integrator import validate_net5 as val  # noqa: E402

YR = 3.15576e7


def tables():
    """States (every output of every case) and the per-case Params they belong to."""
    names, cp, t, y0, ref = val.load()
    states = ref[:, 1:, :4].reshape(-1, 4)
    case = np.repeat(np.arange(len(names)), ref.shape[1] - 1)
    return jnp.asarray(states), jnp.asarray(case), ig.broadcast_params(val.params(cp), len(names))


def make_step(dt, chunk, opt):
    """Jitted: gather the sampled states chunk by chunk and integrate each chunk over dt."""
    def one(args, states, case, ptab):
        idx = args
        y = states[idx]
        p = jax.tree_util.tree_map(lambda a: a[case[idx]], ptab)
        r = ig.integrate(y, p, dt, opt=opt, net=networks.NET5)
        return r.y, r.n_steps, r.n_rejected, r.failed

    def run(idx, states, case, ptab):
        ch = idx.reshape(-1, chunk)
        out = jax.lax.map(lambda a: one(a, states, case, ptab), ch)
        return jax.tree_util.tree_map(lambda a: a.reshape((-1,) + a.shape[2:]), out)
    return jax.jit(run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="cells = n^3")
    ap.add_argument("--dt-yr", type=float, default=1e3)
    ap.add_argument("--chunk", type=int, default=20, help="log2 cells per chunk")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tiers", default=None, help="e.g. 4,1;12,4;0,16 (cap 0: the rest)")
    ap.add_argument("--sharded", action="store_true", help="also 2 devices vs 1, bitwise")
    a = ap.parse_args()
    ncell = a.n**3
    chunk = min(2**a.chunk, ncell)
    states, case, ptab = tables()
    idx = jnp.asarray(np.random.default_rng(0).integers(0, states.shape[0], ncell), jnp.int32)
    opt = ig.Options() if a.tiers is None else ig.Options(tiers=tuple(
        (int(c) or None, int(f)) for c, f in (t.split(",") for t in a.tiers.split(";"))))
    run = make_step(a.dt_yr * YR, chunk, opt)
    t0 = time.perf_counter()
    out = jax.block_until_ready(run(idx, states, case, ptab))
    first = time.perf_counter() - t0
    walls = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(run(idx, states, case, ptab))
        walls.append(time.perf_counter() - t0)
    y, ns, nr, fl = (np.asarray(v) for v in out)
    w = min(walls)
    q = np.percentile(ns, [50, 90, 99])
    print(f"{a.n}^3 dt={a.dt_yr:g} yr chunk=2^{int(np.log2(chunk))} tiers={opt.tiers} "
          f"dev={jax.devices()[0].platform}: {w * 1e3:.1f} ms ({w / ncell * 1e9:.1f} ns/cell; "
          f"reps {[round(x * 1e3, 1) for x in walls]} ms; first call {first:.1f} s)")
    print(f"  substeps accepted: mean {ns.mean():.2f} p50 {q[0]:.0f} p90 {q[1]:.0f} "
          f"p99 {q[2]:.0f} max {ns.max()}; rejected mean {nr.mean():.3f} max {nr.max()}; "
          f"failed {int(fl.sum())}")
    hist = np.bincount(np.minimum(ns, 64))
    print("  histogram (accepted substeps: cells):",
          {k: int(v) for k, v in enumerate(hist) if v})
    if a.sharded:
        from jax.sharding import Mesh, PartitionSpec as P
        from luna.utils.shmap import shard_map
        mesh = Mesh(np.array(jax.devices()[:2]), ("x",))
        cs = min(chunk, ncell // 2)
        run1 = make_step(a.dt_yr * YR, cs, opt)
        y1 = np.asarray(run1(idx, states, case, ptab)[0])
        f = jax.jit(shard_map(lambda i, s, c, p: run1(i, s, c, p)[0], mesh=mesh,
                              in_specs=(P("x"), P(), P(), P()), out_specs=P("x")))
        y2 = np.asarray(f(idx, states, case, ptab))
        print(f"  sharded on 2 devices == one device (chunk 2^{int(np.log2(cs))}) bitwise: "
              f"{np.array_equal(y1, y2)} (max |diff| {np.max(np.abs(y1 - y2)):.3e}); "
              f"== the unsplit {ncell}-cell run (chunk 2^{int(np.log2(chunk))}): "
              f"{np.array_equal(y, y2)}")


if __name__ == "__main__":
    main()
