"""The fused kernels' first pass (k1 substeps) on 2^m cells of a saved state, repeated, for rocprofv3.

    rocprofv3 --kernel-trace [--pmc ...] -- python tools/chem_coupled/prof_fused.py --run DIR --state S"""
import argparse
import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from luna.chemistry import fused  # noqa: E402
from luna.sources import chemistry as ch  # noqa: E402
from tools.chem_coupled import fixed_k as fk  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--log2", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--block", type=int, default=fused.BLOCK)
    ap.add_argument("--warps", type=int, default=fused.WARPS)
    a = ap.parse_args()
    fused.BLOCK, fused.WARPS = a.block, a.warps
    c, dt, y0, n_h, cols, td0, _ = fk.load_cells(a.run, a.state)
    m = 2 ** a.log2
    s = slice(y0.shape[0] // 2, y0.shape[0] // 2 + m)       # mid-box cells
    p = ch.params(c, n_h[s], tuple(v[s] for v in cols), td0[s], per_cell=False)
    net = ch.fixed_net(1e-2)
    f = jax.jit(lambda y0, p: fused.substeps(net, 1, y0, p, dt))
    jax.block_until_ready(f(y0[s], p))
    for _ in range(a.reps):
        t0 = time.perf_counter()
        r = jax.block_until_ready(f(y0[s], p))
        print(f"block {a.block} warps {a.warps}: first pass on {m} cells: {1e3 * (time.perf_counter() - t0):.2f} ms, "
              f"{1e9 * (time.perf_counter() - t0) / m:.3f} ns/cell, flagged "
              f"{int(jnp.sum((r[2] > 0.1) | r[3]))}", flush=True)


if __name__ == "__main__":
    main()
