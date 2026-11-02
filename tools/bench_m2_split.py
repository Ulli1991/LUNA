"""M1 against M2 on the same split structure, one GPU, FP64: one transport
step (x y z z y x sweeps, floored after each) of a grey M1 state and of an M2
state (`radiation.m2_split`) with the trace closure and with the learned
closure frozen over the step, fused kernels and XLA. Steady state: median of
REPS after a warm-up. Both steps are bound by the same CFL, so ms per step
compares the cost per unit of simulated time. The network (once per step) is
also timed alone: XLA's in the XLA rows, the kernel in the fused rows.

    python tools/bench_m2_split.py   # env: SIZES, XLA_SIZES, REPS, OUT, TILES, PARAMS
"""
import csv
import os
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from luna.radiation import m1_pallas, m2_3d, m2_pallas, m2_split, m2_stage  # noqa: E402
from luna.radiation.m1 import m1_split_step  # noqa: E402

SIZES = [int(s) for s in os.environ.get("SIZES", "128,256,512").split(",")]
XLA_SIZES = [int(s) for s in os.environ.get("XLA_SIZES", "128,256,512").split(",") if s]
REPS = int(os.environ.get("REPS", "10"))
OUT = os.environ.get("OUT", "m2_split_bench.csv")
# frozen-closure kernel tiles to try, "1x4x64;1x2x64"; the first is the default
TILES = [tuple(int(v) for v in t.split("x"))
         for t in os.environ.get("TILES", "1x4x64").split(";")]
PARAMS = m2_stage.load_params(os.environ.get("PARAMS", ""))
CHAT = 1.0


def fields(n):
    """A beam through isotropic background light: a state both closures act on."""
    x = (jnp.arange(n) + 0.5) / n - 0.5
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    E = 1.0 + jnp.exp(-(X * X + Y * Y + Z * Z) / 0.02)
    beam = jnp.exp(-(Y * Y + Z * Z) / 0.005)
    U = m2_3d.add(m2_3d.pack(E, jnp.zeros((3,) + E.shape), jnp.einsum("ij,xyz->ijxyz", jnp.eye(3) / 3.0, E)),
                  m2_3d.beam(beam, (1.0, 0.3, 0.1)))
    R = jnp.stack([U[0], CHAT * U[1], CHAT * U[2], CHAT * U[3]])[None]
    return R, U


def timed(f, x):
    out = f(x)
    jax.block_until_ready(out)
    del out
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = f(x)
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
        del out
    return float(np.median(ts))


def main():
    rows = []
    print(f"jax {jax.__version__}, {jax.devices()[0].device_kind}, network widths "
          f"{[int(W.shape[0]) for W, _ in PARAMS]}", flush=True)
    for n in SIZES:
        dx = 1.0 / n
        dt = 0.9 * m2_split.substep_dt(dx, chat=CHAT)
        R, U = jax.jit(fields, static_argnums=0)(n)
        nets = {"xla": jax.jit(lambda u: m2_split.free_components(u, PARAMS)),
                "pallas": jax.jit(lambda u: m2_pallas.free_components(u, PARAMS))}
        for kernel in ("pallas", "xla"):
            if kernel == "xla" and n not in XLA_SIZES:
                continue
            m1_pallas.KERNEL = kernel
            row = dict(n=n, kernel=kernel, net_ms=1e3 * timed(nets[kernel], U))
            try:
                row["m1_ms"] = 1e3 * timed(jax.jit(
                    lambda r: m1_split_step(r, dt, dx, chat=CHAT, boundary=1)), R)
                row["m2_trace_ms"] = 1e3 * timed(jax.jit(
                    lambda u: m2_split.split_step(u, dt, dx, chat=CHAT, boundary=1,
                                                  kernel=kernel)), U)
                for tile in (TILES if kernel == "pallas" else TILES[:1]):
                    m2_pallas.TILE_FROZEN = tile
                    key = "m2_learned_ms" if tile == TILES[0] else f"m2_learned_{'x'.join(map(str, tile))}_ms"
                    row[key] = 1e3 * timed(jax.jit(
                        lambda u: m2_split.split_step(u, dt, dx, chat=CHAT, boundary=1,
                                                      params=PARAMS, kernel=kernel)), U)
                m2_pallas.TILE_FROZEN = TILES[0]
            except Exception as e:                                   # noqa: BLE001
                row["error"] = str(e).split("\n")[0][:160]
            m1 = row.get("m1_ms", float("nan"))
            print(f"{n}^3 {kernel:6s}: " + "  ".join(
                f"{k} {v:8.2f}" + (f" ({v / m1:4.2f}x M1)" if k.startswith("m2") else "")
                for k, v in row.items() if k.endswith("_ms"))
                + (f"  ERROR {row['error']}" if "error" in row else ""), flush=True)
            rows.append(row)
    m1_pallas.KERNEL = "auto"
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("n", "kernel"), k))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
