"""The hydro step's roofline on one GPU: the production MUSCL-Hancock step
(shard_map, wide halo, minmod, HLLC, FP64) at a ladder of grid sizes, with
the XLA divergence and with the fused Pallas kernel (hydro_kernel), steady
state (median of REPS after a warm-up), and the step's bytes as XLA counts
them for the XLA path. PRECISION=fp32 or mixed (recon32) replaces FP64; RIEMANN=hll replaces HLLC;
DIM=1 or 2 runs that dimension (SIZES are then cells per side); LIMITER=1 (MC) or 2 (van Leer);
PFS=1 preserve_face_states; REPORT=1 the step with its face report (as the runner).

    python tools/bench_hydro_roofline.py      # env: SIZES, REPS, OUT, BOUNDARY, PRECISION, RIEMANN, DIM, LIMITER, PFS, REPORT
"""
import csv
import os
import time

import numpy as np

import jax
PRECISION = os.environ.get("PRECISION", "fp64")
if PRECISION not in ("fp64", "fp32", "mixed"):
    raise ValueError(f"PRECISION={PRECISION!r}: fp64, fp32 or mixed")
jax.config.update("jax_enable_x64", PRECISION != "fp32")
import jax.numpy as jnp  # noqa: E402

from luna.hydro.conversions import prim_to_cons  # noqa: E402
from luna.time_integration.integrators import create_muscl_hancock_modular_step  # noqa: E402
from luna.utils.halo_exchange_shardmap import make_halo_shardmap  # noqa: E402

GAMMA = 5.0 / 3.0
SIZES = [int(s) for s in os.environ.get("SIZES", "64,128,256,384,512").split(",")]
REPS = int(os.environ.get("REPS", "10"))
OUT = os.environ.get("OUT", "hydro_roofline.csv")
BOUNDARY = int(os.environ.get("BOUNDARY", "1"))     # 0 periodic, 1 zero-gradient, 4 diode
RIEMANN = os.environ.get("RIEMANN", "hllc")         # hllc | hll
DIM = int(os.environ.get("DIM", "3"))
LIMITER = int(os.environ.get("LIMITER", "0"))       # 0 minmod, 1 MC, 2 van Leer
SWITCHES = {"floors": (1e-10, 1e-10)} if os.environ.get("PFS", "0") == "1" else {}
if os.environ.get("REPORT", "0") == "1":           # the face report the runner reads (setup/face_report.py)
    SWITCHES["face_report"] = True


def sedov_like(n):
    """Uniform gas with a hot sphere (DIM-ball): shocks and smooth flow in one state."""
    x = (np.arange(n) + 0.5) / n - 0.5
    X = np.meshgrid(*([x] * DIM), indexing="ij")
    r = np.sqrt(sum(a * a for a in X))
    P = np.where(r < 0.1, 1e3, 1.0)
    W = np.stack([np.ones_like(r)] + [0.1 * a for a in X] + [P])
    return prim_to_cons(jnp.asarray(W, jnp.float32 if PRECISION == "fp32" else jnp.float64), GAMMA)


def main():
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    axes = ("x", "y", "z")[:DIM]
    mesh = Mesh(np.array(jax.devices()[:1]).reshape((1,) * DIM), axes)
    rows = []
    for n in SIZES:
        U = jax.device_put(sedov_like(n), NamedSharding(mesh, P(None, *axes)))
        dx = 1.0 / n
        dt = 0.2 * dx / 40.0
        for kernel in ("xla", "pallas"):
            step = create_muscl_hancock_modular_step(
                make_halo_shardmap(mesh, DIM), GAMMA, 1, LIMITER, dx, dx, dx, 1 / dx, 1 / dx, 1 / dx,
                BOUNDARY, DIM, 0.4, riemann_solver=RIEMANN, use_shardmap=True, mesh=mesh,
                kernel=kernel, recon32=PRECISION == "mixed", **SWITCHES)[1]
            try:
                out = step(U, dt); jax.block_until_ready(out)
                ts = []
                for _ in range(REPS):
                    t0 = time.perf_counter(); out = step(U, dt); jax.block_until_ready(out)
                    ts.append(time.perf_counter() - t0)
                t = float(np.median(ts))
                gb = float("nan")
                if kernel == "xla":
                    ca = step.lower(U, dt).compile().cost_analysis()
                    ca = ca[0] if isinstance(ca, (list, tuple)) else ca
                    gb = float(ca.get("bytes accessed", 0.0)) / 2 ** 30
                row = dict(N=n, dim=DIM, limiter=LIMITER, precision=PRECISION, kernel=kernel, ms=1e3 * t,
                           mcells=n ** DIM / t / 1e6, xla_gib=gb)
            except Exception as e:                                   # noqa: BLE001
                row = dict(N=n, dim=DIM, limiter=LIMITER, precision=PRECISION, kernel=kernel, ms=float("nan"),
                           mcells=float("nan"), xla_gib=float("nan"))
                print(f"N={n} {kernel}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            rows.append(row)
            print(f"N={n:8d} {DIM}-D lim={LIMITER} {PRECISION:5s} {kernel:6s} {row['ms']:9.2f} ms  {row['mcells']:8.0f} Mcell/s"
                  f"  XLA bytes {row['xla_gib']:7.1f} GiB", flush=True)
        del U
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
