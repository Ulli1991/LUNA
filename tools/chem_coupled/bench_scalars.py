"""Cost of carrying passive scalars through the Pallas hydro step: n_scalars = 0 vs 3, one GPU.
    python tools/chem_coupled/bench_scalars.py --n 512 [--reps 10]
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from luna.hydro.conversions import prim_to_cons  # noqa: E402
from luna.time_integration.integrators import create_muscl_hancock_modular_step  # noqa: E402
from luna.utils.halo_exchange_shardmap import make_halo_shardmap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    n = a.n
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    sh = NamedSharding(mesh, P(None, "x", "y", "z"))
    dx = 1.0 / n

    @jax.jit
    def ic():
        x = (jnp.arange(n) + 0.5) / n
        X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
        k = 2 * jnp.pi
        rho = 1.0 + 0.5 * jnp.sin(k * X) * jnp.cos(k * Y)
        W = jnp.stack([rho, 0.3 * jnp.sin(k * Y), 0.3 * jnp.sin(k * Z), 0.3 * jnp.sin(k * X),
                       1.0 + 0.1 * jnp.cos(k * Z)])
        U = prim_to_cons(W, 5.0 / 3.0)
        return U, jnp.stack([rho * 0.1, rho * 1e-3, rho * 1e-5])
    U, S = (jax.device_put(v, sh) for v in ic())
    dt = 0.2 * dx
    mk = lambda ns: create_muscl_hancock_modular_step(  # noqa: E731
        make_halo_shardmap(mesh, 3), 5.0 / 3.0, 1, 0, dx, dx, dx, 1 / dx, 1 / dx, 1 / dx, 0, 3,
        0.4, riemann_solver="hllc", use_shardmap=True, mesh=mesh, kernel="pallas",
        n_scalars=ns)[1]
    for ns, call in ((0, lambda f: f(U, dt)), (3, lambda f: f(U, S, dt))):
        f = mk(ns)
        jax.block_until_ready(call(f))
        w = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(call(f))
            w.append(time.perf_counter() - t0)
        w = np.array(w) * 1e3
        print(f"{n}^3 n_scalars={ns}: min {w.min():.2f} ms, median {np.median(w):.2f} ms "
              f"({w.min() * 1e6 / n**3:.2f} ns/cell)", flush=True)


if __name__ == "__main__":
    main()
