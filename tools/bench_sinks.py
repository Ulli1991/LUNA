"""Time the sink hooks of one step on a device mesh: the sinks' field on the
gas (evaluated inside the gas kick), the force on the sinks, and `finish`
(accretion, creation, merging), for a slot count and a rule set.

    python tools/bench_sinks.py --N 512 --mesh 2 1 1 --slots 32 256 --active 32 256

Every slot of `--active` is a live sink, spread over the cloud's centre, on a
lognormal gas above the creation threshold: the accretion windows are full.
Prints one line per (slots, active, rules) with the median of `--reps`."""
import argparse
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from luna.gravity.particle_mesh_blocks import _block_origin  # noqa: E402
from luna.setup.state import Particles  # noqa: E402
from luna.species.sinks_sharded import ShardedSinks  # noqa: E402
from luna.utils.shmap import shard_map  # noqa: E402

RULES = {"luna": dict(accretion="threshold", creation="luna", merging=False),
         "flash": dict(accretion="federrath", creation="flash", merging=True)}


def timed(f, reps, what=""):
    t0 = time.perf_counter()
    jax.block_until_ready(f())                 # compile + warm
    print(f"   {what}: first call {time.perf_counter() - t0:.1f} s", flush=True)
    ts = []
    for i in range(reps):
        print(f"   {what}: rep {i}", flush=True) if i == 0 else None
        t0 = time.perf_counter()
        jax.block_until_ready(f())
        ts.append(time.perf_counter() - t0)
    print(f"   {what}: median {1e3 * float(np.median(ts)):.2f} ms over {reps}", flush=True)
    return 1e3 * float(np.median(ts))


def main():
    import faulthandler
    faulthandler.dump_traceback_later(240, repeat=True)     # where a hang sits
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--mesh", type=int, nargs=3, default=(2, 1, 1))
    ap.add_argument("--slots", type=int, nargs="+", default=(32, 256))
    ap.add_argument("--active", type=int, nargs="+", default=None)
    ap.add_argument("--rules", nargs="+", default=("luna", "flash"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--parts", nargs="+", default=("finish", "field", "force"))
    a = ap.parse_args()
    N, L = a.N, 8.0
    dx = L / N
    devs = np.array(jax.devices()[:int(np.prod(a.mesh))]).reshape(a.mesh)
    mesh = Mesh(devs, ("x", "y", "z"))
    PU, Pg = NamedSharding(mesh, P(None, "x", "y", "z")), NamedSharding(mesh, P("x", "y", "z"))
    rho_thr = 1.0

    @jax.jit
    def gas(key):
        k1, k2 = jax.random.split(key)
        rho = jnp.exp(0.5 * jax.random.normal(k1, (N, N, N)))
        v = 0.3 * jax.random.normal(k2, (3, N, N, N))
        U = jnp.concatenate([rho[None], rho[None] * v,
                             (1.0 + 0.5 * rho * jnp.sum(v * v, 0))[None]])
        return jax.lax.with_sharding_constraint(U, PU)
    U = gas(jax.random.PRNGKey(0))
    jax.block_until_ready(U)
    print("gas ready", flush=True)
    x = (jnp.arange(N) + 0.5) * dx
    phi = jax.jit(lambda: jax.lax.with_sharding_constraint(
        -jnp.exp(-((x[:, None, None] - 4) ** 2 + (x[None, :, None] - 4) ** 2
                   + (x[None, None, :] - 4) ** 2) / 4.0), Pg))()
    for n_max in a.slots:
        for n_act in (a.active or [n_max]):
            if n_act > n_max:
                continue
            rng = np.random.default_rng(1)
            pos = np.zeros((n_max, 3)); act = np.zeros(n_max); m = np.zeros(n_max)
            pos[:n_act] = 4.0 + rng.uniform(-2.0, 2.0, (n_act, 3))
            act[:n_act] = 1.0; m[:n_act] = rng.uniform(0.5, 5.0, n_act)
            vel = np.zeros((n_max, 3)); vel[:n_act] = rng.normal(0, 0.3, (n_act, 3))
            for rule in a.rules:
                sk = ShardedSinks(mesh, N3=(N, N, N), box3=(L, L, L), G=4.3e-3, boundary=0,
                                  n_max=n_max, softening=0.0, accretion_radius=0.0,
                                  creation_density=rho_thr, creation_separation=0.0,
                                  create_frac=0.0, sink_sink=True, is_main=False, **RULES[rule])
                part = sk._replicate(Particles(jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(m),
                                               jnp.asarray(act), jnp.zeros((n_max, 3)),
                                               extra=sk.zero_moments()))
                divv = sk.observe(U)
                nan = float("nan")
                t_fin = t_force = t_field = nan
                if "finish" in a.parts:
                    t_fin = timed(lambda: sk.finish(U, part, phi, divv), a.reps, "finish")
                g = tuple(jax.device_put(jnp.zeros((N, N, N)), Pg) for _ in range(3))
                if "force" in a.parts:
                    t_force = timed(lambda: sk.force(part, g), a.reps, "force")
                nloc = tuple(N // p for p in a.mesh)
                field = jax.jit(shard_map(
                    lambda p_, m_, a_: sk._sink_field_on_grid(
                        p_, m_, a_, origin=_block_origin(tuple(a.mesh), nloc)),
                    mesh=mesh, in_specs=(P(), P(), P()),
                    out_specs=(P("x", "y", "z"),) * 3))
                if "field" in a.parts:
                    t_field = timed(lambda: field(part.pos, part.mass, part.active), a.reps, "field")
                print(f"N={N} mesh={tuple(a.mesh)} slots={n_max} active={n_act} rules={rule}: "
                      f"finish {t_fin:.2f} ms, force {t_force:.2f} ms, "
                      f"field on gas {t_field:.2f} ms (x2 per step in the kick)", flush=True)


if __name__ == "__main__":
    main()
    import os
    os._exit(0)          # ROCm: jax's atexit clear_backends can hang after a 2-GPU run
