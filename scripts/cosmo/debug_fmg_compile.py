"""Find where the sharded Particle-Mesh compile blows up -- on GPU.

Times COMPILE (first call) vs RUN (second call) for the replicated FMG and the
shard_map FMG, at increasing grid size, printing each result immediately so we
can watch where it explodes. Then breaks down the other shard_map pieces
(deposit / gather / exchange) at the largest size that compiled.

Run on GPUs (>=2 for a real shard_map):
  PYTHONUNBUFFERED=1 python scripts/cosmo/debug_fmg_compile.py
Env:
  LUNA_DBG_NS   comma list of grid sizes (default "64,128,256")
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.getcwd())

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from gravity.particle_mesh_shardmap import (
    cic_deposit_slab, decompose_to_slabs, exchange_particles_slab,
    gather_accel_slab, make_slab_mesh)
from gravity.poisson_multigrid import (
    _max_levels, solve_poisson_fmg, solve_poisson_fmg_shardmap)


def _t(fn):
    """(compile+run, run-only) wall times."""
    t0 = time.time()
    jax.block_until_ready(fn())
    t1 = time.time()
    jax.block_until_ready(fn())
    t2 = time.time()
    return t1 - t0, t2 - t1


def main():
    ndev = jax.device_count()
    plat = jax.devices()[0].platform
    mesh = make_slab_mesh(jax.devices())
    rng = np.random.default_rng(0)
    Ns = [int(x) for x in os.environ.get("LUNA_DBG_NS", "64,128,256").split(",")]

    print(f"# platform={plat} ndev={ndev}", flush=True)
    print(f"# {'N':>4} {'nlev':>4} {'nx':>3} | {'repl_c':>8} {'repl_r':>7} | "
          f"{'shmap_c':>9} {'shmap_r':>7}", flush=True)
    last_ok = None
    for N in Ns:
        dx = 1.0 / N
        delta = jnp.asarray(rng.standard_normal((N, N, N)))
        deltaS = jax.device_put(delta, NamedSharding(mesh, P("x", "y", "z")))
        nx_loc = N // ndev
        nlev0 = min(_max_levels(N), int(np.log2(nx_loc)) + 1)
        nlevs = [int(x) for x in os.environ.get("LUNA_DBG_NLEV", str(nlev0)).split(",")]
        for nlev in nlevs:
            xloc_coarse = nx_loc >> (nlev - 1)      # per-shard x at coarsest level
            for nx in [int(x) for x in os.environ.get("LUNA_DBG_NX", "0").split(",")]:
                print(f"  {N:>4} {nlev:>4} {nx:>3} | xloc_c={xloc_coarse:>3} | "
                      f"compiling...", flush=True)
                sc, sr = _t(lambda nlev=nlev, nx=nx: solve_poisson_fmg_shardmap(
                    deltaS, dx, 1.0, boundary=0, mesh=mesh, n_levels=nlev,
                    n_extra_vcycles=nx))
                print(f"  {N:>4} {nlev:>4} {nx:>3} | xloc_c={xloc_coarse:>3} | "
                      f"compile {sc:>8.1f}  run {sr:>7.2f}", flush=True)
        last_ok = N

    # component breakdown at the largest size that compiled
    N = last_ok
    print(f"\n# other shard_map pieces at N={N}", flush=True)
    dx = 1.0 / N
    npart = (N // 2) ** 3
    pos = rng.uniform(0, 1.0, size=(npart, 3))
    vel = rng.standard_normal((npart, 3)) * 0.01
    mass = np.full(npart, 1.0 / npart)
    pg, vg, mg, ag, C = decompose_to_slabs(pos, vel, mass, N, ndev, dx, 1.0, 1.5)
    posS = jax.device_put(jnp.asarray(pg), NamedSharding(mesh, P("x", None)))
    pS = jax.device_put(jnp.asarray(vg), NamedSharding(mesh, P("x", None)))
    massS = jax.device_put(jnp.asarray(mg), NamedSharding(mesh, P("x")))
    actS = jax.device_put(jnp.asarray(ag), NamedSharding(mesh, P("x")))
    zerosS = jax.device_put(jnp.zeros((N, N, N)), NamedSharding(mesh, P("x", None, None)))
    for name, fn in [
        ("deposit", lambda: cic_deposit_slab(posS, massS, actS, N, N, N, dx, mesh)),
        ("gather", lambda: gather_accel_slab((zerosS, zerosS, zerosS), posS, N, N, N, dx, mesh)),
        ("exchange", lambda: exchange_particles_slab(posS, pS, massS, actS, N, dx, 1.0, mesh)),
    ]:
        c, r = _t(fn)
        print(f"  {name:>9}: compile {c:8.1f}s  run {r:7.3f}s", flush=True)

    # is argsort the exchange bottleneck? time a raw per-device argsort vs a
    # cumsum-based compaction at the same buffer size.
    print("\n# argsort vs cumsum at per-shard exchange buffer size", flush=True)
    C = pg.shape[0] // ndev
    key = jnp.asarray(rng.standard_normal((C,)))
    msk = jnp.asarray(rng.random((C,)) > 0.5)
    jas = jax.jit(lambda k: jnp.argsort(k))
    def _compact(m):                       # O(n) compaction: dest = cumsum(mask)-1
        idx = jnp.cumsum(m.astype(jnp.int32)) - 1
        out = jnp.zeros((C,), jnp.int32).at[jnp.where(m, idx, C - 1)].set(jnp.arange(C))
        return out
    jcs = jax.jit(_compact)
    ca, ra = _t(lambda: jas(key))
    cc, rc = _t(lambda: jcs(msk))
    print(f"  argsort C={C}: compile {ca:6.1f}s  run {ra:7.3f}s", flush=True)
    print(f"  cumsum  C={C}: compile {cc:6.1f}s  run {rc:7.3f}s", flush=True)


if __name__ == "__main__":
    main()
