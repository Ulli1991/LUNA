"""Multi-GPU slab-kernel roofline: time the sharded PM kernels in isolation
(deposit_slab / FMG shardmap cold & warm / grad+gather / particle exchange)
to localize the multi-GPU slowdown seen in the l10 rung. Synthetic lattice
particles (kernel timing is IC-independent, and it skips the IC read).

  LUNA_GRAV_NX     global mesh per side (= particles per side)
  LUNA_GRAV_REPS   timed reps per kernel (default 5)
"""
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.getcwd())
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    from setup import initialize_jax_distributed
    initialize_jax_distributed(enable_x64=True)

import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from gravity.particle_mesh_shardmap import (assemble_slab_from_chunk,
                                            cic_deposit_slab,
                                            exchange_particles_slab,
                                            gather_accel_slab, grad_phi_slab,
                                            make_slab_mesh)
from gravity.poisson_multigrid import _max_levels, solve_poisson_fmg_shardmap

NX = int(os.environ["LUNA_GRAV_NX"])
REPS = int(os.environ.get("LUNA_GRAV_REPS", "5"))
BOX = 100000.0                                   # kpc/h, arbitrary for timing
main = jax.process_index() == 0


def bench(name, fn, *args):
    fn = jax.jit(fn)
    t0 = time.time()
    out = jax.block_until_ready(fn(*args))
    t_compile = time.time() - t0
    ts = []
    for _ in range(REPS):
        t0 = time.time()
        jax.block_until_ready(fn(*args))
        ts.append(time.time() - t0)
    if main:
        print(f"[slab-roofline] nx={NX} ndev={jax.device_count()} "
              f"op={name:10s} compile={t_compile:6.1f}s "
              f"ms={1e3 * float(np.median(ts)):10.3f}", flush=True)
    return out


mesh = make_slab_mesh(jax.devices())
ndev = jax.device_count()
nproc, pid = jax.process_count(), jax.process_index()
dx = BOX / NX
n_total = NX ** 3

# synthetic lattice particles for this process's x-plane range, mid-cell,
# deterministic sub-cell jitter so CIC touches all 8 neighbours
x0, x1 = pid * NX // nproc, (pid + 1) * NX // nproc
xs = (np.arange(x0, x1) + 0.5) * dx
ys = (np.arange(NX) + 0.5) * dx
g = np.stack(np.meshgrid(xs, ys, ys, indexing="ij"), axis=-1).reshape(-1, 3)
rng = np.random.default_rng(42 + pid)
pos_l = (g + (rng.random(g.shape) - 0.5) * 0.6 * dx).astype(np.float64)
vel_l = np.zeros_like(pos_l)
mass_l = np.full(pos_l.shape[0], 1.0)
del g

(posS, pS, massS), actS, C = assemble_slab_from_chunk(
    [pos_l, vel_l, mass_l], [P("x", None), P("x", None), P("x")],
    n_total, ndev, mesh, capacity_factor=1.5)
if main:
    print(f"[slab-roofline] {n_total} particles  mesh={NX}^3  cap_C={C}",
          flush=True)
n_levels = min(_max_levels(NX), int(np.log2(NX // ndev)) + 1)

rho = bench("deposit", lambda p, m, a: cic_deposit_slab(
    p, m, a, NX, NX, NX, dx, mesh), posS, massS, actS)
delta = rho / jnp.mean(rho) - 1.0

phi = bench("fmg_cold", lambda d: solve_poisson_fmg_shardmap(
    d, dx, 1.0, boundary=0, mesh=mesh, n_levels=n_levels,
    n_extra_vcycles=4), delta)

phi = bench("fmg_warm", lambda d, ph: solve_poisson_fmg_shardmap(
    d, dx, 1.0, boundary=0, mesh=mesh, n_levels=n_levels,
    phi_init=ph, warm_cycles=1), delta, phi)

gacc = bench("grad", lambda ph: grad_phi_slab(ph, dx, mesh, order=4), phi)
bench("gather", lambda ga, p: gather_accel_slab(ga, p, NX, NX, NX, dx, mesh),
      gacc, posS)
bench("exchange", lambda p, v, m, a: exchange_particles_slab(
    p, v, m, a, NX, dx, BOX, mesh), posS, pS, massS, actS)
if main:
    print("[slab-roofline] done", flush=True)
