"""Multi-GPU 3D-block PM kernel roofline: time the block-sharded kernels in
isolation (deposit / FMG shardmap cold & warm / grad+gather / exchange), so the
production blocks path can be read kernel by kernel against its budget.

Synthetic lattice particles, built per process for its OWN block only — no
global (ndev*C) host array, so an l10-sized run fits in host memory.

Also reports the mover census: how many particles cross a block face once the
cumulative drift reaches the runner's exchange trigger (0.9 dx). That number is
what sizes ``face_capacity_factor``; the slab path found its default was ~100x
oversized.

    srun -n 8 python -m scripts.validation.runs.gravity_blocks_roofline nx=1024 pdims=2,2,2

  nx         global mesh: one number for a cube, or 512x512x1024 for a
             non-cubic grid (a fixed per-GPU tile when the device count is
             not a cube). Particles = one per cell.
  pdims      block layout px,py,pz (default: balanced over the devices)
  reps       timed reps per kernel (default 5)
  fp32       fp32 particles + fp32 delta/psi (gravity_precision 'single'):
             the fat 1024^3-per-GPU tile does not fit an MI300A in fp64
             (~135 GB); fp32 does (~75 GB). Its own curve, never mixed in.
  step_only  skip the per-kernel section, time the full step only (~8 min
             of 32-node time saved per 64-GPU submission)
"""
import os
import time

import jax

jax.config.update("jax_enable_x64", True)
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    from luna.setup.jax_init import initialize_jax_distributed
    initialize_jax_distributed(enable_x64=True)

import jax.numpy as jnp
import numpy as np
from functools import partial
from luna.utils.shmap import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from luna.gravity.particle_mesh_blocks import (AXES, cic_deposit_blocks,
                                          exchange_particles_blocks,
                                          gather_accel_blocks,
                                          grad_phi_blocks, make_block_mesh,
                                          resort_blocks)
from luna.gravity.poisson_multigrid import solve_poisson_fmg_shardmap
from luna.setup.cli import args
from luna.setup.mesh import balanced_pdims


def _triple(s):
    """'512' -> (512,512,512); '512x512x1024' / '512,512,1024' -> as written."""
    v = [int(q) for q in str(s).replace("x", ",").split(",")]
    return tuple(v * 3) if len(v) == 1 else tuple(v)


ARGS = args(nx="512", pdims="", reps=5, fp32=False, step_only=False)
NG = _triple(ARGS["nx"])
NX = NG[0]                                       # shortest side sets dx
FP32 = ARGS["fp32"]
PDT = np.float32 if FP32 else np.float64
SOLVE_DT = "float32" if FP32 else "float64"
REPS = ARGS["reps"]
DXBOX = 100000.0                                 # kpc/h, arbitrary for timing
main = jax.process_index() == 0

ndev = jax.device_count()
pdims = _triple(ARGS["pdims"]) if ARGS["pdims"] else tuple(balanced_pdims(ndev))
mesh = make_block_mesh(jax.devices(), pdims)
dx = DXBOX / NX
BOX = tuple(NG[a] * dx for a in range(3))        # per-axis extent, cells cubic
n_total = int(np.prod(NG))
n_cells = n_total                                # one particle per cell
nloc = [NG[a] // pdims[a] for a in range(3)]


def bench(name, fn, *args, work=None, unit="Mpart/s", jit=True, reps=None):
    """Time one kernel and report throughput. ``work`` is the GLOBAL item count
    (particles or cells), so `agg` is aggregate throughput and `per-GPU` is the
    weak-scaling number: at a fixed per-GPU tile it should stay FLAT as the
    device count grows, and per-GPU/per-GPU(1 device) is the efficiency."""
    fn = jax.jit(fn) if jit else fn
    t0 = time.time()
    out = jax.block_until_ready(fn(*args))
    t_compile = time.time() - t0
    ts = []
    for _ in range(reps or REPS):
        t0 = time.time()
        jax.block_until_ready(fn(*args))
        ts.append(time.time() - t0)
    if main:
        ms = 1e3 * float(np.median(ts))
        thr = (work / 1e6) / (ms * 1e-3) if work else 0.0
        print(f"[blocks-roofline] ndev={ndev:<3d} pdims={pdims} "
              f"op={name:12s} compile={t_compile:6.1f}s ms={ms:10.3f} "
              f"{unit}={thr:10.1f} per-GPU={thr / ndev:9.1f}", flush=True)
    return out


# --- per-process block-local lattice ---------------------------------------
# Device i of mesh.devices.ravel() owns block unravel_index(i, pdims); the flat
# particle axis is sharded P(("x","y","z")), i.e. row-major over the same order
# assemble_blocks_sharded uses. Build only the blocks this process hosts.
C = int(np.ceil(1.5 * n_total / ndev))
devs = list(mesh.devices.ravel())
multi = jax.process_count() > 1
bufs = [[], [], [], []]
for i, d in enumerate(devs):
    if multi and d.process_index != jax.process_index():
        continue
    b = np.unravel_index(i, pdims)
    ax = [(np.arange(b[a] * nloc[a], (b[a] + 1) * nloc[a]) + 0.5) * dx
          for a in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    rng = np.random.default_rng(42 + i)
    cp = np.zeros((C, 3), PDT)
    cp[:g.shape[0]] = (g + (rng.random(g.shape) - 0.5) * 0.6 * dx).astype(PDT)
    cv = np.zeros((C, 3), PDT)
    cm = np.zeros((C,), PDT)
    ca = np.zeros((C,), PDT)
    cm[:g.shape[0]] = 1.0
    ca[:g.shape[0]] = 1.0
    del g
    for j, c in enumerate((cp, cv, cm, ca)):
        bufs[j].append(jax.device_put(c, d))
flat = NamedSharding(mesh, P(("x", "y", "z")))
flat2 = NamedSharding(mesh, P(("x", "y", "z"), None))
mk = jax.make_array_from_single_device_arrays
posS = mk((ndev * C, 3), flat2, bufs[0])
pS = mk((ndev * C, 3), flat2, bufs[1])
massS = mk((ndev * C,), flat, bufs[2])
actS = mk((ndev * C,), flat, bufs[3])
del bufs
if main:
    print(f"[blocks-roofline] {n_total} particles  mesh={NG}  pdims={pdims} "
          f"block={nloc}  cap_C={C}", flush=True)

# --- mover census at the runner's exchange trigger --------------------------
# The runner exchanges when cumulative drift reaches 0.9 dx; displace every
# particle by that much in a random direction and count face crossers per axis.
@partial(jax.jit, static_argnames=())
@partial(shard_map, mesh=mesh,
         in_specs=(P(("x", "y", "z"), None), P(("x", "y", "z"))),
         out_specs=P())
def _movers(pos_l, act_l):
    key = jax.random.PRNGKey(7)
    u = jax.random.normal(key, pos_l.shape)
    u = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), 1e-30)
    q = jnp.mod(pos_l + 0.9 * dx * u, jnp.asarray(BOX))
    n = jnp.zeros((), jnp.int32)
    for a in range(3):
        npx = int(mesh.shape[AXES[a]])
        if npx == 1:
            continue
        bw = nloc[a] * dx
        d = jax.lax.axis_index(AXES[a])
        tgt = jnp.clip((q[:, a] / bw).astype(jnp.int32), 0, npx - 1)
        n = n + jnp.sum(((act_l > 0) & (tgt != d)).astype(jnp.int32))
    return jax.lax.pmax(n, AXES)


n_mov = int(jax.block_until_ready(_movers(posS, actS)))
if main:
    print(f"[blocks-roofline] movers/device at the 0.9dx trigger: {n_mov} "
          f"({n_mov / C:.5f} of C={C}); face_capacity_factor default 0.125 "
          f"sizes each face buffer at {int(0.125 * C)} slots", flush=True)

# delta via explicit shard_map psum — an auto-sharded jnp.mean spliced into
# the eager shard_map chain WEDGES the ROCm runtime at rank count (silent
# deadlock in this section at 64 GPUs, twice, 2026-08-27; same class as the
# coupled runner's step-2 hang).
@partial(jax.jit, static_argnames=())
@partial(shard_map, mesh=mesh, in_specs=(P("x", "y", "z"),),
         out_specs=P("x", "y", "z"))
def _delta_of(rho_l):
    ssum = jax.lax.psum(jnp.sum(rho_l), AXES)
    cnt = jax.lax.psum(jnp.asarray(rho_l.size, rho_l.dtype), AXES)
    return rho_l / (ssum / cnt) - 1.0        # dtype-preserving


# --- kernels ----------------------------------------------------------------
if not ARGS["step_only"]:
    rho = bench("deposit", lambda p, m, a: cic_deposit_blocks(
        p, m, a, NG, dx, mesh), posS, massS, actS, work=n_total)
    rho = bench("deposit_32", lambda p, m, a: cic_deposit_blocks(
        p, m, a, NG, dx, mesh, dtype=jnp.float32), posS, massS, actS,
        work=n_total)
    rho = bench("deposit_fx", lambda p, m, a: cic_deposit_blocks(
        p, m, a, NG, dx, mesh, dtype=jnp.float32,
        fixed=jnp.int32, fixed_bits=17), posS, massS, actS, work=n_total)
    delta = _delta_of(rho)

    phi = bench("fmg_cold", lambda d: solve_poisson_fmg_shardmap(
        d, dx, 1.0, boundary=0, mesh=mesh, n_extra_vcycles=4), delta,
        work=n_cells, unit="Mcell/s")
    phi = bench("fmg_warm", lambda d, ph: solve_poisson_fmg_shardmap(
        d, dx, 1.0, boundary=0, mesh=mesh, phi_init=ph, warm_cycles=1,
        n_pre=1, n_post=1), delta, phi, work=n_cells, unit="Mcell/s")

    gacc = bench("grad", lambda ph: grad_phi_blocks(ph, dx, mesh, order=4),
                 phi, work=n_cells, unit="Mcell/s")
    bench("gather", lambda ga, p: gather_accel_blocks(ga, p, NG, dx, mesh),
          gacc, posS, work=n_total)

    # Exchange: the old fixed 0.125*C buffer vs the geometric default.
    bench("exchange_012", lambda p, v, m, a: exchange_particles_blocks(
        p, v, m, a, NG, dx, BOX, mesh, face_capacity_factor=0.125),
        posS, pS, massS, actS, work=n_total)
    bench("exchange", lambda p, v, m, a: exchange_particles_blocks(
        p, v, m, a, NG, dx, BOX, mesh), posS, pS, massS, actS, work=n_total)
    # the persistent particle IDs ride in the exchange and the resort
    idsS = jax.jit(lambda a: jnp.where(a > 0, jnp.arange(a.shape[0]), -1),
                   out_shardings=flat)(actS)
    bench("exchange_ids", lambda p, v, m, a, i: exchange_particles_blocks(
        p, v, m, a, NG, dx, BOX, mesh, ids=i), posS, pS, massS, actS, idsS,
        work=n_total)
    bench("resort", lambda p, v, m, a: resort_blocks(p, v, m, a, p, NG, dx, mesh),
          posS, pS, massS, actS, work=n_total, jit=False)
    bench("resort_ids", lambda p, v, m, a, i: resort_blocks(
        p, v, m, a, p, NG, dx, mesh, ids=i), posS, pS, massS, actS, idsS,
        work=n_total, jit=False)
    if main:
        from luna.gravity.particle_mesh_blocks import _face_capacity
        print(f"[blocks-roofline] face buffer: old {int(0.125 * C)} slots vs "
              f"geometric {_face_capacity(NG, mesh, C, None)} slots, for "
              f"{n_mov} movers", flush=True)
else:
    # step-only: build the warm seed uncounted (one fixed deposit + cold FMG)
    rho = cic_deposit_blocks(posS, massS, actS, NG, dx, mesh,
                             dtype=jnp.float32, fixed=jnp.int32,
                             fixed_bits=17)
    delta = _delta_of(rho)
    phi = solve_poisson_fmg_shardmap(delta, dx, 1.0, boundary=0, mesh=mesh,
                                     n_extra_vcycles=4)

# --- FULL production step ---------------------------------------------------
# The runner's per-step sequence, NOT fused into one jit: _run_blocks keeps the
# MG solve out of the step programs on purpose (fusing it asked XLA's SPMD
# partitioner to swallow the multigrid pyramid and the compile exploded at
# 1024^3). So this times the same chain of separately-jitted calls the runner
# makes, with one block_until_ready at the end -- that IS the production step.
mass_unit = n_total / float(BOX[0] * BOX[1] * BOX[2])


def step_warm(pos, mass, act, psi):
    rho = cic_deposit_blocks(pos, mass, act, NG, dx, mesh, dtype=jnp.float32,
                             fixed=jnp.int32, fixed_bits=17,
                             mass_unit=mass_unit)
    # runner parity: field_stats hands the solve an fp64 delta and the carried
    # psi is fp64 too -- the fori_loop carry must match on BOTH.
    psi = solve_poisson_fmg_shardmap(_delta_of(rho.astype(SOLVE_DT)), dx, 1.0,
                                     boundary=0, mesh=mesh, phi_init=psi,
                                     warm_cycles=1, n_pre=1, n_post=1)
    g = gather_accel_blocks(grad_phi_blocks(psi, dx, mesh, order=4),
                            pos, NG, dx, mesh)
    return g, psi


if main:
    print("[blocks-roofline] step_warm: compiling the fp64 warm-solve variant",
          flush=True)
bench("step_warm", step_warm, posS, massS, actS, phi.astype(SOLVE_DT),
      reps=max(REPS, 20),   # a real plateau, like the hydro 30-step runs
      work=n_total, jit=False)
if main:
    print("[blocks-roofline] done", flush=True)
