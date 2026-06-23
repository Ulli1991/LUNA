"""Microbenchmark + parameter sweep for the sharded FMG Poisson solver.

Isolates solve_poisson_fmg_shardmap on an Nx^3 slab-sharded grid (multi-GPU),
times the steady-state solve, and sweeps the convergence knobs
(n_coarsest, n_extra_vcycles, n_pre/n_post) to find a faster setting that
keeps accuracy. Accuracy = relative L2 of phi vs the production reference
(n_coarsest=50, n_extra=2, n_pre=n_post=2) AND the true residual
||lap(phi) - rhs|| / ||rhs||.

  BENCH_NX (default 512), runs on however many devices JAX sees.
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.getcwd())
from functools import partial

from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from gravity.particle_mesh_shardmap import make_slab_mesh
from gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap
from gravity.poisson_multigrid import (_max_levels, laplacian_7pt,
                                       solve_poisson_fmg_shardmap)

Nx = int(os.environ.get("BENCH_NX", "512"))
mesh = make_slab_mesh(jax.devices())
ndev = int(mesh.shape["x"])
nx_loc = Nx // ndev
n_levels = min(_max_levels(Nx), int(np.log2(nx_loc)) + 1)
dx = 1.0 / Nx
fourpiG = 1.0
spec = P("x", "y", "z")
shd = NamedSharding(mesh, spec)
is_main = jax.process_index() == 0

# cosmo-like source: low-k structure + small-scale noise, zero mean, built per
# x-slab on host to avoid a full Nx^3 host array, then assembled sharded.
rng = np.random.default_rng(0)
xs = np.arange(Nx) / Nx
yz_lowk = (np.sin(2 * np.pi * np.arange(Nx) / Nx)[:, None]
           + np.cos(4 * np.pi * np.arange(Nx) / Nx)[None, :])

def _slab(i):
    lo = i * nx_loc
    blk = np.empty((nx_loc, Nx, Nx), dtype=np.float64)
    for k in range(nx_loc):
        ph = np.sin(2 * np.pi * xs[lo + k])
        blk[k] = ph * yz_lowk + 0.1 * rng.standard_normal((Nx, Nx))
    return jnp.asarray(blk)

delta = jax.make_array_from_single_device_arrays(
    (Nx, Nx, Nx), shd,
    [jax.device_put(_slab(i), d) for i, d in enumerate(mesh.devices.ravel())])
delta = delta - jnp.mean(delta)            # zero mean (periodic solvability)
jax.block_until_ready(delta)

# sharded residual ||lap(phi) - fourpiG*delta|| / ||fourpiG*delta||
halo_fn = make_scalar_halo_3d_shardmap(mesh)(0)

@jax.jit
@partial(shard_map, mesh=mesh, in_specs=(spec, spec), out_specs=P())
def _resid_norm(phi, rho):
    lap = laplacian_7pt(phi, dx, halo_fn)
    num = jnp.sum((lap - fourpiG * rho) ** 2)
    den = jnp.sum((fourpiG * rho) ** 2)
    return jax.lax.psum(jnp.stack([num, den]), ("x", "y", "z"))

def resid(phi):
    nd = _resid_norm(phi, delta)
    return float(jnp.sqrt(nd[0] / nd[1]))

def timed(**kw):
    f = lambda: solve_poisson_fmg_shardmap(
        delta, dx, fourpiG, boundary=0, mesh=mesh, n_levels=n_levels, **kw)
    phi = f(); jax.block_until_ready(phi)        # compile
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); jax.block_until_ready(f()); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), phi

if is_main:
    print(f"FMG bench: Nx={Nx}^3, {ndev} devices, nx_loc={nx_loc}, n_levels={n_levels}", flush=True)
    print(f"{'setting':<42} {'t(ms)':>9} {'resid':>10} {'relL2':>10}", flush=True)

# reference = production cosmo settings
ref_kw = dict(n_extra_vcycles=2, n_pre=2, n_post=2, n_coarsest=50)
t_ref, phi_ref = timed(**ref_kw)
r_ref = resid(phi_ref)
if is_main:
    print(f"{'REF n_coarse=50 n_extra=2 n_pre=n_post=2':<42} {t_ref*1e3:>9.1f} {r_ref:>10.2e} {'--':>10}", flush=True)

trials = [
    dict(n_extra_vcycles=2, n_pre=2, n_post=2, n_coarsest=20),
    dict(n_extra_vcycles=2, n_pre=2, n_post=2, n_coarsest=10),
    dict(n_extra_vcycles=1, n_pre=2, n_post=2, n_coarsest=20),
    dict(n_extra_vcycles=1, n_pre=2, n_post=2, n_coarsest=10),
    dict(n_extra_vcycles=1, n_pre=1, n_post=1, n_coarsest=10),
    dict(n_extra_vcycles=0, n_pre=2, n_post=2, n_coarsest=20),
    dict(n_extra_vcycles=1, n_pre=2, n_post=2, n_coarsest=5),
]
for kw in trials:
    t, phi = timed(**kw)
    rel = float(jnp.linalg.norm(phi - phi_ref) / jnp.linalg.norm(phi_ref))
    if is_main:
        tag = f"n_coarse={kw['n_coarsest']} n_extra={kw['n_extra_vcycles']} n_pre=n_post={kw['n_pre']}"
        print(f"{tag:<42} {t*1e3:>9.1f} {resid(phi):>10.2e} {rel:>10.2e}"
              f"   ({t_ref/t:.2f}x)", flush=True)
print("=== done ===", flush=True) if is_main else None
