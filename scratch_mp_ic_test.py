"""Multi-PROCESS (4 ranks, 1 CPU device each) test of the chunked multi-node IC
ingestion on the REAL l8 (256^3) MUSIC baryon IC: each rank reads its index-chunk,
assembles its slab via make_array_from_single_device_arrays, settles with one
drift exchange, then checks (collectively): no particles lost, mass conserved, and
every active particle landed in its OWNING x-slab (mis-owned==0 => the x-major
chunk≈slab assumption holds and a single-hop settle suffices)."""
import os
import sys
os.environ["JAX_PLATFORM_NAME"] = "cpu"
sys.path.insert(0, "/raven/u/uli/jax-hydro")
from setup.distributed_setup import initialize_jax_distributed
initialize_jax_distributed(enable_x64=True)
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec as P

from cosmology.music_ic import music_part_count, read_music_dm_ic
from gravity.particle_mesh_shardmap import (assemble_slab_from_chunk,
                                            exchange_particles_slab, make_slab_mesh)

IC = "/ptmp/uli/jax-hydro/music/ics_arepo_l8_bary.hdf5"
Nx = 256
nproc, pid = jax.process_count(), jax.process_index()
mesh = make_slab_mesh(jax.devices())
ndev = int(mesh.shape["x"])
box = 25000.0
dx = box / Nx
nx_loc = Nx // ndev
slab_w = nx_loc * dx

N = music_part_count(IC, 1)
lo = pid * N // nproc
hi = N if pid == nproc - 1 else (pid + 1) * N // nproc
dm = read_music_dm_ic(IC, 1, lo, hi)

(posS, pS, massS), actS, C = assemble_slab_from_chunk(
    [np.asarray(dm.pos), np.asarray(dm.vel), np.asarray(dm.mass)],
    [P("x", None), P("x", None), P("x")], N, ndev, mesh, capacity_factor=1.3)
# how far is the chunk from the slab BEFORE the settle?
pre = np.asarray(posS.addressable_shards[0].data)
pre_act = np.asarray(actS.addressable_shards[0].data) > 0
pre_owner = np.clip((np.mod(pre[pre_act, 0], box) / slab_w).astype(int), 0, ndev - 1)
pre_mis = int(np.sum(pre_owner != pid))

posS, pS, massS, actS, nbad, occ = exchange_particles_slab(
    posS, pS, massS, actS, Nx, dx, box, mesh)

loc = np.asarray(posS.addressable_shards[0].data)
act = np.asarray(actS.addressable_shards[0].data) > 0
mass = np.asarray(massS.addressable_shards[0].data)
owner = np.clip((np.mod(loc[act, 0], box) / slab_w).astype(int), 0, ndev - 1)
mis = int(np.sum(owner != pid))
nloc = int(act.sum())
lmass = float(mass[act].sum())

g = lambda v: float(np.asarray(multihost_utils.process_allgather(jnp.array(v))).sum())
tot_act = int(g(nloc)); tot_mass = g(lmass)
tot_mis = int(g(mis)); tot_premis = int(g(pre_mis))
if pid == 0:
    print(f"N={N} nproc={nproc} cap_C={C} nx_loc={nx_loc}")
    print(f"pre-settle mis-owned (chunk vs slab): {tot_premis} ({100*tot_premis/N:.3f}%)")
    print(f"post-settle: active={tot_act} (want {N})  mis-owned={tot_mis}  "
          f"nbad={int(nbad)}  mass={tot_mass:.6e}")
    ok = (tot_act == N) and (tot_mis == 0)
    print("PASS" if ok else f"NEEDS-MULTIHOP (mis={tot_mis}, lost={N-tot_act})")
