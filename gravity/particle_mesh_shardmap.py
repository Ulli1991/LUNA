"""Grid-sharded Particle-Mesh primitives for the cosmological DM N-body, built
for grids beyond what a single device can hold (>~512^3).

Decomposition: a 1-D slab decomposition along x. The device mesh is 3-D,
``(ndev, 1, 1)`` over axes ``("x", "y", "z")``, so the *grid* shards along x
only and the y/z halo ppermutes inside `solve_poisson_fmg_shardmap` collapse to
identity (size-1 axes). This is what lets us reuse the gas shard_map multigrid
solver completely unchanged, while keeping the *particle* bookkeeping 1-D: a
particle ever only crosses into the adjacent (ring) slab in one step, so the
drift-time exchange is a single left/right ppermute rather than a 26-neighbour
problem.

Invariant maintained across the whole run: the global particle buffer has shape
``(ndev * C, ...)`` sharded ``P("x", ...)``, and device d's local chunk holds
exactly the particles whose comoving x lies in slab d (``[d*nx_loc, (d+1)*nx_loc)``
cells), padded out to capacity C with inactive slots. Every primitive here
relies on that invariant via ``jax.lax.axis_index("x")``.

Primitives (each wraps its own shard_map; compose on globally-sharded arrays):
  * cic_deposit_slab   particles -> local rho slab, +1-cell x-ghost add (ppermute)
  * grad_phi_slab      phi slab -> -grad(phi) slab (2-cell x-halo, 4th order)
  * gather_accel_slab  force slabs + particles -> accel at particles (1-cell halo)
  * exchange_particles_slab   redistribute particles to their owning slab

All y/z indexing is periodic (modulo Ny/Nz); x periodicity is carried by the
ppermute ring. dx is assumed cubic (dx=dy=dz), as in the rest of the PM code.
"""
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


def make_slab_mesh(devices):
    """3-D mesh ``(ndev,1,1)`` over ("x","y","z") — slab decomposition along x."""
    devs = np.asarray(devices)
    ndev = devs.shape[0]
    return Mesh(devs.reshape(ndev, 1, 1), ("x", "y", "z"))


def decompose_to_slabs(pos, vel, mass, Nx, ndev, dx, box, capacity_factor=1.5):
    """Host-side: bucket particles into per-slab fixed-capacity buffers.

    Returns (pos_g, vel_g, mass_g, active_g, C) as numpy arrays of global shape
    (ndev*C, ...), laid out so device d's contiguous chunk [d*C:(d+1)*C] holds
    exactly the slab-d particles followed by inactive padding. `vel` carries
    whatever per-particle vector travels with the position (e.g. canonical
    momentum); pass None to skip. Capacity C = ceil(capacity_factor * N/ndev),
    rounded so ndev*C buffers every slab with headroom for clustering.
    """
    n = pos.shape[0]
    nx_loc = Nx // ndev
    slab_w = nx_loc * dx
    xw = np.mod(pos[:, 0], box)
    owner = np.clip((xw / slab_w).astype(np.int64), 0, ndev - 1)
    counts = np.bincount(owner, minlength=ndev)
    C = int(np.ceil(capacity_factor * n / ndev))
    C = max(C, int(counts.max()))                         # never under-size a slab
    pos_g = np.zeros((ndev * C, 3), dtype=pos.dtype)
    vel_g = None if vel is None else np.zeros((ndev * C, vel.shape[1]), dtype=vel.dtype)
    mass_g = np.zeros((ndev * C,), dtype=mass.dtype)
    active_g = np.zeros((ndev * C,), dtype=np.float64)
    for d in range(ndev):
        sel = np.nonzero(owner == d)[0]
        base = d * C
        pos_g[base:base + sel.size] = pos[sel]
        if vel is not None:
            vel_g[base:base + sel.size] = vel[sel]
        mass_g[base:base + sel.size] = mass[sel]
        active_g[base:base + sel.size] = 1.0
    return pos_g, vel_g, mass_g, active_g, C


def assemble_slab_from_chunk(local_arrays, specs, N_total, ndev, mesh,
                             capacity_factor=1.15):
    """Multi-PROCESS IC assembly (one GPU per process). Each process passes only
    its OWN index-chunk of particles -- which, by MUSIC's x-major Lagrangian
    ordering, is already ~its x-slab at z_init -- as host arrays in `local_arrays`
    (each (n_local, ...)), with matching `specs` (PartitionSpecs). Pads each to the
    per-slab capacity C = ceil(capacity_factor * N_total / ndev), places it on this
    process's single local device, and stitches the global (ndev*C, ...) sharded
    jax arrays via make_array_from_single_device_arrays -- so NO full (ndev*C)
    host array is ever built and NO full-array device_put is issued (that pattern
    deadlocks multi-process). Returns (sharded_arrays, active_sharded, C). A
    drift exchange_particles_slab afterwards settles the sub-cell boundary
    crossers into their owning slab.
    """
    from jax.sharding import NamedSharding
    C = int(np.ceil(capacity_factor * N_total / ndev))
    n_local = int(local_arrays[0].shape[0])
    m = min(n_local, C)
    ldev = mesh.local_devices()[0]                        # 1 GPU / process
    out = []
    for a_host, spec in zip(local_arrays, specs):
        buf = np.zeros((C,) + a_host.shape[1:], dtype=a_host.dtype)
        buf[:m] = np.asarray(a_host)[:m]
        shard = jax.device_put(jnp.asarray(buf), ldev)
        gshape = (ndev * C,) + tuple(a_host.shape[1:])
        out.append(jax.make_array_from_single_device_arrays(
            gshape, NamedSharding(mesh, spec), [shard]))
    act = np.zeros((C,), dtype=np.float64)
    act[:m] = 1.0
    act_shard = jax.device_put(jnp.asarray(act), ldev)
    actS = jax.make_array_from_single_device_arrays(
        (ndev * C,), NamedSharding(mesh, P("x")), [act_shard])
    return out, actS, C


def _ring_perms(ndev):
    """ppermute permutations for the 1-D ring along x."""
    to_right = [(i, (i + 1) % ndev) for i in range(ndev)]   # i sends to i+1
    to_left = [(i, (i - 1) % ndev) for i in range(ndev)]    # i sends to i-1
    return to_right, to_left


# ---------------------------------------------------------------------------
# CIC deposit  (particles -> grid density), slab-local with x-ghost exchange
# ---------------------------------------------------------------------------
def cic_deposit_slab(pos, mass, active, Nx, Ny, Nz, dx, mesh):
    """Deposit particle masses to a grid density, sharded as x-slabs.

    `pos`/`mass`/`active` are globally sharded P("x",None)/P("x"). Returns the
    global density (Nx,Ny,Nz) sharded P("x",None,None). Each device deposits its
    own (in-slab) particles into a ghosted (nx_loc+2, Ny, Nz) block, then
    ppermute-adds the two x-ghost planes onto the neighbour slabs' edge cells.
    """
    ndev = int(mesh.shape["x"])
    nx_loc = Nx // ndev
    to_right, to_left = _ring_perms(ndev)
    inv_dV = 1.0 / (dx ** 3)

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P("x", None), P("x"), P("x")),
             out_specs=P("x", None, None))
    def _body(pos_l, mass_l, active_l):
        d = jax.lax.axis_index("x")
        x0 = d * nx_loc                                   # global cell offset of slab
        w = (mass_l * active_l) * inv_dV
        fx = pos_l[:, 0] / dx - 0.5
        fy = pos_l[:, 1] / dx - 0.5
        fz = pos_l[:, 2] / dx - 0.5
        i0 = jnp.floor(fx).astype(jnp.int32)
        j0 = jnp.floor(fy).astype(jnp.int32)
        k0 = jnp.floor(fz).astype(jnp.int32)
        tx = fx - i0
        ty = fy - j0
        tz = fz - k0
        il = i0 - x0 + 1                                  # local x-index into ghosted block
        il = jnp.clip(il, 0, nx_loc)                      # inactive slots -> harmless (w=0)
        rho_g = jnp.zeros((nx_loc + 2, Ny, Nz), dtype=w.dtype)
        for di in (0, 1):
            wx = tx if di else (1.0 - tx)
            ix = il + di
            for dj in (0, 1):
                wy = ty if dj else (1.0 - ty)
                jy = (j0 + dj) % Ny
                for dk in (0, 1):
                    wz = tz if dk else (1.0 - tz)
                    kz = (k0 + dk) % Nz
                    rho_g = rho_g.at[ix, jy, kz].add(wx * wy * wz * w)
        # exchange the two x-ghost planes onto neighbour edge cells.
        # index 0 == cell "-1" == left neighbour's last real cell;
        # index nx_loc+1 == cell "nx_loc" == right neighbour's first real cell.
        rho_loc = rho_g[1:-1]
        to_left_plane = rho_g[0:1]                        # send to d-1
        to_right_plane = rho_g[-1:]                       # send to d+1
        recv_my_last = jax.lax.ppermute(to_left_plane, "x", to_left)   # from d+1: my cell nx_loc-1
        recv_my_first = jax.lax.ppermute(to_right_plane, "x", to_right)  # from d-1: my cell 0
        rho_loc = rho_loc.at[-1:].add(recv_my_last)
        rho_loc = rho_loc.at[0:1].add(recv_my_first)
        return rho_loc

    return _body(pos, mass, active)


# ---------------------------------------------------------------------------
# Force field  -grad(phi) on the grid, slab-local with a 2-cell x-halo
# ---------------------------------------------------------------------------
def grad_phi_slab(phi, dx, mesh, order=4):
    """-grad(phi) on the sharded grid. Returns (gx,gy,gz), each P("x",None,None).

    x: ppermute halo (2 cells for 4th order, 1 for 2nd); y,z: periodic roll.
    Same stencils as ParticleMeshGravity.grad_phi so results match the
    single-device path.
    """
    ndev = int(mesh.shape["x"])
    to_right, to_left = _ring_perms(ndev)
    inv_12dx = 1.0 / (12.0 * dx)
    inv_2dx = 0.5 / dx
    h = 2 if order == 4 else 1

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P("x", None, None),),
             out_specs=(P("x", None, None), P("x", None, None), P("x", None, None)))
    def _body(phi_l):
        # x-halo of width h via ppermute (left ghost = right edge of left nbr).
        left_edge = phi_l[:h]
        right_edge = phi_l[-h:]
        left_ghost = jax.lax.ppermute(right_edge, "x", to_right)   # from d-1
        right_ghost = jax.lax.ppermute(left_edge, "x", to_left)    # from d+1
        pp = jnp.concatenate([left_ghost, phi_l, right_ghost], axis=0)
        if order == 4:
            c = inv_12dx
            gx = (pp[4:] - 8.0 * pp[3:-1] + 8.0 * pp[1:-3] - pp[:-4]) * c
            # y,z periodic via roll on the local full-length axes
            gy = (jnp.roll(phi_l, -2, 1) - 8.0 * jnp.roll(phi_l, -1, 1)
                  + 8.0 * jnp.roll(phi_l, 1, 1) - jnp.roll(phi_l, 2, 1)) * c
            gz = (jnp.roll(phi_l, -2, 2) - 8.0 * jnp.roll(phi_l, -1, 2)
                  + 8.0 * jnp.roll(phi_l, 1, 2) - jnp.roll(phi_l, 2, 2)) * c
            return gx, gy, gz
        gx = -(pp[2:] - pp[:-2]) * inv_2dx
        gy = -(jnp.roll(phi_l, -1, 1) - jnp.roll(phi_l, 1, 1)) * inv_2dx
        gz = -(jnp.roll(phi_l, -1, 2) - jnp.roll(phi_l, 1, 2)) * inv_2dx
        return gx, gy, gz

    return _body(phi)


# ---------------------------------------------------------------------------
# CIC gather  (force grid -> particles), slab-local with a 1-cell x-halo
# ---------------------------------------------------------------------------
def gather_accel_slab(g, pos, Nx, Ny, Nz, dx, mesh):
    """Sample the force field (gx,gy,gz) at the local particles -> accel (M,3).

    `g` is a 3-tuple of grids P("x",None,None); `pos` is P("x",None). Each
    device gathers its own particles from a 1-cell x-haloed copy of its slab.
    """
    ndev = int(mesh.shape["x"])
    nx_loc = Nx // ndev
    to_right, to_left = _ring_perms(ndev)

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P("x", None, None), P("x", None, None), P("x", None, None),
                       P("x", None)),
             out_specs=P("x", None))
    def _body(gx_l, gy_l, gz_l, pos_l):
        d = jax.lax.axis_index("x")
        x0 = d * nx_loc

        def halo1(f):
            left_ghost = jax.lax.ppermute(f[-1:], "x", to_right)   # from d-1
            right_ghost = jax.lax.ppermute(f[:1], "x", to_left)    # from d+1
            return jnp.concatenate([left_ghost, f, right_ghost], axis=0)

        gxh, gyh, gzh = halo1(gx_l), halo1(gy_l), halo1(gz_l)
        fx = pos_l[:, 0] / dx - 0.5
        fy = pos_l[:, 1] / dx - 0.5
        fz = pos_l[:, 2] / dx - 0.5
        i0 = jnp.floor(fx).astype(jnp.int32)
        j0 = jnp.floor(fy).astype(jnp.int32)
        k0 = jnp.floor(fz).astype(jnp.int32)
        tx = fx - i0
        ty = fy - j0
        tz = fz - k0
        il = jnp.clip(i0 - x0 + 1, 0, nx_loc)             # +1: into 1-ghost block
        ax = jnp.zeros_like(fx)
        ay = jnp.zeros_like(fx)
        az = jnp.zeros_like(fx)
        for di in (0, 1):
            wx = tx if di else (1.0 - tx)
            ix = il + di
            for dj in (0, 1):
                wy = ty if dj else (1.0 - ty)
                jy = (j0 + dj) % Ny
                for dk in (0, 1):
                    wz = tz if dk else (1.0 - tz)
                    kz = (k0 + dk) % Nz
                    wgt = wx * wy * wz
                    ax = ax + wgt * gxh[ix, jy, kz]
                    ay = ay + wgt * gyh[ix, jy, kz]
                    az = az + wgt * gzh[ix, jy, kz]
        return jnp.stack([ax, ay, az], axis=1)

    return _body(g[0], g[1], g[2], pos)


# ---------------------------------------------------------------------------
# Particle exchange on drift  (1-D ring redistribution, fixed shapes)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _build_exchange_body(Nx, dx, box, mesh):
    """Build and CACHE the jitted shard_map exchange kernel, keyed on the static
    (Nx, dx, box, mesh). Without the cache a fresh jit closure is created on every
    call, so XLA recompiles the kernel on every step."""
    ndev = int(mesh.shape["x"])
    nx_loc = Nx // ndev
    slab_w = nx_loc * dx                                  # slab width in length units
    to_right, to_left = _ring_perms(ndev)

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P("x", None), P("x", None), P("x"), P("x")),
             out_specs=(P("x", None), P("x", None), P("x"), P("x"),
                        P(), P()))
    def _body(pos_l, p_l, mass_l, active_l):
        C = pos_l.shape[0]
        d = jax.lax.axis_index("x")
        xw = jnp.mod(pos_l[:, 0], box)
        act = active_l > 0
        # target slab and ring distance to this slab (handles periodic wrap):
        # diff 0 = stay, 1 = right neighbour, ndev-1 = left neighbour. Anything
        # else is a (disallowed) multi-hop move -> kept as stay, flagged by n_bad.
        target = jnp.clip((xw / slab_w).astype(jnp.int32), 0, ndev - 1)
        diff = jnp.mod(target - d, ndev)
        go_right = act & (diff == 1)
        go_left = act & (diff == ndev - 1)
        stay = act & (~go_left) & (~go_right)

        # Ship the FULL local buffers (+ their boolean keep-masks) to each ring
        # neighbour and let the RECEIVER mask -- this removes the 3 pre-pack
        # argsorts entirely (the ppermute volume is identical: still C per buffer).
        # A particle that stays is carried in its neighbour's arrival buffer too,
        # but with keep=False there, so it is counted exactly once.
        posRr = jax.lax.ppermute(pos_l, "x", to_right)
        pRr = jax.lax.ppermute(p_l, "x", to_right)
        mRr = jax.lax.ppermute(mass_l, "x", to_right)
        kRr = jax.lax.ppermute(go_right.astype(jnp.float32), "x", to_right) > 0
        posLl = jax.lax.ppermute(pos_l, "x", to_left)
        pLl = jax.lax.ppermute(p_l, "x", to_left)
        mLl = jax.lax.ppermute(mass_l, "x", to_left)
        kLl = jax.lax.ppermute(go_left.astype(jnp.float32), "x", to_left) > 0

        # merge stay + arrivals-from-left(d-1 went right) + arrivals-from-right(d+1 went left)
        pos_all = jnp.concatenate([pos_l, posRr, posLl], axis=0)
        p_all = jnp.concatenate([p_l, pRr, pLl], axis=0)
        mass_all = jnp.concatenate([mass_l, mRr, mLl], axis=0)
        act_all = jnp.concatenate([stay, kRr, kLl], axis=0)
        # O(C) stable compaction (replaces the O(3C log 3C) argsort): the running
        # active-count is each active particle's destination slot; scatter into a
        # zero buffer, dropping inactive (slot via False) and any >C overflow.
        slot = jnp.cumsum(act_all.astype(jnp.int32)) - 1
        dest = jnp.where(act_all, slot, C)                # inactive / overflow(>=C) -> dropped
        def _take(a):
            z = jnp.zeros((C,) + a.shape[1:], a.dtype)
            return z.at[dest].set(a, mode="drop")
        pos_n = _take(pos_all)
        p_n = _take(p_all)
        mass_n = _take(mass_all)
        nact = jnp.maximum(slot[-1] + 1, 0)               # total active across the 3C
        # If a slab's active count exceeds capacity C, the slots with dest>=C were
        # silently dropped by the scatter above. Count that overflow so a capacity
        # violation can NEVER be silent: any nonzero n_bad now means particles were
        # lost/misplaced and the caller must raise dm_capacity_factor (or shorten dt).
        n_overflow = jnp.maximum(nact - C, 0).astype(jnp.int32)
        act_n = (jnp.arange(C) < jnp.minimum(nact, C)).astype(active_l.dtype)  # front min(nact,C)

        # diagnostics: still-out-of-slab (multi-hop) + capacity overflow, and peak occupancy
        xw2 = jnp.mod(pos_n[:, 0], box)
        tgt2 = jnp.clip((xw2 / slab_w).astype(jnp.int32), 0, ndev - 1)
        bad = (act_n > 0) & (tgt2 != d)
        n_bad = jax.lax.psum(jnp.sum(bad.astype(jnp.int32)) + n_overflow, "x")
        occ = jnp.sum(act_n > 0).astype(jnp.int32)
        max_occ = jax.lax.pmax(occ, "x")
        return pos_n, p_n, mass_n, act_n, n_bad, max_occ

    return _body


def exchange_particles_slab(pos, p, mass, active, Nx, dx, box, mesh):
    """Redistribute particles to their owning x-slab after a drift.

    Each device classifies its active particles as stay / go-left / go-right by
    comparing wrapped x against its slab bounds, ships the leavers to the ring
    neighbour via ppermute (fixed-capacity, masked), and compacts the merged set
    (stay + arrivals) back into its capacity-C buffer. Returns the redistributed
    (pos,p,mass,active) plus diagnostics (n_out_of_slab_after, max_occupancy).

    Assumes a particle crosses at most one slab boundary per call (true for a
    CFL/acceleration-limited step with slabs many cells wide). Residual
    out-of-slab particles are reported so the caller can catch a violation. The
    jitted kernel is built once and cached (see _build_exchange_body)."""
    return _build_exchange_body(Nx, dx, box, mesh)(pos, p, mass, active)
