"""Grid-sharded Particle-Mesh primitives for the cosmological DM N-body:
1-D slab decomposition along x on a ``(ndev, 1, 1)`` mesh over ("x","y","z").

INVARIANT: the global particle buffer is ``(ndev * C, ...)`` sharded
``P("x", ...)``, and device d's chunk holds exactly the particles whose
comoving x lies in slab d, padded to capacity C with inactive slots — every
primitive relies on this. y/z indexing is periodic; x periodicity rides the
ppermute ring. dx assumed cubic.
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
    Returns (pos_g, vel_g, mass_g, active_g, C), numpy arrays of shape
    (ndev*C, ...) with chunk d = slab-d particles + inactive padding;
    ``vel=None`` skips the vector."""
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
    """Multi-PROCESS IC assembly (one GPU per process): each process passes
    only its own chunk; the global (ndev*C, ...) sharded arrays are stitched
    via make_array_from_single_device_arrays -- no full host array, no
    full-array device_put (which deadlocks multi-process).
    Returns (sharded_arrays, active_sharded, C)."""
    from jax.sharding import NamedSharding
    C = int(np.ceil(capacity_factor * N_total / ndev))
    n_local = int(local_arrays[0].shape[0])
    m = min(n_local, C)
    ldev = mesh.local_devices[0]                          # 1 GPU / process
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
def cic_deposit_slab(pos, mass, active, Nx, Ny, Nz, dx, mesh,
                     dtype=None, fixed=None, fixed_bits=17, mass_unit=None):
    """CIC-deposit particle masses to a grid density sharded as x-slabs:
    inputs sharded P("x", ...); returns (Nx, Ny, Nz) sharded P("x", None,
    None), x-ghost planes ppermute-added onto neighbour slabs.

    `dtype`/`fixed`/`fixed_bits`/`mass_unit` mirror
    ParticleMeshGravity.cic_deposit: `fixed` = jnp.int32/int64 switches to
    fixed-point integer accumulation (bitwise reproducible, exact in mass —
    see _cic_deposit_fixed; integer atomics are also far faster than the
    fp64 float scatter this kernel used before: 984 -> ~20 ms at 1024^3/8
    GH200). `mass_unit` = global mean density (sum m / box^3); if None it is
    computed with a psum (mass is conserved, so pass it in for zero cost)."""
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
        il = jnp.clip(il, 0, nx_loc)
        # Inactive padding slots all sit at pos=0 -> the SAME grid cell; their
        # w=0 atomic adds still serialize on that one address (~400 ms of the
        # 421 ms deposit at 201M slots / 67M padding, 1024^3/8 GPU). Route them
        # out of bounds so mode="drop" skips the scatter entirely.
        il = jnp.where(active_l > 0, il, nx_loc + 2)
        if fixed is not None:
            # fixed-point: integer units of S per mean-density cell, corner
            # weights split by a rounding tree so they sum exactly to W.
            S = float(1 << fixed_bits)
            ftype = jnp.float32 if fixed == jnp.int32 else jnp.float64
            mu = (jax.lax.psum(jnp.sum(w), "x") / float(Nx * Ny * Nz)
                  if mass_unit is None else mass_unit)
            txf, tyf, tzf = (q.astype(ftype) for q in (tx, ty, tz))
            W = jnp.rint((w / mu).astype(ftype) * ftype(S)).astype(fixed)
            rho_g = jnp.zeros((nx_loc + 2, Ny, Nz), dtype=fixed)
            for di in (0, 1):
                lo = jnp.rint(W.astype(ftype) * (1.0 - txf)).astype(fixed)
                Wx = lo if di == 0 else W - lo
                ix = il + di
                for dj in (0, 1):
                    lo = jnp.rint(Wx.astype(ftype) * (1.0 - tyf)).astype(fixed)
                    Wxy = lo if dj == 0 else Wx - lo
                    jy = (j0 + dj) % Ny
                    for dk in (0, 1):
                        lo = jnp.rint(Wxy.astype(ftype) * (1.0 - tzf)).astype(fixed)
                        upd = lo if dk == 0 else Wxy - lo
                        kz = (k0 + dk) % Nz
                        rho_g = rho_g.at[ix, jy, kz].add(upd, mode="drop")
        else:
            if dtype is not None:
                w_, tx, ty, tz = (q.astype(dtype) for q in (w, tx, ty, tz))
            else:
                w_ = w
            rho_g = jnp.zeros((nx_loc + 2, Ny, Nz), dtype=w_.dtype)
            for di in (0, 1):
                wx = tx if di else (1.0 - tx)
                ix = il + di
                for dj in (0, 1):
                    wy = ty if dj else (1.0 - ty)
                    jy = (j0 + dj) % Ny
                    for dk in (0, 1):
                        wz = tz if dk else (1.0 - tz)
                        kz = (k0 + dk) % Nz
                        rho_g = rho_g.at[ix, jy, kz].add(wx * wy * wz * w_,
                                                         mode="drop")
        # exchange the two x-ghost planes onto neighbour edge cells (integer
        # planes in the fixed path — exact through the ppermute).
        # index 0 == cell "-1" == left neighbour's last real cell;
        # index nx_loc+1 == cell "nx_loc" == right neighbour's first real cell.
        rho_loc = rho_g[1:-1]
        to_left_plane = rho_g[0:1]                        # send to d-1
        to_right_plane = rho_g[-1:]                       # send to d+1
        recv_my_last = jax.lax.ppermute(to_left_plane, "x", to_left)   # from d+1: my cell nx_loc-1
        recv_my_first = jax.lax.ppermute(to_right_plane, "x", to_right)  # from d-1: my cell 0
        rho_loc = rho_loc.at[-1:].add(recv_my_last)
        rho_loc = rho_loc.at[0:1].add(recv_my_first)
        if fixed is not None:
            out = jnp.float32 if dtype is None and fixed == jnp.int32 else \
                (dtype or jnp.float64)
            mu_out = jnp.asarray(mu, dtype=out)
            return rho_loc.astype(out) * (mu_out / out(S))
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
def _build_exchange_body(Nx, dx, box, mesh, C_send):
    """Build and CACHE the jitted shard_map exchange kernel, keyed on the static
    (Nx, dx, box, mesh, C_send). Without the cache a fresh jit closure is created
    on every call, so XLA recompiles the kernel on every step."""
    ndev = int(mesh.shape["x"])
    nx_loc = Nx // ndev
    slab_w = nx_loc * dx                                  # slab width in length units
    to_right, to_left = _ring_perms(ndev)

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P("x", None), P("x", None), P("x"), P("x")),
             out_specs=(P("x", None), P("x", None), P("x"), P("x"),
                        P(), P(), P()))
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

        # Pack the movers into fixed C_send-slot send buffers and ppermute ONLY
        # those. Movers are a thin skin of the slab (one drift << one cell), so
        # shipping the full C-slot buffers — the previous scheme — moved ~1000x
        # more bytes than needed and saturated the inter-node links (1245 ms of
        # a 2286 ms step at 1024^3/8 GPU). Packing uses the same O(C) cumsum
        # compaction as the merge below. A mover beyond C_send is dropped by
        # the scatter and counted in n_bad — never silent (the caller's
        # conservation watchdog also trips): raise dm_exchange_send_div's
        # divisor... i.e. LOWER it, or raise capacity, if that ever fires.
        def _pack(sel):
            slot = jnp.cumsum(sel.astype(jnp.int32)) - 1
            dest = jnp.where(sel, slot, C_send)           # inactive/overflow -> dropped
            n = jnp.sum(sel.astype(jnp.int32))
            def take(a):
                z = jnp.zeros((C_send,) + a.shape[1:], a.dtype)
                return z.at[dest].set(a, mode="drop")
            return take, jnp.minimum(n, C_send), jnp.maximum(n - C_send, 0)

        if ndev == 2:
            # Degenerate ring: the left and right neighbour are the SAME device
            # (to_left == to_right), and every mover has diff == 1 == ndev-1.
            # One packed buffer holds all movers; sent once, inserted once.
            takeR, nR, ovR = _pack(act & (diff == 1))
            nL = jnp.zeros((), jnp.int32)
            ovL = jnp.zeros((), jnp.int32)
        else:
            takeR, nR, ovR = _pack(go_right)
            takeL, nL, ovL = _pack(go_left)
        posRr = jax.lax.ppermute(takeR(pos_l), "x", to_right)
        pRr = jax.lax.ppermute(takeR(p_l), "x", to_right)
        mRr = jax.lax.ppermute(takeR(mass_l), "x", to_right)
        nRr = jax.lax.ppermute(nR.reshape(1), "x", to_right)
        kRr = jnp.arange(C_send) < nRr[0]                 # arrivals sit in [0:nRr)

        if ndev == 2:
            pos_all = jnp.concatenate([pos_l, posRr], axis=0)
            p_all = jnp.concatenate([p_l, pRr], axis=0)
            mass_all = jnp.concatenate([mass_l, mRr], axis=0)
            act_all = jnp.concatenate([stay, kRr], axis=0)
        else:
            posLl = jax.lax.ppermute(takeL(pos_l), "x", to_left)
            pLl = jax.lax.ppermute(takeL(p_l), "x", to_left)
            mLl = jax.lax.ppermute(takeL(mass_l), "x", to_left)
            nLl = jax.lax.ppermute(nL.reshape(1), "x", to_left)
            kLl = jnp.arange(C_send) < nLl[0]

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
        n_bad = jax.lax.psum(jnp.sum(bad.astype(jnp.int32)) + n_overflow
                             + ovR + ovL, "x")
        occ = jnp.sum(act_n > 0).astype(jnp.int32)
        max_occ = jax.lax.pmax(occ, "x")
        # gross movers this call (both directions, incl. any send overflow) —
        # the data that sizes the send buffer, instead of guessing send_div
        n_move = jax.lax.pmax(nR + ovR + nL + ovL, "x")
        return pos_n, p_n, mass_n, act_n, n_bad, max_occ, n_move

    return _body


def exchange_particles_slab(pos, p, mass, active, Nx, dx, box, mesh,
                            send_div=64):
    """Redistribute particles to their owning x-slab after a drift (packed
    movers ppermuted to ring neighbours, compacted into the capacity-C
    buffer). ASSUMES a particle crosses at most one slab boundary per call.
    `send_div`: send-buffer slots per neighbour = C // send_div — at 64 that
    is several full x-cell-layers of particles, >>any single drift; a send
    overflow is counted in n_bad, never silent. Returns the redistributed
    (pos, p, mass, active) plus diagnostics (n_out_of_slab_after,
    max_occupancy, max_gross_movers_per_device)."""
    ndev = int(mesh.shape["x"])
    C = pos.shape[0] // ndev
    C_send = max(C // int(send_div), 1)
    return _build_exchange_body(Nx, dx, box, mesh, C_send)(pos, p, mass, active)
