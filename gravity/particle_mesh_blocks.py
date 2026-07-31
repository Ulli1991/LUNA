"""3D-block particle-mesh machinery: DM particles owned by the hydro's
(px,py,pz) meshblock decomposition — one mesh for gas, density, potential
and forces (docs/SCALING_DESIGN.md; HACC-style ownership).

Generalizes the 1-D slab path (particle_mesh_shardmap): deposit into a
ghosted local block folded onto neighbours per axis, gradient/gather via
the shared scalar halo, and a per-axis ring-exchange cascade for particle
migration (a diagonal move resolves over successive axis passes).
Conservation is the caller's watchdog: every exchange returns overflow
counts so a capacity violation can never be silent.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from utils.halo_exchange_shardmap import _pad_axis

AXES = ("x", "y", "z")


def make_block_mesh(devices, pdims):
    """3D device mesh (px,py,pz) matching the hydro's meshblock layout."""
    devs = np.asarray(devices).reshape(*pdims)
    return Mesh(devs, AXES)


def decompose_to_blocks(pos, vel, mass, N, pdims, dx, box,
                        capacity_factor=1.5):
    """Host-side: bucket particles into per-block fixed-capacity buffers in
    device (row-major block) order. Returns (pos_g, vel_g, mass_g, active_g,
    C) shaped (ndev*C, ...)."""
    n = pos.shape[0]
    ndev = int(np.prod(pdims))
    nloc = [N // p for p in pdims]
    bw = [nl * dx for nl in nloc]
    bidx = [np.clip((np.mod(pos[:, a], box) / bw[a]).astype(np.int64),
                    0, pdims[a] - 1) for a in range(3)]
    owner = (bidx[0] * pdims[1] + bidx[1]) * pdims[2] + bidx[2]
    counts = np.bincount(owner, minlength=ndev)
    C = int(np.ceil(capacity_factor * n / ndev))
    C = max(C, int(counts.max()))
    pos_g = np.zeros((ndev * C, 3), dtype=pos.dtype)
    vel_g = None if vel is None else np.zeros((ndev * C, vel.shape[1]),
                                              dtype=vel.dtype)
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


def assemble_blocks_sharded(pos, vel, mass, N, pdims, dx, box, mesh,
                            capacity_factor=1.5):
    """Build the sharded per-block particle buffers WITHOUT any global
    (ndev*C) host array. Single-process: per-device chunks from a local
    bucketing. Multi-process (1 GPU/process): each process keeps only its
    own block's rows — no host OOM, no cross-host device_put.
    Returns (posS, pS, massS, actS, C)."""
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    ndev = int(np.prod(pdims))
    nloc = [N // p for p in pdims]
    n = pos.shape[0]
    C = int(np.ceil(capacity_factor * n / ndev))
    bidx = [np.clip((np.mod(pos[:, a], box) / (nloc[a] * dx)).astype(np.int64),
                    0, pdims[a] - 1) for a in range(3)]
    owner = (bidx[0] * pdims[1] + bidx[1]) * pdims[2] + bidx[2]
    C = max(C, int(np.bincount(owner, minlength=ndev).max()))
    devs = list(mesh.devices.ravel())
    multi = jax.process_count() > 1

    def _chunk(d):
        sel = np.nonzero(owner == d)[0]
        m = sel.size
        cp = np.zeros((C, 3), pos.dtype)
        cv = np.zeros((C, vel.shape[1]), vel.dtype)
        cm = np.zeros((C,), mass.dtype)
        ca = np.zeros((C,), np.float64)
        cp[:m] = pos[sel]
        cv[:m] = vel[sel]
        cm[:m] = mass[sel]
        ca[:m] = 1.0
        return cp, cv, cm, ca

    bufs = [[], [], [], []]
    for i, d in enumerate(devs):
        if multi and d.process_index != jax.process_index():
            continue
        for j, c in enumerate(_chunk(i)):
            bufs[j].append(jax.device_put(c, d))
    flat = NamedSharding(mesh, P(("x", "y", "z")))
    flat2 = NamedSharding(mesh, P(("x", "y", "z"), None))
    mk = jax.make_array_from_single_device_arrays
    posS = mk((ndev * C, 3), flat2, bufs[0])
    pS = mk((ndev * C, vel.shape[1]), flat2, bufs[1])
    massS = mk((ndev * C,), flat, bufs[2])
    actS = mk((ndev * C,), flat, bufs[3])
    return posS, pS, massS, actS, C


def _block_origin(pdims, nloc):
    """Per-axis global cell offset of this device's block (traced)."""
    return [jax.lax.axis_index(AXES[a]) * nloc[a] for a in range(3)]


def _fold_ghosts(rho_g, mesh):
    """Fold the 1-cell ghost shell of a ghosted local block onto the owning
    neighbours, axis by axis (later axes carry the corners). Inverse of a
    halo pad; ppermute along sharded axes, periodic self-wrap otherwise."""
    for a, name in enumerate(AXES):
        npx = int(mesh.shape[name])
        to_right = [(i, (i + 1) % npx) for i in range(npx)]
        to_left = [(i, (i - 1) % npx) for i in range(npx)]
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        core = [slice(None)] * 3
        lo[a] = slice(0, 1)
        hi[a] = slice(-1, None)
        core[a] = slice(1, -1)
        left_plane = rho_g[tuple(lo)]
        right_plane = rho_g[tuple(hi)]
        if npx > 1:
            recv_last = jax.lax.ppermute(left_plane, name, to_left)
            recv_first = jax.lax.ppermute(right_plane, name, to_right)
        else:                                   # periodic wrap onto self
            recv_last, recv_first = left_plane, right_plane
        rho_g = rho_g[tuple(core)]
        end = [slice(None)] * 3
        start = [slice(None)] * 3
        end[a] = slice(-1, None)
        start[a] = slice(0, 1)
        rho_g = rho_g.at[tuple(end)].add(recv_last)
        rho_g = rho_g.at[tuple(start)].add(recv_first)
    return rho_g


def cic_deposit_blocks(pos, mass, active, N, dx, mesh, dtype=None):
    """CIC deposit onto the (px,py,pz)-sharded density grid. Inputs sharded
    P("x") flat-per-block; output P("x","y","z"). ``dtype`` float32 halves
    scatter cost (weights from cell-relative offsets stay exact)."""
    pdims = [int(mesh.shape[a]) for a in AXES]
    nloc = [N // p for p in pdims]
    inv_dV = 1.0 / dx ** 3

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P(("x", "y", "z"), None), P(("x", "y", "z")),
                       P(("x", "y", "z"))),
             out_specs=P("x", "y", "z"))
    def _body(pos_l, mass_l, active_l):
        org = _block_origin(pdims, nloc)
        w = (mass_l * active_l) * inv_dV
        f = [pos_l[:, a] / dx - 0.5 for a in range(3)]
        i0 = [jnp.floor(f[a]).astype(jnp.int32) for a in range(3)]
        t = [f[a] - i0[a] for a in range(3)]
        # local ghosted index; clip keeps inactive slots harmless (w=0)
        il = [jnp.clip(i0[a] - org[a] + 1, 0, nloc[a]) for a in range(3)]
        if dtype is not None:
            w = w.astype(dtype)
            t = [q.astype(dtype) for q in t]
        rho_g = jnp.zeros((nloc[0] + 2, nloc[1] + 2, nloc[2] + 2),
                          dtype=w.dtype)
        for di in (0, 1):
            wx = t[0] if di else (1.0 - t[0])
            for dj in (0, 1):
                wy = t[1] if dj else (1.0 - t[1])
                for dk in (0, 1):
                    wz = t[2] if dk else (1.0 - t[2])
                    rho_g = rho_g.at[il[0] + di, il[1] + dj,
                                     il[2] + dk].add(wx * wy * wz * w)
        return _fold_ghosts(rho_g, mesh)

    return _body(pos, mass, active)


def _halo(phi_l, mesh, width):
    """Width-w halo on all three axes inside a shard_map body (periodic)."""
    for a, name in enumerate(AXES):
        phi_l = _pad_axis(phi_l, axis=a, axis_name=name,
                          n_proc=int(mesh.shape[name]), periodic=True,
                          width=width)
    return phi_l


def grad_phi_blocks(phi, dx, mesh, order=4):
    """-grad(phi) on the block-sharded grid; (gx,gy,gz) each P("x","y","z")."""
    inv_12dx = 1.0 / (12.0 * dx)
    inv_2dx = 0.5 / dx
    h = 2 if order == 4 else 1

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P("x", "y", "z"),),
             out_specs=(P("x", "y", "z"),) * 3)
    def _body(phi_l):
        pp = _halo(phi_l, mesh, h)
        c = [slice(h, -h)] * 3

        def sl(axis, off):
            s = list(c)
            s[axis] = slice(h + off, -h + off if -h + off else None)
            return pp[tuple(s)]

        out = []
        for a in range(3):
            if order == 4:
                g = (sl(a, 2) - 8.0 * sl(a, 1) + 8.0 * sl(a, -1)
                     - sl(a, -2)) * inv_12dx
            else:
                g = -(sl(a, 1) - sl(a, -1)) * inv_2dx
            out.append(g)
        return tuple(out)

    return _body(phi)


def gather_accel_blocks(g, pos, N, dx, mesh):
    """Trilinear force interpolation at particle positions; particles may sit
    up to one ghost cell outside their block (post-drift, pre-exchange)."""
    pdims = [int(mesh.shape[a]) for a in AXES]
    nloc = [N // p for p in pdims]

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=((P("x", "y", "z"),) * 3, P(("x", "y", "z"), None)),
             out_specs=P(("x", "y", "z"), None))
    def _body(g_l, pos_l):
        gp = [_halo(q, mesh, 1) for q in g_l]
        org = _block_origin(pdims, nloc)
        f = [pos_l[:, a] / dx - 0.5 for a in range(3)]
        i0 = [jnp.floor(f[a]).astype(jnp.int32) for a in range(3)]
        t = [jnp.clip(f[a] - i0[a], 0.0, 1.0) for a in range(3)]
        il = [jnp.clip(i0[a] - org[a] + 1, 0, nloc[a]) for a in range(3)]
        acc = [0.0, 0.0, 0.0]
        for di in (0, 1):
            wx = t[0] if di else (1.0 - t[0])
            for dj in (0, 1):
                wy = t[1] if dj else (1.0 - t[1])
                for dk in (0, 1):
                    wz = t[2] if dk else (1.0 - t[2])
                    wgt = wx * wy * wz
                    for a in range(3):
                        acc[a] = acc[a] + wgt * gp[a][il[0] + di, il[1] + dj,
                                                      il[2] + dk]
        return jnp.stack(acc, axis=1)

    return _body(tuple(g), pos)


def resort_blocks(pos, p, mass, active, N, dx, mesh):
    """Cell-order (row-major) resort of every block's local buffer: restores
    the deposit/gather memory locality that clustering decays (measured 4-8x;
    WarpX-style sparse sort cadence — docs/SCALING_DESIGN.md). Stable sort;
    inactive slots sink to the tail, so compaction is preserved."""
    pdims = [int(mesh.shape[a]) for a in AXES]
    nloc = [N // p_ for p_ in pdims]

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P(("x", "y", "z"), None), P(("x", "y", "z"), None),
                       P(("x", "y", "z")), P(("x", "y", "z"))),
             out_specs=(P(("x", "y", "z"), None), P(("x", "y", "z"), None),
                        P(("x", "y", "z")), P(("x", "y", "z"))))
    def _body(pos_l, p_l, mass_l, act_l):
        org = _block_origin(pdims, nloc)
        idx = [jnp.clip(jnp.floor(pos_l[:, a] / dx).astype(jnp.int32)
                        - org[a] + 1, 0, nloc[a] + 1) for a in range(3)]
        key = ((idx[0] * (nloc[1] + 2) + idx[1]) * (nloc[2] + 2) + idx[2])
        key = jnp.where(act_l > 0, key, jnp.iinfo(jnp.int32).max)
        order = jnp.argsort(key, stable=True)
        return (pos_l[order], p_l[order], mass_l[order], act_l[order])

    return _body(pos, p, mass, active)


def _exchange_axis(pos, p, mass, active, axis, N, dx, box, mesh, Cf):
    """One ring-exchange pass along ``axis``: movers are packed into fixed
    face buffers of ``Cf`` slots per direction (movers are a few percent of
    the capacity C, so this ships ~C/Cf-fold less than full-buffer shipping;
    send overflow is counted, never silent)."""
    name = AXES[axis]
    npx = int(mesh.shape[name])
    if npx == 1:
        zero = jnp.zeros((), jnp.int32)
        return pos, p, mass, active, zero
    nloc = N // npx
    bw = nloc * dx
    to_right = [(i, (i + 1) % npx) for i in range(npx)]
    to_left = [(i, (i - 1) % npx) for i in range(npx)]

    @partial(jax.jit, static_argnames=())
    @partial(shard_map, mesh=mesh,
             in_specs=(P(("x", "y", "z"), None), P(("x", "y", "z"), None),
                       P(("x", "y", "z")), P(("x", "y", "z"))),
             out_specs=(P(("x", "y", "z"), None), P(("x", "y", "z"), None),
                        P(("x", "y", "z")), P(("x", "y", "z")), P()))
    def _body(pos_l, p_l, mass_l, active_l):
        C = pos_l.shape[0]
        d = jax.lax.axis_index(name)
        xw = jnp.mod(pos_l[:, axis], box)
        act = active_l > 0
        target = jnp.clip((xw / bw).astype(jnp.int32), 0, npx - 1)
        diff = jnp.mod(target - d, npx)
        go_right = act & (diff == 1)
        go_left = act & (diff == npx - 1) & (npx > 2)
        stay = act & (~go_left) & (~go_right)

        def _pack(sel):
            """Movers -> fixed (Cf, ...) face buffers + count (+overflow)."""
            slot = jnp.cumsum(sel.astype(jnp.int32)) - 1
            dest = jnp.where(sel, slot, Cf)

            def tk(a):
                z = jnp.zeros((Cf,) + a.shape[1:], a.dtype)
                return z.at[dest].set(a, mode="drop")

            cnt = jnp.sum(sel.astype(jnp.int32))
            return (tk(pos_l), tk(p_l), tk(mass_l), jnp.minimum(cnt, Cf),
                    jnp.maximum(cnt - Cf, 0))

        bufR = _pack(go_right)
        sendover = bufR[4]
        arrs = [jax.lax.ppermute(q, name, to_right) for q in bufR[:4]]
        if npx > 2:
            bufL = _pack(go_left)
            sendover = sendover + bufL[4]
            arrs_l = [jax.lax.ppermute(q, name, to_left) for q in bufL[:4]]
        kR = jnp.arange(Cf) < arrs[3]
        pos_all = [pos_l, arrs[0]]
        p_all = [p_l, arrs[1]]
        mass_all = [mass_l, arrs[2]]
        act_all = [stay, kR]
        if npx > 2:
            kL = jnp.arange(Cf) < arrs_l[3]
            pos_all.append(arrs_l[0])
            p_all.append(arrs_l[1])
            mass_all.append(arrs_l[2])
            act_all.append(kL)
        pos_a = jnp.concatenate(pos_all, axis=0)
        p_a = jnp.concatenate(p_all, axis=0)
        mass_a = jnp.concatenate(mass_all, axis=0)
        act_a = jnp.concatenate(act_all, axis=0)

        slot = jnp.cumsum(act_a.astype(jnp.int32)) - 1
        dest = jnp.where(act_a, slot, C)

        def _take(a):
            z = jnp.zeros((C,) + a.shape[1:], a.dtype)
            return z.at[dest].set(a, mode="drop")

        pos_n = _take(pos_a)
        p_n = _take(p_a)
        mass_n = _take(mass_a)
        nact = jnp.maximum(slot[-1] + 1, 0)
        n_overflow = jnp.maximum(nact - C, 0).astype(jnp.int32) + sendover
        act_n = (jnp.arange(C) < jnp.minimum(nact, C)).astype(active_l.dtype)
        n_over = jax.lax.psum(n_overflow, AXES)
        return pos_n, p_n, mass_n, act_n, n_over

    return _body(pos, p, mass, active)


def exchange_particles_blocks(pos, p, mass, active, N, dx, box, mesh,
                              face_capacity_factor=0.125):
    """Redistribute particles to their owning 3D block: ring exchange along
    x, then y, then z (a diagonal mover reaches its block over the cascade).
    Assumes <=1 block hop per axis per call. Movers ride in face buffers of
    ``face_capacity_factor * C`` slots per direction. Returns arrays + total
    overflow count (any nonzero = capacity violation — caller must raise)."""
    C = pos.shape[0] // int(np.prod([mesh.shape[a] for a in AXES]))
    Cf = max(1024, int(face_capacity_factor * C))
    n_over_tot = jnp.zeros((), jnp.int32)
    for axis in range(3):
        pos, p, mass, active, n_over = _exchange_axis(
            pos, p, mass, active, axis, N, dx, box, mesh, Cf)
        n_over_tot = n_over_tot + n_over
    return pos, p, mass, active, n_over_tot
