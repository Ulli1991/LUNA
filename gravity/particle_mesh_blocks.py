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

import os
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
    Ng = _ncells(N)
    nloc = [Ng[a] // pdims[a] for a in range(3)]
    bw = [nl * dx for nl in nloc]
    bx = _box3(box)
    bidx = [np.clip((np.mod(pos[:, a], bx[a]) / bw[a]).astype(np.int64),
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
    Ng = _ncells(N)
    nloc = [Ng[a] // pdims[a] for a in range(3)]
    n = pos.shape[0]
    C = int(np.ceil(capacity_factor * n / ndev))
    bx = _box3(box)
    bidx = [np.clip((np.mod(pos[:, a], bx[a]) / (nloc[a] * dx)).astype(np.int64),
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


# Build-once memo for the jitted shard_map programs below: rebuilding on
# every call hands jax.jit a fresh function identity -> retrace per call,
# fatal for per-step eager use in the split KDK. Keyed on every closure
# constant that shapes the traced program.
_PROGRAM_CACHE = {}


def _mesh_key(mesh):
    return (tuple(sorted((k, int(v)) for k, v in mesh.shape.items())),
            tuple(str(d) for d in mesh.devices.flat))


def _ncells(N):
    """Global cell counts per axis. ``N`` may be a scalar (cubic grid) or a
    3-tuple — non-cubic grids are what a fixed per-GPU tile needs when the
    device count is not a cube (e.g. 512x512x1024 on 2 GPUs)."""
    return (int(N),) * 3 if np.isscalar(N) else tuple(int(v) for v in N)


def _box3(box):
    """Physical box extent per axis. A scalar means a cube; a non-cubic grid
    at fixed cell size has a non-cubic box, which is what a fixed per-GPU tile
    needs whenever the device count is not a cube (2, 4, 16, 32 ...)."""
    return ((float(box),) * 3 if np.isscalar(box)
            else tuple(float(v) for v in box))


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


def cic_deposit_blocks(pos, mass, active, N, dx, mesh, dtype=None,
                       fixed=None, fixed_bits=17, mass_unit=None):
    """CIC deposit onto the (px,py,pz)-sharded density grid. Inputs sharded
    P("x") flat-per-block; output P("x","y","z"). ``dtype`` float32 halves
    scatter cost (weights from cell-relative offsets stay exact).

    ``fixed``/``fixed_bits``/``mass_unit`` mirror ``cic_deposit_slab``: `fixed`
    = jnp.int32/int64 accumulates in integer units of 2^fixed_bits per
    mean-density cell, so the deposit is bitwise reproducible and exact in
    mass, and the scatter rides integer atomics (984 -> ~20 ms at 1024^3/8
    GH200 on the slab path). `mass_unit` = global mean density; None computes
    it with a psum. The integer path also routes inactive padding out of
    bounds so ``mode="drop"`` skips those scatters entirely, rather than
    spreading them like the float path below."""
    pdims = [int(mesh.shape[a]) for a in AXES]
    Ng = _ncells(N)
    nloc = [Ng[a] // pdims[a] for a in range(3)]
    inv_dV = 1.0 / dx ** 3

    _pk = ("dep", _mesh_key(mesh), Ng, float(dx), str(dtype),
           str(fixed), int(fixed_bits), mass_unit is not None,
           os.environ.get("LUNA_DEP_SPREAD", "z"))
    _b = _PROGRAM_CACHE.get(_pk)
    if _b is None:
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
            # SPREAD the inactive slots' target cells across the local grid:
            # padding tends to be degenerate (all slots at one point), and
            # 500M zero-weight atomic adds onto 8 addresses SERIALIZE the
            # scatter (measured 2.6 s vs ~0.1 s at 512^3). Zero weight means
            # the field is unchanged; only the traffic pattern matters.
            # Row-major: the LAST axis is contiguous in rho_g, so let it vary
            # fastest with the slot id. An x-fastest spread strides a whole
            # yz-plane per slot and costs 5.7x a real particle (measured
            # 512^3: 0.62 vs 0.11 ns/slot); z-fastest matches the coalesced
            # pattern of lattice-ordered particles.
            sid = jnp.arange(pos_l.shape[0], dtype=jnp.int32)
            if os.environ.get("LUNA_DEP_SPREAD", "z") == "x":   # A/B only
                spread = (sid % nloc[0], (sid // nloc[0]) % nloc[1],
                          (sid // (nloc[0] * nloc[1])) % nloc[2])
            else:
                spread = ((sid // (nloc[2] * nloc[1])) % nloc[0],
                          (sid // nloc[2]) % nloc[1], sid % nloc[2])
            act = active_l > 0
            if fixed is not None:
                # Integer units of S per mean-density cell; the corner weights
                # are split by a rounding tree so the eight of them sum back to
                # W exactly -> mass is conserved to the bit through the scatter
                # AND through _fold_ghosts (integer ppermute + add).
                S = float(1 << fixed_bits)
                ftype = jnp.float32 if fixed == jnp.int32 else jnp.float64
                mu = (jax.lax.psum(jnp.sum(w), AXES)
                      / float(Ng[0] * Ng[1] * Ng[2])
                      if mass_unit is None else mass_unit)
                # padding out of bounds: a w=0 integer add is still an atomic.
                il = [jnp.where(act, il[a], nloc[a] + 2) for a in range(3)]
                tf = [q.astype(ftype) for q in t]
                W = jnp.rint((w / mu).astype(ftype) * ftype(S)).astype(fixed)
                rho_g = jnp.zeros((nloc[0] + 2, nloc[1] + 2, nloc[2] + 2),
                                  dtype=fixed)
                for di in (0, 1):
                    lo = jnp.rint(W.astype(ftype) * (1.0 - tf[0])).astype(fixed)
                    Wx = lo if di == 0 else W - lo
                    for dj in (0, 1):
                        lo = jnp.rint(Wx.astype(ftype)
                                      * (1.0 - tf[1])).astype(fixed)
                        Wxy = lo if dj == 0 else Wx - lo
                        for dk in (0, 1):
                            lo = jnp.rint(Wxy.astype(ftype)
                                          * (1.0 - tf[2])).astype(fixed)
                            upd = lo if dk == 0 else Wxy - lo
                            rho_g = rho_g.at[il[0] + di, il[1] + dj,
                                             il[2] + dk].add(upd, mode="drop")
                rho_g = _fold_ghosts(rho_g, mesh)
                out = (jnp.float32 if dtype is None and fixed == jnp.int32
                       else (dtype or jnp.float64))
                return rho_g.astype(out) * (jnp.asarray(mu, out) / out(S))
            il = [jnp.where(act, il[a], spread[a]) for a in range(3)]
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

        _PROGRAM_CACHE[_pk] = _b = _body
    return _b(pos, mass, active)


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

    _pk = ("grad", _mesh_key(mesh), float(dx), order)
    _b = _PROGRAM_CACHE.get(_pk)
    if _b is None:
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

        _PROGRAM_CACHE[_pk] = _b = _body
    return _b(phi)


def gather_accel_blocks(g, pos, N, dx, mesh):
    """Trilinear force interpolation at particle positions; particles may sit
    up to one ghost cell outside their block (post-drift, pre-exchange)."""
    pdims = [int(mesh.shape[a]) for a in AXES]
    Ng = _ncells(N)
    nloc = [Ng[a] // pdims[a] for a in range(3)]

    _pk = ("gath", _mesh_key(mesh), Ng, float(dx))
    _b = _PROGRAM_CACHE.get(_pk)
    if _b is None:
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

        _PROGRAM_CACHE[_pk] = _b = _body
    return _b(tuple(g), pos)


def resort_blocks(pos, p, mass, active, N, dx, mesh):
    """Cell-order (row-major) resort of every block's local buffer: restores
    the deposit/gather memory locality that clustering decays (measured 4-8x;
    WarpX-style sparse sort cadence — docs/SCALING_DESIGN.md). Stable sort;
    inactive slots sink to the tail, so compaction is preserved."""
    pdims = [int(mesh.shape[a]) for a in AXES]
    Ng = _ncells(N)
    nloc = [Ng[a] // pdims[a] for a in range(3)]

    _pk = ("resort", _mesh_key(mesh), Ng, float(dx))
    _b = _PROGRAM_CACHE.get(_pk)
    if _b is None:
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

        _PROGRAM_CACHE[_pk] = _b = _body
    return _b(pos, p, mass, active)


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
    nloc = _ncells(N)[axis] // npx
    box_a = _box3(box)[axis]
    bw = nloc * dx
    to_right = [(i, (i + 1) % npx) for i in range(npx)]
    to_left = [(i, (i - 1) % npx) for i in range(npx)]

    _pk = ("ex", _mesh_key(mesh), axis, N, float(dx), box_a, Cf)
    _b = _PROGRAM_CACHE.get(_pk)
    if _b is None:
        @partial(jax.jit, static_argnames=())
        @partial(shard_map, mesh=mesh,
                 in_specs=(P(("x", "y", "z"), None), P(("x", "y", "z"), None),
                           P(("x", "y", "z")), P(("x", "y", "z"))),
                 out_specs=(P(("x", "y", "z"), None), P(("x", "y", "z"), None),
                            P(("x", "y", "z")), P(("x", "y", "z")), P()))
        def _body(pos_l, p_l, mass_l, active_l):
            C = pos_l.shape[0]
            d = jax.lax.axis_index(name)
            xw = jnp.mod(pos_l[:, axis], box_a)
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

        _PROGRAM_CACHE[_pk] = _b = _body
    return _b(pos, p, mass, active)


def _face_capacity(N, mesh, C, factor):
    """Slots per face buffer. ``factor`` scales the per-block capacity C; None
    derives it from geometry instead, which is what scale wants: movers are a
    SURFACE quantity while C is a VOLUME one, so any fixed fraction of C is
    oversized by ~nloc and gets worse the bigger the block. The runner
    triggers an exchange once cumulative drift reaches ~1 cell, so crossers
    per face are bounded by one cell-layer of the block; 1.5x that is the
    buffer, and the overflow counter stays the backstop."""
    if factor is not None:
        return max(1024, int(factor * C))
    Ng = _ncells(N)
    pdims = [int(mesh.shape[a]) for a in AXES]
    nloc = [Ng[a] // pdims[a] for a in range(3)]
    layer = max((nloc[(a + 1) % 3] * nloc[(a + 2) % 3]
                 for a in range(3) if pdims[a] > 1), default=0)
    return max(1024, int(1.5 * layer))


def exchange_particles_blocks(pos, p, mass, active, N, dx, box, mesh,
                              face_capacity_factor=None):
    """Redistribute particles to their owning 3D block: ring exchange along
    x, then y, then z (a diagonal mover reaches its block over the cascade).
    Assumes <=1 block hop per axis per call. Movers ride in fixed face buffers
    (sized by ``_face_capacity``; ``face_capacity_factor`` overrides the
    geometric default with a fraction of C). Returns arrays + total overflow
    count (any nonzero = capacity violation — caller must raise)."""
    C = pos.shape[0] // int(np.prod([mesh.shape[a] for a in AXES]))
    Cf = _face_capacity(N, mesh, C, face_capacity_factor)
    n_over_tot = jnp.zeros((), jnp.int32)
    for axis in range(3):
        pos, p, mass, active, n_over = _exchange_axis(
            pos, p, mass, active, axis, N, dx, box, mesh, Cf)
        n_over_tot = n_over_tot + n_over
    return pos, p, mass, active, n_over_tot
