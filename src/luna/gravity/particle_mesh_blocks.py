"""Particle mesh on the hydro's (px, py, pz) block mesh: deposit, gradient, gather, resort and particle exchange.

Every exchange returns an overflow count; the caller must raise on nonzero. See docs/history/SCALING_DESIGN.md.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from luna.utils.shmap import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from luna.gravity.poisson_bc import pad_inplace
from luna.gravity.poisson_bc_shardmap import sharded_ghosts

AXES = ("x", "y", "z")


def make_block_mesh(devices, pdims):
    """3D device mesh (px,py,pz) matching the hydro's meshblock layout."""
    devs = np.asarray(devices).reshape(*pdims)
    return Mesh(devs, AXES)


def decompose_to_blocks(pos, vel, mass, N, pdims, dx, box,
                        capacity_factor=1.5):
    """Host-side per-block fixed-capacity buffers in row-major block order: (pos, vel, mass, active, C), (ndev*C, ...)."""
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
                            capacity_factor=1.5, ids=None):
    """Sharded per-block particle buffers without a global host array: (posS, pS, massS, actS, idsS, C)."""
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
        if ids is None:
            return cp, cv, cm, ca
        ci = np.full((C,), -1, ids.dtype)          # -1: an empty slot
        ci[:m] = ids[sel]
        return cp, cv, cm, ca, ci

    bufs = [[], [], [], [], []]
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
    idsS = None if ids is None else mk((ndev * C,), flat, bufs[4])
    return posS, pS, massS, actS, idsS, C


def block_of_process(mesh):
    """(ix, iy, iz) of this process's block, from its device's place in the mesh (not process_index)."""
    ldev = mesh.local_devices[0]
    where = np.argwhere(mesh.devices == ldev)
    return tuple(int(v) for v in where[0])


def block_rows(pos, N, pdims, dx, box, block, axes=(1, 2)):
    """Host-side mask of the rows of ``pos`` whose cell lies in ``block`` along ``axes``.

    Filter on (y, z) only: rows that crossed an x-slab face are settled by one exchange; filtering x drops them.
    """
    Ng = _ncells(N)
    bx = _box3(box)
    nloc = [Ng[a] // pdims[a] for a in range(3)]
    keep = np.ones(pos.shape[0], dtype=bool)
    for a in axes:
        idx = np.clip((np.mod(pos[:, a], bx[a]) / (nloc[a] * dx)).astype(np.int64),
                      0, pdims[a] - 1)
        keep &= idx == block[a]
    return keep


def exact_block_capacity(pos_local, N, pdims, dx, box):
    """Exact per-block buffer size for a one-time assembly, covering pre-exchange occupancy too."""
    from jax.experimental import multihost_utils
    ndev = int(np.prod(pdims))
    Ng, bx = _ncells(N), _box3(box)
    bw = [(Ng[a] // pdims[a]) * dx for a in range(3)]
    bidx = [np.clip((np.mod(pos_local[:, a], bx[a]) / bw[a]).astype(np.int64),
                    0, pdims[a] - 1) for a in range(3)]
    owner = (bidx[0] * pdims[1] + bidx[1]) * pdims[2] + bidx[2]
    local = np.zeros(ndev + 1, dtype=np.int64)
    local[:ndev] = np.bincount(owner, minlength=ndev)   # final owner of my rows
    local[ndev] = int(pos_local.shape[0])               # my pre-exchange rows
    if jax.process_count() == 1:
        return int(max(local[:ndev].max(), local[ndev]))
    allc = np.asarray(multihost_utils.process_allgather(local))   # (nproc, ndev+1)
    return int(max(allc[:, :ndev].sum(axis=0).max(), allc[:, ndev].max()))


def exact_settle_face_capacity(pos_local, N, pdims, dx, box, block):
    """Face-buffer slots for the one-time IC settle: the global max of movers per axis."""
    from jax.experimental import multihost_utils
    Ng, bx = _ncells(N), _box3(box)
    bw = [(Ng[a] // pdims[a]) * dx for a in range(3)]
    need = np.zeros(3, dtype=np.int64)
    for a in range(3):
        idx = np.clip((np.mod(pos_local[:, a], bx[a]) / bw[a]).astype(np.int64),
                      0, pdims[a] - 1)
        need[a] = int(np.count_nonzero(idx != block[a]))
    if jax.process_count() > 1:
        need = np.asarray(multihost_utils.process_allgather(need)).max(axis=0)
    return max(1024, int(need.max()))


def assemble_blocks_local(local_arrays, specs, n_total, mesh, capacity_factor=1.5,
                          capacity=None):
    """Multi-process per-block buffers from the rows this process owns: (sharded_arrays, active, C).

    Every rank must pass the same ``n_total``; ``capacity`` (exact_block_capacity) overrides the factor.
    """
    from jax.sharding import NamedSharding
    ndev = int(np.prod([mesh.shape[a] for a in AXES]))
    C = int(capacity) if capacity is not None else int(
        np.ceil(capacity_factor * n_total / ndev))
    n_local = int(local_arrays[0].shape[0])
    if n_local > C:
        raise ValueError(f"this block holds {n_local} particles > capacity {C}: "
                         "raise dm_capacity_factor")
    ldev = mesh.local_devices[0]
    out = []
    for a_host, spec in zip(local_arrays, specs):
        buf = np.zeros((C,) + tuple(a_host.shape[1:]), dtype=a_host.dtype)
        buf[:n_local] = np.asarray(a_host)
        shard = jax.device_put(jnp.asarray(buf), ldev)
        out.append(jax.make_array_from_single_device_arrays(
            (ndev * C,) + tuple(a_host.shape[1:]), NamedSharding(mesh, spec), [shard]))
    act = np.zeros((C,), dtype=np.float64)
    act[:n_local] = 1.0
    actS = jax.make_array_from_single_device_arrays(
        (ndev * C,), NamedSharding(mesh, P(AXES)),
        [jax.device_put(jnp.asarray(act), ldev)])
    return out, actS, C


# Build-once memo of the jitted programs: a rebuild per call would retrace every step.
_PROGRAM_CACHE = {}


def _mesh_key(mesh):
    return (tuple(sorted((k, int(v)) for k, v in mesh.shape.items())),
            tuple(str(d) for d in mesh.devices.flat))


def _ncells(N):
    """Global cell counts per axis from a scalar (cube) or a 3-tuple."""
    return (int(N),) * 3 if np.isscalar(N) else tuple(int(v) for v in N)


def _box3(box):
    """Physical box extent per axis from a scalar (cube) or a 3-tuple."""
    return ((float(box),) * 3 if np.isscalar(box)
            else tuple(float(v) for v in box))


def _block_origin(pdims, nloc):
    """Per-axis global cell offset of this device's block (traced)."""
    return [jax.lax.axis_index(AXES[a]) * nloc[a] for a in range(3)]


def _fold_ghosts(rho_g, mesh):
    """Fold the 1-cell ghost shell onto the owning neighbours, axis by axis (the inverse of a halo pad)."""
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
        else:
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
    """CIC density on the block-sharded grid; particles P(('x','y','z')) flat per block, output P('x','y','z').

    fixed: jnp.int32/int64 for fixed-point accumulation, 2^fixed_bits units per mean-density cell.
    mass_unit: the global mean density; None computes it with a psum.
    """
    pdims = [int(mesh.shape[a]) for a in AXES]
    Ng = _ncells(N)
    nloc = [Ng[a] // pdims[a] for a in range(3)]
    inv_dV = 1.0 / dx ** 3

    _pk = ("dep", _mesh_key(mesh), Ng, float(dx), str(dtype),
           str(fixed), int(fixed_bits),
           None if mass_unit is None else float(mass_unit))
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
            il = [_local_cell(i0[a], org[a], Ng[a], nloc[a]) for a in range(3)]
            # Spread the zero-weight padding slots over the grid, z fastest: atomics on one cell serialize.
            sid = jnp.arange(pos_l.shape[0], dtype=jnp.int32)
            spread = ((sid // (nloc[2] * nloc[1])) % nloc[0],
                      (sid // nloc[2]) % nloc[1], sid % nloc[2])
            act = active_l > 0
            if fixed is not None:
                # The rounding tree makes the eight corners sum to W exactly: mass conserved to the bit.
                S = float(1 << fixed_bits)
                ftype = jnp.float32 if fixed == jnp.int32 else jnp.float64
                mu = (jax.lax.psum(jnp.sum(w), AXES)
                      / float(Ng[0] * Ng[1] * Ng[2])
                      if mass_unit is None else mass_unit)
                # padding out of bounds, dropped: a w=0 integer add is still an atomic
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
    return pad_inplace(phi_l, width, sharded_ghosts(mesh, True, width))


def grad_phi_blocks(phi, dx, mesh, order=4, boundary=0):
    """-grad(phi) on the block-sharded grid: (gx, gy, gz), each P('x','y','z')."""
    inv_12dx = 1.0 / (12.0 * dx)
    inv_2dx = 0.5 / dx
    h = 2 if order == 4 else 1
    boundary = int(boundary)

    _pk = ("grad", _mesh_key(mesh), float(dx), order, boundary)
    _b = _PROGRAM_CACHE.get(_pk)
    if _b is None:
        @partial(jax.jit, static_argnames=())
        @partial(shard_map, mesh=mesh,
                 in_specs=(P("x", "y", "z"),),
                 out_specs=(P("x", "y", "z"),) * 3)
        def _body(phi_l):
            if boundary == 0:
                pp = _halo(phi_l, mesh, h)
            else:
                from luna.gravity.isolated import pad_sharded
                pp = pad_sharded(phi_l, mesh, h, boundary)
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
    """Trilinear force at particle positions, which may sit up to one ghost cell outside their block."""
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
            il = [_local_cell(i0[a], org[a], Ng[a], nloc[a]) for a in range(3)]
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


def _local_cell(i0, org, Ng, nloc):
    """Lower CIC corner of a particle in its block's 1-cell-ghosted local grid, offset taken minimum-image."""
    # Fold only offsets outside the ghost window [-1, nloc]; a symmetric fold breaks when nloc > Ng/2.
    r = i0 - org
    r = jnp.where(r < -1, r + Ng, r)
    r = jnp.where(r > nloc, r - Ng, r)
    return jnp.clip(r + 1, 0, nloc)


def resort_blocks(pos, p, mass, active, g, N, dx, mesh, ids=None):
    """Stable cell-order resort of every block's buffer, inactive slots last: (pos, p, mass, active, g, ids).

    ``g`` and ``ids`` are permuted with the particles so each keeps its own force.
    """
    pdims = [int(mesh.shape[a]) for a in AXES]
    Ng = _ncells(N)
    nloc = [Ng[a] // pdims[a] for a in range(3)]

    _pk = ("resort", _mesh_key(mesh), Ng, float(dx), ids is None)
    _b = _PROGRAM_CACHE.get(_pk)
    if _b is None:
        from luna.utils.sorting import argsort_uint, bits_for, native_argsort_ok
        native_ok = native_argsort_ok()          # eager self-check, must run before tracing
        Pk, Pn = P(("x", "y", "z"), None), P(("x", "y", "z"))
        specs = (Pk, Pk, Pn, Pn, Pk) + (() if ids is None else (Pn,))

        @partial(jax.jit, static_argnames=())
        @partial(shard_map, mesh=mesh, in_specs=specs, out_specs=specs)
        def _body(pos_l, p_l, mass_l, act_l, g_l, *ids_l):
            org = _block_origin(pdims, nloc)
            idx = [jnp.clip(jnp.floor(pos_l[:, a] / dx).astype(jnp.int32)
                            - org[a] + 1, 0, nloc[a] + 1) for a in range(3)]
            key = ((idx[0] * (nloc[1] + 2) + idx[1]) * (nloc[2] + 2) + idx[2])
            n_cells = (nloc[0] + 2) * (nloc[1] + 2) * (nloc[2] + 2)
            key = jnp.where(act_l > 0, key, n_cells)          # inactive: past every cell
            # the native argsort is wrong on some ROCm builds; see luna/utils/sorting.py
            order = argsort_uint(key, bits_for(n_cells), native_ok)
            return tuple(a[order] for a in (pos_l, p_l, mass_l, act_l, g_l) + ids_l)

        _PROGRAM_CACHE[_pk] = _b = _body
    out = _b(pos, p, mass, active, g, *(() if ids is None else (ids,)))
    return tuple(out) + ((None,) if ids is None else ())


def _exchange_axis(cols, active, axis, N, dx, box, mesh, Cf):
    """One ring-exchange pass along ``axis`` through face buffers of ``Cf`` slots: (cols, active, overflow).

    cols: (pos, p, mass, ...), moved together; pos decides the owner.
    """
    name = AXES[axis]
    npx = int(mesh.shape[name])
    if npx == 1:
        zero = jnp.zeros((), jnp.int32)
        return cols, active, zero
    nloc = _ncells(N)[axis] // npx
    box_a = _box3(box)[axis]
    bw = nloc * dx
    to_right = [(i, (i + 1) % npx) for i in range(npx)]
    to_left = [(i, (i - 1) % npx) for i in range(npx)]
    ndims = tuple(c.ndim for c in cols)

    _pk = ("ex", _mesh_key(mesh), axis, N, float(dx), box_a, Cf, ndims)
    _b = _PROGRAM_CACHE.get(_pk)
    if _b is None:
        S = ("x", "y", "z")
        specs = tuple(P(S, None) if n == 2 else P(S) for n in ndims)

        @partial(jax.jit, static_argnames=())
        @partial(shard_map, mesh=mesh, in_specs=(specs, P(S)),
                 out_specs=(specs, P(S), P()))
        def _body(cols_l, active_l):
            pos_l = cols_l[0]
            C = pos_l.shape[0]
            d = jax.lax.axis_index(name)
            xw = jnp.mod(pos_l[:, axis], box_a)
            act = active_l > 0
            target = jnp.clip((xw / bw).astype(jnp.int32), 0, npx - 1)
            diff = jnp.mod(target - d, npx)
            go_right = act & (diff == 1)
            go_left = act & (diff == npx - 1) & (npx > 2)
            stay = act & (~go_left) & (~go_right)

            def _send(sel, perm):
                """Ship the movers to the neighbour: (received columns, received mask, overflow)."""
                slot = jnp.cumsum(sel.astype(jnp.int32)) - 1
                dest = jnp.where(sel, slot, Cf)
                bufs = [jnp.zeros((Cf,) + a.shape[1:], a.dtype).at[dest].set(a, mode="drop")
                        for a in cols_l]
                cnt = jnp.sum(sel.astype(jnp.int32))
                got = [jax.lax.ppermute(q, name, perm) for q in bufs]
                n_got = jax.lax.ppermute(jnp.minimum(cnt, Cf), name, perm)
                return got, jnp.arange(Cf) < n_got, jnp.maximum(cnt - Cf, 0)

            parts, masks = [list(cols_l)], [stay]
            got, keep, sendover = _send(go_right, to_right)
            parts.append(got)
            masks.append(keep)
            if npx > 2:
                got, keep, over = _send(go_left, to_left)
                parts.append(got)
                masks.append(keep)
                sendover = sendover + over
            act_a = jnp.concatenate(masks, axis=0)
            slot = jnp.cumsum(act_a.astype(jnp.int32)) - 1
            dest = jnp.where(act_a, slot, C)
            new = tuple(
                jnp.zeros((C,) + parts[0][j].shape[1:], parts[0][j].dtype).at[dest].set(
                    jnp.concatenate([q[j] for q in parts], axis=0), mode="drop")
                for j in range(len(cols_l)))
            nact = jnp.maximum(slot[-1] + 1, 0)
            n_overflow = jnp.maximum(nact - C, 0).astype(jnp.int32) + sendover
            act_n = (jnp.arange(C) < jnp.minimum(nact, C)).astype(active_l.dtype)
            n_over = jax.lax.psum(n_overflow, AXES)
            return new, act_n, n_over

        _PROGRAM_CACHE[_pk] = _b = _body
    return _b(tuple(cols), active)


def _face_capacity(N, mesh, C, factor, capacity=None):
    """Slots per face buffer: counted, a fraction ``factor`` of C, or by default 1.5 cell layers of the block."""
    if capacity is not None:
        return max(1024, int(capacity))
    if factor is not None:
        return max(1024, int(factor * C))
    Ng = _ncells(N)
    pdims = [int(mesh.shape[a]) for a in AXES]
    nloc = [Ng[a] // pdims[a] for a in range(3)]
    layer = max((nloc[(a + 1) % 3] * nloc[(a + 2) % 3]
                 for a in range(3) if pdims[a] > 1), default=0)
    return max(1024, int(1.5 * layer))


def exchange_particles_blocks(pos, p, mass, active, N, dx, box, mesh,
                              face_capacity_factor=None, face_capacity=None, ids=None):
    """Move particles to their owning block by ring exchanges along x, y, z: (pos, p, mass, active, ids, overflow).

    Assumes at most one block hop per axis; a nonzero overflow is a capacity violation the caller must raise.
    """
    C = pos.shape[0] // int(np.prod([mesh.shape[a] for a in AXES]))
    Cf = _face_capacity(N, mesh, C, face_capacity_factor, face_capacity)
    n_over_tot = jnp.zeros((), jnp.int32)
    cols = (pos, p, mass) + (() if ids is None else (ids,))
    for axis in range(3):
        cols, active, n_over = _exchange_axis(cols, active, axis, N, dx, box, mesh, Cf)
        n_over_tot = n_over_tot + n_over
    pos, p, mass = cols[:3]
    return pos, p, mass, active, (cols[3] if ids is not None else None), n_over_tot
