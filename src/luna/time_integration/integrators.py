"""The MUSCL-Hancock hydro step factory, with halo exchange inside the jitted step.
Which kernel path runs for which switches: docs/HYDRO_KERNEL_PATHS.md.
"""

import jax
import jax.numpy as jnp

from luna.hydro.conversions import cons_to_prim, cons_to_prim_floored


def cell_dt(vs, cs, dxs, CFL):
    """Per-cell CFL dt, the RAMSES rule: CFL / sum_d (|v_d| + c) / dx_d."""
    return CFL / sum((jnp.abs(v) + cs) / dx for v, dx in zip(vs, dxs))


def _make_compute_dt_fn(gamma, CFL, spatial_dim, dx, dy, dz, mesh=None, floors=None):
    """Return compute_dt(U) -> CFL time step, a shard_map pmin when ``mesh`` is given; floors: (smallr, smallc), RAMSES primitives."""
    # With a mesh this must be shard_map, not GSPMD: mixing the two executables hangs ROCm.
    if spatial_dim not in (1, 2, 3):
        raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")
    dxs = (dx, dy, dz)[:spatial_dim]

    def compute_dt_from_U(U):
        W = cons_to_prim(U, gamma) if floors is None else cons_to_prim_floored(U, gamma, *floors)
        cs = jnp.sqrt(gamma * W[spatial_dim + 1] / W[0])
        dt_local = cell_dt([W[1 + d] for d in range(spatial_dim)], cs, dxs, CFL)
        # NaN -> -inf: the cross-device min drops NaN, so the runners' finite-dt guard would miss it.
        return jnp.min(jnp.where(jnp.isnan(dt_local), -jnp.inf, dt_local))

    if mesh is None:
        return jax.jit(compute_dt_from_U)
    from luna.utils.shmap import shard_map
    from jax.sharding import PartitionSpec as P
    axes = ("x", "y", "z")[:spatial_dim]

    def _dt_sharded(U):
        return jax.lax.pmin(compute_dt_from_U(U), axes)

    return jax.jit(shard_map(_dt_sharded, mesh=mesh,
                             in_specs=(P(None, *axes),), out_specs=P()))


def _hydro_paths(order, spatial_dim, boundary, fused, recon32):
    """(fused, recon32, wide) as the step resolves them; see docs/HYDRO_KERNEL_PATHS.md."""
    fused = bool(fused and order >= 1 and spatial_dim == 3 and boundary in (0, 1))
    recon32 = bool(recon32 and order >= 1 and spatial_dim == 3 and not fused)
    # 1D/2D diode stays narrow (its ghost-centred wall face is first order).
    wide = bool((not fused) and order >= 1 and spatial_dim in (1, 2, 3)
                and (boundary in (0, 1, 2) or (boundary == 4 and spatial_dim == 3)))
    return fused, recon32, wide


FACE_REPORT_CELLS = 8       # cells per shard a face report locates


def _face_report(own, floored, shape, axes, sizes=None):
    """{"n": first-order face states, "n_floor": floored ones, "cells": (K, ndim) global indices of the first K cells
    with one (-1 pads), "faces": (K,) their face bitmasks}, one entry per shard, gathered over the mesh axes (None:
    one device); own/floored: per-cell int8 bitmasks, bit 2 d the +d face, 2 d + 1 the -d face. The cells are found
    only on a step with one (lax.cond): a clean step costs one reduction."""
    k, nd = FACE_REPORT_CELLS, len(shape)
    if own is None:                     # ORDER=0: no reconstructed face
        return _gather({"n": jnp.zeros((1,), jnp.int32), "n_floor": jnp.zeros((1,), jnp.int32),
                        "cells": jnp.full((k, nd), -1, jnp.int32), "faces": jnp.zeros((k,), jnp.int32)},
                       axes, sizes)
    n = jnp.sum(jax.lax.population_count(own), dtype=jnp.int32)

    def where():
        nz = jnp.nonzero(own > 0, size=k, fill_value=-1)
        idx = jnp.stack(nz, axis=1).astype(jnp.int32)
        faces = jnp.where(nz[0] >= 0, own[tuple(jnp.maximum(i, 0) for i in nz)], 0).astype(jnp.int32)
        if axes is not None:
            off = jnp.stack([jax.lax.axis_index(a) * n_d for a, n_d in zip(axes, shape)]).astype(jnp.int32)
            idx = jnp.where(idx >= 0, idx + off, -1)
        return idx, faces

    # + 0 * n: varying over the shards like the true branch (shard_map's cond needs equal types)
    cells, faces = jax.lax.cond(n > 0, where, lambda: (jnp.full((k, nd), -1, jnp.int32) + 0 * n,
                                                       jnp.zeros((k,), jnp.int32) + 0 * n))
    n_floor = (jnp.zeros((), jnp.int32) if floored is None
               else jnp.sum(jax.lax.population_count(floored), dtype=jnp.int32))
    return _gather({"n": n[None], "n_floor": n_floor[None], "cells": cells, "faces": faces}, axes, sizes)


def _gather(rep, axes, sizes):
    """The report's per-shard entries stacked over the mesh, on every device (axes None: as is)."""
    if axes is None:
        return rep
    # every shard's entries on every device, replicated (each process can read it, multi-node): its slot of a
    # zero array, summed over the mesh (all_gather's result stays typed per shard for out_specs P())
    slot, n_sh = 0, 1
    for a, m in zip(axes, sizes):
        slot, n_sh = slot * m + jax.lax.axis_index(a), n_sh * m

    def gather(v):
        full = jnp.zeros((n_sh,) + v.shape, v.dtype).at[slot].set(v)
        return jax.lax.psum(full, tuple(axes)).reshape((n_sh * v.shape[0],) + v.shape[1:])
    return jax.tree.map(gather, rep)


def _report_from_masks(rec, shape, axes, sizes=None):
    """_face_report from slopes.recording_faces masks: each per-cell (maybe with a ghost ring), cropped to the shard."""
    def crop(m):
        return m[tuple(slice((m.shape[d] - n) // 2, (m.shape[d] - n) // 2 + n) for d, n in enumerate(shape))]
    own = floored = None                # bit k: the k-th recorded face state, + then - per axis (the kernels' order)
    for k, (bad, fl) in enumerate(rec):
        b = crop(bad).astype(jnp.int8) << k
        own = b if own is None else own | b
        if fl is not None:
            f = crop(fl).astype(jnp.int8) << k
            floored = f if floored is None else floored | f
    return _face_report(own, floored, shape, axes, sizes)


def _report_from_pallas(r, axes, floors, sizes=None):
    """_face_report from the Pallas per-cell report (first-order bitmask, floored bitmask << 8 with floors)."""
    return _face_report((r & 63).astype(jnp.int8), ((r >> 8) & 63).astype(jnp.int8) if floors else None,
                        r.shape, axes, sizes)


def create_muscl_hancock_modular_step(
    pad_halo_fn,
    gamma,
    order,
    limiter,
    dx,
    dy,
    dz,
    inv_dx,
    inv_dy,
    inv_dz,
    boundary,
    spatial_dim,
    CFL,
    sharding=None,
    riemann_solver="hllc",
    use_shardmap=False,
    mesh=None,
    fused=False,
    recon32=False,
    kernel="xla",
    autodiff=False,
    n_scalars=0,
    mass_flux=False,
    floors=None,
    allow_first_order=False,
    face_report=False,
):
    """Return (compute_dt_fn(U) -> CFL dt, step_fn(U, dt) -> U_new, or step_fn(U, S, dt) -> (U_new, S_new) with n_scalars scalars S = rho X); use_shardmap needs ``mesh``; pad_halo_fn=None: local jnp.pad halo, or the ppermute halo under shard_map; mass_flux: step_fn(U, dt) -> (U_new, (Fx, Fy, Fz)), the face mass fluxes, N_d + 1 faces per shard along d (3-D PLM, XLA or Pallas); floors: (smallr, smallc), the RAMSES positivity floors on primitives, faces and Riemann inputs (preserve_face_states; U itself is not floored); allow_first_order: an unphysical Hancock face state (rho <= 0 or p <= 0) takes the cell-centre state (audit C8), else NaN (the run refuses); face_report: every step_fn also returns, last, the face report {"n", "n_floor": per-shard counts, "cells": (shards x K, ndim) global cell indices} (_face_report)."""
    if pad_halo_fn is None:
        if use_shardmap:
            if mesh is None:
                raise ValueError("use_shardmap=True requires `mesh` to be passed in")
            from luna.utils.halo_exchange_shardmap import make_halo_shardmap
            pad_halo_fn = make_halo_shardmap(mesh, spatial_dim)
        else:
            from luna.utils.halo_local import pad_halo_local as pad_halo_fn
    compute_dt_fn = _make_compute_dt_fn(gamma, CFL, spatial_dim, dx, dy, dz,
                                        mesh=mesh if use_shardmap else None, floors=floors)

    from luna.hydro.muscl_hancock import muscl_hancock_divergence

    # Under shard_map each shard is local, so the wrap face is not visible from both sides.
    exact_wrap = (boundary == 0) and (not use_shardmap)

    from luna.hydro import muscl_hancock_pallas as mhp
    # 'pallas' refuses what the kernels cannot run (as requested, before _hydro_paths drops it)
    pallas = mhp.select(kernel, mhp.unsupported(
        spatial_dim=spatial_dim, order=order, limiter=limiter, riemann_solver=riemann_solver,
        boundary=boundary, fused=fused, recon32=recon32, autodiff=autodiff, n_scalars=n_scalars))

    fused, recon32, wide = _hydro_paths(order, spatial_dim, boundary, fused, recon32)
    if mass_flux and (n_scalars or spatial_dim != 3 or order < 1 or fused):
        raise NotImplementedError(
            "face mass fluxes (gravity_energy='flux'): the 3-D PLM step (XLA or Pallas), no hydro_fused, no "
            f"passive scalars; got hydro_kernel={kernel!r} (Pallas: {bool(pallas)}), "
            f"SPATIAL_DIMENSION={spatial_dim}, ORDER={order}, hydro_fused={fused}, n_scalars={n_scalars}")
    from luna.hydro.slopes import face_spec, recording_faces
    gkw = {"face_guard": face_spec(floors, gamma, allow_first_order)}
    axes = ("x", "y", "z")[:spatial_dim] if use_shardmap else None
    sizes = tuple(int(mesh.shape[a]) for a in axes) if use_shardmap else None

    def divergence(W_pad, dt, shape, **kw):
        """(dU or (dU, F), face report or None): the Pallas or XLA divergence."""
        if pallas:
            out = mhp.divergence(W_pad, dt, gamma=gamma, inv=(inv_dx, inv_dy, inv_dz), recon32=recon32,
                                 riemann_solver=riemann_solver, limiter=limiter, report=face_report,
                                 **gkw, **kw)
            return (out[0], _report_from_pallas(out[1], axes, floors, sizes)) if face_report else (out, None)
        kw.pop("n_scalars", None)
        with recording_faces() as rec:
            out = muscl_hancock_divergence(
                W_pad, dt, gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
                spatial_dim=spatial_dim, inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz,
                boundary=boundary, order=order, exact_wrap=exact_wrap, recon32=recon32,
                wide=wide, **gkw, **kw)
        return out, (_report_from_masks(rec, shape, axes, sizes) if face_report else None)

    def tail(rep):
        return (rep,) if face_report else ()

    def to_prim(U):
        return cons_to_prim(U, gamma) if floors is None else cons_to_prim_floored(U, gamma, *floors)

    def _constrain(U_new):
        if sharding is not None and not use_shardmap:
            U_new = jax.lax.with_sharding_constraint(U_new, sharding)
        return U_new

    def _step_flux(U, dt):
        W = to_prim(U)
        W_pad = (pad_halo_fn(W, boundary, spatial_dim, width=2) if wide
                 else pad_halo_fn(W, boundary, spatial_dim))
        (dU, F), rep = divergence(W_pad, dt, U.shape[1:], mass_flux=True)
        return (_constrain(U + dt * dU), F) + tail(rep)

    def _step_inner(U, dt):
        W = to_prim(U)
        W_pad = (pad_halo_fn(W, boundary, spatial_dim, width=2) if (fused or wide)
                 else pad_halo_fn(W, boundary, spatial_dim))
        dU, rep = divergence(W_pad, dt, U.shape[1:], **({} if pallas else {"fused": fused}))
        U_new = _constrain(U + dt * dU)
        return (U_new,) + tail(rep) if face_report else U_new

    def _step_scalars(U, S, dt):
        W = to_prim(U)
        W = jnp.concatenate([W, S / W[0]])
        dU, rep = divergence(pad_halo_fn(W, boundary, spatial_dim, width=2), dt, U.shape[1:],
                             n_scalars=S.shape[0])
        nu = U.shape[0]
        return (U + dt * dU[:nu], S + dt * dU[nu:]) + tail(rep)

    if use_shardmap:
        from jax.sharding import PartitionSpec as P
        rep_spec = {"n": P(), "n_floor": P(), "cells": P(), "faces": P()}    # gathered in _face_report
    if n_scalars:
        if not (pallas and use_shardmap):
            raise NotImplementedError("passive scalars: the Pallas kernel under shard_map only "
                                      "(HLLC, the wide path; hydro_kernel='pallas')")
        from luna.utils.shmap import shard_map
        from jax.sharding import PartitionSpec as P
        spec = P(None, *("x", "y", "z")[:spatial_dim])
        return compute_dt_fn, jax.jit(shard_map(_step_scalars, mesh=mesh,
                                                in_specs=(spec, spec, P()),
                                                out_specs=(spec, spec) + tail(rep_spec)))

    if use_shardmap:
        if mesh is None:
            raise ValueError("use_shardmap=True requires `mesh` to be passed in")
        from luna.utils.shmap import shard_map
        from jax.sharding import PartitionSpec as P

        if spatial_dim == 3:
            in_spec = P(None, "x", "y", "z")
        elif spatial_dim == 2:
            in_spec = P(None, "x", "y")
        else:
            in_spec = P(None, "x")
        if mass_flux:
            face = P("x", "y", "z")
            return compute_dt_fn, jax.jit(shard_map(_step_flux, mesh=mesh, in_specs=(in_spec, P()),
                                                    out_specs=(in_spec, (face, face, face)) + tail(rep_spec)))
        time_step_muscl_hancock_modular = jax.jit(
            shard_map(
                _step_inner,
                mesh=mesh,
                in_specs=(in_spec, P()),
                out_specs=(in_spec,) + tail(rep_spec) if face_report else in_spec,
            )
        )
    elif mass_flux:
        return compute_dt_fn, jax.jit(_step_flux)
    else:
        time_step_muscl_hancock_modular = jax.jit(_step_inner)

    return compute_dt_fn, time_step_muscl_hancock_modular


def create_boundary_mass_fn(pad_halo_fn, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy,
                            inv_dz, boundary, spatial_dim, *, mesh, riemann_solver="hllc",
                            fused=False, recon32=False, floors=None, allow_first_order=False):
    """Return face_mass(U, dt, acc) -> acc + (2, 6) mass in / out through the faces x_lo .. z_hi; floors, allow_first_order: as the step."""
    fused, recon32, wide = _hydro_paths(order, spatial_dim, boundary, fused, recon32)
    if spatial_dim != 3 or not (wide or fused):
        raise NotImplementedError(
            "boundary mass budget: the 3-D two-ghost hydro path only (ORDER >= 1, "
            f"BOUNDARY 0/1/4); got SPATIAL_DIMENSION={spatial_dim}, BOUNDARY={boundary}, "
            f"ORDER={order}")
    from jax.sharding import PartitionSpec as P

    from luna.hydro.muscl_hancock import domain_face_fluxes_3d
    from luna.hydro.slopes import face_spec
    from luna.utils.shmap import shard_map
    axes = ("x", "y", "z")
    n = {a: int(mesh.shape[a]) for a in axes}
    area = (dy * dz, dx * dz, dx * dy)

    def _local(U, dt, acc):
        W = cons_to_prim(U, gamma) if floors is None else cons_to_prim_floored(U, gamma, *floors)
        W_pad = pad_halo_fn(W, boundary, 3, width=2)
        faces = domain_face_fluxes_3d(W_pad, dt, gamma=gamma, limiter=limiter,
                                      riemann_solver=riemann_solver, inv_dx=inv_dx,
                                      inv_dy=inv_dy, inv_dz=inv_dz, recon32=recon32,
                                      face_guard=face_spec(floors, gamma, allow_first_order))
        rows_in, rows_out = [], []
        for d, a in enumerate(axes):
            i = jax.lax.axis_index(a)
            for side, F in enumerate(faces[d]):
                edge = (i == 0) if side == 0 else (i == n[a] - 1)
                f = F[0] * (dt * area[d])
                inward = f if side == 0 else -f     # +x flux enters at x_lo
                rows_in.append(jnp.where(edge, jnp.sum(jnp.maximum(inward, 0.0)), 0.0))
                rows_out.append(jnp.where(edge, jnp.sum(jnp.maximum(-inward, 0.0)), 0.0))
        return acc + jax.lax.psum(jnp.stack([jnp.stack(rows_in), jnp.stack(rows_out)]), axes)

    return jax.jit(shard_map(_local, mesh=mesh, in_specs=(P(None, *axes), P(), P()),
                             out_specs=P()))
