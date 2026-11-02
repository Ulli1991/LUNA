"""MUSCL-Hancock flux divergence dU/dt = -∇·F(U), first and second order.

Which branch runs for which switches: docs/HYDRO_KERNEL_PATHS.md.
"""

import jax.numpy as jnp

from luna.hydro.jacobian import (
    apply_primitive_jacobian_1d,
    apply_primitive_jacobian_2d,
    apply_primitive_jacobian_3d_x,
    apply_primitive_jacobian_3d_y,
    apply_primitive_jacobian_3d_z,
)
from luna.hydro.slopes import (
    compute_slope_x_3d_from_padded,
    compute_slope_y_3d_from_padded,
    compute_slope_z_3d_from_padded,
    compute_slopes_from_padded,
    face_guard_fn,
    face_rule,
)


def _pick_riemann(riemann_solver: str):
    """Return the flux callable for the requested solver."""
    if riemann_solver == "hll":
        from luna.hydro.fluxes import hll_flux_primitive as flux_fn
    else:
        from luna.hydro.fluxes import hllc_flux_primitive as flux_fn
    return flux_fn


def first_order_divergence(W_pad, *, gamma, riemann_solver, spatial_dim, inv_dx, inv_dy, inv_dz):
    """First-order (Godunov) flux divergence -∇·F from a 1-ghost primitive pad."""
    flux_fn = _pick_riemann(riemann_solver)

    if spatial_dim == 1:
        Fx = flux_fn(W_pad[:, :-1], W_pad[:, 1:], axis=0, gamma=gamma)
        return -(Fx[:, 1:] - Fx[:, :-1]) * inv_dx

    if spatial_dim == 2:
        Fx = flux_fn(W_pad[:, :-1, 1:-1], W_pad[:, 1:, 1:-1], axis=0, gamma=gamma)
        Fy = flux_fn(W_pad[:, 1:-1, :-1], W_pad[:, 1:-1, 1:], axis=1, gamma=gamma)
        return -((Fx[:, 1:, :] - Fx[:, :-1, :]) * inv_dx + (Fy[:, :, 1:] - Fy[:, :, :-1]) * inv_dy)

    if spatial_dim == 3:
        Fx = flux_fn(W_pad[:, :-1, 1:-1, 1:-1], W_pad[:, 1:, 1:-1, 1:-1], axis=0, gamma=gamma)
        Fy = flux_fn(W_pad[:, 1:-1, :-1, 1:-1], W_pad[:, 1:-1, 1:, 1:-1], axis=1, gamma=gamma)
        Fz = flux_fn(W_pad[:, 1:-1, 1:-1, :-1], W_pad[:, 1:-1, 1:-1, 1:], axis=2, gamma=gamma)
        return -(
            (Fx[:, 1:, :, :] - Fx[:, :-1, :, :]) * inv_dx
            + (Fy[:, :, 1:, :] - Fy[:, :, :-1, :]) * inv_dy
            + (Fz[:, :, :, 1:] - Fz[:, :, :, :-1]) * inv_dz
        )

    raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")


_SLOPE_3D = (
    compute_slope_x_3d_from_padded,
    compute_slope_y_3d_from_padded,
    compute_slope_z_3d_from_padded,
)


def _axis_face_slices(d):
    """Index tuples (all-but-last, all-but-first, first, last) along axis ``d`` of a rank-4 array."""
    lo = [slice(None)] * 4
    hi = [slice(None)] * 4
    first = [slice(None)] * 4
    last = [slice(None)] * 4
    lo[1 + d] = slice(None, -1)
    hi[1 + d] = slice(1, None)
    first[1 + d] = slice(0, 1)
    last[1 + d] = slice(-1, None)
    return tuple(lo), tuple(hi), tuple(first), tuple(last)


def _axis_bc_slabs(W_pad, d):
    """Left/right ghost slabs of a one-ghost pad along axis ``d``."""
    bc = [slice(1, -1)] * 3
    bc[d] = slice(0, 1)
    WL_bc = W_pad[(slice(None), *bc)]
    bc[d] = slice(-1, None)
    WR_bc = W_pad[(slice(None), *bc)]
    return WL_bc, WR_bc


def _muscl_hancock_3d(W_pad, dt, *, gamma, limiter, riemann_solver, inv_dx, inv_dy, inv_dz, boundary,
                      exact_wrap=False, recon32=False, wide=False, mass_flux=False, face_guard=False):
    """3D unsplit MUSCL-Hancock divergence, built one direction at a time to bound memory.

    recon32: slopes, predictor and face states in FP32; fluxes and update in the state dtype.
    wide: ``W_pad`` has two ghost layers, so every seam face sees two reconstructed states.
    mass_flux: also return the mass flux on every face, (Fx, Fy, Fz) with N_d + 1 faces along axis d.
    face_guard: False, or (smallr, smallc, gamma) for the RAMSES face floors (slopes.ramses_face).
    """
    import jax.lax as lax

    flux_fn = _pick_riemann(riemann_solver)
    guard = face_guard_fn(face_guard)
    Wp = W_pad.astype(jnp.float32) if recon32 else W_pad
    Wc = Wp[:, 1:-1, 1:-1, 1:-1]

    dWx = compute_slope_x_3d_from_padded(Wp, limiter)
    A_dW = apply_primitive_jacobian_3d_x(Wc, dWx * inv_dx, gamma)
    A_dW = lax.optimization_barrier(A_dW)

    dWy = compute_slope_y_3d_from_padded(Wp, limiter)
    A_dW = A_dW + apply_primitive_jacobian_3d_y(Wc, dWy * inv_dy, gamma)

    dWz = compute_slope_z_3d_from_padded(Wp, limiter)
    A_dW = A_dW + apply_primitive_jacobian_3d_z(Wc, dWz * inv_dz, gamma)

    W_half = Wc - 0.5 * jnp.asarray(dt, Wp.dtype) * A_dW

    def _sweep(d, Wp, W_half, W_pad):
        """Axis-``d`` contribution to dU (slope recomputed locally)."""
        dW = _SLOPE_3D[d](Wp, limiter)
        Wr = guard((W_half + 0.5 * dW).astype(W_pad.dtype), Wc, 4).astype(W_pad.dtype)
        Wl = guard((W_half - 0.5 * dW).astype(W_pad.dtype), Wc, 4).astype(W_pad.dtype)
        if wide:
            slo, shi = [slice(1, -1)] * 3, [slice(1, -1)] * 3
            slo[d], shi[d] = slice(0, -1), slice(1, None)
            F = flux_fn(Wr[(slice(None),) + tuple(slo)],
                        Wl[(slice(None),) + tuple(shi)], axis=d, gamma=gamma)
            a, b = [slice(None)] * 4, [slice(None)] * 4
            a[1 + d], b[1 + d] = slice(0, -1), slice(1, None)
            dU_d = (F[tuple(a)] - F[tuple(b)]) * (inv_dx, inv_dy, inv_dz)[d]
            return (dU_d, F[0]) if mass_flux else dU_d
        lo, hi, first, last = _axis_face_slices(d)
        F_int = flux_fn(Wr[lo], Wl[hi], axis=d, gamma=gamma)
        if exact_wrap:
            # Periodic wrap is one face: one flux used at both ends, so the sum telescopes.
            F_face = flux_fn(Wr[last], Wl[first], axis=d, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            # Cell-centred ghost on the outer side: first order, not telescoping at a seam.
            WL_bc, WR_bc = _axis_bc_slabs(W_pad, d)
            F_l = flux_fn(WL_bc, Wl[first], axis=d, gamma=gamma)
            F_r = flux_fn(Wr[last], WR_bc, axis=d, gamma=gamma)
        dU_d = jnp.concatenate(
            [F_l - F_int[first], F_int[lo] - F_int[hi], F_int[last] - F_r],
            axis=1 + d,
        ) * (inv_dx, inv_dy, inv_dz)[d]
        if mass_flux:
            return dU_d, jnp.concatenate([F_l[0], F_int[0], F_r[0]], axis=d)
        return dU_d

    # The y/z inputs go through the barrier with x, else XLA fuses all three sweeps.
    ins = (Wp, W_half, W_pad)
    if mass_flux:
        (dU, fx), *ins = lax.optimization_barrier((_sweep(0, *ins), *ins))
        (dy, fy), (dz, fz) = _sweep(1, *ins), _sweep(2, *ins)
        return dU + dy + dz, (fx, fy, fz)
    dU, *ins = lax.optimization_barrier((_sweep(0, *ins), *ins))
    dU = dU + _sweep(1, *ins) + _sweep(2, *ins)
    return dU


def domain_face_fluxes_3d(W_pad2, dt, *, gamma, limiter, riemann_solver,
                          inv_dx, inv_dy, inv_dz, recon32=False, face_guard=False):
    """Wide-step fluxes on the block's six outer faces, ((x_lo, x_hi), (y_lo, y_hi), (z_lo, z_hi)).

    Same arithmetic as ``_muscl_hancock_3d(wide=True)``; each flux is (n_vars, *transverse).
    """
    flux_fn = _pick_riemann(riemann_solver)
    guard = face_guard_fn(face_guard)
    slopes = (compute_slope_x_3d_from_padded, compute_slope_y_3d_from_padded,
              compute_slope_z_3d_from_padded)
    jac = (apply_primitive_jacobian_3d_x, apply_primitive_jacobian_3d_y,
           apply_primitive_jacobian_3d_z)
    inv = (inv_dx, inv_dy, inv_dz)
    out = []
    for d in range(3):
        pair = []
        for side in (slice(0, 4), slice(-4, None)):
            s = [slice(None)] * 4
            s[1 + d] = side
            slab = W_pad2[tuple(s)]
            Wp = slab.astype(jnp.float32) if recon32 else slab
            Wc = Wp[:, 1:-1, 1:-1, 1:-1]
            A_dW = jac[0](Wc, slopes[0](Wp, limiter) * inv[0], gamma)
            A_dW = A_dW + jac[1](Wc, slopes[1](Wp, limiter) * inv[1], gamma)
            A_dW = A_dW + jac[2](Wc, slopes[2](Wp, limiter) * inv[2], gamma)
            W_half = Wc - 0.5 * jnp.asarray(dt, Wp.dtype) * A_dW
            dW = slopes[d](Wp, limiter)
            Wr = guard((W_half + 0.5 * dW).astype(slab.dtype), Wc, 4).astype(slab.dtype)
            Wl = guard((W_half - 0.5 * dW).astype(slab.dtype), Wc, 4).astype(slab.dtype)
            lo, hi = [slice(1, -1)] * 3, [slice(1, -1)] * 3
            lo[d], hi[d] = slice(0, 1), slice(1, 2)
            F = flux_fn(Wr[(slice(None),) + tuple(lo)], Wl[(slice(None),) + tuple(hi)],
                        axis=d, gamma=gamma)
            pair.append(jnp.squeeze(F, axis=1 + d))
        out.append(tuple(pair))
    return tuple(out)


def _muscl_hancock_3d_fused(W_pad2, dt, *, gamma, limiter, riemann_solver,
                            inv_dx, inv_dy, inv_dz, face_guard=False):
    """Concat-free 3D MUSCL-Hancock divergence from a two-ghost pad (boundary 0/1 only)."""
    flux_fn = _pick_riemann(riemann_solver)
    guard = face_guard_fn(face_guard)
    inv = (inv_dx, inv_dy, inv_dz)
    Wg = W_pad2[:, 1:-1, 1:-1, 1:-1]          # interior + one ghost ring
    dWs = [compute_slope_x_3d_from_padded(W_pad2, limiter),
           compute_slope_y_3d_from_padded(W_pad2, limiter),
           compute_slope_z_3d_from_padded(W_pad2, limiter)]
    A_dW = apply_primitive_jacobian_3d_x(Wg, dWs[0] * inv_dx, gamma)
    A_dW = A_dW + apply_primitive_jacobian_3d_y(Wg, dWs[1] * inv_dy, gamma)
    A_dW = A_dW + apply_primitive_jacobian_3d_z(Wg, dWs[2] * inv_dz, gamma)
    W_half = Wg - 0.5 * dt * A_dW

    dU = None
    for d in range(3):
        Wr = guard(W_half + 0.5 * dWs[d], Wg, 4)
        Wl = guard(W_half - 0.5 * dWs[d], Wg, 4)
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[d] = slice(0, -1)
        hi[d] = slice(1, None)
        F = flux_fn(Wr[(slice(None),) + tuple(lo)],
                    Wl[(slice(None),) + tuple(hi)], axis=d, gamma=gamma)
        a = [slice(None)] * 4
        b = [slice(None)] * 4
        a[1 + d] = slice(0, -1)
        b[1 + d] = slice(1, None)
        contrib = (F[tuple(a)] - F[tuple(b)]) * inv[d]
        dU = contrib if dU is None else dU + contrib
    return dU


# Face-flux variants: per-axis (n_vars, N_d+1, ...) flux arrays, used by 1D/2D and AMR reflux.


def _muscl_hancock_1d_fluxes(W_pad, dt, dWx, *, gamma, riemann_solver, inv_dx, boundary,
                             wide=False, face_guard=False):
    """1D MUSCL-Hancock face fluxes ``(Fx,)``, Fx of shape (n_vars, N+1)."""
    flux_fn = _pick_riemann(riemann_solver)
    guard = face_guard_fn(face_guard)
    Wc = W_pad[:, 1:-1]
    A_dW = apply_primitive_jacobian_1d(Wc, dWx * inv_dx, gamma)
    W_half = Wc - 0.5 * dt * A_dW
    Wr_x = guard(W_half + 0.5 * dWx, Wc, 2)
    Wl_x = guard(W_half - 0.5 * dWx, Wc, 2)
    if wide:
        return (flux_fn(Wr_x[:, :-1], Wl_x[:, 1:], axis=0, gamma=gamma),)
    if boundary == 0:
        WL_bc = Wr_x[:, -1:]
        WR_bc = Wl_x[:, 0:1]
    else:
        WL_bc = W_pad[:, 0:1]
        WR_bc = W_pad[:, -1:]
    WL = jnp.concatenate([WL_bc, Wr_x], axis=1)
    WR = jnp.concatenate([Wl_x, WR_bc], axis=1)
    Fx = flux_fn(WL, WR, axis=0, gamma=gamma)
    return (Fx,)


def _muscl_hancock_2d_fluxes(W_pad, dt, dWx, dWy, *, gamma, riemann_solver, inv_dx, inv_dy,
                             boundary, wide=False, face_guard=False):
    """2D MUSCL-Hancock face fluxes (Fx, Fy)."""
    flux_fn = _pick_riemann(riemann_solver)
    guard = face_guard_fn(face_guard)
    Wc = W_pad[:, 1:-1, 1:-1]
    A_dW = apply_primitive_jacobian_2d(Wc, dWx * inv_dx, dWy * inv_dy, gamma)
    W_half = Wc - 0.5 * dt * A_dW

    Wr_x = guard(W_half + 0.5 * dWx, Wc, 3)
    Wl_x = guard(W_half - 0.5 * dWx, Wc, 3)
    if wide:
        Wr_y = guard(W_half + 0.5 * dWy, Wc, 3)
        Wl_y = guard(W_half - 0.5 * dWy, Wc, 3)
        return (flux_fn(Wr_x[:, :-1, 1:-1], Wl_x[:, 1:, 1:-1], axis=0, gamma=gamma),
                flux_fn(Wr_y[:, 1:-1, :-1], Wl_y[:, 1:-1, 1:], axis=1, gamma=gamma))
    if boundary == 0:
        WL_x_bc = Wr_x[:, -1:, :]
        WR_x_bc = Wl_x[:, 0:1, :]
    else:
        WL_x_bc = W_pad[:, 0:1, 1:-1]
        WR_x_bc = W_pad[:, -1:, 1:-1]
    WL_x = jnp.concatenate([WL_x_bc, Wr_x], axis=1)
    WR_x = jnp.concatenate([Wl_x, WR_x_bc], axis=1)
    Fx = flux_fn(WL_x, WR_x, axis=0, gamma=gamma)

    Wr_y = guard(W_half + 0.5 * dWy, Wc, 3)
    Wl_y = guard(W_half - 0.5 * dWy, Wc, 3)
    if boundary == 0:
        WL_y_bc = Wr_y[:, :, -1:]
        WR_y_bc = Wl_y[:, :, 0:1]
    else:
        WL_y_bc = W_pad[:, 1:-1, 0:1]
        WR_y_bc = W_pad[:, 1:-1, -1:]
    WL_y = jnp.concatenate([WL_y_bc, Wr_y], axis=2)
    WR_y = jnp.concatenate([Wl_y, WR_y_bc], axis=2)
    Fy = flux_fn(WL_y, WR_y, axis=1, gamma=gamma)

    return Fx, Fy


def _muscl_hancock_3d_fluxes(W_pad, dt, *, gamma, limiter, riemann_solver,
                             inv_dx, inv_dy, inv_dz, boundary, exact_wrap=False, face_guard=False):
    """3D face fluxes (Fx, Fy, Fz), each with N+1 faces along its axis; face_guard: slopes.face_guard_fn spec."""
    import jax.lax as lax

    flux_fn = _pick_riemann(riemann_solver)
    guard = face_guard_fn(face_guard)
    Wc = W_pad[:, 1:-1, 1:-1, 1:-1]

    dWx = compute_slope_x_3d_from_padded(W_pad, limiter)
    A_dW = apply_primitive_jacobian_3d_x(Wc, dWx * inv_dx, gamma)
    A_dW = lax.optimization_barrier(A_dW)

    dWy = compute_slope_y_3d_from_padded(W_pad, limiter)
    A_dW = A_dW + apply_primitive_jacobian_3d_y(Wc, dWy * inv_dy, gamma)
    A_dW = lax.optimization_barrier(A_dW)

    dWz = compute_slope_z_3d_from_padded(W_pad, limiter)
    A_dW = A_dW + apply_primitive_jacobian_3d_z(Wc, dWz * inv_dz, gamma)

    W_half = Wc - 0.5 * dt * A_dW
    W_half = lax.optimization_barrier(W_half)

    def _faces(d):
        """Axis-``d`` face fluxes, N+1 faces."""
        lo, hi, first, last = _axis_face_slices(d)
        dW = _SLOPE_3D[d](W_pad, limiter)
        Wr = guard(W_half + 0.5 * dW, Wc, 4)
        Wl = guard(W_half - 0.5 * dW, Wc, 4)
        F_int = flux_fn(Wr[lo], Wl[hi], axis=d, gamma=gamma)
        if exact_wrap:
            F_face = flux_fn(Wr[last], Wl[first], axis=d, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            WL_bc, WR_bc = _axis_bc_slabs(W_pad, d)
            F_l = flux_fn(WL_bc, Wl[first], axis=d, gamma=gamma)
            F_r = flux_fn(Wr[last], WR_bc, axis=d, gamma=gamma)
        return jnp.concatenate([F_l, F_int, F_r], axis=1 + d)

    return _faces(0), _faces(1), _faces(2)


def muscl_hancock_face_fluxes(
    W_pad,
    dt,
    *,
    gamma,
    limiter,
    riemann_solver,
    spatial_dim,
    inv_dx,
    inv_dy,
    inv_dz,
    boundary,
    exact_wrap=False,
    face_guard=False,
):
    """Second-order face fluxes (Fx, Fy[, Fz]); Fd has N_d + 1 entries along axis d; face_guard: slopes.face_guard_fn spec."""
    if spatial_dim == 3:
        return _muscl_hancock_3d_fluxes(
            W_pad, dt, gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
            inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz, boundary=boundary,
            exact_wrap=exact_wrap, face_guard=face_guard,
        )
    if spatial_dim == 2:
        dWx, dWy, _ = compute_slopes_from_padded(W_pad, limiter)
        return _muscl_hancock_2d_fluxes(
            W_pad, dt, dWx, dWy, gamma=gamma, riemann_solver=riemann_solver,
            inv_dx=inv_dx, inv_dy=inv_dy, boundary=boundary, face_guard=face_guard,
        )
    if spatial_dim == 1:
        dWx, _, _ = compute_slopes_from_padded(W_pad, limiter)
        return _muscl_hancock_1d_fluxes(
            W_pad, dt, dWx, gamma=gamma, riemann_solver=riemann_solver,
            inv_dx=inv_dx, boundary=boundary, face_guard=face_guard,
        )
    raise NotImplementedError(f"face-flux variant supports spatial_dim in (1,2,3); got {spatial_dim}")


def riemann_face_flux(W_L, W_R, *, axis, gamma, riemann_solver):
    """Riemann flux from a pair of reconstructed face states (variables on axis 0)."""
    return _pick_riemann(riemann_solver)(W_L, W_R, axis=axis, gamma=gamma)


def muscl_hancock_boundary_states(
    W_pad, dt, *, gamma, limiter, spatial_dim, inv_dx, inv_dy, inv_dz, face_guard=False,
):
    """Reconstructed states ((L_d, R_d), ...) at the first/last interior cell along each axis.

    Each is a face slab (n_vars, *transverse), from the same predictor as the face fluxes; face_guard: as
    there (slopes.face_rule, not recorded: these faces are the face fluxes' own, counted there).
    """
    rule = face_rule(face_guard)

    def _fo(Wf, Wc):
        return rule(Wf, Wc, Wf.ndim)[0]

    if spatial_dim == 1:
        dWx, _, _ = compute_slopes_from_padded(W_pad, limiter)
        Wc = W_pad[:, 1:-1]
        A_dW = apply_primitive_jacobian_1d(Wc, dWx * inv_dx, gamma)
        W_half = Wc - 0.5 * dt * A_dW
        Lx = _fo(W_half - 0.5 * dWx, Wc)[:, 0]
        Rx = _fo(W_half + 0.5 * dWx, Wc)[:, -1]
        return ((Lx, Rx),)

    if spatial_dim == 2:
        dWx, dWy, _ = compute_slopes_from_padded(W_pad, limiter)
        Wc = W_pad[:, 1:-1, 1:-1]
        A_dW = apply_primitive_jacobian_2d(Wc, dWx * inv_dx, dWy * inv_dy, gamma)
        W_half = Wc - 0.5 * dt * A_dW
        Lx = _fo(W_half - 0.5 * dWx, Wc)[:, 0, :]
        Rx = _fo(W_half + 0.5 * dWx, Wc)[:, -1, :]
        Ly = _fo(W_half - 0.5 * dWy, Wc)[:, :, 0]
        Ry = _fo(W_half + 0.5 * dWy, Wc)[:, :, -1]
        return ((Lx, Rx), (Ly, Ry))

    if spatial_dim == 3:
        Wc = W_pad[:, 1:-1, 1:-1, 1:-1]
        dWx = compute_slope_x_3d_from_padded(W_pad, limiter)
        dWy = compute_slope_y_3d_from_padded(W_pad, limiter)
        dWz = compute_slope_z_3d_from_padded(W_pad, limiter)
        A_dW = (
            apply_primitive_jacobian_3d_x(Wc, dWx * inv_dx, gamma)
            + apply_primitive_jacobian_3d_y(Wc, dWy * inv_dy, gamma)
            + apply_primitive_jacobian_3d_z(Wc, dWz * inv_dz, gamma)
        )
        W_half = Wc - 0.5 * dt * A_dW
        Lx = _fo(W_half - 0.5 * dWx, Wc)[:, 0, :, :]
        Rx = _fo(W_half + 0.5 * dWx, Wc)[:, -1, :, :]
        Ly = _fo(W_half - 0.5 * dWy, Wc)[:, :, 0, :]
        Ry = _fo(W_half + 0.5 * dWy, Wc)[:, :, -1, :]
        Lz = _fo(W_half - 0.5 * dWz, Wc)[:, :, :, 0]
        Rz = _fo(W_half + 0.5 * dWz, Wc)[:, :, :, -1]
        return ((Lx, Rx), (Ly, Ry), (Lz, Rz))

    raise NotImplementedError(f"boundary states support spatial_dim in (1,2,3); got {spatial_dim}")


def divergence_from_face_fluxes(fluxes, *, spatial_dim, inv_dx, inv_dy, inv_dz):
    """dU/dt = -div F from per-axis face fluxes."""
    if spatial_dim == 1:
        (Fx,) = fluxes
        return (Fx[:, :-1] - Fx[:, 1:]) * inv_dx
    if spatial_dim == 2:
        Fx, Fy = fluxes
        return (Fx[:, :-1, :] - Fx[:, 1:, :]) * inv_dx + (Fy[:, :, :-1] - Fy[:, :, 1:]) * inv_dy
    if spatial_dim == 3:
        Fx, Fy, Fz = fluxes
        return (
            (Fx[:, :-1, :, :] - Fx[:, 1:, :, :]) * inv_dx
            + (Fy[:, :, :-1, :] - Fy[:, :, 1:, :]) * inv_dy
            + (Fz[:, :, :, :-1] - Fz[:, :, :, 1:]) * inv_dz
        )
    raise NotImplementedError(f"spatial_dim in (1,2,3); got {spatial_dim}")


def muscl_hancock_divergence(
    W_pad,
    dt,
    *,
    gamma,
    limiter,
    riemann_solver,
    spatial_dim,
    inv_dx,
    inv_dy,
    inv_dz,
    boundary,
    order,
    exact_wrap=False,
    fused=False,
    recon32=False,
    wide=False,
    mass_flux=False,
    face_guard=False,
):
    """dU/dt = -∇·F(U): MUSCL-Hancock, or first-order Godunov when order=0; mass_flux: (dU, face mass fluxes); face_guard: slopes.face_guard_fn spec on the face states."""
    if mass_flux and (order == 0 or fused or spatial_dim != 3):
        raise NotImplementedError("face mass fluxes: the 3-D PLM step without hydro_fused only "
                                  f"(got ORDER={order}, hydro_fused={fused}, SPATIAL_DIMENSION={spatial_dim})")
    if order == 0:
        return first_order_divergence(
            W_pad,
            gamma=gamma,
            riemann_solver=riemann_solver,
            spatial_dim=spatial_dim,
            inv_dx=inv_dx,
            inv_dy=inv_dy,
            inv_dz=inv_dz,
        )

    if fused:
        if spatial_dim != 3:
            raise NotImplementedError("fused path is 3D-only")
        return _muscl_hancock_3d_fused(
            W_pad, dt, gamma=gamma, limiter=limiter,
            riemann_solver=riemann_solver,
            inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz, face_guard=face_guard)

    if spatial_dim == 3:
        return _muscl_hancock_3d(
            W_pad,
            dt,
            gamma=gamma,
            limiter=limiter,
            riemann_solver=riemann_solver,
            inv_dx=inv_dx,
            inv_dy=inv_dy,
            inv_dz=inv_dz,
            boundary=boundary,
            exact_wrap=exact_wrap,
            recon32=recon32,
            wide=wide,
            mass_flux=mass_flux,
            face_guard=face_guard,
        )

    dWx, dWy, dWz = compute_slopes_from_padded(W_pad, limiter)

    if spatial_dim == 1:
        fluxes = _muscl_hancock_1d_fluxes(
            W_pad,
            dt,
            dWx,
            gamma=gamma,
            riemann_solver=riemann_solver,
            inv_dx=inv_dx,
            boundary=boundary,
            wide=wide,
            face_guard=face_guard,
        )
    elif spatial_dim == 2:
        fluxes = _muscl_hancock_2d_fluxes(
            W_pad,
            dt,
            dWx,
            dWy,
            gamma=gamma,
            riemann_solver=riemann_solver,
            inv_dx=inv_dx,
            inv_dy=inv_dy,
            boundary=boundary,
            wide=wide,
            face_guard=face_guard,
        )
    else:
        raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")
    return divergence_from_face_fluxes(
        fluxes, spatial_dim=spatial_dim, inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz
    )
