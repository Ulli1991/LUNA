"""MUSCL-Hancock flux divergence dU/dt = -∇·F(U) (first- and second-order).

Second order adds the unsplit predictor (W_half = Wc - ½dt·A·∇W) + PLM
reconstruction; the 3D branch accumulates per-direction to bound memory.
"""

import jax.numpy as jnp

from hydro.jacobian import (
    apply_primitive_jacobian_1d,
    apply_primitive_jacobian_2d,
    apply_primitive_jacobian_3d_x,
    apply_primitive_jacobian_3d_y,
    apply_primitive_jacobian_3d_z,
)
from hydro.slopes import (
    compute_slope_x_3d_from_padded,
    compute_slope_y_3d_from_padded,
    compute_slope_z_3d_from_padded,
    compute_slopes_from_padded,
)


def _pick_riemann(riemann_solver: str):
    """Return the flux callable for the requested solver."""
    if riemann_solver == "lhllc":
        from hydro.fluxes_lhllc import lhllc_flux_primitive as flux_fn
    elif riemann_solver == "hll":
        from hydro.fluxes import hll_flux_primitive as flux_fn
    else:
        from hydro.fluxes import hllc_flux_primitive as flux_fn
    return flux_fn


# --------------------------------------------------------------------------
# First-order (plain Godunov on PRIMITIVES)
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Second-order MUSCL-Hancock per spatial dimension
# --------------------------------------------------------------------------


# Per-axis single-direction slope kernels, indexed by axis d (0=x, 1=y, 2=z).
_SLOPE_3D = (
    compute_slope_x_3d_from_padded,
    compute_slope_y_3d_from_padded,
    compute_slope_z_3d_from_padded,
)


def _axis_face_slices(d):
    """Rank-4 index tuples for axis ``d`` of a (n_vars, X, Y, Z) array:
    (all-but-last, all-but-first, first-only, last-only)."""
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
    """Left/right halo slabs of a one-ghost padded (n_vars, X+2, Y+2, Z+2)
    array along axis ``d`` (interior slice on the other axes)."""
    bc = [slice(1, -1)] * 3
    bc[d] = slice(0, 1)
    WL_bc = W_pad[(slice(None), *bc)]
    bc[d] = slice(-1, None)
    WR_bc = W_pad[(slice(None), *bc)]
    return WL_bc, WR_bc


def _muscl_hancock_3d(W_pad, dt, *, gamma, limiter, riemann_solver, inv_dx, inv_dy, inv_dz, boundary,
                      exact_wrap=False, recon32=False):
    """3D unsplit MUSCL-Hancock (full transverse predictor). Predictor and
    flux sweeps are built one direction at a time with
    ``optimization_barrier`` between contributions so only one direction's
    intermediates are alive; bit-equivalent to the triple-slope version up
    to floating-point reduction order.

    ``recon32``: run slopes/predictor/face states in FP32 (halves the
    dominant memory traffic, ~2x step speed at 512^3 FP64); face states are
    cast back so the Riemann fluxes and the update stay in the state dtype
    (exact FP64 flux telescoping)."""
    import jax.lax as lax

    flux_fn = _pick_riemann(riemann_solver)
    Wp = W_pad.astype(jnp.float32) if recon32 else W_pad
    Wc = Wp[:, 1:-1, 1:-1, 1:-1]

    # ---- Predictor: build full A_dW one direction at a time, with barriers.
    dWx = compute_slope_x_3d_from_padded(Wp, limiter)
    A_dW = apply_primitive_jacobian_3d_x(Wc, dWx * inv_dx, gamma)
    A_dW = lax.optimization_barrier(A_dW)

    dWy = compute_slope_y_3d_from_padded(Wp, limiter)
    A_dW = A_dW + apply_primitive_jacobian_3d_y(Wc, dWy * inv_dy, gamma)
    A_dW = lax.optimization_barrier(A_dW)

    dWz = compute_slope_z_3d_from_padded(Wp, limiter)
    A_dW = A_dW + apply_primitive_jacobian_3d_z(Wc, dWz * inv_dz, gamma)

    W_half = Wc - 0.5 * jnp.asarray(dt, Wp.dtype) * A_dW
    W_half = lax.optimization_barrier(W_half)

    # ---- Flux phase: per-direction sweeps, slope recomputed inside each.
    def _sweep(d):
        """Axis-``d`` flux-divergence contribution (slope recomputed locally)."""
        lo, hi, first, last = _axis_face_slices(d)
        dW = _SLOPE_3D[d](Wp, limiter)
        Wr = (W_half + 0.5 * dW).astype(W_pad.dtype)
        Wl = (W_half - 0.5 * dW).astype(W_pad.dtype)
        F_int = flux_fn(Wr[lo], Wl[hi], axis=d, gamma=gamma)
        if exact_wrap:
            # Single-process periodic: the wrap seam is ONE physical face
            # (between cell N-1 and cell 0). Compute its flux once from the
            # reconstructed states on both sides and use it at both ends —
            # the flux sum then telescopes exactly (mass/momentum/energy
            # conservation to machine precision) and the seam stays 2nd order.
            F_face = flux_fn(Wr[last], Wl[first], axis=d, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            # Boundary states for the BC / shard face. Use the halo W (cell-
            # centered) — correct under shard_map (the halo brings the actual
            # neighbour shard's edge cell) and for physical BCs (the ghost
            # carries the BC state). Costs half-order accuracy and exact
            # telescoping at that single face; under shard_map the proper fix
            # is exchanging reconstructed face states (future work).
            WL_bc, WR_bc = _axis_bc_slabs(W_pad, d)
            F_l = flux_fn(WL_bc, Wl[first], axis=d, gamma=gamma)
            F_r = flux_fn(Wr[last], WR_bc, axis=d, gamma=gamma)
        return jnp.concatenate(
            [F_l - F_int[first], F_int[lo] - F_int[hi], F_int[last] - F_r],
            axis=1 + d,
        ) * (inv_dx, inv_dy, inv_dz)[d]

    # Strictly sequential (x then y then z): each sweep returns dU
    # contribution, gets summed, barrier forces materialisation before the
    # next sweep starts.
    dU = lax.optimization_barrier(_sweep(0))
    dU = lax.optimization_barrier(dU + _sweep(1))
    dU = dU + _sweep(2)
    return dU


def _muscl_hancock_3d_fused(W_pad2, dt, *, gamma, limiter, riemann_solver,
                            inv_dx, inv_dy, inv_dz):
    """Concat-free 3D MUSCL-Hancock — needs a TWO-ghost pad: reconstructs on
    interior + one ghost ring, one flux call per axis over all N+1 faces.
    Restrictions: order=1 (MUSCL) path only; caller gates on boundary 0/1."""
    flux_fn = _pick_riemann(riemann_solver)
    inv = (inv_dx, inv_dy, inv_dz)
    Wg = W_pad2[:, 1:-1, 1:-1, 1:-1]          # interior + 1 ghost ring
    dWs = [compute_slope_x_3d_from_padded(W_pad2, limiter),
           compute_slope_y_3d_from_padded(W_pad2, limiter),
           compute_slope_z_3d_from_padded(W_pad2, limiter)]
    A_dW = apply_primitive_jacobian_3d_x(Wg, dWs[0] * inv_dx, gamma)
    A_dW = A_dW + apply_primitive_jacobian_3d_y(Wg, dWs[1] * inv_dy, gamma)
    A_dW = A_dW + apply_primitive_jacobian_3d_z(Wg, dWs[2] * inv_dz, gamma)
    W_half = Wg - 0.5 * dt * A_dW

    dU = None
    for d in range(3):
        Wr = W_half + 0.5 * dWs[d]
        Wl = W_half - 0.5 * dWs[d]
        lo = [slice(1, -1)] * 3
        hi = [slice(1, -1)] * 3
        lo[d] = slice(0, -1)                   # N+1 left face states
        hi[d] = slice(1, None)                 # N+1 right face states
        F = flux_fn(Wr[(slice(None),) + tuple(lo)],
                    Wl[(slice(None),) + tuple(hi)], axis=d, gamma=gamma)
        a = [slice(None)] * 4
        b = [slice(None)] * 4
        a[1 + d] = slice(0, -1)
        b[1 + d] = slice(1, None)
        contrib = (F[tuple(a)] - F[tuple(b)]) * inv[d]
        dU = contrib if dU is None else dU + contrib
    return dU


# --------------------------------------------------------------------------
# Face-flux variants (1D/2D divergence + AMR coarse-fine reflux)
# --------------------------------------------------------------------------
#
# These return the per-axis face-flux arrays F (shape ``(n_vars, N_d+1, ...)``
# along axis d) instead of the assembled divergence. The flux divergence
# ``dU/dt = -div.F`` is recovered by :func:`divergence_from_face_fluxes`;
# the 1D/2D branches of :func:`muscl_hancock_divergence` use exactly this
# fluxes+assembly pair. They also let the AMR layer capture the
# *block-boundary* face fluxes (F at index 0 and -1 along each axis) at
# coarse-fine interfaces and apply a Berger-Colella flux correction.
#
# Only the second-order (MUSCL-Hancock) path is provided — the only order AMR
# hydro uses.


def _muscl_hancock_1d_fluxes(W_pad, dt, dWx, *, gamma, riemann_solver, inv_dx, boundary):
    """1D MUSCL-Hancock face fluxes ``(Fx,)`` (shape ``(n_vars, N+1)``): PLM
    predictor + Riemann solve, no divergence assembly."""
    flux_fn = _pick_riemann(riemann_solver)
    Wc = W_pad[:, 1:-1]
    A_dW = apply_primitive_jacobian_1d(Wc, dWx * inv_dx, gamma)
    W_half = Wc - 0.5 * dt * A_dW
    Wr_x = W_half + 0.5 * dWx
    Wl_x = W_half - 0.5 * dWx
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


def _muscl_hancock_2d_fluxes(W_pad, dt, dWx, dWy, *, gamma, riemann_solver, inv_dx, inv_dy, boundary):
    flux_fn = _pick_riemann(riemann_solver)
    Wc = W_pad[:, 1:-1, 1:-1]
    A_dW = apply_primitive_jacobian_2d(Wc, dWx * inv_dx, dWy * inv_dy, gamma)
    W_half = Wc - 0.5 * dt * A_dW

    # ---- X faces (n_vars, Nx+1, Ny) ----
    Wr_x = W_half + 0.5 * dWx
    Wl_x = W_half - 0.5 * dWx
    if boundary == 0:
        WL_x_bc = Wr_x[:, -1:, :]
        WR_x_bc = Wl_x[:, 0:1, :]
    else:
        WL_x_bc = W_pad[:, 0:1, 1:-1]
        WR_x_bc = W_pad[:, -1:, 1:-1]
    WL_x = jnp.concatenate([WL_x_bc, Wr_x], axis=1)
    WR_x = jnp.concatenate([Wl_x, WR_x_bc], axis=1)
    Fx = flux_fn(WL_x, WR_x, axis=0, gamma=gamma)

    # ---- Y faces (n_vars, Nx, Ny+1) ----
    Wr_y = W_half + 0.5 * dWy
    Wl_y = W_half - 0.5 * dWy
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
                             inv_dx, inv_dy, inv_dz, boundary, exact_wrap=False):
    """3D face fluxes — same predictor + reconstruction as ``_muscl_hancock_3d``,
    but each axis assembles the full ``(N+1)`` face-flux array rather than
    streaming the divergence. The block-boundary faces (F[...,0] / F[...,-1])
    are the coarse-fine interface fluxes the reflux step consumes."""
    import jax.lax as lax

    flux_fn = _pick_riemann(riemann_solver)
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
        """Axis-``d`` full (N+1) face-flux array (slope recomputed locally)."""
        lo, hi, first, last = _axis_face_slices(d)
        dW = _SLOPE_3D[d](W_pad, limiter)
        Wr = W_half + 0.5 * dW
        Wl = W_half - 0.5 * dW
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
):
    """Second-order MUSCL-Hancock face fluxes (no divergence assembly).

    Returns a tuple of per-axis face-flux arrays ``(Fx, Fy[, Fz])``; ``Fd`` has
    shape ``(n_vars, ..., N_d + 1, ...)`` (one extra cell along axis ``d``).
    """
    if spatial_dim == 3:
        return _muscl_hancock_3d_fluxes(
            W_pad, dt, gamma=gamma, limiter=limiter, riemann_solver=riemann_solver,
            inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz, boundary=boundary,
            exact_wrap=exact_wrap,
        )
    if spatial_dim == 2:
        dWx, dWy, _ = compute_slopes_from_padded(W_pad, limiter)
        return _muscl_hancock_2d_fluxes(
            W_pad, dt, dWx, dWy, gamma=gamma, riemann_solver=riemann_solver,
            inv_dx=inv_dx, inv_dy=inv_dy, boundary=boundary,
        )
    if spatial_dim == 1:
        dWx, _, _ = compute_slopes_from_padded(W_pad, limiter)
        return _muscl_hancock_1d_fluxes(
            W_pad, dt, dWx, gamma=gamma, riemann_solver=riemann_solver,
            inv_dx=inv_dx, boundary=boundary,
        )
    raise NotImplementedError(f"face-flux variant supports spatial_dim in (1,2,3); got {spatial_dim}")


def riemann_face_flux(W_L, W_R, *, axis, gamma, riemann_solver):
    """Single Riemann flux from a pair of reconstructed face states (var axis 0,
    elementwise over the rest). Used by the AMR same-level seam exchange to
    compute ONE shared flux per block-block face from both blocks' boundary
    reconstructions, so neighbouring blocks telescope (conservation)."""
    return _pick_riemann(riemann_solver)(W_L, W_R, axis=axis, gamma=gamma)


def muscl_hancock_boundary_states(
    W_pad, dt, *, gamma, limiter, spatial_dim, inv_dx, inv_dy, inv_dz,
):
    """Reconstructed primitive face states at the block's boundary cells:
    per-axis tuple ``((L_d, R_d), ...)`` with ``L_d``/``R_d`` the left/right
    reconstructions at the first/last interior cell along axis ``d``, each a
    face slab ``(n_vars, *transverse)``. Uses the same predictor as
    ``muscl_hancock_face_fluxes``, so seam fluxes match the interior."""
    if spatial_dim == 1:
        dWx, _, _ = compute_slopes_from_padded(W_pad, limiter)
        Wc = W_pad[:, 1:-1]
        A_dW = apply_primitive_jacobian_1d(Wc, dWx * inv_dx, gamma)
        W_half = Wc - 0.5 * dt * A_dW
        Lx = (W_half - 0.5 * dWx)[:, 0]
        Rx = (W_half + 0.5 * dWx)[:, -1]
        return ((Lx, Rx),)

    if spatial_dim == 2:
        dWx, dWy, _ = compute_slopes_from_padded(W_pad, limiter)
        Wc = W_pad[:, 1:-1, 1:-1]
        A_dW = apply_primitive_jacobian_2d(Wc, dWx * inv_dx, dWy * inv_dy, gamma)
        W_half = Wc - 0.5 * dt * A_dW
        Lx = (W_half - 0.5 * dWx)[:, 0, :]
        Rx = (W_half + 0.5 * dWx)[:, -1, :]
        Ly = (W_half - 0.5 * dWy)[:, :, 0]
        Ry = (W_half + 0.5 * dWy)[:, :, -1]
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
        Lx = (W_half - 0.5 * dWx)[:, 0, :, :]
        Rx = (W_half + 0.5 * dWx)[:, -1, :, :]
        Ly = (W_half - 0.5 * dWy)[:, :, 0, :]
        Ry = (W_half + 0.5 * dWy)[:, :, -1, :]
        Lz = (W_half - 0.5 * dWz)[:, :, :, 0]
        Rz = (W_half + 0.5 * dWz)[:, :, :, -1]
        return ((Lx, Rx), (Ly, Ry), (Lz, Rz))

    raise NotImplementedError(f"boundary states support spatial_dim in (1,2,3); got {spatial_dim}")


def divergence_from_face_fluxes(fluxes, *, spatial_dim, inv_dx, inv_dy, inv_dz):
    """Recover ``dU/dt = -div.F`` from :func:`muscl_hancock_face_fluxes` output.

    Reproduces the assembly in the matching ``_muscl_hancock_*`` divergence
    functions, so ``divergence_from_face_fluxes(muscl_hancock_face_fluxes(...))``
    equals ``muscl_hancock_divergence(..., order=2)``.
    """
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


# --------------------------------------------------------------------------
# Top-level dispatch
# --------------------------------------------------------------------------


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
):
    """Compute dU/dt = -∇·F(U) using MUSCL-Hancock (second-order) or plain
    Godunov on primitives (first-order, order=0). All arguments after the
    keyword separator are static / closure-friendly for JIT.
    """
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

    # fused concat-free path: W_pad carries TWO ghost layers (see
    # _muscl_hancock_3d_fused). Opt-in via integrators (LUNA_HYDRO_FUSED=1).
    if fused:
        if spatial_dim != 3:
            raise NotImplementedError("fused path is 3D-only")
        return _muscl_hancock_3d_fused(
            W_pad, dt, gamma=gamma, limiter=limiter,
            riemann_solver=riemann_solver,
            inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz)

    # 3D path computes its slopes internally direction-by-direction (memory).
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
        )

    # 1D and 2D paths: face fluxes + divergence assembly (same states, same
    # Riemann calls as the deleted standalone divergence variants).
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
        )
    else:
        raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")
    return divergence_from_face_fluxes(
        fluxes, spatial_dim=spatial_dim, inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz
    )
