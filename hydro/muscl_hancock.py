"""MUSCL-Hancock flux divergence (first- and second-order).

Provides the per-step flux divergence dU/dt = -∇·F(U) used by the production
integrator. The second-order variant adds the MUSCL-Hancock predictor
(Jacobian form: W_half = Wc - ½dt·A·∇W) and PLM reconstruction.

Memory-optimized: the 3D second-order branch accumulates per-direction into
a single running dU so direction-specific intermediates only live during
their own block. The HLLC flux helpers themselves (in hydro/fluxes.py) are
per-component to avoid stacked star-state arrays.
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


def _muscl_hancock_1d(W_pad, dt, dWx, *, gamma, riemann_solver, inv_dx, boundary):
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
    return -(Fx[:, 1:] - Fx[:, :-1]) * inv_dx


def _muscl_hancock_2d(W_pad, dt, dWx, dWy, *, gamma, riemann_solver, inv_dx, inv_dy, boundary):
    flux_fn = _pick_riemann(riemann_solver)
    Wc = W_pad[:, 1:-1, 1:-1]
    A_dW = apply_primitive_jacobian_2d(Wc, dWx * inv_dx, dWy * inv_dy, gamma)
    W_half = Wc - 0.5 * dt * A_dW

    # ---- X direction ----
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
    dU = (Fx[:, :-1, :] - Fx[:, 1:, :]) * inv_dx

    # ---- Y direction ----
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
    dU = dU + (Fy[:, :, :-1] - Fy[:, :, 1:]) * inv_dy

    return dU


def _muscl_hancock_3d(W_pad, dt, *, gamma, limiter, riemann_solver, inv_dx, inv_dy, inv_dz, boundary,
                      exact_wrap=False):
    """3D unsplit MUSCL-Hancock — full transverse predictor, streaming flux phase.

    Predictor (A_dW = A_x dWx + A_y dWy + A_z dWz) is built one direction at
    a time into a running accumulator with `optimization_barrier` between
    contributions. This forces XLA to commit the partial A_dW and release
    the previous direction's slope + Jacobian intermediates before computing
    the next. Slopes are never alive concurrently for the predictor.

    W_half = Wc - 0.5*dt*A_dW carries the full transverse correction (this
    is the unsplit Colella scheme, NOT dim-split — bit-equivalent to the
    original triple-slope version up to floating-point reduction order).

    Flux phase: each direction's sweep is a self-contained closure that
    recomputes its own slope from W_pad, reconstructs Wr/Wl from W_half,
    computes fluxes (interior + 2 boundary), and returns its per-direction
    flux divergence. The three sweep results are summed sequentially with
    barriers between, so per-direction Wr/Wl/F intermediates from the
    previous sweep are released before the next starts.

    Per-axis flux split into (interior, left bdy, right bdy) so the
    (Nx+1)-sized WL/WR concatenated arrays are never materialised.
    """
    import jax.lax as lax

    flux_fn = _pick_riemann(riemann_solver)
    Wc = W_pad[:, 1:-1, 1:-1, 1:-1]

    # ---- Predictor: build full A_dW one direction at a time, with barriers.
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

    # ---- Flux phase: per-direction sweeps, slope recomputed inside each.
    def _x_sweep():
        dW = compute_slope_x_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dW
        Wl = W_half - 0.5 * dW
        F_int = flux_fn(Wr[:, :-1, :, :], Wl[:, 1:, :, :], axis=0, gamma=gamma)
        if exact_wrap:
            # Single-process periodic: the wrap seam is ONE physical face
            # (between cell N-1 and cell 0). Compute its flux once from the
            # reconstructed states on both sides and use it at both ends —
            # the flux sum then telescopes exactly (mass/momentum/energy
            # conservation to machine precision) and the seam stays 2nd order.
            F_face = flux_fn(Wr[:, -1:, :, :], Wl[:, 0:1, :, :], axis=0, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            # Boundary states for the BC / shard face. Use the halo W (cell-
            # centered) — correct under shard_map (the halo brings the actual
            # neighbour shard's edge cell) and for physical BCs (the ghost
            # carries the BC state). Costs half-order accuracy and exact
            # telescoping at that single face; under shard_map the proper fix
            # is exchanging reconstructed face states (future work).
            WL_bc = W_pad[:, 0:1, 1:-1, 1:-1]
            WR_bc = W_pad[:, -1:, 1:-1, 1:-1]
            F_l = flux_fn(WL_bc, Wl[:, 0:1, :, :], axis=0, gamma=gamma)
            F_r = flux_fn(Wr[:, -1:, :, :], WR_bc, axis=0, gamma=gamma)
        return jnp.concatenate(
            [F_l - F_int[:, 0:1, :, :], F_int[:, :-1, :, :] - F_int[:, 1:, :, :], F_int[:, -1:, :, :] - F_r],
            axis=1,
        ) * inv_dx

    def _y_sweep():
        dW = compute_slope_y_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dW
        Wl = W_half - 0.5 * dW
        F_int = flux_fn(Wr[:, :, :-1, :], Wl[:, :, 1:, :], axis=1, gamma=gamma)
        if exact_wrap:
            F_face = flux_fn(Wr[:, :, -1:, :], Wl[:, :, 0:1, :], axis=1, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            WL_bc = W_pad[:, 1:-1, 0:1, 1:-1]
            WR_bc = W_pad[:, 1:-1, -1:, 1:-1]
            F_l = flux_fn(WL_bc, Wl[:, :, 0:1, :], axis=1, gamma=gamma)
            F_r = flux_fn(Wr[:, :, -1:, :], WR_bc, axis=1, gamma=gamma)
        return jnp.concatenate(
            [F_l - F_int[:, :, 0:1, :], F_int[:, :, :-1, :] - F_int[:, :, 1:, :], F_int[:, :, -1:, :] - F_r],
            axis=2,
        ) * inv_dy

    def _z_sweep():
        dW = compute_slope_z_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dW
        Wl = W_half - 0.5 * dW
        F_int = flux_fn(Wr[:, :, :, :-1], Wl[:, :, :, 1:], axis=2, gamma=gamma)
        if exact_wrap:
            F_face = flux_fn(Wr[:, :, :, -1:], Wl[:, :, :, 0:1], axis=2, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            WL_bc = W_pad[:, 1:-1, 1:-1, 0:1]
            WR_bc = W_pad[:, 1:-1, 1:-1, -1:]
            F_l = flux_fn(WL_bc, Wl[:, :, :, 0:1], axis=2, gamma=gamma)
            F_r = flux_fn(Wr[:, :, :, -1:], WR_bc, axis=2, gamma=gamma)
        return jnp.concatenate(
            [F_l - F_int[:, :, :, 0:1], F_int[:, :, :, :-1] - F_int[:, :, :, 1:], F_int[:, :, :, -1:] - F_r],
            axis=3,
        ) * inv_dz

    # Strictly sequential: each sweep returns dU contribution, gets summed,
    # barrier forces materialisation before the next sweep starts.
    dU = lax.optimization_barrier(_x_sweep())
    dU = lax.optimization_barrier(dU + _y_sweep())
    dU = dU + _z_sweep()
    return dU


def _muscl_hancock_3d_fused(W_pad2, dt, *, gamma, limiter, riemann_solver,
                            inv_dx, inv_dy, inv_dz):
    """Concat-free 3D MUSCL-Hancock: needs a TWO-ghost pad.

    Reconstructs (slopes + predictor) on interior PLUS one ghost ring, then
    each axis does ONE flux call over all N+1 faces and assembles the
    divergence by slicing -- no `concatenate`, no boundary special-casing
    (ghosts carry the BC / neighbour-shard content, so shard seams get full
    second-order reconstructed states). Measured +17% over the concat-
    assembled path at 512^3 FP64 on MI300A (scaling/hydro_audit.py); the
    slope/predictor overspill into the ghost ring costs (1+2/N)^3 extra work.

    Restrictions: order=1 (MUSCL) path only; caller gates on boundary 0/1
    (reflective/diode outer-ghost fixes are single-layer in the halo).
    """
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
# Face-flux variants (for AMR coarse-fine reflux)
# --------------------------------------------------------------------------
#
# These return the per-axis face-flux arrays F (shape ``(n_vars, N_d+1, ...)``
# along axis d) instead of the assembled divergence. The flux divergence
# ``dU/dt = -div.F`` is recovered by :func:`divergence_from_face_fluxes`, which
# reproduces :func:`muscl_hancock_divergence` (order=2) exactly. They exist so
# the AMR layer can capture the *block-boundary* face fluxes (F at index 0 and
# -1 along each axis) at coarse-fine interfaces and apply a Berger-Colella flux
# correction. The production divergence path above is left untouched.
#
# Only the second-order (MUSCL-Hancock) path is provided — the only order AMR
# hydro uses. The state construction mirrors the matching ``_muscl_hancock_*``
# divergence functions line-for-line, so the two stay bit-consistent.


def _muscl_hancock_1d_fluxes(W_pad, dt, dWx, *, gamma, riemann_solver, inv_dx, boundary):
    """1D face fluxes for the AMR path — identical predictor + Riemann solve as
    :func:`_muscl_hancock_1d`, but returns the face-flux array ``(Fx,)`` (shape
    ``(n_vars, N+1)``) instead of assembling the divergence, so the AMR reflux /
    seam layer can consume the raw faces. Bit-consistent with the divergence
    variant (same slopes, same Jacobian half-step, same ``flux_fn``)."""
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

    def _x_faces():
        dW = compute_slope_x_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dW
        Wl = W_half - 0.5 * dW
        F_int = flux_fn(Wr[:, :-1, :, :], Wl[:, 1:, :, :], axis=0, gamma=gamma)
        if exact_wrap:
            F_face = flux_fn(Wr[:, -1:, :, :], Wl[:, 0:1, :, :], axis=0, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            WL_bc = W_pad[:, 0:1, 1:-1, 1:-1]
            WR_bc = W_pad[:, -1:, 1:-1, 1:-1]
            F_l = flux_fn(WL_bc, Wl[:, 0:1, :, :], axis=0, gamma=gamma)
            F_r = flux_fn(Wr[:, -1:, :, :], WR_bc, axis=0, gamma=gamma)
        return jnp.concatenate([F_l, F_int, F_r], axis=1)

    def _y_faces():
        dW = compute_slope_y_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dW
        Wl = W_half - 0.5 * dW
        F_int = flux_fn(Wr[:, :, :-1, :], Wl[:, :, 1:, :], axis=1, gamma=gamma)
        if exact_wrap:
            F_face = flux_fn(Wr[:, :, -1:, :], Wl[:, :, 0:1, :], axis=1, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            WL_bc = W_pad[:, 1:-1, 0:1, 1:-1]
            WR_bc = W_pad[:, 1:-1, -1:, 1:-1]
            F_l = flux_fn(WL_bc, Wl[:, :, 0:1, :], axis=1, gamma=gamma)
            F_r = flux_fn(Wr[:, :, -1:, :], WR_bc, axis=1, gamma=gamma)
        return jnp.concatenate([F_l, F_int, F_r], axis=2)

    def _z_faces():
        dW = compute_slope_z_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dW
        Wl = W_half - 0.5 * dW
        F_int = flux_fn(Wr[:, :, :, :-1], Wl[:, :, :, 1:], axis=2, gamma=gamma)
        if exact_wrap:
            F_face = flux_fn(Wr[:, :, :, -1:], Wl[:, :, :, 0:1], axis=2, gamma=gamma)
            F_l = F_face
            F_r = F_face
        else:
            WL_bc = W_pad[:, 1:-1, 1:-1, 0:1]
            WR_bc = W_pad[:, 1:-1, 1:-1, -1:]
            F_l = flux_fn(WL_bc, Wl[:, :, :, 0:1], axis=2, gamma=gamma)
            F_r = flux_fn(Wr[:, :, :, -1:], WR_bc, axis=2, gamma=gamma)
        return jnp.concatenate([F_l, F_int, F_r], axis=3)

    return _x_faces(), _y_faces(), _z_faces()


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
    """Reconstructed primitive face states at the block's boundary cells.

    Returns a per-axis tuple ``((L_0, R_0), (L_1, R_1), ...)`` where ``L_d`` is
    the left-reconstruction ``W_half - 0.5 dW_d`` at the *first* interior cell
    along axis ``d`` and ``R_d`` the right-reconstruction ``W_half + 0.5 dW_d``
    at the *last*; each is a face slab ``(n_vars, *transverse)`` (axis ``d``
    collapsed). The predictor ``W_half`` is identical to the one in
    ``muscl_hancock_face_fluxes`` (same slopes, same Jacobian sum order), so a
    seam flux built from these states matches the block's own interior
    reconstruction at that face.
    """
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
        )

    # 1D and 2D paths still precompute slopes (small N, memory not critical).
    dWx, dWy, dWz = compute_slopes_from_padded(W_pad, limiter)

    if spatial_dim == 1:
        return _muscl_hancock_1d(
            W_pad,
            dt,
            dWx,
            gamma=gamma,
            riemann_solver=riemann_solver,
            inv_dx=inv_dx,
            boundary=boundary,
        )
    if spatial_dim == 2:
        return _muscl_hancock_2d(
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
    raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")
