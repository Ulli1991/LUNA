"""MUSCL-Hancock flux divergence (first- and second-order).

Provides the per-step flux divergence dU/dt = -∇·F(U) used by the production
TIME_SOLVER=1 path. The second-order variant adds the MUSCL-Hancock predictor
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
    """Return hllc_flux_primitive callable for the requested solver."""
    if riemann_solver == "lhllc":
        from hydro.fluxes_lhllc import lhllc_flux_primitive as flux_fn
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


def _muscl_hancock_3d(W_pad, dt, *, gamma, limiter, riemann_solver, inv_dx, inv_dy, inv_dz, boundary):
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
        # Boundary states for the wrap (or BC) face. Use the halo W (cell-
        # centered) for *all* boundary types — this is correct under
        # shard_map (the halo brings the actual neighbour shard's edge cell).
        # For single-process periodic, this loses half-order accuracy at the
        # wrap face vs the previous `Wr[-1:]` reconstruction, but only at
        # that single face (one cell per axis) — negligible for smooth
        # flows. Previous behaviour `Wr[:, -1:, :, :]` was a *local* wrap,
        # which under multi-process injected a small spurious velocity at
        # every shard boundary every step (Jeans linear-theory breakdown).
        WL_bc = W_pad[:, 0:1, 1:-1, 1:-1]
        WR_bc = W_pad[:, -1:, 1:-1, 1:-1]
        F_int = flux_fn(Wr[:, :-1, :, :], Wl[:, 1:, :, :], axis=0, gamma=gamma)
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
        WL_bc = W_pad[:, 1:-1, 0:1, 1:-1]
        WR_bc = W_pad[:, 1:-1, -1:, 1:-1]
        F_int = flux_fn(Wr[:, :, :-1, :], Wl[:, :, 1:, :], axis=1, gamma=gamma)
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
        WL_bc = W_pad[:, 1:-1, 1:-1, 0:1]
        WR_bc = W_pad[:, 1:-1, 1:-1, -1:]
        F_int = flux_fn(Wr[:, :, :, :-1], Wl[:, :, :, 1:], axis=2, gamma=gamma)
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
