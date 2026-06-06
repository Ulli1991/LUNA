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
    """3D MUSCL-Hancock divergence, restructured for memory.

    The previous version held dWx, dWy, dWz alive concurrently (3*N^3
    slopes) plus their finite-difference intermediates (6*N^3) during the
    slope-limiter step — over 200 GB of transient slope state at 1024^3
    fp32. Working set OOMed at 8 GPUs even after fusing all JITs and the
    Riemann reconstruction.

    New structure:
      * Predictor (A_dW) is built direction-by-direction. Each direction's
        slope is computed, contributed to A_dW, then released before the
        next direction starts. Peak slope state shrinks from 3*N^3 to
        1*N^3.
      * Flux phase recomputes each direction's slope inside its own scan
        branch. lax.scan forces serial execution so Wr/Wl/F_*/dU
        intermediates are reused across the three iterations. Slope
        recompute is cheap (2 subtractions + a limiter call) but saves the
        per-direction transients from being alive after the predictor.

    Per-axis flux is still split into (interior, left bdy, right bdy) so
    the (Nx+1)-sized WL/WR concatenated arrays are never materialised.
    """
    import jax.lax as lax

    flux_fn = _pick_riemann(riemann_solver)
    Wc = W_pad[:, 1:-1, 1:-1, 1:-1]

    # ---- Predictor: A_dW = A_x dWx + A_y dWy + A_z dWz, one direction at a time.
    dWx = compute_slope_x_3d_from_padded(W_pad, limiter)
    A_dW = apply_primitive_jacobian_3d_x(Wc, dWx * inv_dx, gamma)
    # dWx is now unreferenced and XLA can release it before the next slope.

    dWy = compute_slope_y_3d_from_padded(W_pad, limiter)
    A_dW = A_dW + apply_primitive_jacobian_3d_y(Wc, dWy * inv_dy, gamma)

    dWz = compute_slope_z_3d_from_padded(W_pad, limiter)
    A_dW = A_dW + apply_primitive_jacobian_3d_z(Wc, dWz * inv_dz, gamma)

    W_half = Wc - 0.5 * dt * A_dW
    # A_dW, dWz no longer needed.

    # ---- Flux phase: per-direction branches, slope recomputed inside each.
    def _x_contrib(_):
        dWx_local = compute_slope_x_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dWx_local
        Wl = W_half - 0.5 * dWx_local
        if boundary == 0:
            WL_bc = Wr[:, -1:, :, :]
            WR_bc = Wl[:, 0:1, :, :]
        else:
            WL_bc = W_pad[:, 0:1, 1:-1, 1:-1]
            WR_bc = W_pad[:, -1:, 1:-1, 1:-1]
        F_int = flux_fn(Wr[:, :-1, :, :], Wl[:, 1:, :, :], axis=0, gamma=gamma)
        F_l = flux_fn(WL_bc, Wl[:, 0:1, :, :], axis=0, gamma=gamma)
        F_r = flux_fn(Wr[:, -1:, :, :], WR_bc, axis=0, gamma=gamma)
        return jnp.concatenate(
            [F_l - F_int[:, 0:1, :, :], F_int[:, :-1, :, :] - F_int[:, 1:, :, :], F_int[:, -1:, :, :] - F_r],
            axis=1,
        ) * inv_dx

    def _y_contrib(_):
        dWy_local = compute_slope_y_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dWy_local
        Wl = W_half - 0.5 * dWy_local
        if boundary == 0:
            WL_bc = Wr[:, :, -1:, :]
            WR_bc = Wl[:, :, 0:1, :]
        else:
            WL_bc = W_pad[:, 1:-1, 0:1, 1:-1]
            WR_bc = W_pad[:, 1:-1, -1:, 1:-1]
        F_int = flux_fn(Wr[:, :, :-1, :], Wl[:, :, 1:, :], axis=1, gamma=gamma)
        F_l = flux_fn(WL_bc, Wl[:, :, 0:1, :], axis=1, gamma=gamma)
        F_r = flux_fn(Wr[:, :, -1:, :], WR_bc, axis=1, gamma=gamma)
        return jnp.concatenate(
            [F_l - F_int[:, :, 0:1, :], F_int[:, :, :-1, :] - F_int[:, :, 1:, :], F_int[:, :, -1:, :] - F_r],
            axis=2,
        ) * inv_dy

    def _z_contrib(_):
        dWz_local = compute_slope_z_3d_from_padded(W_pad, limiter)
        Wr = W_half + 0.5 * dWz_local
        Wl = W_half - 0.5 * dWz_local
        if boundary == 0:
            WL_bc = Wr[:, :, :, -1:]
            WR_bc = Wl[:, :, :, 0:1]
        else:
            WL_bc = W_pad[:, 1:-1, 1:-1, 0:1]
            WR_bc = W_pad[:, 1:-1, 1:-1, -1:]
        F_int = flux_fn(Wr[:, :, :, :-1], Wl[:, :, :, 1:], axis=2, gamma=gamma)
        F_l = flux_fn(WL_bc, Wl[:, :, :, 0:1], axis=2, gamma=gamma)
        F_r = flux_fn(Wr[:, :, :, -1:], WR_bc, axis=2, gamma=gamma)
        return jnp.concatenate(
            [F_l - F_int[:, :, :, 0:1], F_int[:, :, :, :-1] - F_int[:, :, :, 1:], F_int[:, :, :, -1:] - F_r],
            axis=3,
        ) * inv_dz

    def _step(dU_carry, axis_idx):
        contrib = lax.switch(axis_idx, [_x_contrib, _y_contrib, _z_contrib], None)
        return dU_carry + contrib, None

    # Seed the scan with the X-direction contribution rather than a zeros
    # buffer (saves a full N^3 zero seed at 1024^3 fp32).
    dU, _ = lax.scan(_step, _x_contrib(None), jnp.arange(1, 3))
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
