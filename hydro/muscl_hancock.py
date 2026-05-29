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
    apply_primitive_jacobian_3d,
)
from hydro.slopes import compute_slopes_from_padded


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


def _muscl_hancock_3d(
    W_pad, dt, dWx, dWy, dWz, *, gamma, riemann_solver, inv_dx, inv_dy, inv_dz, boundary
):
    """3D MUSCL-Hancock divergence with axis-sequential accumulation.

    Predictor (W_half) uses all three slopes once up-front. The directional
    flux contributions are then computed via jax.lax.scan over axis_idx in
    {0, 1, 2}, which forces XLA to reuse the per-axis intermediate buffers
    (Wr_*, Wl_*, WL_*, WR_*, F*) across iterations instead of holding all
    three sets alive simultaneously. Bit-identical to the unrolled version
    when XLA fuses identically.
    """
    import jax.lax as lax

    flux_fn = _pick_riemann(riemann_solver)
    Wc = W_pad[:, 1:-1, 1:-1, 1:-1]
    A_dW = apply_primitive_jacobian_3d(Wc, dWx * inv_dx, dWy * inv_dy, dWz * inv_dz, gamma)
    W_half = Wc - 0.5 * dt * A_dW

    # Per-axis branches — each lifts the original per-direction block
    # verbatim so the order of floating-point ops is preserved.
    def _x_branch(W_pad, W_half, dW_axis):
        Wr = W_half + 0.5 * dW_axis
        Wl = W_half - 0.5 * dW_axis
        if boundary == 0:
            WL_bc = Wr[:, -1:, :, :]
            WR_bc = Wl[:, 0:1, :, :]
        else:
            WL_bc = W_pad[:, 0:1, 1:-1, 1:-1]
            WR_bc = W_pad[:, -1:, 1:-1, 1:-1]
        WL = jnp.concatenate([WL_bc, Wr], axis=1)
        WR = jnp.concatenate([Wl, WR_bc], axis=1)
        F = flux_fn(WL, WR, axis=0, gamma=gamma)
        return (F[:, :-1, :, :] - F[:, 1:, :, :]) * inv_dx

    def _y_branch(W_pad, W_half, dW_axis):
        Wr = W_half + 0.5 * dW_axis
        Wl = W_half - 0.5 * dW_axis
        if boundary == 0:
            WL_bc = Wr[:, :, -1:, :]
            WR_bc = Wl[:, :, 0:1, :]
        else:
            WL_bc = W_pad[:, 1:-1, 0:1, 1:-1]
            WR_bc = W_pad[:, 1:-1, -1:, 1:-1]
        WL = jnp.concatenate([WL_bc, Wr], axis=2)
        WR = jnp.concatenate([Wl, WR_bc], axis=2)
        F = flux_fn(WL, WR, axis=1, gamma=gamma)
        return (F[:, :, :-1, :] - F[:, :, 1:, :]) * inv_dy

    def _z_branch(W_pad, W_half, dW_axis):
        Wr = W_half + 0.5 * dW_axis
        Wl = W_half - 0.5 * dW_axis
        if boundary == 0:
            WL_bc = Wr[:, :, :, -1:]
            WR_bc = Wl[:, :, :, 0:1]
        else:
            WL_bc = W_pad[:, 1:-1, 1:-1, 0:1]
            WR_bc = W_pad[:, 1:-1, 1:-1, -1:]
        WL = jnp.concatenate([WL_bc, Wr], axis=3)
        WR = jnp.concatenate([Wl, WR_bc], axis=3)
        F = flux_fn(WL, WR, axis=2, gamma=gamma)
        return (F[:, :, :, :-1] - F[:, :, :, 1:]) * inv_dz

    dW_stack = jnp.stack([dWx, dWy, dWz])  # (3, 5, Nx, Ny, Nz)

    def _step(dU_carry, axis_idx):
        dW_axis = dW_stack[axis_idx]
        contrib = lax.switch(axis_idx, [_x_branch, _y_branch, _z_branch], W_pad, W_half, dW_axis)
        return dU_carry + contrib, None

    dU0 = jnp.zeros_like(W_half)
    dU, _ = lax.scan(_step, dU0, jnp.arange(3))
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
    if spatial_dim == 3:
        return _muscl_hancock_3d(
            W_pad,
            dt,
            dWx,
            dWy,
            dWz,
            gamma=gamma,
            riemann_solver=riemann_solver,
            inv_dx=inv_dx,
            inv_dy=inv_dy,
            inv_dz=inv_dz,
            boundary=boundary,
        )
    raise NotImplementedError(f"Spatial dimension {spatial_dim} not supported")
