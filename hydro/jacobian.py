"""Primitive-variable Jacobians for the unsplit MUSCL-Hancock predictor:
W^{n+1/2} = W^n − (Δt/2) · Σ_d A_d(W^n) · ∂W/∂x_d (transverse terms
included, Saltzman-style unsplit). The _3d_{x,y,z} variants compute one
direction at a time so the memory-conscious path holds one slope array.
"""

import jax.numpy as jnp


def apply_primitive_jacobian_1d(W, dW, gamma):
    """A_x · dW for 1D Euler in primitives W = [ρ, vx, P]; returns (3, ...)."""
    rho, vx, P = W
    drho, dvx, dP = dW

    # Sound speed squared (only c² is needed in Jacobian)
    c2 = gamma * P / rho

    # A_x * dW for primitive variables
    # A_x = [vx   ρ    0  ]
    #       [0    vx  1/ρ ]
    #       [0   ρc²  vx  ]

    result_rho = vx * drho + rho * dvx
    result_vx = vx * dvx + (1.0 / rho) * dP
    result_P = rho * c2 * dvx + vx * dP

    return jnp.stack([result_rho, result_vx, result_P])


def apply_primitive_jacobian_2d(W, dWx, dWy, gamma):
    """A_x · dWx + A_y · dWy for 2D Euler in primitives W = [ρ, vx, vy, P];
    returns (4, Nx, Ny)."""
    rho, vx, vy, P = W
    drho_dx, dvx_dx, dvy_dx, dP_dx = dWx
    drho_dy, dvx_dy, dvy_dy, dP_dy = dWy

    # Sound speed squared (only c² is needed in Jacobian)
    c2 = gamma * P / rho
    inv_rho = 1.0 / rho

    # A_x * dWx where A_x = [vx   ρ    0    0  ]
    #                        [0    vx   0   1/ρ ]
    #                        [0    0    vx   0  ]
    #                        [0   ρc²   0    vx ]

    Ax_dWx_rho = vx * drho_dx + rho * dvx_dx
    Ax_dWx_vx = vx * dvx_dx + inv_rho * dP_dx
    Ax_dWx_vy = vx * dvy_dx
    Ax_dWx_P = rho * c2 * dvx_dx + vx * dP_dx

    # A_y * dWy where A_y = [vy   0    ρ    0  ]
    #                        [0    vy   0    0  ]
    #                        [0    0    vy  1/ρ ]
    #                        [0    0   ρc²   vy ]

    Ay_dWy_rho = vy * drho_dy + rho * dvy_dy
    Ay_dWy_vx = vy * dvx_dy
    Ay_dWy_vy = vy * dvy_dy + inv_rho * dP_dy
    Ay_dWy_P = rho * c2 * dvy_dy + vy * dP_dy

    # Sum contributions from both directions
    result_rho = Ax_dWx_rho + Ay_dWy_rho
    result_vx = Ax_dWx_vx + Ay_dWy_vx
    result_vy = Ax_dWx_vy + Ay_dWy_vy
    result_P = Ax_dWx_P + Ay_dWy_P

    return jnp.stack([result_rho, result_vx, result_vy, result_P])


def _apply_primitive_jacobian_3d_axis(W, dW, gamma, d):
    """A_d · ∂_d W for one axis ``d`` (0=x, 1=y, 2=z) of the 3D predictor.
    Shared body of the three per-direction wrappers below."""
    rho = W[0]
    P = W[4]
    vn = W[1 + d]
    rho_c2 = gamma * P
    inv_rho = 1.0 / rho
    rows = [vn * dW[0] + rho * dW[1 + d]]
    for k in range(3):
        row = vn * dW[1 + k]
        if k == d:
            row = row + inv_rho * dW[4]
        rows.append(row)
    rows.append(rho_c2 * dW[1 + d] + vn * dW[4])
    return jnp.stack(rows)


def apply_primitive_jacobian_3d_x(W, dWx, gamma):
    """A_x · ∂_x W contribution to the predictor. Computed alone (no Y/Z
    slopes needed alive) so the memory-conscious MUSCL-Hancock 3D path
    can build A_dW direction-by-direction."""
    return _apply_primitive_jacobian_3d_axis(W, dWx, gamma, 0)


def apply_primitive_jacobian_3d_y(W, dWy, gamma):
    """A_y · ∂_y W contribution to the predictor (see the x variant)."""
    return _apply_primitive_jacobian_3d_axis(W, dWy, gamma, 1)


def apply_primitive_jacobian_3d_z(W, dWz, gamma):
    """A_z · ∂_z W contribution to the predictor (see the x variant)."""
    return _apply_primitive_jacobian_3d_axis(W, dWz, gamma, 2)


def apply_primitive_jacobian_3d(W, dWx, dWy, dWz, gamma):
    """A_x · dWx + A_y · dWy + A_z · dWz for 3D Euler in primitives
    W = [ρ, vx, vy, vz, P]; returns (5, Nx, Ny, Nz)."""
    rho, vx, vy, vz, P = W
    drho_dx, dvx_dx, dvy_dx, dvz_dx, dP_dx = dWx
    drho_dy, dvx_dy, dvy_dy, dvz_dy, dP_dy = dWy
    drho_dz, dvx_dz, dvy_dz, dvz_dz, dP_dz = dWz

    # Sound speed squared (only c² is needed in Jacobian)
    rho_c2 = gamma * P
    inv_rho = 1.0 / rho

    # Fused per-component expressions — no 15 named direction-x-component
    # intermediates kept alive simultaneously. XLA can choose to fuse the
    # whole stack body into a single elementwise kernel.
    result_rho = (
        vx * drho_dx + rho * dvx_dx + vy * drho_dy + rho * dvy_dy + vz * drho_dz + rho * dvz_dz
    )
    result_vx = vx * dvx_dx + inv_rho * dP_dx + vy * dvx_dy + vz * dvx_dz
    result_vy = vx * dvy_dx + vy * dvy_dy + inv_rho * dP_dy + vz * dvy_dz
    result_vz = vx * dvz_dx + vy * dvz_dy + vz * dvz_dz + inv_rho * dP_dz
    result_P = (
        rho_c2 * dvx_dx + vx * dP_dx + rho_c2 * dvy_dy + vy * dP_dy + rho_c2 * dvz_dz + vz * dP_dz
    )

    return jnp.stack([result_rho, result_vx, result_vy, result_vz, result_P])
