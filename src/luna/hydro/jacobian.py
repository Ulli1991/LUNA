"""Primitive-variable Jacobian products A_d · ∂_d W for the unsplit MUSCL-Hancock predictor."""

import jax.numpy as jnp


def apply_primitive_jacobian_1d(W, dW, gamma):
    """A_x · dW for 1D Euler, W = [ρ, vx, P]."""
    return jnp.stack(primitive_jacobian_rows_1d(W, dW, gamma))


def primitive_jacobian_rows_1d(W, dW, gamma):
    """The rows of A_x · dW (1D) as a list; W and dW unpack row by row (arrays or Pallas tiles)."""
    rho, vx, P = W
    drho, dvx, dP = dW

    c2 = gamma * P / rho

    # A_x = [vx   ρ    0  ]
    #       [0    vx  1/ρ ]
    #       [0   ρc²  vx  ]

    result_rho = vx * drho + rho * dvx
    result_vx = vx * dvx + (1.0 / rho) * dP
    result_P = rho * c2 * dvx + vx * dP

    return [result_rho, result_vx, result_P]


def apply_primitive_jacobian_2d(W, dWx, dWy, gamma):
    """A_x · dWx + A_y · dWy for 2D Euler, W = [ρ, vx, vy, P]."""
    return jnp.stack(primitive_jacobian_rows_2d(W, dWx, dWy, gamma))


def primitive_jacobian_rows_2d(W, dWx, dWy, gamma):
    """The rows of A_x · dWx + A_y · dWy (2D) as a list, as primitive_jacobian_rows_1d."""
    rho, vx, vy, P = W
    drho_dx, dvx_dx, dvy_dx, dP_dx = dWx
    drho_dy, dvx_dy, dvy_dy, dP_dy = dWy

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

    result_rho = Ax_dWx_rho + Ay_dWy_rho
    result_vx = Ax_dWx_vx + Ay_dWy_vx
    result_vy = Ax_dWx_vy + Ay_dWy_vy
    result_P = Ax_dWx_P + Ay_dWy_P

    return [result_rho, result_vx, result_vy, result_P]


def _apply_primitive_jacobian_3d_axis(W, dW, gamma, d):
    """A_d · ∂_d W for axis ``d`` (0=x, 1=y, 2=z) of 3D Euler."""
    return jnp.stack(primitive_jacobian_rows_3d_axis(W, dW, gamma, d))


def primitive_jacobian_rows_3d_axis(W, dW, gamma, d):
    """The five rows of A_d · ∂_d W (3D) as a list; rows of W and dW past the fifth are ignored."""
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
    return rows


def apply_primitive_jacobian_3d_x(W, dWx, gamma):
    """A_x · ∂_x W for 3D Euler."""
    return _apply_primitive_jacobian_3d_axis(W, dWx, gamma, 0)


def apply_primitive_jacobian_3d_y(W, dWy, gamma):
    """A_y · ∂_y W for 3D Euler."""
    return _apply_primitive_jacobian_3d_axis(W, dWy, gamma, 1)


def apply_primitive_jacobian_3d_z(W, dWz, gamma):
    """A_z · ∂_z W for 3D Euler."""
    return _apply_primitive_jacobian_3d_axis(W, dWz, gamma, 2)
