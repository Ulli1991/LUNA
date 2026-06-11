"""
Primitive-variable Jacobians for the unsplit MUSCL-Hancock predictor.

The predictor advances cell-centred PRIMITIVE states by a half step:

    W^{n+1/2} = W^n − (Δt/2) · Σ_d  A_d(W^n) · ∂W/∂x_d

with the sum running over all spatial directions — the transverse terms
are included, making the scheme unsplit (Saltzman 1994). This is the
same predictor as RAMSES's `muscl` scheme (Teyssier 2002, A&A 385, 337,
§2.3.1: PLM slopes on primitives + unsplit transverse derivatives, then
a Riemann solve on the reconstructed face states).

The apply_primitive_jacobian_3d_{x,y,z} variants compute one direction's
contribution at a time so the memory-conscious 3D path never holds all
three slope arrays alive simultaneously.

(A flux-difference predictor variant — evaluating the nonlinear flux at
reconstructed face states instead of the Jacobian-times-slope product —
briefly lived here as primitive_flux_* but was never wired into an
integrator; removed, see git history.)
"""

import jax.numpy as jnp


def apply_primitive_jacobian_1d(W, dW, gamma):
    """
    Compute A_x * dW for 1D Euler equations in primitive variables.

    W = [ρ, vx, P]
    A_x * dW advances the solution in time using characteristic information.

    Parameters
    ----------
    W : array, shape (3, ...)
        Primitive variables [ρ, vx, P]
    dW : array, shape (3, ...)
        Slopes [dρ, dvx, dP]
    gamma : float
        Adiabatic index

    Returns
    -------
    A_x * dW : array, shape (3, ...)
    """
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
    """
    Compute A_x * dWx + A_y * dWy for 2D Euler equations in primitive variables.

    W = [ρ, vx, vy, P]
    This combines contributions from both spatial directions for the predictor step.

    Parameters
    ----------
    W : array, shape (4, Nx, Ny)
        Primitive variables [ρ, vx, vy, P]
    dWx : array, shape (4, Nx, Ny)
        X-direction slopes [dρ/dx, dvx/dx, dvy/dx, dP/dx]
    dWy : array, shape (4, Nx, Ny)
        Y-direction slopes [dρ/dy, dvx/dy, dvy/dy, dP/dy]
    gamma : float
        Adiabatic index

    Returns
    -------
    A_x * dWx + A_y * dWy : array, shape (4, Nx, Ny)
    """
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


def apply_primitive_jacobian_3d_x(W, dWx, gamma):
    """A_x · ∂_x W contribution to the predictor. Computed alone (no Y/Z
    slopes needed alive) so the memory-conscious MUSCL-Hancock 3D path
    can build A_dW direction-by-direction."""
    rho, vx, vy, vz, P = W
    drho_dx, dvx_dx, dvy_dx, dvz_dx, dP_dx = dWx
    rho_c2 = gamma * P
    inv_rho = 1.0 / rho
    return jnp.stack([
        vx * drho_dx + rho * dvx_dx,
        vx * dvx_dx + inv_rho * dP_dx,
        vx * dvy_dx,
        vx * dvz_dx,
        rho_c2 * dvx_dx + vx * dP_dx,
    ])


def apply_primitive_jacobian_3d_y(W, dWy, gamma):
    rho, vx, vy, vz, P = W
    drho_dy, dvx_dy, dvy_dy, dvz_dy, dP_dy = dWy
    rho_c2 = gamma * P
    inv_rho = 1.0 / rho
    return jnp.stack([
        vy * drho_dy + rho * dvy_dy,
        vy * dvx_dy,
        vy * dvy_dy + inv_rho * dP_dy,
        vy * dvz_dy,
        rho_c2 * dvy_dy + vy * dP_dy,
    ])


def apply_primitive_jacobian_3d_z(W, dWz, gamma):
    rho, vx, vy, vz, P = W
    drho_dz, dvx_dz, dvy_dz, dvz_dz, dP_dz = dWz
    rho_c2 = gamma * P
    inv_rho = 1.0 / rho
    return jnp.stack([
        vz * drho_dz + rho * dvz_dz,
        vz * dvx_dz,
        vz * dvy_dz,
        vz * dvz_dz + inv_rho * dP_dz,
        rho_c2 * dvz_dz + vz * dP_dz,
    ])


def apply_primitive_jacobian_3d(W, dWx, dWy, dWz, gamma):
    """
    Compute A_x * dWx + A_y * dWy + A_z * dWz for 3D Euler equations.

    W = [ρ, vx, vy, vz, P]

    Parameters
    ----------
    W : array, shape (5, Nx, Ny, Nz)
        Primitive variables [ρ, vx, vy, vz, P]
    dWx, dWy, dWz : arrays, shape (5, Nx, Ny, Nz)
        Slopes in each direction
    gamma : float
        Adiabatic index

    Returns
    -------
    A_x * dWx + A_y * dWy + A_z * dWz : array, shape (5, Nx, Ny, Nz)
    """
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
