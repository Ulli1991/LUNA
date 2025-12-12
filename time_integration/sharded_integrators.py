"""
Time integration schemes for sharded arrays with automatic sharding.

These factory functions create time stepping functions that work with
sharded arrays distributed across multiple devices, similar to the pmap
version in integrators.py but for automatic sharding.
"""
import jax


def create_rk2_sharded_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                             dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim):
    """
    Create an RK2 time stepper for sharded arrays with automatic sharding.

    RK2 handles time integration externally, so the flux function does NOT
    receive dt (no MUSCL-Hancock predictor step).

    Parameters
    ----------
    pad_halo_fn : callable
        Halo exchange function with signature (U, boundary, spatial_dim) -> U_padded
    flux_div_fn : callable
        Flux divergence function (flux_divergence_from_padded)
    gamma : float
        Adiabatic index
    order : int
        Spatial order (0 or 1)
    limiter : int
        Limiter type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing
    boundary : int
        Boundary condition type (captured in closure)
    spatial_dim : int
        Spatial dimension (captured in closure)

    Returns
    -------
    time_step_fn : callable
        RK2 time stepping function with signature (U, dt) -> U_new
    """
    @jax.jit(donate_argnums=(0,))
    def time_step_rk2(U, dt):
        """RK2 time step with sharded arrays."""
        # Stage 1
        U_pad = pad_halo_fn(U, boundary, spatial_dim)
        dU1 = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)
        U1 = U + dt * dU1

        # Stage 2
        U1_pad = pad_halo_fn(U1, boundary, spatial_dim)
        dU2 = flux_div_fn(U1_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        return U + 0.5 * dt * (dU1 + dU2)

    return time_step_rk2


def create_muscl_hancock_sharded_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                                       dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim):
    """
    Create a MUSCL-Hancock time stepper for sharded arrays with automatic sharding.

    MUSCL-Hancock is a predictor-corrector method where the predictor step
    is built into the flux function itself (using dt for half-timestep evolution).

    Parameters
    ----------
    pad_halo_fn : callable
        Halo exchange function with signature (U, boundary, spatial_dim) -> U_padded
    flux_div_fn : callable
        Flux divergence function (flux_divergence_from_padded) that accepts dt parameter
    gamma : float
        Adiabatic index
    order : int
        Spatial order (0 or 1)
    limiter : int
        Limiter type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing
    boundary : int
        Boundary condition type (captured in closure)
    spatial_dim : int
        Spatial dimension (captured in closure)

    Returns
    -------
    time_step_fn : callable
        MUSCL-Hancock time stepping function with signature (U, dt) -> U_new
    """
    @jax.jit(donate_argnums=(0,))
    def time_step_muscl_hancock(U, dt):
        """MUSCL-Hancock predictor-corrector time step with sharded arrays."""
        # Pad with halo exchange
        U_pad = pad_halo_fn(U, boundary, spatial_dim)

        # Compute flux divergence with predictor step (dt is passed!)
        dU = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz, dt)

        # Update state
        return U + dt * dU

    return time_step_muscl_hancock


def create_rk4_sharded_step(pad_halo_fn, flux_div_fn, gamma, order, limiter,
                             dx, dy, dz, inv_dx, inv_dy, inv_dz, boundary, spatial_dim):
    """
    Create an RK4 time stepper for sharded arrays with automatic sharding.

    RK4 handles time integration externally, so the flux function does NOT
    receive dt (no MUSCL-Hancock predictor step).

    Parameters
    ----------
    pad_halo_fn : callable
        Halo exchange function with signature (U, boundary, spatial_dim) -> U_padded
    flux_div_fn : callable
        Flux divergence function (flux_divergence_from_padded)
    gamma : float
        Adiabatic index
    order : int
        Spatial order (0 or 1)
    limiter : int
        Limiter type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing
    boundary : int
        Boundary condition type (captured in closure)
    spatial_dim : int
        Spatial dimension (captured in closure)

    Returns
    -------
    time_step_fn : callable
        RK4 time stepping function with signature (U, dt) -> U_new
    """
    @jax.jit(donate_argnums=(0,))
    def time_step_rk4(U, dt):
        """RK4 time step with sharded arrays."""
        # Stage 1
        U_pad = pad_halo_fn(U, boundary, spatial_dim)
        k1 = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        # Stage 2
        U1 = U + 0.5 * dt * k1
        U1_pad = pad_halo_fn(U1, boundary, spatial_dim)
        k2 = flux_div_fn(U1_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        # Stage 3
        U2 = U + 0.5 * dt * k2
        U2_pad = pad_halo_fn(U2, boundary, spatial_dim)
        k3 = flux_div_fn(U2_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        # Stage 4
        U3 = U + dt * k3
        U3_pad = pad_halo_fn(U3, boundary, spatial_dim)
        k4 = flux_div_fn(U3_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        return U + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return time_step_rk4


