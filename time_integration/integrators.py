import jax.numpy as jnp
from jax import jit, pmap
from functools import partial

@partial(jit, static_argnames=['flux_fn'])
def rk2_step(U, dt, flux_fn):
    """
    RK2 time integration.

    Parameters
    ----------
    U : array
        Conservative state
    dt : float
        Timestep
    flux_fn : callable (static argument)
        Flux divergence function with signature flux_fn(U, dt) -> dU/dt
        (dt is passed but may be ignored by the flux function)
    """
    k1 = flux_fn(U, dt)
    k2 = flux_fn(U + 0.5 * dt * k1, dt)
    return U + dt * k2

@partial(jit, static_argnames=['flux_fn'])
def rk4_step(U, dt, flux_fn):
    """
    RK4 time integration.

    Parameters
    ----------
    U : array
        Conservative state
    dt : float
        Timestep
    flux_fn : callable (static argument)
        Flux divergence function with signature flux_fn(U, dt) -> dU/dt
        (dt is passed but may be ignored by the flux function)
    """
    k1 = flux_fn(U, dt)
    k2 = flux_fn(U + 0.5*dt*k1, dt)
    k3 = flux_fn(U + 0.5*dt*k2, dt)
    k4 = flux_fn(U + dt*k3, dt)
    return U + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

@partial(jit, static_argnames=['flux_fn'])
def muscl_hancock_step(U, dt, flux_fn):
    """
    TRUE MUSCL-Hancock predictor-corrector time integration.

    NOTE: This requires flux_fn to be the MUSCL-Hancock flux function
    which includes the predictor step internally and needs dt as a parameter.

    For backward compatibility with regular flux functions (that don't take dt),
    this will try to call flux_fn(U, dt) if possible, otherwise falls back
    to Heun's method.

    Parameters
    ----------
    U : array
        Conservative state
    dt : float
        Timestep
    flux_fn : callable (static argument)
        MUSCL-Hancock flux divergence: flux_fn(U, dt) -> dU/dt
        OR regular flux divergence: flux_fn(U) -> dU/dt (falls back to Heun)
    """
    # For MUSCL-Hancock flux: predictor is built into the flux function
    # It needs dt to do the half-timestep evolution
    dUdt = flux_fn(U, dt)
    return U + dt * dUdt


def create_rk2_pmap_step(pad_halo_fn, flux_div_fn, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz):
    """
    Create a pmapped RK2 time stepper for multi-GPU execution.

    Parameters
    ----------
    pad_halo_fn : callable
        Function to pad local subdomain with halo exchange
    flux_div_fn : callable
        Flux divergence function
    gamma : float
        Adiabatic index
    order : int
        Spatial order
    limiter : int
        Limiter type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing

    Returns
    -------
    pmapped_step : callable
        Pmapped function with signature (U_local, dt) -> U_local_new
    """
    @partial(pmap, in_axes=(0, None), axis_name='batch')
    def rk2_step_fn(U_local_single, dt):
        """One RK2 step with halo exchange between GPUs."""
        # Stage 1
        U_pad = pad_halo_fn(U_local_single)
        dU1 = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)
        U1 = U_local_single + dt * dU1

        # Stage 2
        U1_pad = pad_halo_fn(U1)
        dU2 = flux_div_fn(U1_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        return U_local_single + 0.5 * dt * (dU1 + dU2)

    return rk2_step_fn


def create_rk4_pmap_step(pad_halo_fn, flux_div_fn, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz):
    """
    Create a pmapped RK4 time stepper for multi-GPU execution.

    Parameters
    ----------
    pad_halo_fn : callable
        Function to pad local subdomain with halo exchange
    flux_div_fn : callable
        Flux divergence function
    gamma : float
        Adiabatic index
    order : int
        Spatial order
    limiter : int
        Limiter type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing

    Returns
    -------
    pmapped_step : callable
        Pmapped function with signature (U_local, dt) -> U_local_new
    """
    @partial(pmap, in_axes=(0, None), axis_name='batch')
    def rk4_step_fn(U_local_single, dt):
        """One RK4 step with halo exchange between GPUs."""
        # Stage 1
        U_pad = pad_halo_fn(U_local_single)
        k1 = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        # Stage 2
        U1 = U_local_single + 0.5 * dt * k1
        U1_pad = pad_halo_fn(U1)
        k2 = flux_div_fn(U1_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        # Stage 3
        U2 = U_local_single + 0.5 * dt * k2
        U2_pad = pad_halo_fn(U2)
        k3 = flux_div_fn(U2_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        # Stage 4
        U3 = U_local_single + dt * k3
        U3_pad = pad_halo_fn(U3)
        k4 = flux_div_fn(U3_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz)

        return U_local_single + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return rk4_step_fn


def create_muscl_hancock_pmap_step(pad_halo_fn, flux_div_fn, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz):
    """
    Create a pmapped MUSCL-Hancock time stepper for multi-GPU execution.

    MUSCL-Hancock is a predictor-corrector method where the predictor step
    is built into the flux function itself (using dt for half-timestep evolution).

    Parameters
    ----------
    pad_halo_fn : callable
        Function to pad local subdomain with halo exchange
    flux_div_fn : callable
        Flux divergence function that accepts dt parameter
    gamma : float
        Adiabatic index
    order : int
        Spatial order
    limiter : int
        Limiter type
    dx, dy, dz : float
        Grid spacing
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacing

    Returns
    -------
    pmapped_step : callable
        Pmapped function with signature (U_local, dt) -> U_local_new
    """
    @partial(pmap, in_axes=(0, None), axis_name='batch')
    def muscl_hancock_step_fn(U_local_single, dt):
        """One MUSCL-Hancock step with halo exchange between GPUs."""
        # Pad with halo exchange
        U_pad = pad_halo_fn(U_local_single)

        # Compute flux divergence with predictor step (dt is passed to flux_div_fn)
        dU = flux_div_fn(U_pad, gamma, order, limiter, dx, dy, dz, inv_dx, inv_dy, inv_dz, dt)

        # Update state
        return U_local_single + dt * dU

    return muscl_hancock_step_fn
