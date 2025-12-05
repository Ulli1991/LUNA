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
        Flux divergence function with signature flux_fn(U) -> dU/dt
    """
    k1 = flux_fn(U)
    k2 = flux_fn(U + 0.5 * dt * k1)
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
        Flux divergence function with signature flux_fn(U) -> dU/dt
    """
    k1 = flux_fn(U)
    k2 = flux_fn(U + 0.5*dt*k1)
    k3 = flux_fn(U + 0.5*dt*k2)
    k4 = flux_fn(U + dt*k3)
    return U + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

@partial(jit, static_argnames=['flux_fn'])
def muscl_hancock_step(U, dt, flux_fn):
    """
    MUSCL-Hancock predictor-corrector time integration.

    Parameters
    ----------
    U : array
        Conservative state
    dt : float
        Timestep
    flux_fn : callable (static argument)
        Flux divergence function with signature flux_fn(U) -> dU/dt
    """
    k1 = flux_fn(U)
    k2 = flux_fn(U + dt * k1)
    return U + dt * 0.5 * (k1 + k2)


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
