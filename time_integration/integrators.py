import jax.numpy as jnp
from jax import jit
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
