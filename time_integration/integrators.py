import jax.numpy as jnp
from jax import jit
from functools import partial

from hydro.divergence import flux_divergence

@partial(jit, static_argnums=(3, 4, 5))
def rk2_step(U, dt, gamma, order, limiter, boundary, dx, dy=None, dz=None):
    k1 = flux_divergence(U, gamma, order, limiter, boundary, dx, dy, dz) 
    k2 = flux_divergence(U + 0.5 * dt * k1, gamma, order, limiter, boundary, dx, dy, dz)
    return U + dt * k2

@partial(jit, static_argnums=(3, 4, 5))
def rk4_step(U, dt, gamma, order, limiter, boundary, dx, dy=None, dz=None):
    k1 = flux_divergence(U, gamma, order, limiter, boundary, dx, dy, dz)
    k2 = flux_divergence(U + 0.5*dt*k1, gamma, order, limiter, boundary, dx, dy, dz)
    k3 = flux_divergence(U + 0.5*dt*k2, gamma, order, limiter, boundary, dx, dy, dz)
    k4 = flux_divergence(U + dt*k3, gamma, order, limiter, boundary, dx, dy, dz)
    return U + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

@partial(jit, static_argnums=(3, 4, 5))
def muscl_hancock_step(U, dt, gamma, order, limiter, boundary, dx, dy=None, dz=None):
    k1 = flux_divergence(U, gamma, order, limiter, boundary, dx, dy, dz)
    k2 = flux_divergence(U + dt * k1, gamma, order, limiter, boundary, dx, dy, dz)
    return U + dt * 0.5 * (k1 + k2)

@partial(jit, static_argnums=(3, 4, 5, 6))
def time_step(U, dt, gamma, order, limiter, time_solver, boundary, dx, dy=None, dz=None):
    """
    Choose the appropriate time integration method based on the config setting.
    """
    if time_solver == 0:  # RK2
        return rk2_step(U, dt, gamma, order, limiter, boundary, dx, dy, dz)
    elif time_solver == 1:  # Predictor-Corrector (Hancock)
        return muscl_hancock_step(U, dt, gamma, order, limiter, boundary, dx, dy, dz)  # Change to muscl_hancock_step
    elif time_solver == 2:  # RK4
        return rk4_step(U, dt, gamma, order, limiter, boundary, dx, dy, dz)
    else:
        raise ValueError(f"Invalid TIME_SOLVER value: {time_solver}")
