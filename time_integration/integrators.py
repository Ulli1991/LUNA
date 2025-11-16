import jax.numpy as jnp
from jax import jit
from hydro.divergence import flux_divergence

def rk2_step(U, dt, dx, dy, gamma, order, limiter):
    k1 = flux_divergence(U, dx, dy, gamma, order, limiter) 
    k2 = flux_divergence(U + 0.5 * dt * k1, dx, dy, gamma, order, limiter)
    return U + dt * k2

def rk4_step(U, dt, dx, dy, gamma, order, limiter):
    k1 = flux_divergence(U, dx, dy, gamma, order, limiter)
    k2 = flux_divergence(U + 0.5*dt*k1, dx, dy, gamma, order, limiter)
    k3 = flux_divergence(U + 0.5*dt*k2, dx, dy, gamma, order, limiter)
    k4 = flux_divergence(U + dt*k3, dx, dy, gamma, order, limiter)
    return U + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def muscl_hancock_step(U, dt, dx, dy, gamma, order, limiter):
    k1 = flux_divergence(U, dx, dy, gamma, order, limiter)
    k2 = flux_divergence(U + dt * k1, dx, dy, gamma, order, limiter)
    return U + dt * 0.5 * (k1 + k2)

def time_step(U, dt, dx, dy, gamma, order, limiter, time_solver):
    """
    Choose the appropriate time integration method based on the config setting.
    """
    if time_solver == 0:  # RK2
        return rk2_step(U, dt, dx, dy, gamma, order, limiter)
    elif time_solver == 1:  # Predictor-Corrector (Hancock)
        return muscl_hancock_step(U, dt, dx, dy, gamma, order, limiter)  # Change to muscl_hancock_step
    elif time_solver == 2:  # RK4
        return rk4_step(U, dt, dx, dy, gamma, order, limiter)
    else:
        raise ValueError(f"Invalid TIME_SOLVER value: {time_solver}")
