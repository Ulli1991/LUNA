# main_autodiff_grad.py
import jax
import jax.numpy as jnp
from functools import partial
from config import Config
from ic_generation.ic import initial_conditions
from time_integration.integrators import rk2_step, rk4_step, muscl_hancock_step
from hydro.divergence import flux_divergence
from time_integration.cfl import compute_dt

# -------------------------
# Autodiff-ready simulation
# -------------------------
def run_simulation(U0, cfg: Config):
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else None
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else None

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy if dy else None
    inv_dz = 1.0 / dz if dz else None

    flux_fn = partial(
        flux_divergence,
        gamma=cfg.gamma,
        order=cfg.ORDER,
        limiter=cfg.LIMITER,
        boundary=cfg.BOUNDARY,
        dx=dx, dy=dy, dz=dz,
        inv_dx=inv_dx, inv_dy=inv_dy, inv_dz=inv_dz
    )

    time_solver_map = {
        0: rk2_step,
        1: muscl_hancock_step,
        2: rk4_step
    }
    time_step_fn = partial(time_solver_map[cfg.TIME_SOLVER], flux_fn=flux_fn)

    def step_fn(carry, _):
        t, U = carry
        dt = compute_dt(U, cfg.gamma, cfg.CFL, dx, dy, dz)
        dt = jnp.minimum(dt, cfg.t_end - t)
        U_next = time_step_fn(U, dt)
        t_next = t + dt
        return (t_next, U_next), None

    n_steps = 10000  # upper bound
    (t_final, U_final), _ = jax.lax.scan(step_fn, (cfg.t_start, U0), None, length=n_steps)
    return U_final

# -------------------------
# Observable: perturbation energy in the sound wave
# -------------------------
def observable(U_final):
    rho = U_final[0]
    u = U_final[1]
    E = U_final[2]  # assuming ideal gas with [rho, rho*u, E]
    # For a simple 1D soundwave, total perturbation energy
    kinetic = 0.5 * rho * u**2
    return jnp.sum(kinetic + (E - 0.5*rho*u**2))  # sum of kinetic + internal

# -------------------------
# Run with gradient
# -------------------------
def main():
    cfg = Config()

    U0 = initial_conditions(cfg)
    sim_fn = jax.jit(run_simulation)
    grad_fn = jax.jit(jax.grad(lambda U: observable(sim_fn(U, cfg))))

    grads = grad_fn(U0)
    print("Gradient shape:", grads.shape)
    print("Gradient (example slice):", grads[:, :10])  # first 10 cells

if __name__ == "__main__":
    main()

