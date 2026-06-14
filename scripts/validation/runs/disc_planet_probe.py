"""Diagnostic probe for the Disc-Planet-2D NaN: step the IC with the EXACT
production step + isothermal projection + particle drain, checking finiteness
after each sub-operation, and report the first non-finite cell (operation, index,
radius, local rho/P/v). Pinpoints WHERE the cold disc blows up.

Run on CPU for a few steps:  JAX_PLATFORMS=cpu python -m scripts.validation.runs.disc_planet_probe
"""
import os
import numpy as np

from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed


def prim(U, gamma):
    import jax.numpy as jnp
    rho = U[0]
    vx, vy = U[1] / rho, U[2] / rho
    P = (gamma - 1.0) * (U[3] - 0.5 * rho * (vx ** 2 + vy ** 2))
    return np.asarray(rho), np.asarray(P), np.asarray(vx), np.asarray(vy)


def report(tag, U, cfg, coord):
    import jax.numpy as jnp
    finite = bool(jnp.all(jnp.isfinite(U)))
    rho, P, vx, vy = prim(U, cfg.gamma)
    bad = ~np.isfinite(rho) | ~np.isfinite(P) | ~np.isfinite(vx) | ~np.isfinite(vy)
    msg = (f"[{tag}] finite={finite}  rho[{rho.min():.3e},{rho.max():.3e}]  "
           f"P[{np.nanmin(P):.3e},{np.nanmax(P):.3e}]  "
           f"|v|max={np.nanmax(np.hypot(vx,vy)):.3e}")
    if bad.any():
        ij = np.argwhere(bad)
        i0, j0 = ij[0]
        X, Y = np.meshgrid(coord, coord, indexing="ij")
        R = np.hypot(X[i0, j0], Y[i0, j0])
        msg += f"\n    FIRST BAD cell ({i0},{j0}) R={R:.3f}  ({len(ij)} bad cells)"
    print(msg, flush=True)
    return finite


def main():
    Nx = int(os.environ.get("LUNA_DP_NX", "128"))
    cs = float(os.environ.get("LUNA_DP_CS", "0.05"))
    L = 6.0
    overrides = dict(
        test_problem="Disc-Planet-2D", SPATIAL_DIMENSION=2, BOUNDARY=4,
        Nx=Nx, Ny=Nx, Lx=L, Ly=L, gamma=5.0 / 3.0,
        ORDER=2, LIMITER=1, CFL=0.3, isothermal_cs2=cs * cs,
        particles_enabled=True, GRAVITATIONAL_G=1.0,
        particle_star_mass=1.0, particle_planet_q=1.0e-3, particle_planet_a=1.0,
        particle_softening=0.6 * cs,
        particle_sink_enabled=True, particle_sink_radius=0.0347,
        particle_sink_rate=20.0, particle_sink_on_star=False,
        disc_damping_enabled=True, disc_damp_R_in=0.5, disc_damp_R_out=2.2,
        disc_damp_rate=10.0,
        disc_sigma0=1.0, disc_sigma_index=0.5, disc_R_in=0.4, disc_R_out=2.5,
        disc_floor=1.0e-3, production_mode=True,
    )
    initialize_jax_distributed(enable_x64=True)
    cfg = replace(Config(), **overrides)

    import jax
    from utils.grid_params import GridParams
    from setup.distributed_setup import (create_distributed_mesh,
                                         initialize_simulation_distributed,
                                         create_solver_distributed)
    from setup.runner_distributed import _isothermal_project
    from sources.particles import setup_particles

    mesh, _ = create_distributed_mesh(cfg)
    U, _, _ = initialize_simulation_distributed(cfg, mesh)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, _, sharding = create_solver_distributed(cfg, grid, mesh)
    ps = setup_particles(cfg, sharding, True)
    pos = ps["pos"]

    dx = L / Nx
    coord = (np.arange(Nx) + 0.5) * dx - 0.5 * L
    nsteps = int(os.environ.get("LUNA_DP_PROBE_STEPS", "12"))
    every = max(1, nsteps // 20)
    print(f"Probe: Nx={Nx}, cs={cs}, cs2={cs*cs:.4e}, soft={0.6*cs:.3f}, nsteps={nsteps}")
    report("IC", U, cfg, coord)

    for s in range(nsteps):
        dt = float(compute_dt_fn(U))
        U1 = step_fn(U, dt)
        U1 = _isothermal_project(U1, cfg.isothermal_cs2, cfg.gamma)
        U1, pos = ps["apply"](U1, pos, dt)
        if not bool(np.isfinite(dt)) or (s % every == 0) or s == nsteps - 1:
            if not report(f"step {s:4d} dt={dt:.3e}", U1, cfg, coord):
                break
        elif not bool(__import__("jax").numpy.all(
                __import__("jax").numpy.isfinite(U1))):
            report(f"step {s:4d} dt={dt:.3e} NaN", U1, cfg, coord)
            break
        U = U1


if __name__ == "__main__":
    main()
