"""Run exactly ONE LUNA timestep from the soundwave eigenmode IC, dump result.

Bypasses the runner's warmup so the comparison with AthenaK is clean. The
warmup is intentional for production runs (JIT warm-up) but it advances U,
which would make a 1-step diagnostic ambiguous.

Saves an HDF5 file with the post-step density and the IC for reference,
plus the dt actually used. Designed to be runnable on the login node (CPU)
or on a single GPU.
"""
import os
import sys
import argparse
import numpy as np
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nx", type=int, default=128)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--out", type=str, default="one_step_luna.h5")
    parser.add_argument("--enable-x64", action="store_true", default=True)
    args = parser.parse_args()

    # Set float64 BEFORE any jax operations.
    import jax
    jax.config.update("jax_enable_x64", args.enable_x64)
    import jax.numpy as jnp
    from dataclasses import replace
    from template_config import Config
    from ic_generation.ic import initial_conditions
    from utils.grid_params import GridParams
    from time_integration.sharded_integrators_modular import (
        create_muscl_hancock_modular_step,
    )
    from utils.halo_exchange_auto import pad_with_halo_auto
    from hydro.divergence import flux_divergence_from_padded

    Nx = args.Nx
    LX = 1.0
    GAMMA = 5.0 / 3.0
    cfg = replace(
        Config(),
        test_problem="Soundwave",
        SPATIAL_DIMENSION=1,
        BOUNDARY=0,
        Nx=Nx,
        Lx=LX,
        gamma=GAMMA,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=1,        # MonCen
        CFL=args.cfl,
        soundwave_single_mode=True,
    )

    # Build initial state (cell-centered, single right-going eigenmode).
    U0 = initial_conditions(cfg)
    print(f"Nx={Nx}, U0.shape={U0.shape}, dtype={U0.dtype}")

    grid = GridParams.from_config(cfg)

    def pad_halo_fn(U, boundary, spatial_dim):
        return pad_with_halo_auto(U, boundary, spatial_dim)

    compute_dt_fn, step_fn = create_muscl_hancock_modular_step(
        pad_halo_fn, flux_divergence_from_padded,
        cfg.gamma, cfg.ORDER, cfg.LIMITER,
        grid.dx, grid.dy, grid.dz, grid.inv_dx, grid.inv_dy, grid.inv_dz,
        cfg.BOUNDARY, cfg.SPATIAL_DIMENSION, cfg.CFL,
        sharding=None,
    )

    dt_cfl = float(compute_dt_fn(U0))
    print(f"dt_cfl = {dt_cfl:.16e}")

    U1 = step_fn(U0, dt_cfl)
    jax.block_until_ready(U1)
    U0_np = np.asarray(U0)
    U1_np = np.asarray(U1)

    # Cell centers
    dx = LX / Nx
    x = (np.arange(Nx) + 0.5) * dx

    with h5py.File(args.out, "w") as f:
        f.attrs["Nx"] = Nx
        f.attrs["Lx"] = LX
        f.attrs["dt"] = dt_cfl
        f.attrs["gamma"] = GAMMA
        f.attrs["cfl"] = args.cfl
        f.create_dataset("x", data=x)
        f.create_dataset("rho_initial", data=U0_np[0])
        f.create_dataset("mom_initial", data=U0_np[1])
        f.create_dataset("E_initial",   data=U0_np[2])
        f.create_dataset("rho_after_step", data=U1_np[0])
        f.create_dataset("mom_after_step", data=U1_np[1])
        f.create_dataset("E_after_step",   data=U1_np[2])
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
