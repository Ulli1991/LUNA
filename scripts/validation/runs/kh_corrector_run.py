"""Solver-in-the-loop CNN corrector on KH in AUTODIFF mode (migration of #41).

The production 2D Kelvin-Helmholtz solver, autodiff_mode=True,
autodiff_task="corrector_training": a CNN flux corrector is trained (reverse
mode, through the production coarse step_fn) to make the coarse rollout track a
fine reference (production solver at 4x res, downsampled). Dispatched through
main.run.

Env: LUNA_KHC_NX (coarse N), LUNA_KHC_NSTEPS, LUNA_KHC_NITER, LUNA_KHC_DT.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dataclasses import replace

from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


def main():
    Nc = int(os.environ.get("LUNA_KHC_NX", "32"))
    Ncy = int(os.environ.get("LUNA_KHC_NY", str(Nc)))     # 1:2 box -> set 2*Nc
    n_steps = int(os.environ.get("LUNA_KHC_NSTEPS", "200"))
    n_iter = int(os.environ.get("LUNA_KHC_NITER", "100"))
    dt = float(os.environ.get("LUNA_KHC_DT", "0.004"))
    Ly = 2.0                  # shear layers at y=0.5,1.5; Lx follows aspect (square cells)
    Lx = Ly * Nc / Ncy

    overrides = dict(
        test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, BOUNDARY=0,
        Nx=Nc, Ny=Ncy, Lx=Lx, Ly=Ly, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, CFL=0.4,
        # ---- AUTODIFF: solver-in-the-loop corrector training ----
        autodiff_mode=True,
        autodiff_task="corrector_training",
        autodiff_n_steps=n_steps,
        autodiff_n_iter=n_iter,
        autodiff_fixed_dt=dt,
        PLOT_AFTER_SIM=False, production_mode=False,
        output_dir=f"snapshots/kh_corrector_Nx{Nc}",
    )
    initialize_jax_distributed(enable_x64=True)
    cfg = replace(Config(), **overrides)
    result = run(cfg)
    if isinstance(result, dict) and "corr_err" in result:
        ce = result["corr_err"]; pe = result["coarse_err"]
        print(f"RESULT corrector: coarse-only={pe[-1] if pe else 'nan'} "
              f"+CNN={ce[-1] if ce else 'nan'}", flush=True)


if __name__ == "__main__":
    main()
