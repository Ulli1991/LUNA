"""Coupled turbulence + self-gravity IC inversion — diffhydro Fig.13 (top) analog.

The reverse-mode showcase of turb_autodiff_run.py, but with SELF-GRAVITY active
in the loop: the production Driven-Turbulence-3D setup (OU forcing + soft
isothermal thermostat) is evolved with the differentiable FMG-Poisson self-gravity
(custom-vjp adjoint), and reverse-mode AD recovers the initial low-k density
perturbation from the evolved, turbulently-stirred AND gravitationally-amplified
state.

This is diffhydro's headline multiphysics inversion: the gradient must propagate
through turbulent advection *and* gravitational collapse simultaneously — longer-
range couplings and more history dependence than either alone.

Env: LUNA_TG_NX, LUNA_TG_NSTEPS, LUNA_TG_NITER, LUNA_TG_G (grav. strength),
     LUNA_TG_MESH ("2x2x1"), LUNA_TG_SHARDMAP.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dataclasses import replace

from hydro.turbulence import OUDriverConfig
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run

LX = LY = LZ = 1.0
GAMMA = 5.0 / 3.0
RHO0 = P0 = 1.0
CS2_ISO = P0 / RHO0
M_TARGET = 0.3
V_RMS = M_TARGET * (CS2_ISO ** 0.5)
K_MIN, K_MAX = 1.0, 2.0
T_TURNOVER = LX / V_RMS
T_AC = 0.5 * T_TURNOVER
RMS_ACC = V_RMS / T_AC
OU_SEED = 7


def main():
    Nx = int(os.environ.get("LUNA_TG_NX", "32"))
    n_steps = int(os.environ.get("LUNA_TG_NSTEPS", "200"))
    n_iter = int(os.environ.get("LUNA_TG_NITER", "80"))
    G = float(os.environ.get("LUNA_TG_G", "1.0"))
    use_shardmap = os.environ.get("LUNA_TG_SHARDMAP", "0") == "1"
    mesh_env = os.environ.get("LUNA_TG_MESH", "")            # e.g. "2x2x1"
    mesh_shape = tuple(int(v) for v in mesh_env.split("x")) if mesh_env else None

    ou_cfg = OUDriverConfig(
        k_min=K_MIN, k_max=K_MAX, solenoidal_weight=1.0,
        autocorrelation_time=T_AC, rms_acceleration=RMS_ACC,
        seed=OU_SEED, spectrum="parabolic", Lx=LX, Ly=LY, Lz=LZ,
    )
    overrides = dict(
        test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=Nx, Ny=Nx, Nz=Nx, Lx=LX, Ly=LY, Lz=LZ, gamma=GAMMA,
        ORDER=1, LIMITER=1, CFL=0.4,
        ou_config=ou_cfg, ou_seed=OU_SEED, isothermal_cs2=CS2_ISO,
        # ---- coupled physics: turbulence AND self-gravity (differentiable) ----
        SELF_GRAVITY=True, GRAVITATIONAL_G=G, POISSON_BOUNDARY=0,
        # ---- AUTODIFF RUN MODE: reverse-mode IC inversion ----
        autodiff_mode=True,
        autodiff_task="ic_inversion",
        autodiff_n_steps=n_steps,
        autodiff_fixed_dt=0.3 * (LX / Nx) / 2.0,   # fixed dt (cs+v ~ <2): clean grads
        autodiff_n_iter=n_iter,
        autodiff_ic_field="density",
        MESH_SHAPE=mesh_shape,
        PLOT_AFTER_SIM=False, production_mode=False,
        output_dir=f"snapshots/turbgrav_autodiff_Nx{Nx}",
    )
    initialize_jax_distributed(enable_x64=True)
    cfg = replace(Config(), **overrides)
    result = run(cfg).extra
    if isinstance(result, dict) and "ic_residual" in result:
        print(f"RESULT turbgrav ic_residual={result['ic_residual']:.3e}", flush=True)


if __name__ == "__main__":
    main()
