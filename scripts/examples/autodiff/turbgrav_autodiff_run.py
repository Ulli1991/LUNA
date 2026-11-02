"""Coupled turbulence + self-gravity IC inversion (diffhydro Fig. 13 analog).

    python -m scripts.examples.autodiff.turbgrav_autodiff_run nx=32 n_steps=200 g=1

The turb_autodiff setup with self-gravity in the loop: OU forcing + a soft
isothermal thermostat, the differentiable FMG-Poisson self-gravity
(custom-vjp adjoint), and reverse-mode AD recovering the initial low-k
density perturbation through turbulent advection and gravitational collapse.

Parameters: nx, n_steps, g (gravitational strength), and the task's options
(ic_inversion.py): n_iter, lr, kmax, slope.
"""
import functools
from dataclasses import replace


from luna.hydro.turbulence import OUDriverConfig  # noqa: E402
from luna.setup.cli import script_main            # noqa: E402
from luna.ic_generation import driven_turbulence_3d  # noqa: E402
from luna.template_config import Config           # noqa: E402
from scripts.examples.autodiff import ic_inversion

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


TASK = functools.partial(ic_inversion.task, n_iter=80, field="density")


def config(nx=32, n_steps=200, g=1.0):
    ou_cfg = OUDriverConfig(k_min=K_MIN, k_max=K_MAX, solenoidal_weight=1.0,
                            autocorrelation_time=T_AC, rms_acceleration=RMS_ACC,
                            seed=OU_SEED, spectrum="parabolic", Lx=LX, Ly=LY, Lz=LZ)
    return replace(
        Config(),
        test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=LX, Ly=LY, Lz=LZ, gamma=GAMMA,
        problem=driven_turbulence_3d.Params(rho0=RHO0, P0=P0),         # dimensionless 1, 1 here
        ORDER=1, LIMITER=1,
        ou_config=ou_cfg, ou_seed=OU_SEED, isothermal_cs2=CS2_ISO,
        SELF_GRAVITY=True, GRAVITATIONAL_G=g, POISSON_BOUNDARY=0,
        autodiff_mode=True, autodiff_task="ic_inversion", autodiff_n_steps=n_steps,
        autodiff_fixed_dt=0.3 * (LX / nx) / 2.0,     # fixed dt (cs + v < 2): clean grads
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/turbgrav_autodiff_Nx{nx}", enable_x64=True,
    )


def report(res):
    r = res.extra
    if isinstance(r, dict) and "ic_residual" in r:
        print(f"RESULT turbgrav ic_residual={r['ic_residual']:.3e}", flush=True)


if __name__ == "__main__":
    script_main(config, report=report, task=TASK)
