"""IC inversion (the ic_inversion example): the gravity, cooling and Sedov cases.

    python -m scripts.examples.autodiff.ic_inversion_run problem=jeans nx=32
    python -m scripts.examples.autodiff.ic_inversion_run problem=cooling nx=32 mach=0.3
    python -m scripts.examples.autodiff.ic_inversion_run problem=sedov nx=32 MESH_SHAPE=2,1,1

  jeans:   Jeans-Instability-3D + self-gravity; recover an IC density
           perturbation evolved under the differentiable FMG Poisson solve.
  cooling: a broadband density IC advected by OU-driven turbulence while
           radiative cooling acts (diffhydro sec. 7.5): advection keeps the
           large scales recoverable, cooling and mixing erase the small ones.
  sedov:   recover an IC energy perturbation from the blast.

All differentiate w.r.t. the full IC field (reverse mode) and emit the
recovery + r(k) figure. Parameters: problem, nx, n_steps, mach and n_turn
(cooling case), and the task's options (ic_inversion.py): n_iter, lr, kmax,
slope, field (energy for sedov, else density).
"""
import functools
from dataclasses import replace


from luna.hydro.turbulence import OUDriverConfig  # noqa: E402
from luna.setup.cli import script_main            # noqa: E402
from luna.ic_generation import driven_turbulence_3d, jeans3d  # noqa: E402
from luna.template_config import Config           # noqa: E402
from luna.units import UnitSystem                 # noqa: E402
from scripts.examples.autodiff import ic_inversion

KPC = 3.08568e21
MSUN = 1.98892e33


def config(problem="jeans", nx=32, n_steps=150, mach=0.3, n_turn=1.0):
    common = dict(
        SPATIAL_DIMENSION=3, Nx=nx, Ny=nx, Nz=nx, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1,
        autodiff_mode=True, autodiff_task="ic_inversion",
        autodiff_n_steps=n_steps,
        autodiff_fixed_dt=0.0, autodiff_dt_mode="diff",
        PLOT_AFTER_SIM=False, enable_x64=True,
    )
    if problem == "jeans":
        L = 1.0
        ov = dict(
            test_problem="Jeans-Instability-3D", BOUNDARY=0,
            Lx=L, Ly=L, Lz=L, t_end=0.3,             # a fraction of the dynamical time
            problem=jeans3d.Params(rho0=1.0, P0=1.0, amp=0.0, kx=1, ky=0, kz=0),
            SELF_GRAVITY=True, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
            # fixed dt: "diff" adaptive dt differentiates through a min over
            # cells whose argmin kinks corrupt the gradient at 64^3
            autodiff_fixed_dt=0.3 * (L / nx) / 2.0,
            output_dir=f"snapshots/jeans_autodiff_Nx{nx}",
        )
    elif problem == "cooling":
        # the old cooling box's regime (n_H ~ 0.05, T ~ 2e5 K near the SD93
        # peak, above the 1e4 K floor), now driven so the IC is advected; no
        # thermostat: gamma = 5/3 + radiative cooling set the thermodynamics
        units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                           UnitVelocity_in_cm_per_s=1.0e5)
        gamma = 5.0 / 3.0
        rho0, p0 = 1.76e-4, 0.485
        cs = (gamma * p0 / rho0) ** 0.5                      # ~68 km/s
        L = 0.05                                             # 50 pc box (kpc)
        v_rms = mach * cs
        t_ac = 0.5 * L / v_rms
        ou_cfg = OUDriverConfig(k_min=1.0, k_max=2.0, solenoidal_weight=1.0,
                                autocorrelation_time=t_ac, rms_acceleration=v_rms / t_ac,
                                seed=7, spectrum="parabolic", Lx=L, Ly=L, Lz=L)
        ov = dict(
            test_problem="Driven-Turbulence-3D", BOUNDARY=0, units=units,
            Lx=L, Ly=L, Lz=L, gamma=gamma, problem=driven_turbulence_3d.Params(rho0=rho0, P0=p0),
            ou_config=ou_cfg, ou_seed=7,
            autodiff_fixed_dt=0.3 * (L / nx) / (cs * (1.0 + mach)),
            SELF_GRAVITY=False, cooling_enabled=True, cooling_table="sd93_solar",
            output_dir=f"snapshots/cooling_autodiff_Nx{nx}",
        )
    elif problem == "sedov":
        units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                           UnitVelocity_in_cm_per_s=1.0e5)
        L = 50.0e-3                                          # 50 pc in kpc
        ov = dict(
            test_problem="Sedov-Blast-3D", BOUNDARY=1, units=units,
            Lx=L, Ly=L, Lz=L, t_end=1.0e-5,                  # ~10 kyr
            output_dir=f"snapshots/sedov_autodiff_Nx{nx}",
        )
    else:
        raise ValueError(f"problem={problem!r}: jeans, cooling or sedov")
    return replace(Config(), **{**common, **ov})


def report(res):
    r = res.extra
    if isinstance(r, dict) and "ic_residual" in r:
        print(f"RESULT ic_residual={r['ic_residual']:.3e}", flush=True)


def main(argv=None):
    """`problem=` picks the case and the perturbed field; the task's options
    follow on the same line."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    problem = dict(a.split("=", 1) for a in argv if "=" in a).get("problem", "jeans")
    task = functools.partial(ic_inversion.task, n_iter=80,
                             field="energy" if problem == "sedov" else "density")
    return script_main(config, argv, report=report, task=task)


if __name__ == "__main__":
    main()
