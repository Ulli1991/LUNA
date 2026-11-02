"""The M82 wind in adjoint mode: the production M82 parameter file with
autodiff_mode=True, running the eta_recovery example.

    python -m scripts.examples.autodiff.m82_autodiff_run nx=64 task=eta_recovery n_iter=60

Parameters: nx (Ny = nx, Nz = 2 nx), t_end_myr, selfgrav (differentiable FMG
self-gravity), fixed_dt (> 0 forces a fixed dt; 0 = adaptive stop-gradient
CFL dt), n_steps (step budget, must exceed the steps to t_end), task
(eta_recovery), dt_mode (stopgrad | diff), grad_mode (reverse | forward), and
the task's options (n_iter). MESH_SHAPE follows as a field.
The tests import `config` for the same setup.
"""
from dataclasses import replace


from luna.setup.cli import script_main            # noqa: E402
from luna.ic_generation import m82_disk_3d  # noqa: E402
from luna.template_config import Config           # noqa: E402
from luna.units import UnitSystem                 # noqa: E402

KPC = 3.08568e21
MSUN = 1.98892e33
MYR = 1.0e6 * 3.15576e7


def config(nx=8, t_end_myr=50.0, selfgrav=False, fixed_dt=0.0, n_steps=4000,
           task="eta_recovery", dt_mode="stopgrad", grad_mode="reverse"):
    units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                       UnitVelocity_in_cm_per_s=1.0e5)
    t_end = t_end_myr * MYR / units.UnitTime_in_s
    return replace(
        Config(),
        test_problem="M82-Disk-3D", SPATIAL_DIMENSION=3, BOUNDARY=1,
        Nx=nx, Ny=nx, Nz=2 * nx, Lx=10.0, Ly=10.0, Lz=20.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, units=units, riemann_solver="hllc",
        cc85_enabled=True, cc85_R_star_pc=300.0, cc85_SFR_Msun_per_yr=20.0,
        cc85_eta_M=0.6, cc85_eta_E=0.9, cooling_mu=0.6,
        m82_use_external_gravity=True,
        problem=m82_disk_3d.Params(M_gas=0.25, R_scale_kpc=1.6, R_trunc_kpc=4.5, T_K=1.0e4,
                                   n_ambient_cgs=1.0e-4, T_ambient_K=1.0e6),
        m82_stellar_MN_M=1.0, m82_stellar_MN_a_kpc=0.8, m82_stellar_MN_b_kpc=0.15,
        m82_halo_NFW_M_vir=50.0, m82_halo_NFW_c=20.0,
        cooling_enabled=True, cooling_table="sd93_solar",
        t_end=t_end,
        SELF_GRAVITY=selfgrav, GRAVITATIONAL_G=units.GravityConstant, POISSON_BOUNDARY=0,
        autodiff_mode=True, autodiff_task=task, autodiff_n_steps=n_steps,
        autodiff_fixed_dt=fixed_dt, autodiff_dt_mode=dt_mode, autodiff_grad_mode=grad_mode,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/m82_autodiff_Nx{nx}", enable_x64=True,
    )


def report(res):
    print("RESULT", res.extra, flush=True)


def main(argv=None):
    """`task=` picks the example; its options follow on the same line."""
    import functools
    import sys
    from luna.autodiff.rollout import example_task
    argv = sys.argv[1:] if argv is None else argv
    name = dict(a.split("=", 1) for a in argv if "=" in a).get("task", "eta_recovery")
    return script_main(config, argv, report=report,
                       task=functools.partial(example_task(name), n_iter=60))


if __name__ == "__main__":
    main()
