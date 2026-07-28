"""M82 wind in AUTODIFF mode: the production M82 parameter file with
autodiff_mode=True, dispatched through main.run (the SAME entry path as the
HPC code) — which selects run_simulation_autodiff instead of
run_simulation_distributed. Same IC, hydro step, source machinery, and
sharding as the HPC M82 run; just the differentiable scan-loop runner.
Reports d ln L_X / d ln eta.

Env: LUNA_M82_NX, LUNA_M82_TEND (Myr), LUNA_M82_SELFGRAV, LUNA_M82_FWDONLY.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dataclasses import replace

from units import UnitSystem
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run

KPC = 3.08568e21
MSUN = 1.98892e33


MYR = 1.0e6 * 3.15576e7


def main():
    Nx = int(os.environ.get("LUNA_M82_NX", "8"))
    selfgrav = os.environ.get("LUNA_M82_SELFGRAV", "0") == "1"
    t_end_Myr = float(os.environ.get("LUNA_M82_TEND", "50.0"))
    # adaptive (stop_gradient CFL dt) by default; set LUNA_M82_FIXEDDT>0 to force fixed dt
    fixed_dt = float(os.environ.get("LUNA_M82_FIXEDDT", "0.0"))
    # max step budget (coasts once t_end reached); must exceed steps-to-t_end
    n_steps_max = int(os.environ.get("LUNA_M82_NSTEPS", "4000"))
    task = os.environ.get("LUNA_M82_TASK", "xray_sensitivity")
    n_iter = int(os.environ.get("LUNA_M82_NITER", "60"))
    dt_mode = os.environ.get("LUNA_M82_DTMODE", "stopgrad")   # stopgrad | diff
    grad_mode = os.environ.get("LUNA_M82_GRADMODE", "reverse")  # reverse | forward
    use_shardmap = os.environ.get("LUNA_M82_SHARDMAP", "0") == "1"
    mesh_env = os.environ.get("LUNA_M82_MESH", "")            # e.g. "2x2x2" (x-sep: sbatch --export eats commas)
    mesh_shape = tuple(int(v) for v in mesh_env.split("x")) if mesh_env else None

    units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                       UnitVelocity_in_cm_per_s=1.0e5)
    t_end_code = t_end_Myr * MYR / units.UnitTime_in_s

    overrides = dict(
        test_problem="M82-Disk-3D", SPATIAL_DIMENSION=3, BOUNDARY=1,
        Nx=Nx, Ny=Nx, Nz=2 * Nx, Lx=10.0, Ly=10.0, Lz=20.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, CFL=0.3, units=units, riemann_solver="hllc",
        cc85_enabled=True, cc85_R_star_pc=300.0, cc85_SFR_Msun_per_yr=20.0,
        cc85_eta_M=0.6, cc85_eta_E=0.9, cc85_n_ambient_cgs=1.0e-4,
        cc85_T_ambient_K=1.0e6, cc85_mu=0.6,
        m82_use_external_gravity=True, m82_disk_M_gas=0.25,
        m82_disk_R_scale_kpc=1.6, m82_disk_R_trunc_kpc=4.5, m82_disk_T_K=1.0e4,
        m82_stellar_MN_M=1.0, m82_stellar_MN_a_kpc=0.8, m82_stellar_MN_b_kpc=0.15,
        m82_halo_NFW_M_vir=50.0, m82_halo_NFW_c=20.0,
        cooling_enabled=True, cooling_table="sd93_solar",
        t_end=t_end_code,
        # self-gravity (differentiable FMG Poisson), optional
        SELF_GRAVITY=selfgrav,
        GRAVITATIONAL_G=units.GravityConstant,
        POISSON_BOUNDARY=0,
        # ---- AUTODIFF RUN MODE (adaptive stop_gradient CFL dt) ----
        autodiff_mode=True,
        autodiff_task=task,
        autodiff_n_steps=n_steps_max,
        autodiff_fixed_dt=fixed_dt,
        autodiff_dt_mode=dt_mode,
        autodiff_grad_mode=grad_mode,
        autodiff_n_iter=n_iter,
        MESH_SHAPE=mesh_shape,
        PLOT_AFTER_SIM=False, production_mode=False,
        output_dir=f"snapshots/m82_autodiff_Nx{Nx}",
    )
    # Same entry path as the HPC code: build cfg, then main.run dispatches on
    # cfg.autodiff_mode to the differentiable runner.
    initialize_jax_distributed(enable_x64=True)
    cfg = replace(Config(), **overrides)
    result = run(cfg).extra
    if isinstance(result, dict) and "dlnLX_dlneta" in result:
        print("RESULT dlnLX/dlneta =", result["dlnLX_dlneta"], flush=True)
    elif isinstance(result, dict) and "all_finite" in result:
        print(f"RESULT forward-only all_finite={result['all_finite']}", flush=True)


if __name__ == "__main__":
    main()
