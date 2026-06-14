"""IC inversion through the orchestrator for the GRAVITY and SEDOV examples
(migrations of #42 and #38) — autodiff_task="ic_inversion" via main.run.

  * gravity (Jeans-Instability-3D + self-gravity): recover an IC density
    perturbation that evolves under the differentiable FMG Poisson self-gravity.
  * sedov  (Sedov-Blast-3D): recover an IC ENERGY perturbation from the blast.

Both differentiate w.r.t. the full IC field (reverse mode) and emit the
recovery + r(k) figure.

Env: LUNA_AD_PROBLEM (jeans|sedov), LUNA_AD_NX, LUNA_AD_NSTEPS, LUNA_AD_NITER.
"""
import os
import sys

sys.path.insert(0, "/raven/u/uli/jax-hydro")

from dataclasses import replace

from units import UnitSystem
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run

KPC = 3.08568e21
MSUN = 1.98892e33


def main():
    problem = os.environ.get("LUNA_AD_PROBLEM", "jeans")
    Nx = int(os.environ.get("LUNA_AD_NX", "32"))
    n_steps = int(os.environ.get("LUNA_AD_NSTEPS", "150"))
    n_iter = int(os.environ.get("LUNA_AD_NITER", "80"))

    common = dict(
        SPATIAL_DIMENSION=3, Nx=Nx, Ny=Nx, Nz=Nx, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, CFL=0.4,
        autodiff_mode=True, autodiff_task="ic_inversion",
        autodiff_n_steps=n_steps, autodiff_n_iter=n_iter,
        PLOT_AFTER_SIM=False, production_mode=False,
    )

    # adaptive (stop-grad-free "diff") dt for both: no CFL guessing, can't crash
    common.update(autodiff_fixed_dt=0.0, autodiff_dt_mode="diff")

    if problem == "jeans":
        L = 1.0
        ov = dict(
            test_problem="Jeans-Instability-3D", BOUNDARY=0,
            Lx=L, Ly=L, Lz=L, t_end=0.3,    # a fraction of the dynamical time
            jeans_rho0=1.0, jeans_P0=1.0, jeans_amp=0.0,
            jeans_kx=1, jeans_ky=0, jeans_kz=0,
            SELF_GRAVITY=True, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
            autodiff_ic_field="density",
            output_dir=f"snapshots/jeans_autodiff_Nx{Nx}",
        )
    elif problem == "cooling":
        # Reverse-mode IC reconstruction through the RADIATIVE COOLING solver
        # (covers the diffhydro "radiatively cooling flows" autodiff context).
        # Uniform box (no gravity) at n_H~0.05 cm^-3, T~2e5 K — cooling-active
        # (above the 1e4 K floor, near the SD93 peak). The orchestrator perturbs
        # the energy IC; cooling (Λ(T) ∝ n² and strongly T-dependent) imprints it
        # on the evolved state, and reverse-mode recovers it.
        units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                           UnitVelocity_in_cm_per_s=1.0e5)
        MYR = 1.0e6 * 3.15576e7
        L = float(os.environ.get("LUNA_AD_L_KPC", "0.05"))      # 50 pc box
        t_Myr = float(os.environ.get("LUNA_AD_TEND_MYR", "0.1"))  # ~half a t_cool
        ov = dict(
            test_problem="Jeans-Instability-3D", BOUNDARY=0, units=units,
            Lx=L, Ly=L, Lz=L, t_end=t_Myr * MYR / units.UnitTime_in_s,
            jeans_rho0=1.76e-4, jeans_P0=0.485, jeans_amp=0.0,
            jeans_kx=1, jeans_ky=0, jeans_kz=0, jeans_eigenmode=False,
            SELF_GRAVITY=False,
            cooling_enabled=True, cooling_table="sd93_solar",
            autodiff_ic_field="energy",
            output_dir=f"snapshots/cooling_autodiff_Nx{Nx}",
        )
    else:  # sedov
        units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                           UnitVelocity_in_cm_per_s=1.0e5)
        L = 50.0e-3                    # 50 pc in kpc
        ov = dict(
            test_problem="Sedov-Blast-3D", BOUNDARY=1, units=units,
            Lx=L, Ly=L, Lz=L, t_end=1.0e-5,   # ~10 kyr (blast develops)
            autodiff_ic_field="energy",
            output_dir=f"snapshots/sedov_autodiff_Nx{Nx}",
        )

    initialize_jax_distributed(enable_x64=True)
    cfg = replace(Config(), **common, **ov)
    result = run(cfg)
    if isinstance(result, dict) and "ic_residual" in result:
        print(f"RESULT {problem} ic_residual={result['ic_residual']:.3e}",
              flush=True)


if __name__ == "__main__":
    main()
