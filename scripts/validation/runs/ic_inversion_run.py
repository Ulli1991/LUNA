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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dataclasses import replace

from units import UnitSystem
from template_config import Config
from hydro.turbulence import OUDriverConfig
from setup.distributed_setup import initialize_jax_distributed
from main import run

KPC = 3.08568e21
MSUN = 1.98892e33


def main():
    problem = os.environ.get("LUNA_AD_PROBLEM", "jeans")
    Nx = int(os.environ.get("LUNA_AD_NX", "32"))
    n_steps = int(os.environ.get("LUNA_AD_NSTEPS", "150"))
    n_iter = int(os.environ.get("LUNA_AD_NITER", "80"))
    use_shardmap = os.environ.get("LUNA_AD_SHARDMAP", "0") == "1"
    mesh_env = os.environ.get("LUNA_AD_MESH", "")            # e.g. "2x2x1"
    mesh_shape = tuple(int(v) for v in mesh_env.split("x")) if mesh_env else None

    common = dict(
        SPATIAL_DIMENSION=3, Nx=Nx, Ny=Nx, Nz=Nx, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, CFL=0.4,
        autodiff_mode=True, autodiff_task="ic_inversion",
        autodiff_n_steps=n_steps, autodiff_n_iter=n_iter,
        MESH_SHAPE=mesh_shape,
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
            # FIXED dt (like turbgrav/cooling): "diff"-mode adaptive dt
            # differentiates through a min-over-cells whose argmin kinks scale
            # with resolution — at 64^3 the dt-gradient path corrupts the whole
            # gradient (FD check fails, optimization diverges). cs=1.29, v<<cs.
            autodiff_fixed_dt=0.3 * (L / Nx) / 2.0,
            autodiff_ic_field="density",
            output_dir=f"snapshots/jeans_autodiff_Nx{Nx}",
        )
    elif problem == "cooling":
        # Reverse-mode IC reconstruction through a TURBULENT, RADIATIVELY-COOLING
        # FLOW — diffhydro §7.5 ("Turbulent field with Cooling"). The physically
        # correct setup: a broadband density IC is ADVECTED by an OU-driven
        # turbulent flow while radiative cooling acts (Λ(T) ∝ n²). Advection
        # transports large-scale structure (recoverable) while cooling + turbulent
        # mixing erase small-scale information — so r(k) stays ~1 to a genuine k*
        # then collapses at the dissipative/mixing limit.
        #
        # (The old setup — perturbing energy in a STATIC uniform box then cooling —
        # is ill-posed: with no flow, cooling is a pointwise contraction toward the
        # 10^4 K floor, a many-to-one map with no recoverable inverse. Measured
        # residual 1.18 > 1.0 null, non-monotonic r(k). Fixed by adding the flow.)
        units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                           UnitVelocity_in_cm_per_s=1.0e5)
        # PHYSICALLY-CONSISTENT turbulent cooling flow. The regime is the old
        # cooling box's (forward-stable, cooling-active: n_H~0.05 cm^-3, T~2e5 K,
        # near the SD93 peak, above the 1e4 K floor) — but now DRIVEN turbulent so
        # a broadband density IC is advected through it. NO isothermal thermostat
        # (that fought the cooling and NaN'd); gamma=5/3 adiabatic + radiative
        # cooling set the thermodynamics.
        gamma = 5.0 / 3.0
        rho0, p0 = 1.76e-4, 0.485                  # code units -> n_H~0.05, T~2e5 K
        cs = (gamma * p0 / rho0) ** 0.5            # ~68 km/s (code velocity)
        L = 0.05                                   # 50 pc box (kpc)
        mach = float(os.environ.get("LUNA_AD_MACH", "0.3"))    # subsonic: clean grads
        v_rms = mach * cs
        t_turnover = L / v_rms                     # code time
        t_ac = 0.5 * t_turnover
        n_turn = float(os.environ.get("LUNA_AD_NTURN", "1.0"))  # evolve ~1 turnover
        ou_cfg = OUDriverConfig(
            k_min=1.0, k_max=2.0, solenoidal_weight=1.0,
            autocorrelation_time=t_ac, rms_acceleration=v_rms / t_ac,
            seed=7, spectrum="parabolic", Lx=L, Ly=L, Lz=L,
        )
        ov = dict(
            test_problem="Driven-Turbulence-3D", BOUNDARY=0, units=units,
            Lx=L, Ly=L, Lz=L, gamma=gamma,
            ou_config=ou_cfg, ou_seed=7,               # NO isothermal_cs2: cooling rules
            autodiff_fixed_dt=0.3 * (L / Nx) / (cs * (1.0 + mach)),  # CFL-safe fixed dt
            SELF_GRAVITY=False,
            cooling_enabled=True, cooling_table="sd93_solar",
            autodiff_ic_field="density",              # recover the advected IC
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
    # merge first (ov overrides common) so a per-problem key that also appears in
    # common — e.g. gamma / autodiff_fixed_dt for the turbulent-cooling setup —
    # doesn't trip replace()'s "multiple values for keyword" error.
    cfg = replace(Config(), **{**common, **ov})
    result = run(cfg)
    if isinstance(result, dict) and "ic_residual" in result:
        print(f"RESULT {problem} ic_residual={result['ic_residual']:.3e}",
              flush=True)


if __name__ == "__main__":
    main()
