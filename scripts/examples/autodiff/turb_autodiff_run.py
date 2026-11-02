"""Driven ISM turbulence in adjoint mode: the reverse-mode showcase.

    python -m scripts.examples.autodiff.turb_autodiff_run nx=32 n_steps=200 n_iter=80
    python -m scripts.examples.autodiff.turb_autodiff_run nx=64 cool=none MESH_SHAPE=2,1,1

A ~100 pc periodic two-phase ISM box (Arepo units) at n_H ~ 1 cm^-3,
T ~ 1e4 K, driven by OU forcing and cooled by the combined Koyama-Inutsuka +
Sutherland-Dopita curve. Reverse-mode AD through the production solver
recovers the initial low-k density perturbation from the evolved state.

Parameters: nx, n_steps, n_h (cm^-3), t_k (K), l_pc (box), mach,
cool (combined | koyama_inutsuka | sd93_solar | none), and the task's options
(ic_inversion.py): n_iter, lr, kmax, slope.
"""
import functools
from dataclasses import replace


from luna.hydro.turbulence import OUDriverConfig  # noqa: E402
from luna.setup.cli import script_main            # noqa: E402
from luna.ic_generation import driven_turbulence_3d  # noqa: E402
from luna.template_config import Config           # noqa: E402
from luna.units import UnitSystem                 # noqa: E402
from scripts.examples.autodiff import ic_inversion

KPC = 3.08568e21
MSUN = 1.98892e33
M_H = 1.6726e-24            # g
K_B = 1.3807e-16            # erg/K
MU_RHO = 1.4                # mean mass per H (incl. He): rho from n_H
MU_P = 0.6                  # mean mass per particle: pressure
GAMMA = 5.0 / 3.0


TASK = functools.partial(ic_inversion.task, n_iter=80, field="density")


def config(nx=32, n_steps=200, n_h=1.0, t_k=1.0e4, l_pc=100.0, mach=2.0,
           cool="combined"):
    cool = cool or "none"          # the command line reads `cool=none` as None
    units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                       UnitVelocity_in_cm_per_s=1.0e5)
    u_dens = (1.0e10 * MSUN) / KPC ** 3
    u_pres = u_dens * 1.0e10
    L = l_pc * 1.0e-3                                         # pc -> kpc
    rho0 = (n_h * MU_RHO * M_H) / u_dens
    P0 = ((n_h * MU_RHO) / MU_P * K_B * t_k) / u_pres
    cs = (GAMMA * P0 / rho0) ** 0.5                           # km/s
    v_rms = mach * cs
    T_turn = L / v_rms
    T_ac = 0.5 * T_turn
    dt = 0.3 * (L / nx) / (cs + v_rms)                        # CFL-safe fixed dt
    ou_cfg = OUDriverConfig(k_min=1.0, k_max=2.0, solenoidal_weight=1.0,
                            autocorrelation_time=T_ac, rms_acceleration=v_rms / T_ac,
                            seed=7, spectrum="parabolic", Lx=L, Ly=L, Lz=L)
    print(f"ISM turb box: L={l_pc} pc, n_H={n_h} cm^-3, T={t_k:.1e} K -> "
          f"rho0={rho0:.3e}, P0={P0:.3e}, cs={cs:.1f} km/s, Mach={mach}, cooling={cool}",
          flush=True)
    return replace(
        Config(),
        test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=L, Ly=L, Lz=L, gamma=GAMMA, units=units,
        problem=driven_turbulence_3d.Params(rho0=rho0, P0=P0),         # the state printed above, applied
        ORDER=1, LIMITER=1, ou_config=ou_cfg, ou_seed=7,
        cooling_enabled=(cool != "none"),
        cooling_table=(cool if cool != "none" else "sd93_solar"),
        autodiff_mode=True, autodiff_task="ic_inversion", autodiff_n_steps=n_steps,
        autodiff_fixed_dt=dt,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/turb_autodiff_Nx{nx}", enable_x64=True,
    )


def report(res):
    r = res.extra
    if isinstance(r, dict) and "ic_residual" in r:
        print(f"RESULT ic_residual={r['ic_residual']:.3e}", flush=True)


if __name__ == "__main__":
    script_main(config, report=report, task=TASK)
