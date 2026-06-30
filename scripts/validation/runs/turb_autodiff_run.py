"""Driven ISM turbulence in AUTODIFF mode — the REVERSE-mode showcase.

PHYSICAL-UNITS (Arepo system) two-phase ISM box: an ~100 pc periodic box at
n_H ~ 1 cm^-3, T ~ 1e4 K, driven by OU forcing and cooled by the combined
Koyama-Inutsuka (cold/warm ISM) + Sutherland-Dopita (hot) cooling curve, so the
gas relaxes toward the realistic two-phase ISM. Reverse-mode AD through the
production solver (OU forcing + radiative cooling in the loop) recovers the
initial low-k density perturbation from the evolved turbulent, multiphase state.

Dispatched through main.run (same entry path as the HPC code).

Env: LUNA_TURB_NX, LUNA_TURB_NSTEPS, LUNA_TURB_NITER, LUNA_TURB_MESH,
     LUNA_TURB_NH (cm^-3), LUNA_TURB_T (K), LUNA_TURB_LPC (box, pc),
     LUNA_TURB_MACH, LUNA_TURB_COOL (combined|ki|sd93_solar|none).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from dataclasses import replace

from hydro.turbulence import OUDriverConfig
from template_config import Config
from units import UnitSystem
from setup.distributed_setup import initialize_jax_distributed
from main import run

# ---- Arepo unit system: 1 kpc / 1e10 Msun / 1 km/s ----
KPC = 3.08568e21
MSUN = 1.98892e33
M_H = 1.6726e-24            # g
K_B = 1.3807e-16            # erg/K
MU_RHO = 1.4               # mean mass per H (incl. He) -> rho from n_H
MU_P = 0.6                 # mean mass per particle -> pressure
GAMMA = 5.0 / 3.0


def main():
    Nx = int(os.environ.get("LUNA_TURB_NX", "32"))
    n_steps = int(os.environ.get("LUNA_TURB_NSTEPS", "200"))
    n_iter = int(os.environ.get("LUNA_TURB_NITER", "80"))
    use_shardmap = os.environ.get("LUNA_TURB_SHARDMAP", "0") == "1"
    mesh_env = os.environ.get("LUNA_TURB_MESH", "")
    mesh_shape = tuple(int(v) for v in mesh_env.split("x")) if mesh_env else None

    n_H = float(os.environ.get("LUNA_TURB_NH", "1.0"))         # cm^-3
    T_K = float(os.environ.get("LUNA_TURB_T", "1.0e4"))        # K
    L_pc = float(os.environ.get("LUNA_TURB_LPC", "100.0"))     # box, pc
    mach = float(os.environ.get("LUNA_TURB_MACH", "2.0"))
    cool = os.environ.get("LUNA_TURB_COOL", "combined")        # combined|ki|sd93_solar|none

    units = UnitSystem(UnitLength_in_cm=KPC, UnitMass_in_g=1.0e10 * MSUN,
                       UnitVelocity_in_cm_per_s=1.0e5)
    u_dens = (1.0e10 * MSUN) / KPC ** 3                        # g/cm^3
    u_pres = u_dens * (1.0e5) ** 2                             # erg/cm^3

    # ISM box in code units
    L = (L_pc * 1.0e-3)                                        # pc -> kpc (code length)
    rho0 = (n_H * MU_RHO * M_H) / u_dens
    n_tot = (n_H * MU_RHO) / MU_P                              # total particles / cm^3
    P0 = (n_tot * K_B * T_K) / u_pres
    cs = (GAMMA * P0 / rho0) ** 0.5                            # code velocity (km/s)
    v_rms = mach * cs
    T_turn = L / v_rms
    T_ac = 0.5 * T_turn
    rms_acc = v_rms / T_ac
    dt = 0.3 * (L / Nx) / (cs + v_rms)                         # CFL-safe fixed dt

    ou_cfg = OUDriverConfig(
        k_min=1.0, k_max=2.0, solenoidal_weight=1.0,
        autocorrelation_time=T_ac, rms_acceleration=rms_acc,
        seed=7, spectrum="parabolic", Lx=L, Ly=L, Lz=L,
    )
    overrides = dict(
        test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=Nx, Ny=Nx, Nz=Nx, Lx=L, Ly=L, Lz=L, gamma=GAMMA, units=units,
        ORDER=1, LIMITER=1, CFL=0.4,
        ou_config=ou_cfg, ou_seed=7,
        # ---- physical ISM thermodynamics: radiative cooling (NOT isothermal) ----
        cooling_enabled=(cool != "none"),
        cooling_table=(cool if cool != "none" else "sd93_solar"),
        # ---- AUTODIFF RUN MODE: reverse-mode IC inversion ----
        autodiff_mode=True,
        autodiff_task="ic_inversion",
        autodiff_n_steps=n_steps,
        autodiff_fixed_dt=dt,
        autodiff_n_iter=n_iter,
        autodiff_ic_field="density",
        MESH_SHAPE=mesh_shape,
        PLOT_AFTER_SIM=False, production_mode=False,
        output_dir=f"snapshots/turb_autodiff_Nx{Nx}",
    )
    initialize_jax_distributed(enable_x64=True)
    cfg = replace(Config(), **overrides)
    if cfg.MESH_SHAPE is None or True:
        print(f"ISM turb box: L={L_pc} pc, n_H={n_H} cm^-3, T={T_K:.1e} K -> "
              f"rho0={rho0:.3e}, P0={P0:.3e}, cs={cs:.1f} km/s, Mach={mach}, "
              f"cooling={cool}", flush=True)
    result = run(cfg)
    if isinstance(result, dict) and "ic_residual" in result:
        print(f"RESULT ic_residual={result['ic_residual']:.3e}", flush=True)


if __name__ == "__main__":
    main()
