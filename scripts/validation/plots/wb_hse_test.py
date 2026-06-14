"""HSE-preservation test for the well-balanced integrator.

Initial state = hydrostatic equilibrium U_eq (no perturbation, no velocity).
The well-balanced integrator should keep the state at U_eq to ROUNDOFF for
many sound-crossing times. Without well-balanced, truncation error in
dp/dx + rho*g would drive spurious flows on the order of a few percent in
the same time.

Reports:
  - initial ||U||_inf and ||U_eq||_inf for sanity
  - final ||U - U_eq||_inf per field (rho, mvx, mvy, mvz, E)
  - sound-crossing fraction reached
"""
import glob, os
import numpy as np
import h5py

from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run
from ic_generation.solar_convection_2d import solar_convection_2d_equilibrium


NX = 64
NY = 64
LX = 1.0
LY = 1.0
GAMMA = 5.0 / 3.0
GRAVITY_Z = -1.0          # downward (along -y in 2D)
SOLAR_RHO0 = 1.0
SOLAR_P0   = 1.0
T_END = 5.0               # several sound-crossing times for cs~1, L=1
OUTDIR = f"snapshots/wb_hse_2d_Nx{NX}"


def main():
    overrides = dict(
        test_problem="Solar-Convection-2D",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,            # periodic xy (true closed-z BC is a follow-up)
        Nx=NX, Ny=NY,
        Lx=LX, Ly=LY,
        gamma=GAMMA,
        ORDER=1,
        LIMITER=0,             # minmod
        CFL=0.4,
        t_end=T_END,
        time_bet_snaps=T_END / 5.0,    # 6 snapshots
        gravity_z=GRAVITY_Z,
        use_well_balanced=True,
        riemann_solver="lhllc",
        # Polytropic atmosphere parameters
        solar_rho0=SOLAR_RHO0,
        solar_p0=SOLAR_P0,
        # Gamma=gamma -> adiabatic, isentropic, Schwarzschild-stable
        solar_pert_amp=0.0,            # no perturbation - this is the HSE test
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=OUTDIR,
        clear_output_dir=True,
        enable_x64=True,
    )
    print(f"WB HSE test: Nx={NX}, t_end={T_END}, g_z={GRAVITY_Z}, gamma=Gamma={GAMMA}")
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))

    # Load the final snapshot and compare against U_eq.
    snaps = sorted(glob.glob(os.path.join(OUTDIR, "snapdir_*")))
    if not snaps:
        raise SystemExit("No snapshots saved.")
    last = snaps[-1]
    files = sorted(glob.glob(os.path.join(last, "*.hdf5")))
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx_ = int(a["Nx_global"]); Ny_ = int(a["Ny_global"])
        t_final = float(a["time"])
    rho = np.empty((Nx_, Ny_))
    vx  = np.empty_like(rho); vy = np.empty_like(rho); p = np.empty_like(rho)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"])
            shp = c["density"].shape
            rho[xs:xs+shp[0], ys:ys+shp[1]] = c["density"][:]
            vx [xs:xs+shp[0], ys:ys+shp[1]] = c["velocity_x"][:]
            vy [xs:xs+shp[0], ys:ys+shp[1]] = c["velocity_y"][:]
            p  [xs:xs+shp[0], ys:ys+shp[1]] = c["pressure"][:]

    class _C:
        pass
    cfg = _C()
    cfg.Nx, cfg.Ny = NX, NY
    cfg.Lx, cfg.Ly = LX, LY
    cfg.gamma = GAMMA
    cfg.gravity_z = GRAVITY_Z
    cfg.solar_rho0 = SOLAR_RHO0
    cfg.solar_p0 = SOLAR_P0
    cfg.solar_Gamma = GAMMA
    U_eq = np.asarray(solar_convection_2d_equilibrium(cfg))
    rho_eq = U_eq[0]
    p_eq = (GAMMA - 1.0) * U_eq[3]

    print(f"\n=== HSE preservation at t={t_final} ===")
    print(f"  rho_eq max = {rho_eq.max():.6e},  rho_eq min = {rho_eq.min():.6e}")
    print(f"  |rho - rho_eq|_inf = {np.max(np.abs(rho - rho_eq)):.3e}")
    print(f"  |vx|_inf           = {np.max(np.abs(vx)):.3e}")
    print(f"  |vy|_inf           = {np.max(np.abs(vy)):.3e}")
    print(f"  |p - p_eq|_inf     = {np.max(np.abs(p - p_eq)):.3e}")


if __name__ == "__main__":
    main()
