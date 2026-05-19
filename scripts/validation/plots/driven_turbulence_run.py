"""Driven-turbulence 3D run, subsonic regime, env-var Nx.

Same OU forcing parameters at every resolution so the driven k-modes and
stochastic realisation are identical across the convergence sweep; only the
grid resolution changes. Adapted from Federrath/Schmidt (2010).

    LUNA_TURB_NX  : grid resolution (default 128)
    LUNA_TURB_OUT : output dir (default snapshots/turb_Nx{Nx})
"""
import os
from scripts.validation.runner import run_one
from hydro.turbulence import OUDriverConfig


# Box & physics. LUNA runs gamma=5/3 internally for HLLC robustness, but the
# runner enforces p = CS2_ISO * rho after every step (operator-split soft
# thermostat) so the EFFECTIVE physics is isothermal: no compressive heating,
# density PDF follows the true isothermal solution.
LX = LY = LZ = 1.0
GAMMA = 5.0 / 3.0
RHO0 = 1.0
P0 = 1.0
CS2_ISO = P0 / RHO0                       # isothermal sound speed squared = p/rho
CS0 = CS2_ISO ** 0.5                      # = 1.0

# Target subsonic Mach number (rms velocity / isothermal sound speed).
M_TARGET = 0.3
V_RMS_TARGET = M_TARGET * CS0             # = 0.30

# OU forcing parameters (same at every Nx so the driven realisation matches).
K_MIN = 1.0
K_MAX = 2.0                               # ~17 modes; large-scale driving only
SOLENOIDAL_WEIGHT = 1.0                   # pure solenoidal (subsonic standard)
T_TURNOVER = LX / V_RMS_TARGET            # ~2.58
T_AC = 0.5 * T_TURNOVER                   # autocorrelation time
RMS_ACC = V_RMS_TARGET / T_AC             # steady-state v_rms ~ rms_acc * T_ac
OU_SEED = 7
SPECTRUM = "parabolic"

# Time integration: 5 turnover times is enough to reach statistical steady
# state for subsonic driving; snapshots every 0.25 T_turnover.
T_END = 5.0 * T_TURNOVER
TIME_BET_SNAPS = 0.25 * T_TURNOVER         # ~21 snapshots


def main():
    Nx = int(os.environ.get("LUNA_TURB_NX", "128"))
    out_dir = os.environ.get("LUNA_TURB_OUT", f"snapshots/turb_Nx{Nx}")
    ou_cfg = OUDriverConfig(
        k_min=K_MIN, k_max=K_MAX,
        solenoidal_weight=SOLENOIDAL_WEIGHT,
        autocorrelation_time=T_AC,
        rms_acceleration=RMS_ACC,
        seed=OU_SEED,
        spectrum=SPECTRUM,
        Lx=LX, Ly=LY, Lz=LZ,
    )
    overrides = dict(
        test_problem="Driven-Turbulence-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=0,                # periodic
        Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=LX, Ly=LY, Lz=LZ,
        gamma=GAMMA,
        TIME_SOLVER=1,             # MUSCL-Hancock
        ORDER=1,                   # PLM
        LIMITER=1,                 # MonCen (smooth-flow friendly)
        CFL=0.4,
        t_end=T_END,
        time_bet_snaps=TIME_BET_SNAPS,
        ou_config=ou_cfg,
        ou_seed=OU_SEED,
        isothermal_cs2=CS2_ISO,    # soft thermostat: enforces p = CS2_ISO * rho
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=out_dir,
        enable_x64=True,
    )
    print(f"Driven-turbulence run: Nx={Nx}, M_target={M_TARGET}, "
          f"v_rms_target={V_RMS_TARGET:.3f}, T_turnover={T_TURNOVER:.3f}, "
          f"t_end={T_END:.3f}, T_ac={T_AC:.3f}, rms_acc={RMS_ACC:.3f}")
    run_one(overrides)


if __name__ == "__main__":
    main()
