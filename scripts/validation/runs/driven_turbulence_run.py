"""Driven turbulence in 3D, subsonic regime.

    python -m scripts.validation.runs.driven_turbulence_run nx=128
    python -m scripts.validation.runs.driven_turbulence_run nx=256 enable_x64=0 \\
        output_dir=snapshots/turb256_fp32

The same OU forcing at every resolution, so the driven k-modes and the
stochastic realisation are identical across a convergence sweep. Adapted from
Federrath/Schmidt (2010). gamma = 5/3 inside the solver with p = cs^2 rho
enforced after every step, so the effective physics is isothermal.

Parameter: nx. enable_x64 and output_dir are Config fields.
"""
from dataclasses import replace

from luna.hydro.turbulence import OUDriverConfig
from luna.setup.cli import script_main
from luna.ic_generation import driven_turbulence_3d
from luna.template_config import Config

LX = LY = LZ = 1.0
GAMMA = 5.0 / 3.0
RHO0, P0 = 1.0, 1.0
CS2_ISO = P0 / RHO0
CS0 = CS2_ISO ** 0.5
M_TARGET = 0.3                            # rms velocity / isothermal sound speed
V_RMS_TARGET = M_TARGET * CS0
K_MIN, K_MAX = 1.0, 2.0                   # ~17 modes, large-scale driving only
SOLENOIDAL_WEIGHT = 1.0
T_TURNOVER = LX / V_RMS_TARGET            # ~2.58
T_AC = 0.5 * T_TURNOVER                   # autocorrelation time
RMS_ACC = V_RMS_TARGET / T_AC             # steady-state v_rms ~ rms_acc * T_ac
OU_SEED = 7
SPECTRUM = "parabolic"
T_END = 5.0 * T_TURNOVER                  # statistical steady state
TIME_BET_SNAPS = 0.25 * T_TURNOVER        # ~21 snapshots


def config(nx=128):
    ou_cfg = OUDriverConfig(k_min=K_MIN, k_max=K_MAX, solenoidal_weight=SOLENOIDAL_WEIGHT,
                            autocorrelation_time=T_AC, rms_acceleration=RMS_ACC,
                            seed=OU_SEED, spectrum=SPECTRUM, Lx=LX, Ly=LY, Lz=LZ)
    print(f"Driven-turbulence run: Nx={nx}, M_target={M_TARGET}, "
          f"v_rms_target={V_RMS_TARGET:.3f}, T_turnover={T_TURNOVER:.3f}, "
          f"t_end={T_END:.3f}, T_ac={T_AC:.3f}, rms_acc={RMS_ACC:.3f}")
    return replace(
        Config(),
        test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, Lx=LX, Ly=LY, Lz=LZ, gamma=GAMMA,
        problem=driven_turbulence_3d.Params(rho0=RHO0, P0=P0),         # dimensionless 1, 1 here
        ORDER=1, LIMITER=1,
        t_end=T_END, time_bet_snaps=TIME_BET_SNAPS,
        ou_config=ou_cfg, ou_seed=OU_SEED, isothermal_cs2=CS2_ISO,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/turb_Nx{nx}", clear_output_dir=True,
        enable_x64=True, enable_checkpointing=False,
    )


if __name__ == "__main__":
    script_main(config)
