"""Sedov-Taylor 3D blast wave with Arepo-style physical units.

E = 1e51 erg into 8 central cells, n_ISM = 1 cm^-3, T_ISM = 1e4 K, gamma=5/3.
Box: 50 pc cube. Grid: 128^3. Walltime: 10,000 yr.

Unit system (Arepo convention):
    UnitLength_in_cm        = 3.08568e21   (1 kpc)
    UnitMass_in_g           = 1.98892e43   (1e10 Msun)
    UnitVelocity_in_cm_per_s = 1e5         (1 km/s)
    => UnitTime ~ 1 Gyr; so 10 kyr ~ 1.022e-5 code time units.

Just runs the sim and saves snapshots; plot script picks them up afterwards.
"""
import os
from units import UnitSystem
from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


# Physical setup constants
SEC_PER_YEAR = 3.15576e7
KILOPARSEC_CGS = 3.08568e21
SOLAR_MASS_CGS = 1.98892e33

# Arepo unit system
UNIT_LENGTH = KILOPARSEC_CGS          # 1 kpc
UNIT_MASS = 1e10 * SOLAR_MASS_CGS     # 1e10 Msun
UNIT_VELOCITY = 1e5                   # 1 km/s

# Derived UnitTime (UnitLength / UnitVelocity)
UNIT_TIME_S = UNIT_LENGTH / UNIT_VELOCITY        # ~3.086e16 s ~ 0.978 Gyr

# Run config
BOX_SIZE_PC = 50.0                                # 50 pc
BOX_KPC = BOX_SIZE_PC * 1e-3                      # in code units
NX = int(os.environ.get("LUNA_SEDOV_NX", "256"))
T_END_YR = 1.0e4                                  # 10,000 yr
T_END_CODE = T_END_YR * SEC_PER_YEAR / UNIT_TIME_S
N_SNAPS = int(os.environ.get("LUNA_SEDOV_NSNAPS", "10"))   # benchmark sets =1
TIME_BET_SNAPS = T_END_CODE / N_SNAPS


def main():
    units = UnitSystem(
        UnitLength_in_cm=UNIT_LENGTH,
        UnitMass_in_g=UNIT_MASS,
        UnitVelocity_in_cm_per_s=UNIT_VELOCITY,
    )

    overrides = dict(
        test_problem="Sedov-Blast-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=1,                # zero-gradient outflow (Sedov is not periodic)
        Nx=NX, Ny=NX, Nz=NX,
        Lx=BOX_KPC, Ly=BOX_KPC, Lz=BOX_KPC,
        gamma=5.0 / 3.0,
        ORDER=1,                   # PLM
        LIMITER=0,                 # minmod (shock-safe)
        CFL=0.4,                   # 3D unsplit safety
        t_end=T_END_CODE,
        time_bet_snaps=TIME_BET_SNAPS,
        units=units,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        output_dir=os.environ.get("LUNA_SEDOV_OUT", f"snapshots/sedov3d_arepo_Nx{NX}"),
        clear_output_dir=True,
        MESH_SHAPE=(
            tuple(int(s) for s in os.environ["LUNA_MESH_SHAPE"].split(","))
            if os.environ.get("LUNA_MESH_SHAPE") else None
        ),
        USE_SHARDMAP=os.environ.get("LUNA_USE_SHARDMAP", "0") == "1",
        # Precision via LUNA_X64 env (default float32). float64 doubles the
        # working set, so large grids per A100 (e.g. 1024^3 on few GPUs) can OOM
        # / trip the XLA collective-combiner bug — use more GPUs there. The
        # code-vs-code scaling benchmark sets LUNA_X64=1 to match AthenaK (FP64).
        enable_x64=os.environ.get("LUNA_X64", "0") == "1",
        enable_checkpointing=False,
    )
    print(f"Sedov 3D Arepo run: box={BOX_SIZE_PC} pc, Nx={NX}, t_end={T_END_YR} yr "
          f"= {T_END_CODE:.4e} code")
    initialize_jax_distributed(enable_x64=bool(overrides.get("enable_x64", False)))
    overrides = {k: v for k, v in overrides.items() if k != "enable_x64"}
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
