"""Sedov-Taylor 3D blast wave in Arepo-style physical units.

    python -m scripts.validation.runs.sedov3d_run nx=256
    srun -n 8 python -m scripts.validation.runs.sedov3d_run nx=1024 MESH_SHAPE=2,2,2 \\
        enable_x64=1 max_steps=30 write_snapshots=0         # a scaling point

E = 1e51 erg into 8 central cells, n_ISM = 1 cm^-3, T_ISM = 1e4 K, gamma = 5/3,
a 50 pc box run for 10,000 yr (1.022e-5 code time in kpc / 1e10 Msun / km/s).
Snapshots go to the output dir for sedov3d_plot.py.

Parameters: nx and n_snaps. The Config fields follow as field=value:
MESH_SHAPE, enable_x64 (float64), max_steps
(a step cap for timing runs) and write_snapshots=0 for pure timing.
"""
import time as _time
from dataclasses import replace

from luna.setup.cli import script_main
from luna.template_config import Config
from luna.units import UnitSystem

_T0 = _time.time()                     # startup phase timing (rank 0 prints)
SEC_PER_YEAR = 3.15576e7
KPC = 3.08568e21
MSUN = 1.98892e33
UNIT_LENGTH, UNIT_MASS, UNIT_VELOCITY = KPC, 1e10 * MSUN, 1e5      # Arepo units
UNIT_TIME_S = UNIT_LENGTH / UNIT_VELOCITY                           # ~0.978 Gyr
BOX_KPC = 50.0e-3                                                   # 50 pc
T_END_CODE = 1.0e4 * SEC_PER_YEAR / UNIT_TIME_S                     # 10,000 yr


def config(nx=256, n_snaps=10):
    units = UnitSystem(UnitLength_in_cm=UNIT_LENGTH, UnitMass_in_g=UNIT_MASS,
                       UnitVelocity_in_cm_per_s=UNIT_VELOCITY)
    print(f"Sedov 3D Arepo run: box=50 pc, Nx={nx}, t_end=1e4 yr = {T_END_CODE:.4e} code")
    print(f"[startup] imports done t+{_time.time() - _T0:.1f}s "
          f"({_time.strftime('%H:%M:%S')})", flush=True)
    return replace(
        Config(),
        test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3, BOUNDARY=1,
        Nx=nx, Ny=nx, Nz=nx, Lx=BOX_KPC, Ly=BOX_KPC, Lz=BOX_KPC,
        gamma=5.0 / 3.0, ORDER=1, LIMITER=0,
        t_end=T_END_CODE, time_bet_snaps=T_END_CODE / n_snaps, units=units,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/sedov3d_arepo_Nx{nx}", clear_output_dir=True,
        enable_checkpointing=False,
    )


if __name__ == "__main__":
    script_main(config)
