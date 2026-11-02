"""Supersonic driven turbulence with every in-situ product on: PDFs, spectra, structure functions, snipshots, frames.

    python -m scripts.examples.insitu_turbulence_run nx=64
    python -m scripts.examples.insitu_turbulence_run nx=256 MESH_SHAPE=2,2,2

Outputs under output_dir/insitu/: stats/, spectra/, sf/ (one HDF5 per record), snip/<step>/snip.h5
(one virtual global array per field over per-process parts), frames/<field>/ (uint16 + PNG),
header.json. Positions and, for isothermal runs, the energy are not stored: the headers say how
to rebuild them. Parameter: nx.
"""
from dataclasses import replace

from luna.analysis.config import InsituConfig
from luna.hydro.turbulence import OUDriverConfig
from luna.ic_generation import driven_turbulence_3d
from luna.setup.cli import script_main
from luna.template_config import Config

MACH = 4.0                                  # target rms Mach number, cs = 1
T_TURN = 0.5 / MACH                         # L / (2 v_rms)


def config(nx=64):
    ou = OUDriverConfig(k_min=1.0, k_max=3.0, solenoidal_weight=0.5, autocorrelation_time=T_TURN,
                        rms_acceleration=2.0 * MACH ** 2, seed=7)
    ins = InsituConfig(stats_dt=0.01 * T_TURN, spectra_dt=0.1 * T_TURN, sf_dt=0.1 * T_TURN,
                       snip_dt=0.1 * T_TURN, frame_dt=0.01 * T_TURN, snip_factor=4)
    return replace(
        Config(), test_problem="Driven-Turbulence-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=nx, Ny=nx, Nz=nx, gamma=5.0 / 3.0, problem=driven_turbulence_3d.Params(rho0=1.0, P0=1.0),
        isothermal_cs2=1.0, ORDER=1, LIMITER=0, t_end=3.0 * T_TURN,
        ou_config=ou, ou_seed=7, insitu=ins, write_snapshots=False, PLOT_AFTER_SIM=False,
        enable_checkpointing=False, output_dir=f"snapshots/insitu_turb_{nx}", clear_output_dir=True)


if __name__ == "__main__":
    script_main(config)
