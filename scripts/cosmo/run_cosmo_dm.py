"""DM-only cosmological N-body run THROUGH main.run, from a MUSIC IC.

main.run detects CosmoDM and dispatches to setup/runner_cosmo_dm.py, which loads
the MUSIC ICs, wires box + cosmology from the header, and evolves the particles
with a super-comoving DKD leapfrog + the comoving multigrid PM gravity.

  LUNA_DM_IC    MUSIC Arepo-HDF5 IC path
  LUNA_DM_MESH  PM mesh (default 64)
  LUNA_DM_AEND  final scale factor (default 1.0 = z=0)
"""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.getcwd())
from dataclasses import replace

from main import run
from template_config import Config


def main():
    cfg = replace(
        Config(),
        CosmoDM=True, CosmologicalIntegration=True,
        music_ic_path=os.environ.get(
            "LUNA_DM_IC", "/ptmp/uli/jax-hydro/music/ics_arepo.hdf5"),
        Nx=int(os.environ.get("LUNA_DM_MESH", "64")),
        a_end=float(os.environ.get("LUNA_DM_AEND", "1.0")),
        dloga_max=0.02,
        output_dir="/ptmp/uli/jax-hydro/snapshots/cosmo_dm",
        production_mode=True, PLOT_AFTER_SIM=False,
    )
    run(cfg)


if __name__ == "__main__":
    main()
