"""Coupled gas + DM adiabatic cosmological box THROUGH main.run, from a MUSIC
baryon IC. main.run detects CosmoHydro and dispatches to runner_cosmo_hydro.

  LUNA_BARY_IC  MUSIC baryon Arepo-HDF5 IC (PartType0 gas + PartType1 DM)
  LUNA_DM_MESH  PM / hydro mesh (default 256)
  LUNA_DM_AEND  final scale factor (default 1.0 = z=0)
  LUNA_JEANS_N  Truelove Jeans-floor cells (default 4)
  LUNA_DM_OUT   output dir
"""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.getcwd())
from dataclasses import replace

from main import run
from setup.distributed_setup import initialize_jax_distributed
from template_config import Config


def main():
    # multi-node coordination (>4 GPUs): reads JAX_COORDINATOR_ADDRESS +
    # SLURM_NTASKS/PROCID. Single-process (1-node) runs fall back cleanly, so
    # the existing 4-GPU submits are unaffected. Must precede any device use.
    initialize_jax_distributed(enable_x64=True)
    cfg = replace(
        Config(),
        CosmoHydro=True, CosmologicalIntegration=True,
        music_ic_path=os.environ.get(
            "LUNA_BARY_IC", "/ptmp/uli/jax-hydro/music/ics_arepo_l8_bary.hdf5"),
        Nx=int(os.environ.get("LUNA_DM_MESH", "256")),
        a_end=float(os.environ.get("LUNA_DM_AEND", "1.0")),
        dloga_max=0.02,
        cosmo_jeans_ncell=float(os.environ.get("LUNA_JEANS_N", "4")),
        dm_softening=float(os.environ.get("LUNA_SOFT", "0.0")),   # kpc/h; 0 -> PM cell
        dm_domain_decomp=os.environ.get("LUNA_DM_DD", "0") == "1",  # grid-sharded slab (>512^3)
        dm_capacity_factor=float(os.environ.get("LUNA_DM_CAP", "1.5")),
        cosmo_T_floor=float(os.environ.get("LUNA_TFLOOR", "10.0")),
        cosmo_rho_floor=float(os.environ.get("LUNA_RHOFLOOR", "1e-3")),
        cosmo_vmax=float(os.environ.get("LUNA_VMAX", "0.0")),       # |v~| ceiling
        cosmo_Tmax=float(os.environ.get("LUNA_TMAX", "0.0")),       # temperature ceiling [K]
        cosmo_floor_buffer=float(os.environ.get("LUNA_FLOOR_BUF", "1.0")),
        DualEnergy=os.environ.get("LUNA_DUAL", "0") == "1",
        cosmo_debug=os.environ.get("LUNA_DEBUG", "0") == "1",
        cosmo_snap_every=int(os.environ.get("LUNA_SNAP_EVERY", "50")),
        cosmo_profile=os.environ.get("LUNA_PROF", "0") == "1",   # per-phase csv breakdown
        dm_fp32=os.environ.get("LUNA_DM_FP32", "0") == "1",      # fp32 DM particle path
        cosmo_p_floor=0.0,
        POISSON_FMG_EXTRA_VCYCLES=2,
        POISSON_FMG_N_COARSEST=int(os.environ.get("LUNA_FMG_NCOARSE", "10")),  # 50->10: ~1.5x FMG, identical accuracy

        # gas hydro halo: explicit per-shard ppermute shard_map (fast path),
        # matching the DM/gravity shard_map primitives; periodic gas BC (a
        # cosmological box IS periodic -- was defaulting to zero-grad BOUNDARY=1).
        BOUNDARY=0,
        ORDER=1, LIMITER=1, CFL=0.4, riemann_solver="hllc",
        output_dir=os.environ.get(
            "LUNA_DM_OUT", "/ptmp/uli/jax-hydro/snapshots/cosmo_hydro"),
        production_mode=True, PLOT_AFTER_SIM=False,
    )
    run(cfg)


if __name__ == "__main__":
    main()
