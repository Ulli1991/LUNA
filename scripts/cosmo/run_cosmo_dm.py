"""A dark-matter-only cosmological N-body run through main.run, from a MUSIC IC.

    python -m scripts.cosmo.run_cosmo_dm ic=music/ics_arepo.hdf5 mesh=64 a_end=1
    srun -n 8 python -m scripts.cosmo.run_cosmo_dm ic=... mesh=512 write_snapshots=0

The MUSIC header sets box and cosmology; the particles evolve by the
super-comoving KDK leapfrog with the comoving multigrid PM gravity.

Parameters: ic (the Arepo-HDF5 IC path) and mesh (PM cells per side). a_end,
dm_capacity_factor, write_snapshots, output_dir and every other Config field
follow as field=value.
"""
from dataclasses import replace


from luna.setup.cli import script_main            # noqa: E402
from luna.template_config import Config           # noqa: E402


def config(ic="music/ics_arepo.hdf5", mesh=64):
    return replace(
        Config(),
        run="dm", CosmologicalIntegration=True, music_ic_path=ic, Nx=mesh,
        dm_capacity_factor=1.5, POISSON_FMG_EXTRA_VCYCLES=4, dloga_max=0.02,
        output_dir="snapshots/cosmo_dm", PLOT_AFTER_SIM=False,
        enable_x64=True,
    )


if __name__ == "__main__":
    script_main(config)
