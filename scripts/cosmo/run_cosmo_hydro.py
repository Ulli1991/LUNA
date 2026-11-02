"""A coupled gas + dark-matter adiabatic cosmological box through main.run,
from a MUSIC baryon IC (PartType0 gas + PartType1 DM).

    python -m scripts.cosmo.run_cosmo_hydro ic=music/ics_arepo_l8_bary.hdf5 mesh=256
    srun -n 8 python -m scripts.cosmo.run_cosmo_hydro ic=... mesh=512 \\
        cosmo_snapshot_a=0.5,1.0

Parameters: ic and mesh (PM and hydro cells per side). The physics knobs are
Config fields: a_end, cosmo_jeans_ncell (Truelove cells), dm_softening (kpc/h,
0 = the PM cell), dm_capacity_factor, cosmo_T_floor, cosmo_rho_floor (a
fraction of the mean), cosmo_vmax, cosmo_Tmax, cosmo_floor_buffer,
cosmo_snapshot_a (scale factors to snapshot at; the end always is), gravity_precision,
decomposition and output_dir, all as field=value.
"""
from dataclasses import replace


from luna.setup.cli import script_main            # noqa: E402
from luna.template_config import Config           # noqa: E402


def config(ic="music/ics_arepo_l8_bary.hdf5", mesh=256):
    return replace(
        Config(),
        run="gas+dm", CosmologicalIntegration=True, music_ic_path=ic, Nx=mesh,
        dloga_max=0.02, cosmo_jeans_ncell=4.0, dm_capacity_factor=1.5,
        cosmo_T_floor=10.0, cosmo_rho_floor=1e-3,
        cosmo_p_floor=0.0, POISSON_FMG_EXTRA_VCYCLES=2, POISSON_N_COARSEST=10,
        BOUNDARY=0,                     # a cosmological box is periodic
        ORDER=1, LIMITER=1, riemann_solver="hllc",
        output_dir="snapshots/cosmo_hydro",
        PLOT_AFTER_SIM=False, enable_x64=True,
    )


if __name__ == "__main__":
    script_main(config)
