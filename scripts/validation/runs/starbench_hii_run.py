"""STARBENCH: the D-type expansion of an H II region (Bisbas et al. 2015).

    python -m scripts.validation.runs.starbench_hii_run nx=128
    python -m scripts.validation.runs.starbench_hii_run nx=256 chat_over_ci=100

An ionizing source of 1e49 photons/s switched on at t = 0 at the centre of a
uniform neutral medium of 5.21e-21 g/cm^3 (n_H = 3115 cm^-3). The ionized
gas is held at 1e4 K (c_i = 12.85 km/s), the neutral gas at 1e2 K
(c_o = 0.91 km/s), recombination is case-B on the spot, and the Stromgren
radius is 0.314 pc. Run to 0.141 Myr, the benchmark's early-phase end time,
and compare the front radius with the Spitzer and Hosokawa-Inutsuka
solutions, which bracket what the twelve STARBENCH codes found.

Code units here are the parsec, the megayear, and whatever mass makes
rho_0 = 1; luna.radiation.ionization.starbench_units does that conversion
once so there are no factors in this file.

The reduced speed of light is the one free numerical parameter:
`chat_over_ci` multiplies c_i. It must be large enough that the initial
R-type phase -- which really runs at c -- is not the thing being measured;
the light-crossing time of the Stromgren radius at c_hat is printed so the
delay it introduces can be read off against t_end.
"""
import math
from dataclasses import replace

from luna.radiation.ionization import starbench_units
from luna.setup.cli import script_main
from luna.ic_generation import hii_region_3d
from luna.template_config import Config

BOX_PC = 4.0             # 1.27 R_cl of the benchmark cloud, front well inside


def config(nx=128, chat_over_ci=30.0, n_snaps=24, phase="early", box_pc=BOX_PC,
           substeps=0, t_end_frac=1.0, scheme="hancock_split", order=2, cfl=0.4, ion_sub=1, ion_solver="implicit"):
    u = starbench_units(phase=phase)
    c_i = u["cs2_ionized"] ** 0.5
    chat = float(chat_over_ci) * c_i
    dx = box_pc / nx
    # Enough substeps that a radiation half-step covers half the hydro step of ionized gas at
    # rest (cell_dt: CFL dx / (3 c_i)); a substep is at most the transport's own longest one,
    # as luna.sources.radiation computes it (2 cfl dx / chat split, cfl dx / (3 chat) rk2).
    per_sub = 2.0 * cfl * dx / chat if scheme == "hancock_split" else cfl * dx / (3.0 * chat)
    n_sub = int(substeps) or max(1, math.ceil(0.5 * Config().CFL * dx / (3.0 * c_i) / per_sub))
    t_end = u["t_end"] * float(t_end_frac)

    print("STARBENCH D-type expansion, " + phase + " phase")
    print(f"  box {box_pc} pc on {nx}^3  (dx = {dx:.4f} pc, "
          f"R_St = {u['R_St']:.4f} pc = {u['R_St'] / dx:.1f} cells)")
    print(f"  c_i = {c_i:.3f} pc/Myr, c_hat = {chat:.1f} pc/Myr "
          f"({chat_over_ci:g} c_i), {n_sub} transport substeps per half-step")
    print(f"  R_St light crossing at c_hat: {u['R_St'] / chat:.3e} Myr "
          f"of t_end = {t_end:.4f} Myr")
    print(f"  Ndot = {u['photon_rate']:.3e} photons/Myr, "
          f"alpha_B = {u['alpha_B']:.3e}, sigma = {u['sigma']:.3e} (code)")

    return replace(
        Config(),
        test_problem="HII-Region-3D", SPATIAL_DIMENSION=3, BOUNDARY=1,
        Nx=nx, Ny=nx, Nz=nx, Lx=box_pc, Ly=box_pc, Lz=box_pc,
        gamma=5.0 / 3.0, LIMITER=0, enable_x64=True,
        problem=hii_region_3d.Params(rho0=u["rho0"]),
        radiation_enabled=True, rad_ionization=True, rad_groups=1,
        rad_chat=chat, rad_boundary=1, rad_substeps=n_sub, rad_order=int(order),
        rad_scheme=str(scheme), rad_cfl=float(cfl), rad_ion_subcycles=int(ion_sub),
        rad_ion_solver=str(ion_solver),
        rad_force=False, rad_sigma=u["sigma"], rad_alpha_B=u["alpha_B"],
        rad_photon_rate=u["photon_rate"], rad_m_H=u["m_H"],
        rad_cs2_neutral=u["cs2_neutral"], rad_cs2_ionized=u["cs2_ionized"],
        rad_source_radius=2.0 * dx, rad_x0=0.0,
        t_end=t_end, time_bet_snaps=t_end / n_snaps,
        write_snapshots=True, enable_checkpointing=False,
        PLOT_AFTER_SIM=False,
        output_dir=f"snapshots/starbench_{phase}_nx{nx}", clear_output_dir=True,
    )


if __name__ == "__main__":
    script_main(config)
