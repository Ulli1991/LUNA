"""Disc-planet interaction (de Val-Borro et al. 2006 / GANDALF sec. 6.11): the
particle module's headline gravity example.

    python -m scripts.examples.runs.disc_planet_run nx=512 orbits=30 q=1e-3

A 2D globally-isothermal gas disc orbits a central star (a point mass in the
particle module); a planet on a fixed circular Keplerian orbit at a = 1 opens
a gap over a few tens of orbits. The planet is a sink (accretes inside
~0.5 R_Hill). Code units G = M_star = a = 1, so P_orb = 2 pi.

Parameters: nx, orbits, cs (sound speed at a = H/R there), q (planet/star mass
ratio), sink (planet accretion on/off).
"""
from dataclasses import replace

import numpy as np

from luna.setup.cli import script_main
from luna.ic_generation import disc_planet_2d
from luna.template_config import Config


def config(nx=512, orbits=30.0, cs=0.05, q=1.0e-3, sink=True):
    G, M_star, a = 1.0, 1.0, 1.0
    omega = np.sqrt(G * M_star / a ** 3)          # = 1
    P_orb = 2.0 * np.pi / omega
    t_end = orbits * P_orb
    L = 6.0                                       # box [-3, 3]; disc R_out = 2.5
    R_hill = a * (q / 3.0) ** (1.0 / 3.0)
    r_sink = 0.5 * R_hill
    eps = 0.6 * cs * a                            # softening ~0.6 H at a
    print(f"Disc-planet: Nx={nx}, q={q}, H/R(a)={cs:.3f}, P_orb={P_orb:.3f}, "
          f"{orbits:.0f} orbits -> t_end={t_end:.2f}")
    print(f"  R_Hill={R_hill:.4f}, r_sink={r_sink:.4f}, soft={eps:.4f}, "
          f"sink={'on' if sink else 'off'}")
    return replace(
        Config(),
        test_problem="Disc-Planet-2D", SPATIAL_DIMENSION=2, BOUNDARY=4,
        Nx=nx, Ny=nx, Lx=L, Ly=L, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, isothermal_cs2=cs * cs,
        t_end=t_end, time_bet_snaps=P_orb,                     # one snapshot per orbit
        particles_enabled=True, GRAVITATIONAL_G=G,
        particle_star_mass=M_star, particle_planet_q=q, particle_planet_a=a,
        particle_softening=eps,
        particle_sink_enabled=sink, particle_sink_radius=r_sink, particle_sink_rate=20.0,
        particle_sink_on_star=False,              # inner region handled by damping
        # de Val-Borro damping zones relax toward the initial disc near the
        # inner/outer radii: what keeps the cold disc stable
        disc_damping_enabled=True, disc_damp_R_in=0.5, disc_damp_R_out=2.2,
        disc_damp_rate=10.0,
        problem=disc_planet_2d.Params(sigma0=1.0, sigma_index=0.5, R_in=0.4, R_out=2.5,
                                      floor=1.0e-4),
        PLOT_AFTER_SIM=False, clear_output_dir=True,
        enable_x64=True, output_dir=f"snapshots/disc_planet_Nx{nx}",
    )


if __name__ == "__main__":
    script_main(config)
