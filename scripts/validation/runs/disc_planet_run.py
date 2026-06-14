"""Disc-planet interaction (de Val-Borro et al. 2006 / GANDALF §6.11) through
main.run — the particle module's headline gravity example.

A 2D globally-isothermal gas disc orbits a central star (point mass in the
particle module); a q=10⁻³ planet on a fixed circular Keplerian orbit at a=1
opens a gap over a few tens of orbits. The planet is a SINK (accretes gas inside
~0.5 R_Hill). Code units: G = M⋆ = a = 1, so Ω_planet = 1 and P_orb = 2π.

Env:
    LUNA_DP_NX       grid resolution per side (default 512)
    LUNA_DP_ORBITS   number of planet orbits to integrate (default 30)
    LUNA_DP_CS       sound speed at a (H/R at a; default 0.05)
    LUNA_DP_Q        planet/star mass ratio (default 1e-3)
    LUNA_DP_SINK     1 to enable the planet sink (default 1)
"""
import os
import numpy as np

from dataclasses import replace
from template_config import Config
from setup.distributed_setup import initialize_jax_distributed
from main import run


def main():
    Nx = int(os.environ.get("LUNA_DP_NX", "512"))
    n_orbits = float(os.environ.get("LUNA_DP_ORBITS", "30"))
    cs = float(os.environ.get("LUNA_DP_CS", "0.05"))     # H/R at a (vK=1 there)
    q = float(os.environ.get("LUNA_DP_Q", "1.0e-3"))
    sink = os.environ.get("LUNA_DP_SINK", "1") == "1"

    G = 1.0
    M_star = 1.0
    a = 1.0
    omega = np.sqrt(G * M_star / a**3)          # = 1
    P_orb = 2.0 * np.pi / omega
    t_end = n_orbits * P_orb

    L = 6.0                                       # box [-3, 3]; disc R_out=2.5
    cs2 = cs * cs
    R_hill = a * (q / 3.0) ** (1.0 / 3.0)
    r_sink = 0.5 * R_hill
    eps = 0.6 * cs * a                            # softening ~0.6 H at a

    overrides = dict(
        test_problem="Disc-Planet-2D",
        SPATIAL_DIMENSION=2,
        BOUNDARY=4,                               # diode outflow at the disc edges
        Nx=Nx, Ny=Nx, Lx=L, Ly=L,
        gamma=5.0 / 3.0,
        ORDER=2, LIMITER=1, CFL=0.3,
        isothermal_cs2=cs2,                       # globally isothermal disc
        t_end=t_end,
        time_bet_snaps=P_orb,                     # one snapshot per orbit
        # ---- particle module: star + planet (+ sink) ----
        particles_enabled=True,
        GRAVITATIONAL_G=G,
        particle_star_mass=M_star,
        particle_planet_q=q,
        particle_planet_a=a,
        particle_softening=eps,
        particle_sink_enabled=sink,               # optional planet accretion (gap)
        particle_sink_radius=r_sink,
        particle_sink_rate=20.0,
        particle_sink_on_star=False,              # inner region handled by damping zone
        # de Val-Borro boundary damping zones relax toward the initial disc near
        # the inner/outer radii — this is what stabilizes the cold disc.
        disc_damping_enabled=True,
        disc_damp_R_in=0.5, disc_damp_R_out=2.2, disc_damp_rate=10.0,
        # ---- disc IC ----
        disc_sigma0=1.0, disc_sigma_index=0.5,
        disc_R_in=0.4, disc_R_out=2.5, disc_floor=1.0e-4,
        PLOT_AFTER_SIM=False,
        production_mode=True,
        clear_output_dir=True,
        enable_x64=True,
        output_dir=f"snapshots/disc_planet_Nx{Nx}",
    )

    print(f"Disc-planet: Nx={Nx}, q={q}, H/R(a)={cs:.3f}, "
          f"P_orb={P_orb:.3f}, {n_orbits:.0f} orbits → t_end={t_end:.2f}")
    print(f"  R_Hill={R_hill:.4f}, r_sink={r_sink:.4f}, soft={eps:.4f}, "
          f"sink={'on' if sink else 'off'}")
    initialize_jax_distributed(enable_x64=bool(overrides.pop("enable_x64")))
    run(replace(Config(), **overrides))


if __name__ == "__main__":
    main()
