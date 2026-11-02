"""Smoke test: run a 2-body system in PeTar under AMUSE."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from amuse.community.petar.interface import Petar
from amuse.datamodel import Particles
from amuse.units import nbody_system, units


def main():
    conv = nbody_system.nbody_to_si(1000 | units.MSun, 1 | units.parsec)
    print(">> instantiating Petar ...", flush=True)
    # Torch's PeTar setup (mode='cpu'); sockets because MPI_Comm_spawn fails here.
    g = Petar(conv, mode="cpu", redirection="none",
              channel_type="sockets", initialize_mpi=False)
    print(">> Petar instantiated (mode=cpu)", flush=True)
    # set the auto-determined params: AMUSE commits them with zero particles
    g.parameters.epsilon_squared = (15.0 | units.RSun) ** 2
    g.parameters.r_out = 0.001 | units.parsec
    g.parameters.r_search_min = 0.005 | units.parsec
    g.parameters.r_bin = 0.0005 | units.parsec
    g.parameters.dt_soft = 2.0 ** -10 | nbody_system.time
    print(">> parameters set: r_out=", g.parameters.r_out.value_in(units.parsec),
          "pc, dt_soft=", g.parameters.dt_soft.value_in(units.Myr), "Myr", flush=True)

    p = Particles(2)
    p.mass = [10.0, 10.0] | units.MSun
    p.position = [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]] | units.parsec
    p.velocity = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] | units.kms
    g.particles.add_particles(p)
    print(f">> added {len(g.particles)} particles", flush=True)

    g.evolve_model(0.01 | units.Myr)
    print(">> evolved to t =", g.model_time.value_in(units.Myr), "Myr", flush=True)
    print(">> positions [pc]:", g.particles.position.value_in(units.parsec), flush=True)
    g.stop()
    print(">> PETAR SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
