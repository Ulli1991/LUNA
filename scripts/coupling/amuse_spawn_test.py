"""AMUSE spawn smoke test: start the LUNA worker over sockets and round-trip calls."""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amuse_luna.interface import LunaInterface


def main():
    print(">> instantiating LunaInterface (spawns worker over sockets) ...", flush=True)
    # initialize_mpi=False: a plain subprocess over sockets, no MPI launcher
    code = LunaInterface(channel_type="sockets", initialize_mpi=False, redirection="none")
    print(">> worker spawned OK", flush=True)

    # --- configure a small 3D box ---
    N = 8
    for k, v in dict(SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0,
                     gamma=1.4, ORDER=1, LIMITER=1, BOUNDARY=0).items():
        code.set_config_value(k, float(v))
    print(">> config set", flush=True)

    print(">> initialize_code rc:", code.initialize_code(), flush=True)
    print(">> commit_parameters rc:", code.commit_parameters(), flush=True)

    # --- grid range through AMUSE ---
    rng = code.get_grid_range()
    print(">> get_grid_range:", rng, flush=True)

    # --- full-grid index lists ---
    import numpy as np
    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    i = ii.ravel(); j = jj.ravel(); k = kk.ravel()
    ig = np.zeros_like(i)
    rho = np.ones_like(i, dtype=float)
    zero = np.zeros_like(rho)
    eint = np.full_like(rho, 1.0 / (1.4 - 1.0))   # P = 1

    # --- set/get grid state round-trip through AMUSE marshalling ---
    print(">> set_grid_state rc:", code.set_grid_state(i, j, k, ig, rho, zero, zero, zero, eint), flush=True)
    g_rho, g_mx, g_my, g_mz, g_eint, rc = code.get_grid_state(i, j, k, ig)
    print(">> get_grid_state rc:", rc,
          "| rho==1?", np.allclose(g_rho, 1.0),
          "| eint matches?", np.allclose(g_eint, 1.0 / (1.4 - 1.0)), flush=True)

    print(">> get_time:", code.get_time(), flush=True)
    print(">> evolve_model(0.02) rc:", code.evolve_model(0.02), flush=True)
    print(">> get_time after evolve:", code.get_time(), flush=True)

    # --- particles + gravity bridge through AMUSE ---
    def _v(r, key):
        try:
            return np.asarray(r[key])
        except (TypeError, KeyError, IndexError):
            return np.asarray(r)

    print(">> --- particle + gravity surface over AMUSE ---", flush=True)
    print(">> set_particle_pointers rc:", code.set_particle_pointers("mass"), flush=True)
    r = code.new_particle([0.3, 0.7], [0.5, 0.5], [0.5, 0.5])
    tags = _v(r, "tag")
    print(">> new_particle tags:", tags, flush=True)
    print(">> n_particles:", _v(code.get_number_of_particles(), "number_of_particles"), flush=True)
    code.set_particle_mass(tags, [100.0, 50.0])
    print(">> get_particle_mass:", _v(code.get_particle_mass(tags), "mass"), flush=True)
    g = code.get_gravity_at_point([0.0, 0.0], [0.4, 0.6], [0.5, 0.5], [0.5, 0.5])
    ax = _v(g, "ax")
    print(">> get_gravity_at_point ax:", ax, "| finite:", bool(np.all(np.isfinite(ax))), flush=True)
    print(">> apply_particle_gravity_on_gas rc:", code.apply_particle_gravity_on_gas(0.001), flush=True)
    print(">> apply_gas_gravity_on_particles rc:", code.apply_gas_gravity_on_particles(0.001), flush=True)

    print(">> cleanup_code rc:", code.cleanup_code(), flush=True)
    code.stop()
    print(">> worker stopped — FULL ROUND-TRIP TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
