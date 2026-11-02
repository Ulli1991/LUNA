"""Drive Torch's own `torch_sf` (IMF sampling + star spawning) against a LunaWorker.

Run: PYTHONPATH=. python scripts/coupling/torch_sf_imf_test.py"""
import os
import sys
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.environ["TORCH_SRC"])       # a Torch clone's src/

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from amuse.units import units as u  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402
import torch_sf  # noqa: E402  (from torch/src)


class UnitAdapter:
    """Wrap a LunaWorker in AMUSE quantities (code units are pc, Msun, km/s)."""

    def __init__(self, w):
        self.w = w

    def set_particle_pointers(self, p): return self.w.set_particle_pointers(p)
    def get_number_of_particles(self): return self.w.get_number_of_particles()
    def get_particle_tags(self, idx): return self.w.get_particle_tags(idx)

    # the worker may return size-1 arrays
    @staticmethod
    def _s(x): return float(np.ravel(np.asarray(x))[0])

    def get_particle_mass(self, tag): return self._s(self.w.get_particle_mass(tag)) | u.MSun
    def get_sink_mean_cs(self, tag): return self._s(self.w.get_sink_mean_cs(tag)) | u.kms

    def get_particle_position(self, tag):
        x, y, z = self.w.get_particle_position(tag)
        return np.array([self._s(x), self._s(y), self._s(z)]) | u.parsec

    def get_particle_velocity(self, tag):
        vx, vy, vz = self.w.get_particle_velocity(tag)
        return np.array([self._s(vx), self._s(vy), self._s(vz)]) | u.kms

    def set_particle_mass(self, tag, m): return self.w.set_particle_mass(tag, m.value_in(u.MSun))
    def set_particle_oldmass(self, tag, m): return self.w.set_particle_oldmass(tag, m.value_in(u.MSun))

    def add_particles(self, x, y, z):
        return self.w.add_particles(x.value_in(u.parsec), y.value_in(u.parsec), z.value_in(u.parsec))

    def set_particle_velocity(self, tag, vx, vy, vz):
        return self.w.set_particle_velocity(
            tag, vx.value_in(u.kms), vy.value_in(u.kms), vz.value_in(u.kms))

    # radiation-only, called by torch_sf
    def set_particle_sigd(self, tag, s): return 0
    def set_particle_sigh(self, tag, s): return 0


def _inject_sink(w, pos, mass, cs):
    """Put one active sink with accreted-gas moments, so its mean_cs is `cs`, not NaN."""
    N = int(getattr(w.cfg, "sink_max_count", 1000))
    p = np.zeros((N, 3)); p[0] = pos
    m = np.zeros(N); m[0] = mass
    act = np.zeros(N); act[0] = 1.0
    macc = np.zeros(N); macc[0] = mass
    scs = np.zeros(N); scs[0] = mass * cs
    mom = {"macc": jnp.asarray(macc), "sv": jnp.zeros((N, 3)), "sv2": jnp.zeros((N, 3)),
           "scs": jnp.asarray(scs), "sL": jnp.zeros((N, 3))}
    from luna.setup.state import Particles
    w.state.species["sink"] = Particles(
        jnp.asarray(p), jnp.zeros((N, 3)), jnp.asarray(m), jnp.asarray(act),
        extra=mom)


def main():
    cfg = _build_cfg_for_test(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=16, Ny=16, Nz=16, Lx=10.0, Ly=10.0, Lz=10.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
        sink_max_count=1000)
    w = LunaWorker(cfg); w.initialize_code(); w.commit_parameters()

    SINK_MASS = 300.0      # Msun
    _inject_sink(w, [5.0, 5.0, 5.0], mass=SINK_MASS, cs=0.4)
    w.sync_sinks_from_carry()
    print(f"LUNA sink injected: mass={SINK_MASS} Msun, mean_cs=0.4 km/s")

    adapter = UnitAdapter(w)
    state = SimpleNamespace(all_masses={},
                            user={"with_lyc": False, "with_pe_heat": False, "sigd": 1.0e-18})

    torch_sf.queue_stars(state, adapter,
                         min_imf_mass=0.1 | u.MSun, max_imf_mass=50.0 | u.MSun,
                         sample_imf_mass=SINK_MASS | u.MSun, sample_imf_bins=10)
    queued = np.concatenate([v for v in state.all_masses.values()]) if state.all_masses else np.array([])
    print(f"torch_sf.queue_stars -> queued {len(queued)} stars, "
          f"sum={queued.sum():.1f} Msun, max={queued.max():.1f} Msun (IMF sampled by TORCH)")

    formed = torch_sf.make_stars_from_sinks(state, adapter, sink_rad=0.05 | u.parsec)
    w.sync_sinks_to_carry()

    w.set_particle_pointers("mass")
    nstars = w.get_number_of_particles()
    tags = w.get_particle_tags(None)
    masses = np.array([float(np.ravel(np.asarray(w.get_particle_mass(t)))[0]) for t in tags]) if tags else np.array([])
    sel, _, _, smass = w.get_active_sinks()
    print(f"\nformed_stars={formed}")
    print(f"  stars in LUNA store: {nstars}, total stellar mass={masses.sum():.1f} Msun")
    print(f"  individual masses: {np.sort(masses)[::-1][:12]} ...")
    print(f"  sink mass now: {float(smass[0]) if len(smass) else 0.0:.1f} Msun (was {SINK_MASS})")

    ok = (nstars > 0 and masses.min() >= 0.1 and masses.max() <= 50.0
          and float(smass[0]) < SINK_MASS)
    print("\n" + ("✅ PASS: real torch_sf sampled the IMF and spawned individual stars from a LUNA sink"
                  if ok else "❌ FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
