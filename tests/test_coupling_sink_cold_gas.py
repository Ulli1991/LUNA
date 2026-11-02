"""A sink forms stars only while cold gas lies within 2 r_acc of it
(`sink_star_T_max`; Lewis et al. 2023 sec. 3.2, FLASH-Torch). A blocked
sink's mean_cs is NaN: Torch's make_stars_from_sinks and LUNA's form_stars
both skip it."""
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402
from luna.units import BOLTZMANN_CGS, PROTON_MASS_CGS, UnitSystem  # noqa: E402

N = 16
MU = 2.3
PC = 3.085677581e18


def _worker(T_max):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
        sink_star_mu=MU, sink_star_T_max=T_max,
        units=UnitSystem(UnitLength_in_cm=PC, UnitMass_in_g=1.98892e33,
                         UnitVelocity_in_cm_per_s=1.0e5))
    w = LunaWorker(cfg)
    w.initialize_code()
    w.commit_parameters()
    return w


def _cs2(T):
    """P/rho at temperature T [K], in (km/s)^2."""
    return BOLTZMANN_CGS * T / (MU * PROTON_MASS_CGS) / 1.0e10


def _gas(w, T_near, T_far, r_near):
    """rho = 1 at rest; temperature T_near within r_near of the box centre,
    T_far elsewhere."""
    c = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    P = np.where(r < r_near, _cs2(T_near), _cs2(T_far))
    z = np.zeros((N, N, N))
    w.state.U = jnp.asarray(np.stack([z + 1.0, z, z, z, P / (w.gamma - 1.0)]))


def _sink_cs(w):
    from tests.test_coupling_sink_moments import _inject_sink
    _inject_sink(w, [0.5, 0.5, 0.5])
    tags = w.sync_sinks_from_carry()
    w.set_particle_pointers("sink")
    cs = np.asarray(w.get_sink_state(tags)[-1], float)
    w.set_particle_pointers("mass")
    return cs


def test_a_sink_in_warm_gas_is_blocked():
    """Everything at 300 K: blocked with T_max = 100, not without."""
    w = _worker(100.0)
    _gas(w, 300.0, 300.0, 0.0)
    assert np.all(np.isnan(_sink_cs(w)))
    w = _worker(0.0)
    _gas(w, 300.0, 300.0, 0.0)
    assert np.all(np.isfinite(_sink_cs(w)))


def _shell(w, T_in, lo, hi, T_out=1.0e4):
    """rho = 1 at rest; temperature T_in for lo <= r < hi from the box
    centre, T_out elsewhere."""
    c = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt((X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2)
    P = np.where((r >= lo) & (r < hi), _cs2(T_in), _cs2(T_out))
    z = np.zeros((N, N, N))
    w.state.U = jnp.asarray(np.stack([z + 1.0, z, z, z, P / (w.gamma - 1.0)]))


def test_cold_gas_within_two_accretion_radii_unblocks():
    """r_acc defaults to 2 dx. Cold gas only in the shell 2.5-3.5 dx --
    outside r_acc, inside 2 r_acc -- lets the sink form stars; cold gas only
    beyond 2 r_acc does not."""
    dx = 1.0 / N
    w = _worker(100.0)
    _shell(w, 15.0, 2.5 * dx, 3.5 * dx)
    assert np.all(np.isfinite(_sink_cs(w)))
    w = _worker(100.0)
    _shell(w, 15.0, 4.5 * dx, 1.0)
    assert np.all(np.isnan(_sink_cs(w)))
