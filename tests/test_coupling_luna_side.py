"""The coupled worker's side of what Torch reads (audit 2026-09-27): stars carry
their birth time, sinks attract each other, a supernova hands Torch a usable
time step, and the SN and wind injection radius can be fixed in physical
units for a resolution study."""
import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax.numpy as jnp  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402

N = 16


def _worker(**kw):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=0, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0, **kw)
    w = LunaWorker(cfg)
    w.initialize_code()
    w.commit_parameters()
    U = np.zeros(w.state.U.shape)
    U[0] = 1.0
    U[4] = 1.0
    w.state.U = jnp.asarray(U)
    return w


def test_a_star_is_born_at_the_current_time():
    """Torch's stellar clock and the stars' age read creation_time; left at 0
    the first stars began SeBa as old as the time they formed at."""
    w = _worker()
    w.state._s.t = 0.37          # the wrapped State is a plain dataclass
    tags = w.new_particle([0.2, 0.3], [0.5, 0.5], [0.5, 0.5])
    np.testing.assert_array_equal(w._store().get("creation_time", tags), [0.37, 0.37])
    w.state._s.t = 0.52
    t2 = w.new_particle(0.7, 0.5, 0.5)
    np.testing.assert_array_equal(w._store().get("creation_time", t2), [0.52])
    np.testing.assert_array_equal(w._store().get("creation_time", tags), [0.37, 0.37])


def test_the_worker_keeps_sink_sink_gravity_as_configured():
    """Sinks are LUNA's alone -- PeTar never holds them -- so their mutual pull
    is not double counted; the worker used to force it off."""
    assert _worker(sink_sink_gravity=True).cfg.sink_sink_gravity is True
    assert _worker(sink_sink_gravity=False).cfg.sink_sink_gravity is False


def test_a_supernova_returns_the_gas_time_step():
    """Torch caps the next bridge interval with the dt energy_injection returns;
    0 made a zero-length step. It is the CFL step after the blast."""
    from amuse_luna.interface import LunaImplementation

    class Box:
        value = None
    impl = LunaImplementation()
    impl._worker = _worker()
    before = float(impl._worker.get_timestep())
    dt = Box()
    assert impl.energy_injection(50.0, 0.0, 0.0, 0.5, 0.5, 0.5, dt) == 0
    assert np.isfinite(dt.value) and 0.0 < dt.value < before
    np.testing.assert_allclose(dt.value, float(impl._worker.get_timestep()), rtol=0, atol=0)


def _cells(U0, U1):
    return int(np.sum(np.any(np.asarray(U1) != np.asarray(U0), axis=0)))


def test_the_injection_radius_can_be_fixed():
    """feedback_radius (code length) replaces 3 dx for SNe and winds: the same
    sphere in cells as the geometry says, and a different one from 3 dx."""
    from luna.sources.feedback import inject_energy
    dx = 1.0 / N
    x = (np.arange(N) + 0.5) * dx
    c = x[8]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    d = [np.mod(Q - c + 0.5, 1.0) - 0.5 for Q in (X, Y, Z)]
    r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2
    for R, want in ((0.0, int((r2 < (3 * dx) ** 2).sum())), (5 * dx, int((r2 < (5 * dx) ** 2).sum()))):
        w = _worker(feedback_radius=R)
        U0 = w.state.U
        U1 = inject_energy(U0, w.cfg, c, c, c, 1.0, mass=0.0)
        assert _cells(U0, U1) == want, (R, _cells(U0, U1), want)
    # the worker's wind injector takes the same radius
    w = _worker(feedback_radius=5 * dx)
    tag = w.new_particle(c, c, c)
    w.set_particle_attr("wind_mass", tag, [1e-3])
    w.set_particle_attr("wind_vel", tag, [1.0])
    w.set_particle_mass(tag, [1.0])
    U0 = w.state.U
    w._inject_winds(0.01)
    assert _cells(U0, w.state.U) == int((r2 < (5 * dx) ** 2).sum())
