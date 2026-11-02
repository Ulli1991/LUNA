"""The floor stack (sources/cosmo_floors.py) on a proper-time run.

The floors used to be switched on only for comoving runs; a proper-time cloud
at 1024^3 then carried a cell with negative pressure into the CFL and stopped
(fb_1024, 2026-09-26). These check that a proper clock gets the same stack:
the projection exists, it lifts a vacuum cell and a negative-pressure cell and
leaves a healthy one alone, and the Jeans floor is Truelove's
P >= G rho^2 (N dx)^2 / pi."""
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import jax.numpy as jnp  # noqa: E402

from luna.setup.assemble import build_projections  # noqa: E402
from luna.setup.clock import make_clock  # noqa: E402
from luna.setup.run_kind import GAS  # noqa: E402
from luna.units import KB_OVER_MP_KMS2_PER_K  # noqa: E402
from tests.test_regressions import _build_cfg_for_test  # noqa: E402

pytestmark = pytest.mark.fast

GAMMA = 5.0 / 3.0
N = 8


def _cfg(**kw):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=GAMMA,
        ORDER=1, LIMITER=0, GRAVITATIONAL_G=2.0, POISSON_BOUNDARY=0)
    return replace(cfg, **kw)


def _state():
    """rho 1, at rest, P 1 -- then one vacuum cell and one negative-pressure cell."""
    U = np.zeros((5, N, N, N))
    U[0] = 1.0
    U[4] = 1.0 / (GAMMA - 1.0)
    U[0, 1, 1, 1] = -1e-3                     # a rarefaction overshot below zero
    U[1, 1, 1, 1] = 0.5
    U[0, 5, 5, 5] = 1.0
    U[1, 5, 5, 5] = 2.0                       # kinetic 2.0 > E: negative pressure
    U[4, 5, 5, 5] = 1.0
    return jnp.asarray(U)


def _floors(**kw):
    cfg = _cfg(**kw)
    clock = make_clock(cfg)
    assert not clock.comoving
    U = _state()
    proj = build_projections(cfg, clock, None, GAS, SimpleNamespace(U=U), None, is_main=False)
    return cfg, clock, U, proj


def test_a_proper_run_asking_for_floors_gets_them():
    _, _, _, proj = _floors(cosmo_rho_floor=1e-3, cosmo_T_floor=10.0)
    assert any(type(p).__name__ == "CosmoFloors" for p in proj)


def test_no_floors_asked_none_built():
    _, _, _, proj = _floors()
    assert not any(type(p).__name__ == "CosmoFloors" for p in proj)


def test_the_floors_lift_the_bad_cells_and_leave_the_rest():
    cfg, clock, U, proj = _floors(cosmo_rho_floor=1e-3, cosmo_T_floor=10.0, cosmo_mu=2.3)
    (fl,) = [p for p in proj if type(p).__name__ == "CosmoFloors"]
    V = np.asarray(fl(U, clock.scale(0.0)))
    rho_bar = float(np.mean(np.asarray(U[0])))
    ct = KB_OVER_MP_KMS2_PER_K * 10.0 / 2.3
    # the vacuum cell: floor density, no momentum, the temperature floor's energy
    assert V[0, 1, 1, 1] == pytest.approx(1e-3 * rho_bar, rel=1e-12)
    assert np.all(V[1:4, 1, 1, 1] == 0.0)
    assert V[4, 1, 1, 1] == pytest.approx(ct * 1e-3 * rho_bar / (GAMMA - 1.0), rel=1e-12)
    # the negative-pressure cell keeps its momentum, gets the floor pressure
    ke = 0.5 * 2.0 ** 2 / 1.0
    assert V[1, 5, 5, 5] == 2.0
    P = (GAMMA - 1.0) * (V[4, 5, 5, 5] - ke)
    assert P == pytest.approx(ct * 1.0, rel=1e-12)
    # the vacuum cell's 26 neighbours paid for it (tests/test_floor_conservation.py);
    # every other cell untouched
    mask = np.ones((N, N, N), bool)
    mask[0:3, 0:3, 0:3] = mask[5, 5, 5] = False
    np.testing.assert_array_equal(V[:, mask], np.asarray(U)[:, mask])


def test_the_proper_jeans_floor_is_truelove():
    cfg, clock, U, proj = _floors(cosmo_jeans_ncell=4.0)
    (fl,) = [p for p in proj if type(p).__name__ == "CosmoFloors"]
    dx = cfg.Lx / cfg.Nx
    assert fl.jeans_coeff(clock.scale(0.0)) == pytest.approx(
        cfg.GRAVITATIONAL_G * (4.0 * dx) ** 2 / math.pi, rel=1e-12)
