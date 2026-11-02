"""SN/wind feedback energy injection into the gas (Phase 5 / #4 tractable part).

Checks the LUNA feedback-application physics (sources.feedback.inject_energy)
via the worker's energy_injection: injected energy and ejecta mass land in the
gas, conserved, and localized near the injection site.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402
from tests._helpers import check  # noqa: E402


def _worker(N=16, **kw):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=5.0 / 3.0,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0, **kw,
    )
    w = LunaWorker(cfg)
    w.initialize_code()
    w.commit_parameters()
    return w


def _net_momentum(U):
    """max_d |sum p_d| / sum |p_d|: a radial kick about a cell corner at rest gives no net momentum."""
    return max(abs(float(U[d].sum())) / float(np.abs(U[d]).sum()) for d in (1, 2, 3))


def _uniform_gas(w, N, rho=1.0, P=1.0):
    g = w.gamma
    sh = (N, N, N)
    E = P / (g - 1.0)
    w.state.U = jnp.asarray(np.stack([np.full(sh, rho), np.zeros(sh), np.zeros(sh),
                                      np.zeros(sh), np.full(sh, E)]))


def test_thermal_energy_injection_conserves_totals():
    N = 16
    w = _worker(N)
    _uniform_gas(w, N, rho=1.0, P=1.0)
    dV = (1.0 / N) ** 3
    E0 = float(np.sum(np.asarray(w.state.U[4])) * dV)
    M0 = float(np.sum(np.asarray(w.state.U[0])) * dV)

    w.energy_injection(energy=10.0, frac_kin=0.0, mass=2.0, x=0.5, y=0.5, z=0.5, radius=0.2)
    U = np.asarray(w.state.U)
    E1 = float(np.sum(U[4]) * dV)
    M1 = float(np.sum(U[0]) * dV)

    np.testing.assert_allclose(E1 - E0, 10.0, rtol=1e-6)     # all injected energy lands
    np.testing.assert_allclose(M1 - M0, 2.0, rtol=1e-6)      # ejecta mass conserved
    assert np.all(np.isfinite(U))
    np.testing.assert_array_equal(U[1:4], 0.0)                 # thermal: no momentum at all
    # localized: a corner cell far from the centre is unchanged
    assert U[4, 0, 0, 0] == pytest.approx(1.0 / (w.gamma - 1.0), rel=1e-9)


def test_kinetic_injection_adds_momentum_and_energy():
    N = 16
    w = _worker(N)
    _uniform_gas(w, N, rho=1.0, P=1.0)
    dV = (1.0 / N) ** 3
    E0 = float(np.sum(np.asarray(w.state.U[4])) * dV)
    assert float(np.sum(np.abs(np.asarray(w.state.U[1])))) == 0.0   # no momentum yet

    w.energy_injection(energy=10.0, frac_kin=0.5, mass=0.0, x=0.5, y=0.5, z=0.5, radius=0.2)
    U = np.asarray(w.state.U)
    # total energy still increased by ~the injected amount (gas was at rest)
    np.testing.assert_allclose(float(np.sum(U[4]) * dV) - E0, 10.0, rtol=1e-3)
    # radial kinetic kick created a momentum field (net ~0 by symmetry → use L1)
    assert float(np.sum(np.abs(U[1])) + np.sum(np.abs(U[2])) + np.sum(np.abs(U[3]))) > 0
    np.testing.assert_allclose(float(np.sum(U[0]) * dV), 1.0, rtol=1e-13)   # mass = 0: rho unchanged
    check("kinetic injection |sum p| / sum |p|", _net_momentum(U), 1e-13)
    assert np.all(np.isfinite(U))


@pytest.mark.parametrize("add_heat", [False, True])
def test_wind_injection_adds_mass_momentum_energy(add_heat):
    """By default (FLASH-Torch's momentum injection) the energy gained is the
    kinetic energy of the mixed flow and the internal energy stays put; with
    wind_add_heat the ejecta's whole kinetic energy goes in."""
    N = 16
    w = _worker(N, wind_add_heat=add_heat)
    _uniform_gas(w, N, rho=1.0, P=1.0)
    Eint0 = np.asarray(w.state.U[4])
    dV = (1.0 / N) ** 3
    M0 = float(np.sum(np.asarray(w.state.U[0])) * dV)
    E0 = float(np.sum(np.asarray(w.state.U[4])) * dV)

    dm, vterm = 0.5, 3.0
    w.wind_injection(dm=dm, v_term=vterm, x=0.5, y=0.5, z=0.5, radius=0.2)
    U = np.asarray(w.state.U)

    # ejecta mass conserved
    np.testing.assert_allclose(float(np.sum(U[0]) * dV) - M0, dm, rtol=1e-6)
    # radial wind -> momentum field (net ~0 by symmetry, so use L1) and energy up
    assert float(np.sum(np.abs(U[1])) + np.sum(np.abs(U[2])) + np.sum(np.abs(U[3]))) > 0
    check(f"wind injection add_heat={add_heat} |sum p| / sum |p|", _net_momentum(U), 1e-13)
    ke = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
    if add_heat:
        # injected kinetic energy ~ 0.5 dm v_term^2 (gas was at rest)
        np.testing.assert_allclose(float(np.sum(U[4]) * dV) - E0, 0.5 * dm * vterm**2, rtol=0.05)
    else:
        # the gas was at rest: what went in is the mixed flow's kinetic energy, exactly
        np.testing.assert_allclose(U[4] - ke, Eint0, rtol=1e-12, atol=1e-14)
        assert float(np.sum(ke) * dV) < 0.5 * dm * vterm ** 2   # the inelastic mixing lost some
    assert np.all(np.isfinite(U))
