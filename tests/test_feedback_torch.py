"""Torch's feedback reaching the gas the way FLASH delivered it.

Two things Torch does not say out loud:

- Winds are never a call. Torch writes each star's mass-loss rate and
  terminal velocity (set_particle_wind_mass/_vel) and FLASH injected them in
  its own step. Unwired, a coupled run had no winds at all.
- Every SN arrives with frac_kin = -1, FLASH's "from the local conditions"
  (Simpson et al. 2015). Taken literally it injected 2e51 erg of heat.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402

from luna.sources.feedback import (inject_energy, inject_wind,  # noqa: E402
                                   make_wind_injector, simpson_kinetic_fraction)
from luna.units import UnitSystem  # noqa: E402
from tests._helpers import _build_cfg_for_test, check  # noqa: E402

N, L = 24, 4.0
DX = L / N
UNITS = UnitSystem(UnitLength_in_cm=3.085677581e18, UnitMass_in_g=1.98892e33,
                   UnitVelocity_in_cm_per_s=1.0e5)          # pc / Msun / km/s


def _cfg(boundary=1, n=N, **over):
    base = dict(test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=boundary,
                Nx=n, Ny=n, Nz=n, Lx=L, Ly=L, Lz=L, gamma=5.0 / 3.0, units=UNITS)
    base.update(over)
    return _build_cfg_for_test(**base)


def _gas(n=N, rho=1.0):
    rng = np.random.default_rng(0)
    U = np.zeros((5, n, n, n))
    U[0] = rho * (1.0 + 0.1 * rng.random((n, n, n)))
    U[4] = 1.0
    return jnp.asarray(U)


def _stars(cap, rows):
    pos = np.zeros((cap, 3)); vel = np.zeros((cap, 3))
    dm = np.zeros(cap); vt = np.zeros(cap); on = np.zeros(cap, bool)
    for i, (p, v, m, t) in enumerate(rows):
        pos[i], vel[i], dm[i], vt[i], on[i] = p, v, m, t, True
    return tuple(jnp.asarray(a) for a in (pos, vel, dm, vt, on))


def test_the_batched_winds_are_inject_wind_per_star():
    cfg = _cfg(boundary=1)
    rows = [((1.3, 2.1, 1.7), (0.5, -0.2, 0.1), 0.02, 30.0),
            ((2.6, 1.4, 2.9), (0.0, 0.3, 0.0), 0.05, 12.0)]
    U = _gas()
    inject = make_wind_injector((N, N, N), (DX, DX, DX), periodic=False)
    got, n, dm, E = inject(U, *_stars(8, rows))
    want = U
    for p, v, m, t in rows:
        want = inject_wind(want, cfg, *p, m, t, v_star=v)
    for c in range(5):
        dU = want[c] - U[c]
        assert float(jnp.max(jnp.abs(dU))) > 0.0
        assert float(jnp.max(jnp.abs(got[c] - want[c]))) <= 1e-12 * float(jnp.max(jnp.abs(dU)))
    assert int(n) == 2
    np.testing.assert_allclose(float(dm), 0.07, rtol=1e-14)
    np.testing.assert_allclose(float(E), 0.5 * (0.02 * 900.0 + 0.05 * 144.0), rtol=1e-14)


def test_a_wind_at_a_periodic_face_wraps_and_conserves_mass():
    inject = make_wind_injector((N, N, N), (DX, DX, DX), periodic=True)
    U = _gas()
    got, *_ = inject(U, *_stars(4, [((0.02, 2.0, 3.99), (0, 0, 0), 0.04, 20.0)]))
    dm = np.asarray(got[0] - U[0])
    np.testing.assert_allclose(dm.sum() * DX ** 3, 0.04, rtol=1e-13)
    assert dm[-1].sum() > 0.0 and dm[:, :, 0].sum() > 0.0      # across both faces


def test_stars_without_wind_leave_the_gas_alone():
    inject = make_wind_injector((N, N, N), (DX, DX, DX), periodic=True)
    U = _gas()
    pos, vel, dm, vt, on = _stars(4, [((2.0, 2.0, 2.0), (0, 0, 0), 0.0, 20.0),
                                      ((1.0, 1.0, 1.0), (0, 0, 0), 0.03, 20.0)])
    got, n, _, _ = inject(U, pos, vel, dm, vt, on.at[1].set(False))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(U))
    assert int(n) == 0


def test_a_resolved_sn_with_torchs_minus_one_injects_its_energy_once():
    cfg = _cfg(boundary=1)
    U = _gas(rho=10.0)                   # ~300 cm^-3: r_PDS ~1.6 pc > 4.5 dx = 0.75 pc
    E_sn = UNITS.to_code_energy(1e51)
    out = inject_energy(U, cfg, 2.0, 2.0, 2.0, E_sn, mass=0.0, f_kin=-1.0)
    added = float(jnp.sum(out[4] - U[4])) * DX ** 3
    np.testing.assert_allclose(added, E_sn, rtol=1e-12)


def test_the_kinetic_fraction_follows_flash():
    E = UNITS.to_code_energy(1e51)
    assert simpson_kinetic_fraction(_cfg(), 10.0, E) == 0.0         # resolved
    coarse = _cfg(n=4, Lx=400.0, Ly=400.0, Lz=400.0)                 # 100 pc cells
    rho = 0.01                                                       # ~0.3 cm^-3
    n_cm = UNITS.to_cgs_density(rho) / (1.3 * 1.6726e-24)
    t = 26.5 * n_cm ** (-4.0 / 7.0)
    r = 18.5 * n_cm ** (-3.0 / 7.0)
    want = 3.97e-6 * n_cm * r ** 7 * t ** -2 * 100.0 ** -2
    np.testing.assert_allclose(simpson_kinetic_fraction(coarse, rho, E), want, rtol=1e-6)
    assert 0.0 < want


@pytest.mark.parametrize("dt", [0.01])
def test_the_worker_injects_torchs_parked_wind_rates(dt):
    from amuse_luna.worker import LunaWorker
    w = LunaWorker(_cfg(boundary=0, star_max_count=8,
                        ORDER=1, LIMITER=0, enable_x64=True))
    w.initialize_code(build_ic=True)
    w.commit_parameters()
    tag = w.new_particle(2.0, 2.0, 2.0)
    w.set_particle_mass(tag, [30.0])
    w.set_particle_wind_mass(tag, [2.0])             # Msun per code time
    w.set_particle_wind_vel(tag, [25.0])             # km/s
    m0 = float(jnp.sum(w.state.U[0])) * DX ** 3
    U0 = np.asarray(w.state.U)
    w.evolve_model(w.get_time() + dt)
    m1 = float(jnp.sum(w.state.U[0])) * DX ** 3
    np.testing.assert_allclose(m1 - m0, 2.0 * dt, rtol=1e-10)
    # periodic hydro plus a radial wind from a star at rest: the net momentum is unchanged
    U1 = np.asarray(w.state.U)
    check("worker wind |dP|/sqrt(M E)", float(np.abs(U1[1:4].sum((1, 2, 3)) - U0[1:4].sum((1, 2, 3))).max()
                                              / np.sqrt(U0[0].sum() * U0[4].sum())), 1e-13)
    assert "winds n=1 dm=2.000e-02" in w.feedback_report()


@pytest.mark.parametrize("add_heat", [False, True])
def test_winds_add_mass_and_momentum_and_heat_only_when_asked(add_heat):
    """Mass and momentum go in either way. By default the internal energy is
    untouched (FLASH-Torch's momentum injection, add_therm_e off); with
    add_heat the ejecta's whole kinetic energy is added, and what is not in
    the mixed cell's bulk motion becomes heat."""
    cfg = _cfg(boundary=1)
    rows = [((1.3, 2.1, 1.7), (0.5, -0.2, 0.1), 0.02, 30.0),
            ((2.6, 1.4, 2.9), (0.0, 0.3, 0.0), 0.05, 12.0)]
    U = _gas()
    inject = make_wind_injector((N, N, N), (DX, DX, DX), periodic=False, add_heat=add_heat)
    got, *_ = inject(U, *_stars(8, rows))
    want = U
    for p, v, m, t in rows:
        want = inject_wind(want, cfg, *p, m, t, v_star=v, add_heat=add_heat)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12, atol=1e-14)
    dV = DX ** 3
    np.testing.assert_allclose(float(jnp.sum(got[0] - U[0])) * dV, 0.07, rtol=1e-12)
    ke = lambda A: 0.5 * (A[1] ** 2 + A[2] ** 2 + A[3] ** 2) / A[0]     # noqa: E731
    d_eint = np.asarray((got[4] - ke(got)) - (U[4] - ke(U)))
    if add_heat:
        assert d_eint.max() > 1e-3 * float(U[4].max())                 # it heated
        E_ej = 0.5 * (0.02 * (30.0 ** 2) + 0.05 * (12.0 ** 2))           # plus the star-motion terms
        assert float(jnp.sum(got[4] - U[4])) * dV > 0.9 * E_ej
    else:
        assert np.abs(d_eint).max() < 1e-12 * float(U[4].max())         # internal energy as it was


@pytest.mark.parametrize("add_heat", [False, True])
def test_mass_loading_sweeps_up_gas_and_creates_none(add_heat):
    """Mass loading stands for ambient gas the wind sweeps up, and that gas is
    already in the cells: the gas gains exactly the wind's dm and momentum
    dm (v_star + v_term e_r), as without loading. It used to add
    dm v_term / v_load of new gas (+12 Msun of the +13.8 Msun in fb_256).
    With add_heat the sweep-up dissipates (1 - v_load / v_term) of the
    wind's kinetic energy 0.5 dm v_term^2; a wind slower than v_load is
    left alone."""
    rows = [((2.0, 2.0, 2.0), (0.4, -0.3, 0.2), 0.02, 30.0),    # loaded, f = 3
            ((1.0, 3.0, 1.5), (0.0, 0.2, 0.0), 0.05, 5.0)]      # slower than v_load
    U = _gas()
    kw = dict(periodic=True, add_heat=add_heat)
    plain = make_wind_injector((N, N, N), (DX, DX, DX), **kw)
    loaded = make_wind_injector((N, N, N), (DX, DX, DX), v_load=10.0, **kw)
    a, *_ = plain(U, *_stars(4, rows))
    b, *_ = loaded(U, *_stars(4, rows))
    dV = DX ** 3
    tot = lambda A, k: float(jnp.sum(A[k] - U[k])) * dV        # noqa: E731
    np.testing.assert_allclose(tot(b, 0), 0.02 + 0.05, rtol=1e-12)
    p_want = 0.02 * np.array([0.4, -0.3, 0.2]) + 0.05 * np.array([0.0, 0.2, 0.0])
    np.testing.assert_allclose([tot(b, k) for k in (1, 2, 3)], p_want, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(b[:4]), np.asarray(a[:4]), rtol=1e-14, atol=0)
    if add_heat:
        lost = (1.0 - 10.0 / 30.0) * 0.5 * 0.02 * 30.0 ** 2
        np.testing.assert_allclose(tot(a, 4) - tot(b, 4), lost, rtol=1e-12)
    else:
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-14, atol=0)


@pytest.mark.parametrize("add_heat", [False, True])
def test_an_off_node_wind_adds_only_the_stars_momentum(add_heat):
    """The gas gains sum dm v_star, the radial part cancelling, wherever the star sits.
    Off a node the cells' unit vectors did not cancel: 7.4e-3 of momentum, 0.6 %
    of sum dm v_term, came from nowhere. With add_heat the energy is then exactly
    0.5 dm (v_star^2 + v_term^2): no cell sits on the star, so sum |e|^2 = n."""
    rows = [((1.3, 2.1, 1.7), (0.5, -0.2, 0.1), 0.02, 30.0),
            ((2.6, 1.4, 2.9), (0.0, 0.3, 0.0), 0.05, 12.0)]
    U = _gas()
    got, *_ = make_wind_injector((N, N, N), (DX, DX, DX), periodic=True, add_heat=add_heat)(U, *_stars(8, rows))
    p_want = 0.02 * np.array([0.5, -0.2, 0.1]) + 0.05 * np.array([0.0, 0.3, 0.0])
    dp = np.asarray(jnp.sum(got[1:4] - U[1:4], axis=(1, 2, 3))) * DX ** 3
    check(f"off-node winds add_heat={add_heat} |sum dp - sum dm v_star|", float(np.abs(dp - p_want).max()), 1e-15)
    if add_heat:
        E_want = sum(0.5 * m * (np.dot(v, v) + t * t) for _, v, m, t in rows)
        check("off-node winds |dE - 0.5 dm (v*^2 + vt^2)| / E", abs(float(jnp.sum(got[4] - U[4])) * DX ** 3 - E_want)
              / E_want, 1e-13)


@pytest.mark.parametrize("periodic", [True, False])
def test_a_wind_anywhere_stays_radial_and_adds_no_momentum(periodic):
    """Random stars, at an open corner too: zero net radial momentum, every
    cell still pushed outward, and in the worker's inject_wind as well."""
    rng = np.random.default_rng(1)
    rows = [(tuple(rng.random(3) * L), (0.3, -0.1, 0.2), 0.03, 25.0) for _ in range(6)]
    rows.append(((0.05, 0.1, 3.97), (0.0, 0.0, 0.0), 0.01, 20.0))       # the window's corner octant
    U = _gas()
    inject = make_wind_injector((N, N, N), (DX, DX, DX), periodic=periodic)
    for p, v, m, t in rows:
        got, *_ = inject(U, *_stars(1, [(p, v, m, t)]))
        dU = np.asarray(got[:4] - U[:4])
        dp = dU[1:4].sum(axis=(1, 2, 3)) * DX ** 3
        check(f"wind at {np.round(p, 2)} per={periodic} |sum dp - dm v*| / dm v_t",
              float(np.abs(dp - m * np.array(v)).max()) / (m * t), 1e-14)
        if periodic or min(min(p), L - max(p)) > 3.0 * DX:     # away from a wall: still radial
            cells = np.argwhere(dU[0] > 0.0)
            dr = (cells + 0.5) * DX - np.array(p)
            dr -= L * np.round(dr / L)
            u = dU[1:4][:, cells[:, 0], cells[:, 1], cells[:, 2]].T / dU[0][tuple(cells.T)][:, None]
            u -= np.array(v)
            assert ((u * dr).sum(axis=1) > 0.0).all()
    if not periodic:
        cfg = _cfg(boundary=1)
        W = U
        for p, v, m, t in rows:
            W = inject_wind(W, cfg, *p, m, t, v_star=v)
        p_want = sum(m * np.array(v) for _, v, m, _ in rows)
        dp = np.asarray(jnp.sum(W[1:4] - U[1:4], axis=(1, 2, 3))) * DX ** 3
        check("inject_wind, 7 stars, |sum dp - sum dm v*|", float(np.abs(dp - p_want).max()), 1e-15)


def test_a_one_cell_wind_window_is_refused():
    """One cell has no radial pattern of zero net momentum: refused, not dropped."""
    inject = make_wind_injector((N, N, N), (DX, DX, DX), periodic=True, radius=0.3 * DX)
    with pytest.raises(ValueError, match="1-cell wind window"):
        inject(_gas(), *_stars(1, [((7.6 * DX, 12.6 * DX, 10.4 * DX), (0, 0, 0), 0.02, 30.0)]))
