"""FLASH-Torch's wind (wind_mass_load_mode "torch"/"sink") against a cell-by-cell
transliteration of inject_direct.F90, and the mass bookkeeping of both modes."""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from luna.sources.feedback import (TORCH_REF_RADIUS_CELLS, make_torch_wind_injector,  # noqa: E402
                                   torch_load, torch_wind_radius, weaver_radius)

N, L = 24, 4.0
DX = L / N
DV = DX ** 3
T_TARGET = 5.0e6
V_REF = (T_TARGET / 1.38e7) ** 0.5 * 1000.0      # km/s: FLASH's refVel, code velocity km/s
R_REF = TORCH_REF_RADIUS_CELLS * DX


def _gas(seed=0, still=False):
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.5 * rng.random((N, N, N))
    v = np.zeros((3, N, N, N)) if still else 20.0 * rng.standard_normal((3, N, N, N))
    eint = 50.0 + 10.0 * rng.random((N, N, N))              # specific
    U = np.concatenate([rho[None], rho * v, (rho * (eint + 0.5 * (v * v).sum(0)))[None]])
    return U


def _ro(U):
    """Round-off of a sum over the grid mass."""
    return 1e-14 * float(np.sum(np.asarray(U)[0])) * DV


def _overlap(rad, center, cell_bot, cell_top, nsteps=10):
    """overlap.F90 (ishp 1), the three loops vectorised."""
    d = (cell_top - cell_bot) / nsteps
    i = np.arange(nsteps)
    xsq = (cell_bot[0] + (0.5 + i) * d[0] - center[0]) ** 2
    ysq = (cell_bot[1] + (0.5 + i) * d[1] - center[1]) ** 2
    zsq = (cell_bot[2] + (0.5 + i) * d[2] - center[2]) ** 2
    r = np.sqrt(xsq[:, None, None] + ysq[None, :, None] + zsq[None, None, :])
    dd = rad / np.sqrt(3.0)
    ratio = r / dd
    a = (3.0 / np.sqrt(2.0) - 2.0) / (np.sqrt(3.0) - np.sqrt(2.0)) ** 2
    with np.errstate(divide="ignore"):
        w = np.where(ratio <= 1.0, 1.0,
                     np.where(ratio <= np.sqrt(2.0), 3.0 * (rad / np.sqrt(3.0)) / r - 2.0,
                              np.where(r <= rad, a * (np.sqrt(3.0) - ratio) ** 2, 0.0)))
    return w.sum() / nsteps ** 3


def _inject_direct(U, loc, dm, v_w, rad, cons_quant="momentum", add_therm_e=False, ref=V_REF):
    """inject_direct.F90 with mass_load on, one star, periodic box: (U, mass added)."""
    dens, vel = U[0].copy(), U[1:4] / U[0]
    eint = U[4] / U[0] - 0.5 * (vel * vel).sum(0)
    if cons_quant == "momentum":
        mlf = v_w / ref - 1.0
        inj_v, inj_m = v_w / (1.0 + mlf), dm * (1.0 + mlf)
    else:
        mlf = v_w ** 2 / ref ** 2 - 1.0
        inj_v, inj_m = v_w / np.sqrt(1.0 + mlf), dm * (1.0 + mlf)
    injE = 0.5 * dm * v_w ** 2
    cells, sum_ov = [], 0.0
    K = int(np.ceil(rad / DX)) + 2
    c0 = np.floor(np.asarray(loc) / DX).astype(int)
    for di in range(-K, K + 1):
        for dj in range(-K, K + 1):
            for dk in range(-K, K + 1):
                ijk = c0 + np.array([di, dj, dk])
                x = (ijk + 0.5) * DX                        # this image of the cell
                coll = np.maximum(x - 0.5 * DX, np.minimum(loc, x + 0.5 * DX))
                if ((coll - loc) ** 2).sum() > rad ** 2:
                    continue
                ov = _overlap(rad, np.asarray(loc), x - 0.5 * DX, x + 0.5 * DX)
                r = np.sqrt(((x - loc) ** 2).sum())
                u = (x - loc) / r * inj_v if r != 0.0 else np.zeros(3)
                if ov > 0.0:
                    sum_ov += ov
                    cells.append((tuple(np.mod(ijk, N)), ov, u))
    added = 0.0
    for (i, j, k), ov, u in cells:
        dD = ov / sum_ov * inj_m / DV
        oD, oV = dens[i, j, k], vel[:, i, j, k].copy()
        nD = oD + dD
        if cons_quant == "momentum":
            nV = (u * dD + oV * oD) / nD
            dKE = 0.5 * (nV ** 2).sum() * nD - 0.5 * (oV ** 2).sum() * oD
            add = (injE * ov / sum_ov / DV - dKE) if add_therm_e else 0.0
            eint[i, j, k] = eint[i, j, k] * (oD / nD) + add / nD
        else:
            nVsq = oD / nD * np.sign(oV) * oV ** 2 + dD / nD * np.sign(u) * u ** 2
            nV = np.sign(nVsq) * np.sqrt(np.abs(nVsq))
        vel[:, i, j, k], dens[i, j, k] = nV, nD
        added += dD * DV
    out = np.concatenate([dens[None], dens * vel, (dens * (eint + 0.5 * (vel * vel).sum(0)))[None]])
    return out, added


def _one(pos, dm, v_w, cap=4, bg=0.0, t_w=1.0):
    P = np.zeros((cap, 3)); P[0] = pos
    md = np.zeros(cap); md[0] = dm
    vt = np.zeros(cap); vt[0] = v_w
    on = np.zeros(cap, bool); on[0] = True
    return P, md, vt, on, np.full(cap, bg), np.full(cap, t_w)


# -- the formulas -------------------------------------------------------------

@pytest.mark.parametrize("v_w, quant, F", [(2000.0, "momentum", 2000.0 / V_REF),
                                           (2000.0, "energy", (2000.0 / V_REF) ** 2),
                                           (100.0, "momentum", 100.0 / V_REF)])
def test_the_loading_is_inject_directs(v_w, quant, F):
    """F dm at v_ref, no floor at F = 1."""
    got_F, v_inj = torch_load(v_w, V_REF, quant)
    np.testing.assert_allclose(got_F, F, rtol=1e-14)
    np.testing.assert_allclose(v_inj, V_REF, rtol=1e-14)


def test_the_radius_is_weavers_clipped_as_inject_direct_does():
    """Weaver R_1 in cgs equals code units; clipped to [sqrt(3) dx, r_ref]; r_ref while unmeasured."""
    Msun, yr, pc = 1.98892e33, 3.15576e7, 3.085677581e18
    mdot, rho0, v, t = 1e-6 * Msun / yr, 1e-22, 2.0e8, 1.0e6 * yr        # cgs
    r_cgs = 0.74296 * (1.0 / rho0 * mdot) ** (3.0 / 10.0) * v ** (1.0 / 10.0) * t ** (2.0 / 5.0)
    t_code = pc / 1e5
    r_code = weaver_radius(mdot / Msun * t_code, rho0 * pc ** 3 / Msun, v / 1e5, t / t_code)
    np.testing.assert_allclose(r_code * pc, r_cgs, rtol=1e-12)
    dx = 0.05
    args = (mdot / Msun * t_code, rho0 * pc ** 3 / Msun, v / 1e5)
    big, small = weaver_radius(*args, 1.0), weaver_radius(*args, 1e-12)
    assert small < np.sqrt(3.0) * dx < 3.5 * np.sqrt(3.0) * dx < big
    r = torch_wind_radius(args[0], [args[1], args[1], args[1], 0.0], args[2],
                          [1.0, 1e-12, 3e-3, 3e-3], dx, 3.5 * np.sqrt(3.0) * dx, True)
    mid = weaver_radius(*args, 3e-3)
    assert np.sqrt(3.0) * dx < mid < 3.5 * np.sqrt(3.0) * dx
    np.testing.assert_allclose(r, [3.5 * np.sqrt(3.0) * dx, np.sqrt(3.0) * dx, mid,
                                   3.5 * np.sqrt(3.0) * dx], rtol=1e-14)
    fixed = torch_wind_radius(args[0], [args[1]], args[2], [1e-6], dx, 0.3, False)
    np.testing.assert_allclose(fixed, [0.3], rtol=0)


# -- the grid against the Fortran -------------------------------------------

@pytest.mark.parametrize("quant, add_therm_e", [("momentum", False), ("momentum", True),
                                                ("energy", False)])
def test_the_grid_is_inject_directs(quant, add_therm_e):
    """Every cell as inject_direct.F90 leaves it, sphere wrapping a periodic face."""
    U = _gas(1)
    loc, dm, v_w = np.array([0.13, 2.07, 3.91]), 3e-4, 1800.0
    want, added = _inject_direct(U, loc, dm, v_w, R_REF, quant, add_therm_e)
    inj = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, v_ref=V_REF,
                                   cons_quant=quant, add_therm_e=add_therm_e)
    got, info = inj(jnp.asarray(U), *_one(loc, dm / 0.01, v_w), 0.01)
    got = np.asarray(got)
    F = torch_load(v_w, V_REF, quant)[0]
    np.testing.assert_allclose(added, F * dm, rtol=1e-12)
    for k in range(5):
        np.testing.assert_allclose(got[k], want[k], rtol=1e-11, atol=1e-11 * np.abs(want[k]).max())
    assert np.count_nonzero(np.abs(got - U).sum(0)) > 500      # the whole sphere, not a few cells
    np.testing.assert_allclose((got[0] - U[0]).sum() * DV, F * dm, rtol=0, atol=_ro(U))
    np.testing.assert_allclose(info["dm_created"], (F - 1.0) * dm, rtol=1e-12)
    np.testing.assert_allclose(info["dm"], dm, rtol=1e-14)


def test_momentum_and_energy_as_the_modes_say():
    """Momentum mode: radial dm v_w, e_int kept or +0.5 dm v_w^2; energy mode: +0.5 dm v_w^2 kinetic."""
    loc, dm, v_w, dt = np.array([2.01, 1.93, 2.2]), 3e-4, 1800.0, 0.01
    kw = dict(v_ref=V_REF)
    U = _gas(2)

    def ke(A):
        return 0.5 * (A[1] ** 2 + A[2] ** 2 + A[3] ** 2) / A[0]

    mom = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, **kw)
    a = np.asarray(mom(jnp.asarray(U), *_one(loc, dm / dt, v_w), dt)[0])
    ii = np.argwhere(np.abs(a[0] - U[0]) > 0)
    rel = (ii + 0.5) * DX - loc
    e = rel / np.linalg.norm(rel, axis=1)[:, None]
    p_rad = sum(float(np.dot(a[1:4, i, j, k] - U[1:4, i, j, k], e_)) for (i, j, k), e_ in zip(ii, e)) * DV
    np.testing.assert_allclose(p_rad, dm * v_w, rtol=1e-12)
    np.testing.assert_allclose(a[4] - ke(a), U[4] - ke(U), rtol=0, atol=1e-11 * U[4].max())
    heat = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, add_therm_e=True, **kw)
    b = np.asarray(heat(jnp.asarray(U), *_one(loc, dm / dt, v_w), dt)[0])
    np.testing.assert_allclose((b[4] - U[4]).sum() * DV, 0.5 * dm * v_w ** 2, rtol=0, atol=1e-14 * float(np.abs(U[4]).sum()) * DV)
    S = _gas(2, still=True)
    en = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, cons_quant="energy", **kw)
    c = np.asarray(en(jnp.asarray(S), *_one(loc, dm / dt, v_w), dt)[0])
    np.testing.assert_allclose((ke(c) - ke(S)).sum() * DV, 0.5 * dm * v_w ** 2, rtol=1e-12)
    np.testing.assert_allclose(c[4] / c[0] - ke(c) / c[0], S[4] / S[0] - ke(S) / S[0], rtol=1e-13)


def test_the_background_density_is_taken_once_then_weavers_radius():
    """First call measures rho0 over r_ref and injects there; then Weaver R_1."""
    U = _gas(3)
    loc, mdot, v_w, dt = np.array([2.0, 2.0, 2.0]), 0.03, 1800.0, 0.01
    inj = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, v_ref=V_REF, var_radius=True)
    fixed = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, v_ref=V_REF)
    a, info = inj(jnp.asarray(U), *_one(loc, mdot, v_w), dt)
    np.testing.assert_allclose(np.asarray(a), np.asarray(fixed(jnp.asarray(U), *_one(loc, mdot, v_w), dt)[0]),
                               rtol=1e-14)
    # the mean the Fortran takes: sum(overlap rho) / sum(overlap)
    ws, rs = [], []
    for di in range(-8, 9):
        for dj in range(-8, 9):
            for dk in range(-8, 9):
                ijk = np.floor(loc / DX).astype(int) + [di, dj, dk]
                x = (ijk + 0.5) * DX
                ws.append(_overlap(R_REF, loc, x - 0.5 * DX, x + 0.5 * DX))
                rs.append(U[0][tuple(np.mod(ijk, N))])
    bg = float(np.dot(ws, rs) / np.sum(ws))
    np.testing.assert_allclose(info["bg_rho"][0], bg, rtol=1e-12)
    assert info["bg_rho"][1] == 0.0
    t_w = 1.5
    r1 = weaver_radius(mdot, bg, v_w, t_w)
    assert np.sqrt(3.0) * DX < r1 < R_REF
    b, _ = inj(jnp.asarray(U), *_one(loc, mdot, v_w, bg=bg, t_w=t_w), dt)
    want, _ = _inject_direct(U, loc, mdot * dt, v_w, r1)
    np.testing.assert_allclose(np.asarray(b), want, rtol=1e-11, atol=1e-11 * np.abs(want).max())


# -- "sink": the loaded mass from the nearest sink ----------------------------

def _sinks(masses, far=False):
    pos = np.array([[1.0, 1.0, 1.0], [3.5, 3.5, 3.5], [2.1, 2.2, 2.0]])
    if far:
        pos[2] = [0.2, 0.2, 0.2]
    vel = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, -4.0, 5.0]])
    return pos, vel, np.array(masses, float), np.ones(3)


@pytest.mark.parametrize("add_therm_e", [False, True])
def test_the_sink_pays_the_loading_and_the_budget_closes(add_therm_e):
    """(F - 1) dm leaves the nearest sink with its velocity; mass, momentum and energy close."""
    U = _gas(4)
    loc, mdot, v_w, dt = np.array([2.0, 2.0, 2.0]), 0.03, 1800.0, 0.01
    dm = mdot * dt
    F = v_w / V_REF
    need = (F - 1.0) * dm
    s = _sinks([5.0, 5.0, 5.0])
    m0 = s[2].copy()
    inj = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, v_ref=V_REF, source="sink",
                                   add_therm_e=add_therm_e)
    got, info = inj(jnp.asarray(U), *_one(loc, mdot, v_w), dt, sinks=s)
    got = np.asarray(got)
    np.testing.assert_allclose(m0 - s[2], [0.0, 0.0, need], rtol=0, atol=1e-14 * m0.max())
    np.testing.assert_allclose((got[0] - U[0]).sum() * DV - (m0 - s[2]).sum(), dm, rtol=0, atol=_ro(U))
    assert info["dm_created"] == 0.0 and info["n_capped"] == 0
    np.testing.assert_allclose(info["dm_from_sinks"], need, rtol=1e-14)
    # the torch mode's grid, but its (F-1) dm carrying the sink's velocity
    torch = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, v_ref=V_REF,
                                     add_therm_e=add_therm_e)
    t = np.asarray(torch(jnp.asarray(U), *_one(loc, mdot, v_w), dt)[0])
    np.testing.assert_allclose(got[0], t[0], rtol=1e-14)
    dp = (got[1:4] - t[1:4]).reshape(3, -1).sum(1) * DV
    np.testing.assert_allclose(dp, need * s[1][2], rtol=1e-10)
    if add_therm_e:
        np.testing.assert_allclose((got[4] - U[4]).sum() * DV,
                                   0.5 * dm * v_w ** 2 + 0.5 * need * float(s[1][2] @ s[1][2]),
                                   rtol=1e-12)


def test_a_small_sink_is_emptied_not_driven_negative():
    """A small sink gives what it has, never more; the step loads less."""
    U = _gas(5)
    loc, mdot, v_w, dt = np.array([2.0, 2.0, 2.0]), 0.03, 1800.0, 0.01
    dm = mdot * dt
    small = 0.2 * (v_w / V_REF - 1.0) * dm
    s = _sinks([5.0, 5.0, small])
    inj = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, v_ref=V_REF, source="sink")
    got, info = inj(jnp.asarray(U), *_one(loc, mdot, v_w), dt, sinks=s)
    assert s[2][2] == 0.0 and s[2].min() >= 0.0
    np.testing.assert_allclose(s[2][:2], [5.0, 5.0], rtol=0)
    assert info["n_capped"] == 1
    np.testing.assert_allclose(info["dm_from_sinks"], small, rtol=1e-14)
    np.testing.assert_allclose((np.asarray(got)[0] - U[0]).sum() * DV, dm + small, rtol=0, atol=_ro(U))
    # the Fortran at F_eff = 1 + small/dm (its v_ref -> v_w dm / (dm + small)),
    # plus the moved mass's momentum small v_sink
    want, _ = _inject_direct(U, loc, dm, v_w, R_REF, ref=v_w * dm / (dm + small))
    got = np.asarray(got)
    np.testing.assert_allclose(got[0], want[0], rtol=1e-12)
    dp = (got[1:4] - want[1:4]).reshape(3, -1).sum(1) * DV
    np.testing.assert_allclose(dp, small * s[1][2], rtol=1e-8)


def test_no_sink_means_no_loading():
    U = _gas(6)
    loc, mdot, v_w, dt = np.array([2.0, 2.0, 2.0]), 0.03, 1800.0, 0.01
    inj = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, v_ref=V_REF, source="sink")
    s = _sinks([1.0, 1.0, 1.0])
    s = (s[0], s[1], s[2], np.zeros(3))
    got, info = inj(jnp.asarray(U), *_one(loc, mdot, v_w), dt, sinks=s)
    np.testing.assert_allclose((np.asarray(got)[0] - U[0]).sum() * DV, mdot * dt, rtol=0, atol=_ro(U))
    assert info["n_capped"] == 1 and info["dm_from_sinks"] == 0.0


def test_the_nearest_sink_by_the_particles_boundary():
    """Nearest sink through a periodic face only when the particles are periodic."""
    U = _gas(7)
    loc, mdot, v_w, dt = np.array([3.9, 3.9, 3.9]), 0.03, 1800.0, 0.01
    inj = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, v_ref=V_REF, source="sink")
    s = _sinks([5.0, 5.0, 5.0], far=True)
    s[0][1] = [2.1, 2.1, 2.1]
    inj(jnp.asarray(U), *_one(loc, mdot, v_w), dt, sinks=s, sink_periodic=True)
    assert s[2][2] < 5.0 and s[2][1] == 5.0
    s = _sinks([5.0, 5.0, 5.0], far=True)
    s[0][1] = [2.1, 2.1, 2.1]
    inj(jnp.asarray(U), *_one(loc, mdot, v_w), dt, sinks=s, sink_periodic=False)
    assert s[2][1] < 5.0 and s[2][2] == 5.0


# -- sharded, and the worker -------------------------------------------------

@pytest.mark.skipif(jax.device_count() < 8, reason="a (2,2,2) mesh needs 8 devices")
@pytest.mark.parametrize("source, var", [("torch", True), ("sink", False)])
def test_sharded_is_one_device(source, var):
    """A (2,2,2) mesh gives the one-device grid and bookkeeping."""
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    mesh = Mesh(mesh_utils.create_device_mesh((2, 2, 2), jax.devices()[:8]), ("x", "y", "z"))
    sh = NamedSharding(mesh, P(None, "x", "y", "z"))
    U = _gas(8)
    pos = np.array([[2.0, 2.0, 2.0], [0.05, 3.97, 2.01], [1.0, 3.0, 2.0]])
    mdot, vt = np.array([0.03, 0.02, 0.0]), np.array([1800.0, 2500.0, 900.0])
    args = (pos, mdot, vt, np.ones(3, bool), np.zeros(3), np.full(3, 1e-3), 0.01)
    kw = dict(v_ref=V_REF, var_radius=var, source=source)
    one = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, **kw)
    many = make_torch_wind_injector((N, N, N), (DX, DX, DX), True, sharding=sh, **kw)
    s1, s2 = _sinks([5.0, 1e-4, 5.0]), _sinks([5.0, 1e-4, 5.0])
    a, ia = one(jnp.asarray(U), *args, sinks=s1)
    b, ib = many(jax.device_put(jnp.asarray(U), sh), *args, sinks=s2)
    assert b.sharding == sh
    np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(s2[2], s1[2], rtol=0)
    np.testing.assert_allclose(ib["bg_rho"], ia["bg_rho"], rtol=1e-13)
    for k in ("n", "dm", "dm_created", "dm_from_sinks", "n_capped"):
        assert ia[k] == pytest.approx(ib[k], rel=1e-13)
    assert ia["n"] == 2


def _worker(mode, tmp_path=None, **kw):
    from amuse_luna.worker import LunaWorker
    from luna.units import UnitSystem
    from tests._helpers import _build_cfg_for_test
    units = UnitSystem(UnitLength_in_cm=3.085677581e18, UnitMass_in_g=1.98892e33,
                       UnitVelocity_in_cm_per_s=1.0e5)
    cfg = _build_cfg_for_test(**{**dict(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0, Nx=N, Ny=N, Nz=N,
        Lx=L, Ly=L, Lz=L, gamma=5.0 / 3.0, units=units, ORDER=2, LIMITER=0,
        enable_x64=True, star_max_count=8, boundary_mass_budget=True,
        wind_target_temp_K=T_TARGET, wind_mass_load_mode=mode), **kw})
    w = LunaWorker(cfg)
    w.initialize_code(build_ic=True)
    w.commit_parameters()
    return w


@pytest.mark.parametrize("mode", ["torch", "conservative"])
def test_the_worker_books_the_created_mass(mode):
    """Mode "torch" books (F - 1) dm and the budget line closes; "conservative" creates nothing."""
    w = _worker(mode)
    tag = w.new_particle(2.0, 2.0, 2.0)
    w.set_particle_mass(tag, [30.0])
    w.set_particle_wind_mass(tag, [0.03])
    w.set_particle_wind_vel(tag, [1800.0])
    m0 = float(jnp.sum(w.state.U[0])) * DV
    w.mass_budget_report()                               # the reference sum
    w.set_particle_mass(tag, [30.0 - 0.03 * 2e-4])       # Torch: SeBa's loss off the star
    w.evolve_model(w.get_time() + 1e-4)
    w.evolve_model(w.get_time() + 1e-4)
    dm = 0.03 * 2e-4
    F = 1800.0 / V_REF
    gained = float(jnp.sum(w.state.U[0])) * DV - m0
    line = w.mass_budget_report()
    resid = float(line.split("resid ")[1].split()[0])
    if mode == "torch":
        np.testing.assert_allclose(gained, F * dm, rtol=0, atol=1e-14 * m0)
        np.testing.assert_allclose(w._fb["dm_load"], (F - 1.0) * dm, rtol=1e-12)
        assert "loaded" in line and "wind loading" in w.feedback_report()
    else:
        np.testing.assert_allclose(gained, dm, rtol=0, atol=1e-14 * m0)
        assert "dm_load" not in w._fb and "loaded" not in line
    assert abs(resid) < 1e-12 * m0
    if mode == "torch":                       # without the booked term it would not close
        assert w._fb["dm_load"] > 1e6 * abs(resid)


def test_the_worker_takes_the_loading_from_its_sink():
    """The worker reduces the carry's sink before the step; the store agrees; the budget closes."""
    from dataclasses import replace
    from amuse_luna.worker import SINK_SPECIES
    w = _worker("sink", sinks_enabled=True, sink_max_count=4, sink_creation_density=1e30,
                SELF_GRAVITY=True, POISSON_BOUNDARY=0)
    s = w.state._s
    part = s.species[SINK_SPECIES]
    s.species[SINK_SPECIES] = replace(
        part, pos=part.pos.at[1].set(jnp.array([2.3, 2.1, 1.9])),
        p=part.p.at[1].set(jnp.array([0.5, -0.2, 0.1])),
        mass=part.mass.at[1].set(3.0), active=part.active.at[1].set(1.0))
    tag = w.new_particle(2.0, 2.0, 2.0)
    w.set_particle_mass(tag, [30.0])
    w.set_particle_wind_mass(tag, [0.03])
    w.set_particle_wind_vel(tag, [1800.0])
    w.mass_budget_report()
    w.set_particle_mass(tag, [30.0 - 0.03 * 1e-4])       # Torch: SeBa's loss off the star
    w.evolve_model(w.get_time() + 1e-4)
    need = (1800.0 / V_REF - 1.0) * 0.03 * 1e-4
    part = w.state._s.species[SINK_SPECIES]
    # no accretion at this creation density: the sink lost the loading and only that
    np.testing.assert_allclose(3.0 - float(part.mass[1]), need, rtol=0, atol=1e-14 * 3.0)
    np.testing.assert_allclose(w._fb["dm_from_sinks"], need, rtol=1e-12)
    assert w._fb["dm_load"] == 0.0 and w._fb["n_load_capped"] == 0
    w.set_particle_pointers("sink")
    np.testing.assert_allclose(w.get_particle_mass([1]), [float(part.mass[1])], rtol=0)
    w.set_particle_pointers("mass")
    resid = float(w.mass_budget_report().split("resid ")[1].split()[0])
    assert abs(resid) < 1e-12 * float(jnp.sum(w.state.U[0])) * DV
    assert need > 1e6 * abs(resid)


def _sink(w, slot, pos, vel, mass):
    from dataclasses import replace
    from amuse_luna.worker import SINK_SPECIES
    s = w.state._s
    part = s.species[SINK_SPECIES]
    s.species[SINK_SPECIES] = replace(
        part, pos=part.pos.at[slot].set(jnp.array(pos)), p=part.p.at[slot].set(jnp.array(vel)),
        mass=part.mass.at[slot].set(mass), active=part.active.at[slot].set(1.0))


def _resid(w):
    return float(w.mass_budget_report().split("resid ")[1].split()[0])


@pytest.mark.parametrize("mode", ["conservative", "torch", "sink"])
def test_a_coupled_run_closes_the_budget(mode):
    """Torch's calls (star from a sink, SeBa's loss, an SN) over 6 bridge steps: the budget closes to round-off."""
    w = _worker(mode, BOUNDARY=1, sinks_enabled=True, sink_max_count=4,
                sink_creation_density=1e30, SELF_GRAVITY=True, POISSON_BOUNDARY=0)
    _sink(w, 1, [2.3, 2.1, 1.9], [0.5, -0.2, 0.1], 40.0)
    M = float(jnp.sum(w.state.U[0])) * DV + 40.0
    assert _resid(w) == 0.0                                    # the reference line
    w.set_particle_pointers("sink")                            # make_stars_from_sinks
    w.set_particle_mass([1], [10.0])
    w.set_particle_pointers("mass")
    tag = w.new_particle(2.0, 2.0, 2.0)
    w.set_particle_mass(tag, [30.0])
    m_star, dt, mdot = 30.0, 1e-4, 0.03
    resid = []
    for k in range(6):
        if k == 3:                                             # the star goes off: 5 Msun of ejecta
            w.energy_injection(1.0e3, 0.0, 5.0, 2.0, 2.0, 2.0)
            m_star -= 5.0
            mdot = 0.0
        else:                                                  # SeBa's step: the loss off the star
            m_star -= mdot * dt
        w.set_particle_mass(tag, [m_star])
        w.set_particle_wind_mass(tag, [mdot])
        w.set_particle_wind_vel(tag, [1800.0])
        w.evolve_model(w.get_time() + dt)
        resid.append(_resid(w))
    fb = w._fb
    print(f"{mode}: resid {resid}, winds {fb['dm_wind']:.3e} SN {fb['dm_sn']:.3e} "
          f"created {fb.get('dm_load', 0.0):.3e} from sinks {fb.get('dm_from_sinks', 0.0):.3e}")
    assert max(abs(r) for r in resid) < 1e-12 * M
    b = np.asarray(w.state.sources["boundary_mass"])
    # teeth: every term of the sum is far above the tolerance
    assert fb["dm_wind"] > 1e4 * 1e-12 * M and fb["dm_sn"] == 5.0
    assert np.abs(b[0] - b[1]).max() > 1e3 * 1e-12 * M
    np.testing.assert_allclose(fb["dm_star_loss"], 3 * 0.03 * dt + 5.0, rtol=1e-12)
    if mode == "torch":
        assert fb["dm_load"] > 1e4 * 1e-12 * M
    if mode == "sink":
        assert fb["dm_from_sinks"] > 1e4 * 1e-12 * M and fb["dm_load"] == 0.0
