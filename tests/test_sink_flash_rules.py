"""FLASH-Torch's sink rules (Federrath et al. 2010) as LUNA options.

sink_accretion='federrath': a cell's excess over rho_create goes to a sink only
if the gas is bound to it and falls onto it, and to the most-bound sink when
several reach it. sink_creation_checks='flash': a sink is born empty at a local
potential minimum that converges in every direction and is Jeans unstable and
bound. sink_merging: bound, approaching sinks within r_acc merge. Each test is
built so that the rule it names decides the outcome; the mesh must give the
single device's answer.
"""
import numpy as np
import pytest

import jax

import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from luna.setup.state import Particles  # noqa: E402
from luna.species.sinks import Sinks  # noqa: E402
from luna.species.sinks_sharded import ShardedSinks  # noqa: E402

N, L = 16, 1.0
DX = L / N
GAMMA = 5.0 / 3.0
THR = 1.0
KW = dict(N3=(N, N, N), box3=(L, L, L), G=1.0, boundary=0, n_max=4, softening=0.0,
          accretion_radius=2.5 * DX, creation_density=THR, creation_separation=0.0,
          create_frac=0.9, sink_sink=False, gamma=GAMMA, is_main=False)
X1 = (np.arange(N) + 0.5) * DX
X, Y, Z = np.meshgrid(X1, X1, X1, indexing="ij")
C = (8.5 * DX,) * 3                        # a cell centre


def _U(rho, v, eint=1.0):
    rho = np.broadcast_to(np.asarray(rho, float), (N, N, N))
    v = np.broadcast_to(np.asarray(v, float), (3, N, N, N))
    eint = np.broadcast_to(np.asarray(eint, float), (N, N, N))
    return np.stack([rho, rho * v[0], rho * v[1], rho * v[2],
                     eint + 0.5 * rho * (v ** 2).sum(axis=0)])


def _radial(c, v0):
    """v = v0 r_hat about c (0 in the cell at c)."""
    d = np.stack([X - c[0], Y - c[1], Z - c[2]])
    r = np.sqrt((d ** 2).sum(axis=0))
    return v0 * d / np.where(r > 0, r, 1.0)


def _sinks(positions, masses, vels=None, n_max=4):
    pos = np.zeros((n_max, 3)); m = np.zeros(n_max); a = np.zeros(n_max)
    vel = np.zeros((n_max, 3))
    for i, p in enumerate(positions):
        pos[i], m[i], a[i] = p, masses[i], 1.0
        if vels is not None:
            vel[i] = vels[i]
    return jnp.asarray(pos), jnp.asarray(vel), jnp.asarray(m), jnp.asarray(a)


def _accrete(mode, U, sinks, **kw):
    sk = Sinks(**{**KW, "accretion": mode, **kw})
    pos, vel, m, a = sinks
    U2, pos2, vel2, m2, a2, mom = sk._accrete(jnp.asarray(U), pos, vel, m, a, sk.zero_moments())
    return sk, np.asarray(U2), np.asarray(pos2), np.asarray(vel2), np.asarray(m2)


def _excess_where(U, mask):
    return float(np.sum(np.maximum(U[0] - THR, 0.0)[mask])) * DX ** 3


def _dist(c):
    return np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)


# -- accretion ---------------------------------------------------------------

def test_outflowing_gas_is_taken_by_threshold_but_not_by_federrath():
    """Bound gas (a massive sink) moving away from the sink. Threshold takes
    every cell's excess in r_acc; federrath only the inner cell (r < dx),
    which FLASH accretes without checks."""
    U = _U(4.0, _radial(C, 0.05))
    sinks = _sinks([C], [50.0])
    _, _, _, _, m_thr = _accrete("threshold", U, sinks)
    _, _, _, _, m_fed = _accrete("federrath", U, sinks)
    r = _dist(C)
    gain_thr, gain_fed = m_thr[0] - 50.0, m_fed[0] - 50.0
    assert gain_thr == pytest.approx(_excess_where(U, r < KW["accretion_radius"]), rel=1e-12)
    assert gain_fed == pytest.approx(_excess_where(U, r < DX), rel=1e-12)
    assert gain_fed < 0.1 * gain_thr


def test_unbound_fast_gas_is_not_taken():
    """Gas falling in far faster than escape speed from the sink: only the
    inner cell."""
    U = _U(4.0, _radial(C, -50.0))
    _, _, _, _, m = _accrete("federrath", U, _sinks([C], [1.0]))
    assert m[0] == pytest.approx(1.0 + _excess_where(U, _dist(C) < DX), rel=1e-12)


def test_bound_infalling_gas_is_taken_like_threshold():
    """Slow infall onto a massive sink: every cell in r_acc passes both
    checks, so federrath takes what threshold takes."""
    U = _U(4.0, _radial(C, -0.05))
    sinks = _sinks([C], [50.0])
    _, U_thr, _, _, m_thr = _accrete("threshold", U, sinks)
    _, U_fed, _, _, m_fed = _accrete("federrath", U, sinks)
    assert m_fed[0] == pytest.approx(m_thr[0], rel=1e-12)
    np.testing.assert_allclose(U_fed, U_thr, rtol=0, atol=1e-13)
    assert m_fed[0] - 50.0 > 10 * _excess_where(U, _dist(C) < DX)


def _e(M, r2, u2, eps2):
    """Federrath's specific energy of gas about a sink, zero at r_acc."""
    ra2 = KW["accretion_radius"] ** 2
    return (0.5 * u2 - KW["G"] * M * (1 / np.sqrt(r2 + eps2) - 1 / np.sqrt(ra2 + eps2))
            - 2 * np.pi / 3 * KW["G"] * THR * (ra2 - r2))


def test_a_cell_two_sinks_reach_goes_whole_to_the_more_bound():
    """Gas and sinks at rest (so every cell counts as falling in), two sinks
    2 dx apart, one 10x heavier. Each cell's excess goes whole to the sink
    it is most bound to -- which is the light one close to it, the heavy one
    further out -- as a direct evaluation of the energies says."""
    a, b = C, (C[0] + 2 * DX, C[1], C[2])
    rng = np.random.default_rng(1)
    U = _U(3.0 + rng.random((N, N, N)), 0.0)
    M = (20.0, 2.0)
    _, U2, _, _, m2 = _accrete("federrath", U, _sinks([a, b], M))
    eps2 = (2 * DX) ** 2
    rA2, rB2 = _dist(a) ** 2, _dist(b) ** 2
    ra2 = KW["accretion_radius"] ** 2
    eA, eB = _e(M[0], rA2, 0.0, eps2), _e(M[1], rB2, 0.0, eps2)
    okA = (rA2 < ra2) & ((rA2 < DX ** 2) | (eA < 0))
    okB = (rB2 < ra2) & ((rB2 < DX ** 2) | (eB < 0))
    toA = okA & (~okB | (eA <= eB))
    toB = okB & (~okA | (eB < eA))
    assert toA.sum() > 0 and toB.sum() > 0 and (toA & toB).sum() == 0
    assert (okA & okB & toB).sum() > 0                  # contested cells the light one wins
    np.testing.assert_allclose(m2[0] - M[0], _excess_where(U, toA), rtol=1e-12)
    np.testing.assert_allclose(m2[1] - M[1], _excess_where(U, toB), rtol=1e-12)
    np.testing.assert_allclose(U2[0][toA | toB], THR, rtol=0, atol=1e-12)


def test_federrath_conserves_mass_momentum_and_centre_of_mass():
    """Moving gas, moving overlapping sinks: gas lost = mass gained, momentum
    likewise, no cell drained below rho_create (none taken twice), and the
    sinks moved to the centre of mass keep the system's centre of mass."""
    a, b = C, (C[0] + 2 * DX, C[1], C[2])
    rng = np.random.default_rng(2)
    U = _U(3.0 + rng.random((N, N, N)), rng.normal(0.0, 0.05, (3, N, N, N)))
    sinks = _sinks([a, b], [20.0, 2.0], vels=[(0.01, 0, 0), (-0.01, 0.02, 0)])
    _, U2, pos2, vel2, m2 = _accrete("federrath", U, sinks)
    pos, vel, m, _ = (np.asarray(x) for x in sinks)
    dV = DX ** 3
    assert np.sum(m2 - m) > 0.0
    np.testing.assert_allclose(np.sum(m2 - m), np.sum(U[0] - U2[0]) * dV, rtol=1e-12)
    p_gas = (U[1:4] - U2[1:4]).sum(axis=(1, 2, 3)) * dV
    np.testing.assert_allclose((m2[:, None] * vel2 - m[:, None] * vel).sum(0), p_gas,
                               rtol=1e-10, atol=1e-14)
    drained = U2[0] < U[0]
    assert U2[0][drained].min() >= THR - 1e-12
    # the sinks sit away from the box faces: the plain dipole is the centre of mass
    com0 = (m[:, None] * pos).sum(0) + np.stack([(U[0] * q).sum() for q in (X, Y, Z)]) * dV
    com1 = (m2[:, None] * pos2).sum(0) + np.stack([(U2[0] * q).sum() for q in (X, Y, Z)]) * dV
    np.testing.assert_allclose(com1, com0, rtol=1e-12)


# -- creation ----------------------------------------------------------------

def _blob(c=(0.5, 0.5, 0.5), v_scale=-1.0, eint=1.0, omega=0.0):
    r2 = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2
    rho = 0.5 + 30.0 * np.exp(-r2 / 0.01)
    v = v_scale * np.stack([X - c[0], Y - c[1], Z - c[2]])
    v = v + omega * np.stack([-(Y - c[1]), X - c[0], 0 * X])
    phi = -50.0 * np.exp(-r2 / 0.02)
    return _U(rho, v, eint), phi


def _finish(sk, U, phi, sinks=None):
    n = sk.n_max
    pos, vel, m, a = sinks if sinks is not None else (
        jnp.zeros((n, 3)), jnp.zeros((n, 3)), jnp.zeros(n), jnp.zeros(n))
    U = jnp.asarray(U)
    part = Particles(pos, vel, m, a, jnp.zeros((n, 3)), extra=sk.zero_moments())
    U2, p2 = sk.finish(U, part, jnp.asarray(phi), sk.observe(U))
    return np.asarray(U2), p2


def test_flash_creation_makes_one_sink_at_a_bound_converging_minimum():
    """The minimum sits on the corner of 8 cells with equal potential: all 8
    are candidates and FLASH merges them at birth -- one sink. It is born
    empty; its mass is what the accretion then took, it sits at the centre
    of mass of that gas, and the gas lost exactly that."""
    U, phi = _blob()
    sk = Sinks(**{**KW, "creation": "flash", "accretion": "federrath"})
    U2, p = _finish(sk, U, phi)
    act = np.asarray(p.active)
    assert act.sum() == 1.0
    m = float(np.asarray(p.mass)[act > 0][0])
    assert m == pytest.approx((U[0] - U2[0]).sum() * DX ** 3, rel=1e-12)
    dm = (U[0] - U2[0]) * DX ** 3
    com = np.array([(dm * q).sum() for q in (X, Y, Z)]) / dm.sum()
    np.testing.assert_allclose(np.asarray(p.pos)[act > 0][0], com, rtol=1e-12)


def test_flash_creation_needs_a_potential_minimum():
    """The same dense, converging, bound blob on a potential slope: no
    local minimum, no sink."""
    U, _ = _blob()
    sk = Sinks(**{**KW, "creation": "flash"})
    _, p = _finish(sk, U, -200.0 * X)
    assert float(np.asarray(p.active).sum()) == 0.0


def test_flash_creation_needs_convergence_in_every_direction():
    U, phi = _blob(v_scale=+1.0)                  # the same blob, expanding
    sk = Sinks(**{**KW, "creation": "flash"})
    _, p = _finish(sk, U, phi)
    assert float(np.asarray(p.active).sum()) == 0.0


def _E_grav(U, phi, c=(7, 7, 7)):
    """E_grav of the control volume around cell c (evaluated directly)."""
    o = np.arange(-2, 3)
    O2 = o[:, None, None] ** 2 + o[None, :, None] ** 2 + o[None, None, :] ** 2
    ix, iy, iz = ((c[a] + o) % N for a in range(3))
    sel = (ix[:, None, None], iy[None, :, None], iz[None, None, :])
    w = O2 <= 9
    ph = phi[sel]
    return float((w * U[0][sel] * (ph[w].max() - ph)).sum()) * DX ** 3, int(w.sum())


def test_flash_creation_needs_jeans_instability():
    """E_th = 0.75 E_grav: bound (E_th + E_kin < E_grav) but not Jeans
    unstable (2 E_th > E_grav) -- refused. At 0.4 E_grav it forms."""
    U, phi = _blob(v_scale=-0.01)
    Eg, n = _E_grav(U, phi)
    sk = Sinks(**{**KW, "creation": "flash"})
    for f, want in ((0.75, 0.0), (0.4, 1.0)):
        Uf, _ = _blob(v_scale=-0.01, eint=f * Eg / (n * DX ** 3))
        _, p = _finish(sk, Uf, phi)
        assert float(np.asarray(p.active).sum()) == want, f


def test_flash_creation_needs_negative_total_energy():
    """Fast solid-body rotation adds kinetic energy about the centre of mass
    without breaking the convergence check (no radial motion) or the Jeans
    check (cold): only E_th + E_kin <= E_grav refuses it."""
    U, phi = _blob(v_scale=-0.01, omega=400.0)
    sk = Sinks(**{**KW, "creation": "flash"})
    _, p = _finish(sk, U, phi)
    assert float(np.asarray(p.active).sum()) == 0.0
    U, phi = _blob(v_scale=-0.01, omega=0.0)      # the same, not rotating: forms
    _, p = _finish(sk, U, phi)
    assert float(np.asarray(p.active).sum()) == 1.0


def test_flash_creation_uses_local_minima_not_the_global_one():
    """The deepest well already has a sink; a second, shallower one is a
    bound, converging local minimum. LUNA's own rule only ever looks at the
    global minimum; FLASH's forms the second sink."""
    c2 = (0.25, 0.25, 0.25)
    U1, phi1 = _blob()
    U2_, phi2 = _blob(c=c2)
    r2a = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    r2b = (X - c2[0]) ** 2 + (Y - c2[1]) ** 2 + (Z - c2[2]) ** 2
    near_b = r2b < r2a
    U = np.where(near_b[None], U2_, U1)
    phi = np.where(near_b, 0.8 * phi2, phi1)
    have = _sinks([(0.5, 0.5, 0.5)], [5.0])
    for mode, want in (("luna", 1.0), ("flash", 2.0)):
        sk = Sinks(**{**KW, "creation": mode})
        _, p = _finish(sk, U, phi, have)
        assert float(np.asarray(p.active).sum()) == want, mode


# -- merging -----------------------------------------------------------------

def _merge(positions, masses, vels):
    sk = Sinks(**{**KW, "merging": True})
    pos, vel, m, a = _sinks(positions, masses, vels)
    return (m, vel, pos), sk._merge(pos, vel, m, a, sk.zero_moments())


def test_bound_approaching_sinks_merge_conserving_mass_momentum_and_com():
    (m, vel, pos), (pos2, vel2, m2, a2, _) = _merge(
        [C, (C[0] + 1.5 * DX, C[1], C[2])], [3.0, 1.0], [(0.1, 0, 0), (-0.2, 0.05, 0)])
    m, vel, pos, pos2, vel2, m2, a2 = (np.asarray(x) for x in (m, vel, pos, pos2, vel2, m2, a2))
    assert a2.tolist() == [1.0, 0.0, 0.0, 0.0]           # the light one went into the heavy
    assert m2[0] == pytest.approx(4.0, rel=1e-15)
    np.testing.assert_allclose(m2[0] * vel2[0], (m[:, None] * vel).sum(0), rtol=1e-14)
    np.testing.assert_allclose(m2[0] * pos2[0], (m[:, None] * pos).sum(0), rtol=1e-14)


def test_receding_or_unbound_sinks_do_not_merge():
    b = (C[0] + 1.5 * DX, C[1], C[2])
    _, out = _merge([C, b], [3.0, 1.0], [(-0.1, 0, 0), (0.2, 0, 0)])        # receding
    assert np.asarray(out[3]).sum() == 2.0
    _, out = _merge([C, b], [3.0, 1.0], [(100.0, 0, 0), (-100.0, 0, 0)])    # unbound
    assert np.asarray(out[3]).sum() == 2.0


# -- the mesh gives one device's answer ----------------------------------------

def _mesh():
    if jax.device_count() < 8:
        pytest.skip("needs 8 devices (the suite's CPU backend has them)")
    return Mesh(np.array(jax.devices()[:8]).reshape(2, 2, 2), ("x", "y", "z"))


def _both(kw, U, phi, sinks):
    mesh = _mesh()
    sk, sh = Sinks(**kw), ShardedSinks(mesh, **kw)
    n = sk.n_max
    pos, vel, m, a = sinks
    part = Particles(pos, vel, m, a, jnp.zeros((n, 3)), extra=sk.zero_moments())
    U, phi = jnp.asarray(U), jnp.asarray(phi)
    divv = sk.observe(U)
    Ua, pa = sk.finish(U, part, phi, divv)
    put = lambda x, s: jax.device_put(x, NamedSharding(mesh, s))   # noqa: E731
    Ub, pb = sh.finish(put(U, P(None, "x", "y", "z")), sh._replicate(part),
                       put(phi, P("x", "y", "z")), put(divv, P("x", "y", "z")))
    return (np.asarray(U), (np.asarray(Ua), pa), (np.asarray(Ub), pb))


def _same(U0, a, b):
    (Ua, pa), (Ub, pb) = a, b
    assert np.abs(Ua - U0).max() > 0.0
    np.testing.assert_allclose(Ub, Ua, rtol=1e-12, atol=1e-14)
    for k in ("pos", "p", "mass", "active"):
        np.testing.assert_allclose(np.asarray(getattr(pb, k)), np.asarray(getattr(pa, k)),
                                   rtol=1e-12, atol=1e-14)
    for k in pa.extra:
        np.testing.assert_allclose(np.asarray(pb.extra[k]), np.asarray(pa.extra[k]),
                                   rtol=1e-12, atol=1e-14)


# sinks on the block corner, faces, an interior and across the periodic edge
# (the first two overlap)
MESH_SINKS = [(0.5, 0.5, 0.5), (0.5, 0.5 + 1.5 * DX, 0.5), (0.2, 0.2, 0.8), (0.02, 0.97, 0.5)]


def test_federrath_accretion_on_the_mesh_is_one_devices():
    rng = np.random.default_rng(5)
    U = _U(2.0 + 3.0 * rng.random((N, N, N)), rng.normal(0.0, 0.3, (3, N, N, N)))
    sinks = _sinks(MESH_SINKS, [3.0, 1.0, 2.0, 4.0],
                   vels=[(0.1, 0, 0), (0, -0.2, 0.1), (0, 0, 0.3), (-0.1, 0.1, 0)])
    U0, a, b = _both({**KW, "accretion": "federrath"}, U, np.ones((N, N, N)), sinks)
    _same(U0, a, b)
    assert float(np.asarray(a[1].active).sum()) == 4.0


def test_flash_creation_and_merging_on_the_mesh_are_one_devices():
    """A blob on the corner of all eight blocks forms a sink; two existing
    sinks elsewhere are bound and approaching and merge."""
    U, phi = _blob()
    far = [(0.15, 0.15, 0.15), (0.15 + 1.5 * DX, 0.15, 0.15)]
    sinks = _sinks(far, [2.0, 1.0], vels=[(0.05, 0, 0), (-0.05, 0, 0)])
    kw = {**KW, "accretion": "federrath", "creation": "flash", "merging": True}
    U0, a, b = _both(kw, U, phi, sinks)
    _same(U0, a, b)
    assert float(np.asarray(a[1].active).sum()) == 2.0      # one merged, one born


def test_the_modes_are_validated():
    with pytest.raises(ValueError):
        Sinks(**{**KW, "accretion": "bondi"})
    with pytest.raises(ValueError):
        Sinks(**{**KW, "creation": "seren"})


def test_the_candidate_ranking_is_the_k_lowest_by_value_then_index():
    """`_lowest_k` (row minima, then K rows) against a sort of the whole grid,
    on integer values with many ties and cells that are no candidate (inf)."""
    from luna.species.sinks import _lowest_k
    rng = np.random.default_rng(7)
    for shape, K in (((6, 5, 7), 8), ((4, 4, 16), 8), ((3, 2, 2), 8)):
        g = rng.integers(0, 4, shape).astype(float)
        g[rng.random(shape) < 0.5] = np.inf
        v, f = _lowest_k(jnp.asarray(g), K)
        want = np.lexsort((np.arange(g.size), g.reshape(-1)))[:K]
        np.testing.assert_array_equal(np.asarray(f), want)
        np.testing.assert_array_equal(np.asarray(v), g.reshape(-1)[want])
