"""The windowed accretion kernel against the whole-grid one it replaces.

`Sinks._accrete` used to mask the entire grid once per sink SLOT, active or
not: 97% of a 256^3 step with no sinks in the run, and memory scaling as
n_max x N^3. It works on a (2R+1)^3 window now. The arithmetic is meant to be
the same arithmetic, so these compare it against the original, which is
reproduced here rather than described.

Mass is checked explicitly: what leaves the gas has to arrive on the sinks.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from luna.setup.state import Particles  # noqa: E402
from luna.species.sinks import Sinks  # noqa: E402


def _grid(sk):
    """The grid-shaped cell centres the old kernels read (Sinks keeps 1-D)."""
    return jnp.meshgrid(sk.xc, sk.yc, sk.zc, indexing="ij")


def _reference_accrete(sk, U, pos, vel, mass, active):
    """`_accrete` as it was: the full grid, masked, once per slot."""
    X, Y, Z, dV = *_grid(sk), sk.dV
    r_acc, rho_create = sk.r_acc, sk.rho_create
    for i in range(sk.n_max):
        rho = U[0]
        r2 = (X - pos[i, 0]) ** 2 + (Y - pos[i, 1]) ** 2 + (Z - pos[i, 2]) ** 2
        inside = active[i] * (r2 < r_acc * r_acc)
        dm_cell = inside * jnp.maximum(rho - rho_create, 0.0)
        dM = jnp.sum(dm_cell) * dV
        frac = jnp.where(rho > 0, dm_cell / jnp.maximum(rho, 1e-30), 0.0)
        dpx = jnp.sum(frac * U[1]) * dV
        dpy = jnp.sum(frac * U[2]) * dV
        dpz = jnp.sum(frac * U[3]) * dV
        m_old = mass[i]
        m_new = m_old + dM
        px = m_old * vel[i, 0] + dpx
        py = m_old * vel[i, 1] + dpy
        pz = m_old * vel[i, 2] + dpz
        vx = jnp.where(m_new > 0, px / m_new, 0.0)
        vy = jnp.where(m_new > 0, py / m_new, 0.0)
        vz = jnp.where(m_new > 0, pz / m_new, 0.0)
        mass = mass.at[i].set(m_new)
        vel = vel.at[i].set(jnp.array([vx, vy, vz]))
        kf = jnp.where(inside > 0, 1.0 - jnp.minimum(frac, 1.0), 1.0)
        U = U * kf[None]
    return U, pos, vel, mass, active


def _sinks(n_max=4, N=24, boundary=0, r_acc=None, rho_create=1.0):
    return Sinks(N3=(N, N, N), box3=(1.0, 1.0, 1.0), G=1.0, boundary=boundary,
                 n_max=n_max, softening=0.0,
                 accretion_radius=r_acc if r_acc is not None else 2.0 / N,
                 creation_density=rho_create, creation_separation=0.0,
                 create_frac=0.9, sink_sink=False, is_main=False)


def _gas(N, seed=0):
    rng = np.random.default_rng(seed)
    rho = 1.0 + rng.uniform(0.0, 4.0, (N, N, N))
    v = rng.normal(0.0, 0.3, (3, N, N, N))
    e = 1.0 + rng.uniform(0.0, 0.5, (N, N, N))
    U = np.stack([rho, rho * v[0], rho * v[1], rho * v[2],
                  e + 0.5 * rho * (v ** 2).sum(axis=0)])
    return jnp.asarray(U)


def _state(sk, positions):
    n = sk.n_max
    pos = np.zeros((n, 3))
    act = np.zeros(n)
    for i, p in enumerate(positions):
        pos[i] = p
        act[i] = 1.0
    return (jnp.asarray(pos), jnp.zeros((n, 3)), jnp.zeros(n),
            jnp.asarray(act))


@pytest.mark.parametrize("boundary", [0, 1])
def test_window_matches_the_whole_grid(boundary):
    """Interior sinks: the two kernels must agree to round-off."""
    N = 24
    sk = _sinks(N=N, boundary=boundary)
    U = _gas(N)
    pos, vel, mass, act = _state(sk, [(0.5, 0.5, 0.5), (0.3, 0.62, 0.41)])

    Ua, _, va, ma, _ = sk._accrete(U, pos, vel, mass, act)
    Ub, _, vb, mb, _ = _reference_accrete(sk, U, pos, vel, mass, act)

    np.testing.assert_allclose(np.asarray(ma), np.asarray(mb), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(va), np.asarray(vb), rtol=1e-12,
                               atol=1e-14)
    np.testing.assert_allclose(np.asarray(Ua), np.asarray(Ub), rtol=1e-12)
    assert float(np.asarray(ma).sum()) > 0.0, "nothing was accreted"


def test_a_sink_on_a_periodic_face_reaches_across_it():
    """A DELIBERATE difference from the old kernel, not a port artefact.

    The whole-grid version measured `X - pos`, a plain Cartesian distance, so
    in a periodic box a sink sitting near a face never accreted across it --
    the gas one cell away through the boundary was treated as a box-width
    away. The window wraps and measures to the nearest image, so it does.

    The old answer is wrong for a periodic run; it just rarely showed, because
    sinks form at density peaks in the interior. Interior sinks agree to
    round-off (see above) -- this is the only case where the two differ.
    """
    N = 24
    sk = _sinks(N=N, boundary=0)
    U = _gas(N, seed=3)
    pos, vel, mass, act = _state(sk, [(0.01, 0.5, 0.5)])

    Ua, _, _, ma, _ = sk._accrete(U, pos, vel, mass, act)
    _, _, _, mb, _ = _reference_accrete(sk, U, pos, vel, mass, act)

    assert float(ma[0]) > float(mb[0]) * 1.5, (
        "the window should reach across the face and take more than the "
        "old kernel did: {:.6f} vs {:.6f}".format(float(ma[0]), float(mb[0])))

    # and the gas it took is on the far side of the boundary
    drained = np.asarray(U[0]) - np.asarray(Ua[0])
    assert drained[-1, N // 2, N // 2] > 0.0, "nothing taken across the face"

    # mass still balances, which is the property that must not bend
    lost = (float(np.asarray(U[0]).sum())
            - float(np.asarray(Ua[0]).sum())) * sk.dV
    np.testing.assert_allclose(float(ma.sum()), lost, rtol=1e-11)


def test_a_sink_on_a_wall_does_not_reach_through_it():
    """With a non-periodic boundary there is no image to reach, and the window
    must not invent one by wrapping."""
    N = 24
    sk = _sinks(N=N, boundary=1)
    U = _gas(N, seed=3)
    pos, vel, mass, act = _state(sk, [(0.01, 0.5, 0.5)])

    Ua, _, _, ma, _ = sk._accrete(U, pos, vel, mass, act)
    Ub, _, _, mb, _ = _reference_accrete(sk, U, pos, vel, mass, act)

    np.testing.assert_allclose(np.asarray(ma), np.asarray(mb), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(Ua), np.asarray(Ub), rtol=1e-12)
    drained = np.asarray(U[0]) - np.asarray(Ua[0])
    assert drained[-1, N // 2, N // 2] == 0.0, "reached through a wall"


def test_mass_leaving_the_gas_arrives_on_the_sinks():
    """Particles have gone missing on paths like this before."""
    N = 24
    sk = _sinks(N=N, boundary=0)
    U = _gas(N, seed=7)
    pos, vel, mass, act = _state(sk, [(0.5, 0.5, 0.5), (0.2, 0.8, 0.35)])

    Ua, _, _, ma, _ = sk._accrete(U, pos, vel, mass, act)
    lost = (float(np.asarray(U[0]).sum()) - float(np.asarray(Ua[0]).sum())) * sk.dV
    gained = float(np.asarray(ma).sum())
    assert gained > 0.0
    np.testing.assert_allclose(gained, lost, rtol=1e-11)


def test_empty_slots_cost_nothing_and_change_nothing():
    """The capacity is what used to be swept. With no sinks active the gas must
    come back untouched, whatever n_max is."""
    N = 24
    sk = _sinks(n_max=32, N=N, boundary=0)
    U = _gas(N, seed=11)
    n = sk.n_max
    pos = jnp.zeros((n, 3))
    Ua, _, _, ma, _ = sk._accrete(U, pos, jnp.zeros((n, 3)), jnp.zeros(n),
                                  jnp.zeros(n))
    np.testing.assert_array_equal(np.asarray(Ua), np.asarray(U))
    assert float(np.asarray(ma).sum()) == 0.0


def test_the_window_never_wraps_onto_itself():
    """2R+1 > N would gather a cell twice and drain it twice."""
    for N in (8, 9, 16, 24, 64):
        sk = _sinks(N=N, boundary=0, r_acc=10.0)      # absurd on purpose
        assert 2 * sk.R_acc + 1 <= N, N


def _reference_create(sk, U, pos, vel, mass, active, phi, divv):
    """`_maybe_create` as it was: the whole grid for a one-cell decision."""
    X, Y, Z, dV = *_grid(sk), sk.dV
    Nx, Ny, Nz = sk.N3
    rho = U[0]
    idx = jnp.argmin(phi.reshape(-1))
    ix = idx // (Ny * Nz); iy = (idx // Nz) % Ny; iz = idx % Nz
    rho_pk = rho[ix, iy, iz]; divv_pk = divv[ix, iy, iz]
    pk = jnp.array([X[ix, iy, iz], Y[ix, iy, iz], Z[ix, iy, iz]])
    d2 = (pos[:, 0]-pk[0])**2 + (pos[:, 1]-pk[1])**2 + (pos[:, 2]-pk[2])**2
    near = jnp.any(active * (d2 < sk.r_sep * sk.r_sep))
    free_slot = jnp.argmin(active.astype(jnp.int32))
    has_free = jnp.any(active < 0.5)
    r2 = (X - pk[0])**2 + (Y - pk[1])**2 + (Z - pk[2])**2
    rem = sk.create_frac * (r2 < sk.r_acc * sk.r_acc).astype(rho.dtype)
    M0 = jnp.sum(rem * rho) * dV
    p0 = jnp.array([jnp.sum(rem * U[1]), jnp.sum(rem * U[2]),
                    jnp.sum(rem * U[3])]) * dV
    v0 = jnp.where(M0 > 0, p0 / M0, 0.0)
    vx = U[1] / rho; vy = U[2] / rho; vz = U[3] / rho
    v_rel2 = (vx - v0[0])**2 + (vy - v0[1])**2 + (vz - v0[2])**2
    inb = r2 < sk.r_acc * sk.r_acc
    E_kin = jnp.sum(inb * 0.5 * rho * v_rel2) * dV
    E_th = jnp.sum(inb * (U[4] - 0.5 * rho * (vx**2 + vy**2 + vz**2))) * dV
    E_grav = jnp.sum(inb * rho * (jnp.max(jnp.where(inb, phi, -jnp.inf)) - phi)) * dV
    bound = (E_th + E_kin) <= E_grav
    do = (rho_pk > sk.rho_create) & (divv_pk < 0.0) & bound & (~near) & has_free
    s_ = free_slot
    pos = pos.at[s_].set(jnp.where(do, pk, pos[s_]))
    vel = vel.at[s_].set(jnp.where(do, v0, vel[s_]))
    mass = mass.at[s_].set(jnp.where(do, M0, mass[s_]))
    active = active.at[s_].set(jnp.where(do, 1.0, active[s_]))
    U = U * jnp.where(do, 1.0 - rem, 1.0)[None]
    return U, pos, vel, mass, active


def _well(N, centre):
    """A deep potential well at `centre`, converging everywhere: a sink must
    form, so the decision and everything it takes can be compared."""
    c = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r2 = (X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2
    phi = -50.0 * np.exp(-r2 / 0.01)
    return jnp.asarray(phi), -jnp.ones((N, N, N))


@pytest.mark.parametrize("boundary", [0, 1])
def test_windowed_creation_matches_the_whole_grid(boundary):
    """The creation decision is about one cell, so only the argmin needs the
    whole grid. An interior candidate must give exactly what the old code did:
    the same decision, the same mass taken, the same gas left behind."""
    N = 24
    sk = _sinks(N=N, boundary=boundary, rho_create=1.0)
    U = _gas(N, seed=5)
    phi, divv = _well(N, (0.52, 0.47, 0.5))
    n = sk.n_max
    args = (U, jnp.zeros((n, 3)), jnp.zeros((n, 3)), jnp.zeros(n),
            jnp.zeros(n), phi, divv)

    Ua, pa, va, ma, aa = sk._maybe_create(*args)
    Ub, pb, vb, mb, ab = _reference_create(sk, *args)

    assert float(aa.sum()) == 1.0, "no sink formed; the test is not testing"
    np.testing.assert_array_equal(np.asarray(aa), np.asarray(ab))
    np.testing.assert_allclose(np.asarray(ma), np.asarray(mb), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(pa), np.asarray(pb), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(va), np.asarray(vb), rtol=1e-12,
                               atol=1e-14)
    np.testing.assert_allclose(np.asarray(Ua), np.asarray(Ub), rtol=1e-12)

    # and the mass a new sink starts with is exactly the mass the gas lost
    lost = (float(np.asarray(U[0]).sum())
            - float(np.asarray(Ua[0]).sum())) * sk.dV
    np.testing.assert_allclose(float(ma.sum()), lost, rtol=1e-11)


def test_no_candidate_leaves_the_gas_alone():
    """Most steps: nothing is dense enough. The gas must come back untouched."""
    N = 24
    sk = _sinks(N=N, boundary=0, rho_create=1.0e9)
    U = _gas(N, seed=5)
    phi, divv = _well(N, (0.5, 0.5, 0.5))
    n = sk.n_max
    Ua, _, _, ma, aa = sk._maybe_create(
        U, jnp.zeros((n, 3)), jnp.zeros((n, 3)), jnp.zeros(n), jnp.zeros(n),
        phi, divv)
    assert float(aa.sum()) == 0.0
    np.testing.assert_array_equal(np.asarray(Ua), np.asarray(U))



def _core(N, eint):
    """A dense core (above rho_create = 1 within r_acc) at rest in a
    potential well, internal energy `eint` per unit mass, div v < 0."""
    c = (np.arange(N) + 0.5) / N
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r2 = (X - c[12]) ** 2 + (Y - c[12]) ** 2 + (Z - c[12]) ** 2
    U = np.zeros((5, N, N, N))
    U[0] = 1.0 + 30.0 * np.exp(-r2 / 0.01)
    U[4] = eint * U[0]
    phi = -50.0 * np.exp(-r2 / 0.01)
    return jnp.asarray(U), jnp.asarray(phi), -jnp.ones((N, N, N))


@pytest.mark.parametrize("eint,formed", [(0.05, 1.0), (50.0, 0.0)])
def test_only_a_bound_core_makes_a_sink(eint, formed):
    """Bound means bound about the core, thermal energy included: kinetic
    (about its centre of mass) plus thermal energy within the potential
    energy measured from the region's edge (FLASH, Federrath et al. 2010).
    The same dense, converging core in the same well forms a sink when cold
    and none when 1000x hotter; the absolute potential without a thermal
    term used to pass both."""
    N = 24
    sk = _sinks(N=N, boundary=0, rho_create=1.0)
    U, phi, divv = _core(N, eint)
    n = sk.n_max
    _, _, _, _, act = sk._maybe_create(U, jnp.zeros((n, 3)), jnp.zeros((n, 3)),
                                       jnp.zeros(n), jnp.zeros(n), phi, divv)
    assert float(act.sum()) == formed


def _reference_field(sk, pos, mass, active):
    """The sink field as it was: r2 ** -1.5 per slot, to the nearest image
    when the sinks wrap (a periodic box)."""
    X, Y, Z, G, eps2 = *_grid(sk), sk.G, sk.eps ** 2
    img = ((lambda d, L: d - L * jnp.round(d / L)) if sk.wrap else (lambda d, L: d))
    Lx, Ly, Lz = sk.box3
    gx = jnp.zeros_like(X); gy = jnp.zeros_like(X); gz = jnp.zeros_like(X)
    for i in range(sk.n_max):
        ddx = img(X - pos[i, 0], Lx); ddy = img(Y - pos[i, 1], Ly); ddz = img(Z - pos[i, 2], Lz)
        inv_r3 = active[i] * G * mass[i] * (ddx*ddx + ddy*ddy + ddz*ddz + eps2) ** -1.5
        gx = gx - ddx * inv_r3; gy = gy - ddy * inv_r3; gz = gz - ddz * inv_r3
    return gx, gy, gz


def _two_sinks():
    sk = _sinks(n_max=8, N=24)
    pos, _, _, act = _state(sk, [(0.3, 0.4, 0.5), (0.7, 0.6, 0.2)])
    mass = jnp.asarray(np.array([2.0, 5.0] + [0.0] * 6))
    return sk, pos, mass, act


def test_sink_field_matches_the_power_form():
    """rsqrt cubed against r2 ** -1.5: FP64 round-off, measured on the scale
    of the field (two sinks' fields cancel to zero between them, so a
    per-cell relative error is not defined there)."""
    sk, pos, mass, act = _two_sinks()
    for a, b in zip(jax.jit(sk._sink_field_on_grid)(pos, mass, act),
                    _reference_field(sk, pos, mass, act)):
        scale = float(jnp.max(jnp.abs(b)))
        assert scale > 0.0
        assert float(jnp.max(jnp.abs(a - b))) <= 1e-13 * scale


def test_kick_with_the_field_program_matches_the_precomputed_field():
    """The gas kick evaluating the sinks' field program inside itself gives
    the kick fed the precomputed field arrays."""
    from luna.gravity.backend import kick_gas_body, kick_gas_fields_body
    sk, pos, mass, act = _two_sinks()
    U = _gas(24)
    g = tuple(0.1 * c for c in _reference_field(sk, pos[::-1], mass[::-1], act[::-1]))
    fn, args = sk.field_on_gas(Particles(pos, jnp.zeros_like(pos), mass, act,
                                         jnp.zeros_like(pos)))
    kick = jax.jit(kick_gas_fields_body, static_argnums=(4,))
    got = kick(U, g, 0.7, 1e-3, (fn,), (args,))
    want = kick_gas_body(U, g, 0.7, 1e-3, extra=_reference_field(sk, pos, mass, act))
    dU = want - U
    for c in (1, 2, 3, 4):
        assert float(jnp.max(jnp.abs(dU[c]))) > 0.0
        assert float(jnp.max(jnp.abs(got[c] - want[c]))) <= 1e-12 * float(jnp.max(jnp.abs(dU[c])))
    np.testing.assert_array_equal(np.asarray(got[0]), np.asarray(U[0]))
