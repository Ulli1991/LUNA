"""Analytic external potentials: g = -grad phi at the cell centres in 1D-3D, the kick, the refusals."""
import numpy as np
import pytest

from tests._helpers import accuracy_cfg

KW = {1: dict(Nx=16, Lx=2.0), 2: dict(Nx=16, Ny=8, Lx=2.0, Ly=1.0), 3: dict(Nx=8, Ny=8, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0)}


def phi(kind, X, cfg):
    """The potential at the points X."""
    if kind == "sin":
        return cfg.external_phi0 * np.sin(2 * np.pi * X[cfg.external_axis] / cfg.external_wavelength)
    if kind == "uniform":
        return -cfg.external_g * X[cfg.external_axis]
    c = [0.5 * b for b in (cfg.Lx, cfg.Ly, cfg.Lz)[:len(X)]]
    return -cfg.external_GM / np.sqrt(sum((x - ci) ** 2 for x, ci in zip(X, c)) + cfg.external_softening ** 2)


@pytest.mark.parametrize("sdim", (1, 2, 3))
@pytest.mark.parametrize("kind", ("sin", "uniform", "point_mass"))
def test_acceleration_is_minus_grad_phi(kind, sdim):
    """Against a central difference of phi (h = 1e-6) at every cell centre."""
    from luna.sources.external_gravity import analytic_acceleration
    cfg = accuracy_cfg(test_problem="Soundwave", SPATIAL_DIMENSION=sdim, BOUNDARY=0, external_potential=kind,
                       external_axis=sdim - 1, external_phi0=0.7, external_wavelength=0.5, external_g=-2.0,
                       external_GM=0.3, external_softening=0.1, **KW[sdim])
    g = analytic_acceleration(cfg, np.float64)
    shape, box = g.shape[1:], (cfg.Lx, cfg.Ly, cfg.Lz)[:sdim]
    X = np.meshgrid(*[(np.arange(n) + 0.5) * b / n for n, b in zip(shape, box)], indexing="ij")
    for d in range(sdim):
        h = 1e-6
        Xp, Xm = [x.copy() for x in X], [x.copy() for x in X]
        Xp[d] += h
        Xm[d] -= h
        fd = -(phi(kind, Xp, cfg) - phi(kind, Xm, cfg)) / (2 * h)
        assert np.abs(g[d] - fd).max() < 1e-6 * max(1.0, np.abs(fd).max()), (kind, sdim, d)


def test_kick_adds_rho_g_dt_and_its_work():
    """Momentum gains rho g dt; energy gains dt g . (m + m_new)/2; density untouched."""
    import jax.numpy as jnp
    from luna.sources.external_gravity import _kick
    rng = np.random.default_rng(1)
    U = jnp.asarray(np.stack([1 + rng.random(32), rng.standard_normal(32), rng.standard_normal(32), 5 + rng.random(32)]))
    g = jnp.asarray(rng.standard_normal((2, 32)))
    V = np.asarray(_kick(U, 0.01, g))
    U, g = np.asarray(U), np.asarray(g)
    m_new = U[1:3] + 0.01 * U[0] * g
    assert np.array_equal(V[0], U[0])
    assert np.allclose(V[1:3], m_new, rtol=0, atol=1e-15)
    assert np.allclose(V[3], U[3] + 0.01 * (0.5 * (U[1:3] + m_new) * g).sum(0), rtol=0, atol=1e-14)


@pytest.mark.parametrize("bad,match", [(dict(external_potential="gauss"), "expected"),
                                       (dict(external_potential="sin", m82_use_external_gravity=True), "exclusive"),
                                       (dict(external_potential="sin", external_axis=1), "outside")])
def test_refusals(bad, match):
    with pytest.raises(ValueError, match=match):
        accuracy_cfg(test_problem="Soundwave", SPATIAL_DIMENSION=1, Nx=8, Lx=1.0, **bad).validate()


def m82_kick_error(module):
    """Max relative error of the M82 kick's g in an fp64 run against float64 coordinates."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    n = 16
    cfg = accuracy_cfg(test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n, Lx=4.0, Ly=4.0, Lz=4.0,
                       BOUNDARY=1, m82_use_external_gravity=True)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    apply = module.setup_external_gravity(cfg, NamedSharding(mesh, P(None, "x", "y", "z")), False)["apply"]
    U = jnp.zeros((5, n, n, n)).at[0].set(1.0).at[4].set(1.0)
    g = np.asarray(apply(U, 1.0))[1:4]
    x = (np.arange(n) + 0.5) * (4.0 / n) - 2.0
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    ref = np.stack(module.combined_acceleration(X, Y, Z, cfg.m82_stellar_MN_M, cfg.m82_stellar_MN_a_kpc,
                                                cfg.m82_stellar_MN_b_kpc, cfg.m82_halo_NFW_M_vir, cfg.m82_halo_NFW_c))
    return float(np.abs(g - ref).max() / np.abs(ref).max())


def test_m82_gravity_in_the_run_precision():
    """fp64 run: the M82 acceleration from float64 coordinates (they were float32: ~1e-7 relative)."""
    from luna.sources import external_gravity
    assert m82_kick_error(external_gravity) < 1e-13
