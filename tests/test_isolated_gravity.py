"""Isolated gravity (POISSON_BOUNDARY=3) on one device and on a mesh.

Analytic references: a uniform sphere (monopole, exact outside it) and two
off-centre blobs against a direct sum (the quadrupole face values). The face
cells are checked on purpose: the old one-device gradient put a zero-Dirichlet
ghost there (audit C6), an inward pull ~N/4 x GM/R^2, and the old sharded
gradient wrapped periodically (M2). Sharded = one device, level for level.
docs/WALL_M3.md.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp


from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402

from luna.gravity.isolated import solve_isolated_shardmap        # noqa: E402
from luna.gravity.particle_mesh import ParticleMeshGravity        # noqa: E402
from luna.gravity.particle_mesh_blocks import grad_phi_blocks     # noqa: E402
from luna.gravity.poisson_multigrid import (_solve_poisson_fmg_isolated,  # noqa: E402
                                            solve_poisson)
from tests._helpers import _build_cfg_for_test                    # noqa: E402

N, L, G = 32, 1.0, 1.0
DX = L / N
FOURPIG = 4.0 * np.pi * G
LEVELS = 3               # 32 -> 8 on one device; 16 -> 4 per shard on a 2x2x2 mesh


def _cfg(**kw):
    base = dict(test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N,
                Lx=L, Ly=L, Lz=L, BOUNDARY=4, POISSON_BOUNDARY=3, GRAVITATIONAL_G=G,
                POISSON_N_LEVELS=LEVELS, POISSON_FMG_EXTRA_VCYCLES=4)
    base.update(kw)
    return _build_cfg_for_test(**base)


def _coords():
    x = (np.arange(N) + 0.5) * DX
    return np.meshgrid(x, x, x, indexing="ij")


def _sphere(a=0.2, c=(0.5, 0.5, 0.5)):
    X, Y, Z = _coords()
    r = np.sqrt((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2)
    rho = np.where(r <= a, 1.0, 0.0)
    return rho, r, float(rho.sum() * DX ** 3)


def _faces(a):
    """The six face layers of a grid, stacked."""
    return np.concatenate([a[0].ravel(), a[-1].ravel(), a[:, 0].ravel(), a[:, -1].ravel(),
                           a[:, :, 0].ravel(), a[:, :, -1].ravel()])


def _mesh(shape):
    n = int(np.prod(shape))
    if jax.device_count() < n:
        pytest.skip(f"needs {n} devices")
    return Mesh(np.array(jax.devices()[:n]).reshape(shape), ("x", "y", "z"))


def _put(a, mesh):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, P("x", "y", "z")))


# -- one device ------------------------------------------------------------------

def test_uniform_sphere_potential_and_face_force_one_device():
    """Outside the sphere phi = -GM/r and g = -GM r/r^3, the face cells included."""
    rho, r, M = _sphere()
    pm = ParticleMeshGravity(_cfg())
    phi = np.asarray(pm.gas_potential(jnp.asarray(rho)))
    out = r > 0.2 + 2 * DX
    err_phi = np.max(np.abs(phi[out] + G * M / r[out])) / (G * M / 0.5)
    X, Y, Z = _coords()
    gx, gy, gz = (np.asarray(g) for g in pm.grad_phi(jnp.asarray(phi)))
    ex = -G * M * (X - 0.5) / r ** 3
    gmag = G * M / r ** 2
    err_g = np.max(_faces(np.abs(gx - ex) / gmag))
    print(f"one device: phi err {err_phi:.2e}, face g_x err {err_g:.2e}")
    assert err_phi < 1e-2           # staircase sphere; 5.1e-3 measured at 32^3
    assert err_g < 1e-2             # 5.7e-3 measured; C6's halo gave 13, the
                                    # zero-ghost prolongation 7.5e-2
    # the 4th-order gradient on the same potential
    gx4 = np.asarray(pm.grad_phi(jnp.asarray(phi), order=4)[0])
    err_g4 = np.max(_faces(np.abs(gx4 - ex) / gmag))
    print(f"one device: face g_x err (order 4) {err_g4:.2e}")
    assert err_g4 < 1e-2


def test_one_fmg_pass_face_force():
    """The production cold solve: one FMG pass, no extra V-cycles, auto depth.
    The coarse potential must be prolonged by extrapolation -- with the
    corrections' Phi = 0 face ghosts it is 1.3 (x the true force) at 32^3."""
    rho, r, M = _sphere()
    pm = ParticleMeshGravity(_cfg(POISSON_N_LEVELS=None, POISSON_FMG_EXTRA_VCYCLES=0))
    phi = pm.gas_potential(jnp.asarray(rho))
    X, _, _ = _coords()
    gx = np.asarray(pm.grad_phi(phi)[0])
    err = np.max(_faces(np.abs(gx + G * M * (X - 0.5) / r ** 3) / (G * M / r ** 2)))
    print(f"one FMG pass: face g_x err {err:.2e}")
    assert err < 1.5e-2             # 8.0e-3 measured at 32^3 (2.7e-3 at 64^3)


def test_off_centre_blobs_face_potential_matches_direct_sum():
    """The quadrupole face values: two blobs off centre, phi at the faces
    against the direct sum over the cells."""
    X, Y, Z = _coords()
    rho = (np.exp(-((X - 0.35) ** 2 + (Y - 0.6) ** 2 + (Z - 0.5) ** 2) / (2 * 0.04 ** 2))
           + 0.5 * np.exp(-((X - 0.62) ** 2 + (Y - 0.4) ** 2 + (Z - 0.55) ** 2)
                          / (2 * 0.05 ** 2)))
    pm = ParticleMeshGravity(_cfg())
    phi = np.asarray(pm.gas_potential(jnp.asarray(rho)))
    m = rho.ravel() * DX ** 3
    src = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    keep = m > 1e-9 * m.max()          # (no source in a face cell)
    m, src = m[keep], src[keep]
    face = np.zeros_like(rho, bool)
    face[0] = face[-1] = face[:, 0] = face[:, -1] = face[:, :, 0] = face[:, :, -1] = True
    tgt = np.stack([X[face], Y[face], Z[face]], 1)
    ref = np.array([-G * np.sum(m / np.linalg.norm(src - t, axis=1)) for t in tgt])
    err = np.max(np.abs(phi[face] - ref)) / np.max(np.abs(ref))
    print(f"off-centre blobs: face phi err {err:.2e}")
    # 6.1e-3 at 32^3 and 64^3 alike: the l <= 2 truncation of the face values
    # (6.7e-3 on the face plane itself; monopole alone 8.9e-2)
    assert err < 1e-2


def test_warm_cycles_hold_and_reach_the_cold_solution():
    rho, _, _ = _sphere()
    r = jnp.asarray(rho)
    cold = _solve_poisson_fmg_isolated(r, DX, FOURPIG, LEVELS, 1, 4, 2, 2, 50, False)
    held = solve_poisson(r, DX, FOURPIG, 3, n_levels=LEVELS, n_cycles=2, phi_init=cold)
    fresh = solve_poisson(r, DX, FOURPIG, 3, n_levels=LEVELS, n_cycles=12)
    s = float(jnp.max(jnp.abs(cold)))
    d_held = float(jnp.max(jnp.abs(held - cold))) / s
    d_fresh = float(jnp.max(jnp.abs(fresh - cold))) / s
    print(f"warm: held {d_held:.2e}, from zero {d_fresh:.2e}")
    assert d_held < 1e-4 and d_fresh < 1e-4


# -- on a mesh -----------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 2, 2)])
def test_sharded_isolated_solve_equals_one_device(shape):
    mesh = _mesh(shape)
    rho, _, _ = _sphere(c=(0.45, 0.55, 0.5))
    one = np.asarray(_solve_poisson_fmg_isolated(jnp.asarray(rho), DX, FOURPIG, LEVELS,
                                                 1, 4, 2, 2, 50, False))
    sh = np.asarray(solve_isolated_shardmap(_put(rho, mesh), DX, FOURPIG, mesh,
                                            n_levels=LEVELS, n_extra_vcycles=4))
    d = np.max(np.abs(sh - one)) / np.max(np.abs(one))
    # warm cycles from the cold solution, both ways
    w1 = np.asarray(solve_poisson(jnp.asarray(rho), DX, FOURPIG, 3, n_levels=LEVELS,
                                  n_cycles=2, phi_init=jnp.asarray(one)))
    w2 = np.asarray(solve_isolated_shardmap(_put(rho, mesh), DX, FOURPIG, mesh,
                                            n_levels=LEVELS, phi_init=_put(one, mesh),
                                            warm_cycles=2))
    dw = np.max(np.abs(w2 - w1)) / np.max(np.abs(one))
    print(f"{shape}: cold sharded-vs-one {d:.2e}, warm {dw:.2e}")
    assert d < 1e-12 and dw < 1e-12


@pytest.mark.parametrize("boundary", [1, 3])
@pytest.mark.parametrize("order", [2, 4])
def test_sharded_gradient_faces_equal_one_device(boundary, order):
    """grad_phi_blocks at the box faces: edge (1) or extrapolated (3) ghosts,
    as ParticleMeshGravity.grad_phi -- not the periodic wrap (audit M2)."""
    mesh = _mesh((2, 2, 2))
    X, Y, Z = _coords()
    phi = np.sin(2.3 * X) * np.cos(1.7 * Y) + Z ** 2 - 0.3 * X * Y
    pm = ParticleMeshGravity(_cfg(POISSON_BOUNDARY=boundary))
    one = [np.asarray(g) for g in pm.grad_phi(jnp.asarray(phi), order=order)]
    sh = [np.asarray(g) for g in grad_phi_blocks(_put(phi, mesh), DX, mesh, order=order,
                                                 boundary=boundary)]
    d = max(np.max(np.abs(a - b)) for a, b in zip(one, sh))
    print(f"grad bnd={boundary} order={order}: max diff {d:.2e}")
    assert d < 1e-12


def test_bridge_isolated_field_equals_one_device():
    """The coupled worker's sharded bridge (ShardedParticleBridge) solves and
    differentiates with the same isolated faces as its one-device bridge."""
    from luna.gravity.particle_mesh_shardmap import ShardedParticleBridge
    mesh = _mesh((2, 2, 2))
    rho, _, _ = _sphere(c=(0.45, 0.55, 0.5))
    cfg = _cfg()
    one = [np.asarray(g) for g in ParticleMeshGravity(cfg).gas_accel_grid(jnp.asarray(rho))]
    sh = [np.asarray(g) for g in ShardedParticleBridge(cfg, mesh).accel_field(_put(rho, mesh))]
    d = max(np.max(np.abs(a - b)) for a, b in zip(one, sh)) / np.max(np.abs(one[0]))
    print(f"bridge: sharded-vs-one {d:.2e}")
    assert d < 1e-12
