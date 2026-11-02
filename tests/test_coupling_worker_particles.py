"""Particle + gravity-bridge surface of LunaWorker (Torch/AMUSE, Phase 4 cont.).

Framework-agnostic (no AMUSE): exercises particle CRUD, the 'mass'/'sink'
pointer, and the gas↔particle gravity coupling with physics-sanity sign checks.
"""

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tests.test_regressions import _build_cfg_for_test  # noqa: E402
from amuse_luna.worker import LunaWorker  # noqa: E402


def _worker(N=16):
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
        POISSON_N_LEVELS=LEVELS,
    )
    w = LunaWorker(cfg)
    w.initialize_code()
    w.commit_parameters()
    return w


def _set_density(w, rho_field):
    U = np.array(w.state.U)
    U[0] = rho_field
    U[1:] = 0.0
    w.state.U = jnp.asarray(U)


def test_particle_crud_and_pointer_switch():
    w = _worker(8)
    t = w.new_particle(0.5, 0.5, 0.5)
    assert w.get_number_of_particles() == 1
    w.set_particle_mass(t, [3.0])
    np.testing.assert_allclose(w.get_particle_mass(t), [3.0])
    w.set_particle_velocity(t, [1.0], [2.0], [3.0])
    vx, vy, vz = w.get_particle_velocity(t)
    np.testing.assert_allclose([vx[0], vy[0], vz[0]], [1.0, 2.0, 3.0])

    # 'sink' pointer addresses a separate store.
    w.set_particle_pointers("sink")
    assert w.get_number_of_particles() == 0
    w.new_particle([0.2, 0.8], [0.2, 0.8], [0.2, 0.8])
    assert w.get_number_of_particles() == 2
    w.set_particle_pointers("mass")
    assert w.get_number_of_particles() == 1          # stars untouched
    w.delete_particle(t)
    assert w.get_number_of_particles() == 0


def test_gas_gravity_points_toward_overdensity():
    w = _worker(16)
    rho = np.ones((16, 16, 16))
    rho[8, 8, 8] += 300.0           # central overdensity
    _set_density(w, rho)
    # sample acceleration at a point on the +x side of centre
    ax, ay, az = w.get_gravity_at_point([0.0], [0.78], [0.5], [0.5])
    assert ax[0] < 0.0              # pulled back toward the centre (-x)


def test_particle_gravity_kicks_the_gas():
    w = _worker(16)
    _set_density(w, np.ones((16, 16, 16)))
    t = w.new_particle(0.3, 0.5, 0.5)
    w.set_particle_mass(t, [500.0])
    assert np.allclose(np.array(w.state.U[1]), 0.0)   # gas momentum starts at 0
    w.apply_particle_gravity_on_gas(0.01)
    mom = np.array(w.state.U[1:4])
    assert np.any(mom != 0.0)                          # gas was kicked by the particle


def test_gas_gravity_kicks_a_particle():
    w = _worker(16)
    rho = np.ones((16, 16, 16))
    rho[8, 8, 8] += 500.0
    _set_density(w, rho)
    t = w.new_particle(0.78, 0.5, 0.5)                 # off-centre, +x side
    w.apply_gas_gravity_on_particles(0.01)
    vx, vy, vz = w.get_particle_velocity(t)
    assert vx[0] < 0.0                                 # accelerated toward the overdensity


# One pinned multigrid depth for every layout; the default depth on a mesh is tested in
# test_gravity_decomposition.py.
LEVELS = 2


def _stars(w, n=3):
    rng = np.random.default_rng(4)
    tags = [int(np.atleast_1d(w.new_particle(*rng.random(3)))[0]) for _ in range(n)]
    w.set_particle_mass(tags, list(50.0 + 100.0 * rng.random(n)))
    return tags


def _lumpy(N=16, seed=2):
    rng = np.random.default_rng(seed)
    return 1.0 + rng.random((N, N, N))


def test_the_kicks_are_the_bridge_computed_directly():
    """The split, cached kernels against the bridge called directly, end to
    end. The reference is the bridge the worker's layout calls for: one
    device, ParticleMeshGravity on the whole grid; a mesh, ShardedParticleBridge
    (per-shard solve and point samples). The star deposit goes through the
    host CIC + shard-local scatter, so the sum order differs from XLA's
    scatter -- round-off, not more."""
    w = _worker(16)
    _set_density(w, _lumpy())
    _stars(w)
    b, s = w.bridge, w.stars
    U0 = w.state.U
    pos, mass, act = (jnp.asarray(s.arrays[k]) for k in ("position", "mass", "active"))
    mesh = getattr(getattr(U0, "sharding", None), "mesh", None)
    if mesh is not None and mesh.size > 1:
        from luna.gravity.particle_mesh_shardmap import ShardedParticleBridge
        sb = ShardedParticleBridge(w.cfg, mesh)
        rho_s = w._star_density(np.asarray(pos), np.asarray(mass), np.asarray(act))
        field_of, at = sb.accel_field, sb.accel_at_points
    else:
        rho_s = b.particles_potential                          # unused; see below
        field_of, at = None, None
    if field_of is None:
        want_U = b.kick_gas(U0, *b.grad_phi(b.particles_potential(pos, mass, act.astype(mass.dtype))), 0.01)
    else:
        want_U = b.kick_gas(U0, *field_of(rho_s), 0.01)
    w.apply_particle_gravity_on_gas(0.01)
    np.testing.assert_allclose(np.asarray(w.state.U), np.asarray(want_U), rtol=1e-12, atol=1e-13)
    want_a = np.asarray(b.gas_accel_on_particles(w.state.U[0], pos) if field_of is None
                        else at(field_of(w.state.U[0]), pos))
    v0 = np.array(s.arrays["velocity"])
    w.apply_gas_gravity_on_particles(0.01)
    live = s.arrays["active"]
    np.testing.assert_allclose(s.arrays["velocity"][live], v0[live] + 0.01 * want_a[live],
                               rtol=1e-12, atol=1e-13)


def test_the_fields_are_reused_only_while_their_inputs_stand():
    """Two bridge steps' worth of kicks: the fields computed afresh every time
    and the cached ones give the same numbers, and a changed density or a
    moved star is noticed."""
    def run(cache):
        w = _worker(16)
        _set_density(w, _lumpy())
        tags = _stars(w)
        for step in range(2):
            if not cache:
                w._kick_kernels = None
            w.apply_particle_gravity_on_gas(0.01)
            w.apply_gas_gravity_on_particles(0.01)
            # the gas moves and a star moves between kicks, as in a bridge step
            U = np.array(w.state.U)
            U[0] *= 1.0 + 0.01 * (step + 1)
            w.state.U = jnp.asarray(U)
            x, y, z = w.get_particle_position(tags[0])
            w.set_particle_position(tags[0], x + 0.05, y, z)
            if not cache:
                w._kick_kernels = None
            w.apply_particle_gravity_on_gas(0.01)
            w.apply_gas_gravity_on_particles(0.01)
        return np.asarray(w.state.U), np.array(w.stars.arrays["velocity"])
    Ua, va = run(True)
    Ub, vb = run(False)
    np.testing.assert_array_equal(Ua, Ub)
    np.testing.assert_array_equal(va, vb)


def test_the_kicks_do_not_depend_on_the_layout():
    """The same kicks on two 8-device layouts, (2,2,2) and (4,2,1) blocks:
    the star deposit's shard-local scatter, the sharded solve and the
    shard-local point samples must give the same numbers whichever device
    owns a cell. ((8,1,1) slabs are two cells thick at 16^3: too thin for the
    two multigrid levels every layout here uses.)"""
    if jax.device_count() < 8:
        import pytest
        pytest.skip("needs 8 devices")
    from dataclasses import replace
    out = []
    for mesh in ((4, 2, 1), (2, 2, 2)):
        cfg = _build_cfg_for_test(
            test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
            Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
            ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=0,
            POISSON_N_LEVELS=LEVELS)
        w = LunaWorker(replace(cfg, MESH_SHAPE=mesh))
        w.initialize_code()
        w.commit_parameters()
        U = np.array(w.state.U)
        U[0] = _lumpy()
        U[1:] = 0.0
        w.state.U = jnp.asarray(U)
        _stars(w, 5)
        w.apply_particle_gravity_on_gas(0.01)
        w.apply_gas_gravity_on_particles(0.01)
        out.append((np.asarray(w.state.U), np.array(w.stars.arrays["velocity"])))
    np.testing.assert_allclose(out[1][0], out[0][0], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(out[1][1], out[0][1], rtol=1e-11, atol=1e-12)


def test_the_sharded_point_samples_are_the_single_device_ones():
    """ShardedParticleBridge.accel_at_points (each shard adds the CIC corners
    it owns, then a psum) against ParticleMeshGravity.accel_at_points on the
    same field, periodic and not: the same sums, to round-off."""
    if jax.device_count() < 8:
        import pytest
        pytest.skip("needs 8 devices")
    from dataclasses import replace
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.gravity.particle_mesh import ParticleMeshGravity
    from luna.gravity.particle_mesh_shardmap import ShardedParticleBridge
    rng = np.random.default_rng(3)
    for bc in (0, 1):
        cfg = _build_cfg_for_test(
            test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=bc,
            Nx=16, Ny=16, Nz=16, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
            ORDER=1, LIMITER=1, GRAVITATIONAL_G=1.0, POISSON_BOUNDARY=bc)
        mesh = Mesh(np.array(jax.devices()[:8]).reshape(2, 2, 2), ("x", "y", "z"))
        sh = NamedSharding(mesh, P("x", "y", "z"))
        field = tuple(jnp.asarray(rng.normal(size=(16, 16, 16))) for _ in range(3))
        pos = jnp.asarray(rng.uniform(-0.02, 1.02, (40, 3)))       # some past the faces
        ref = ParticleMeshGravity(cfg).accel_at_points(field, pos)
        got = ShardedParticleBridge(replace(cfg, MESH_SHAPE=(2, 2, 2)), mesh).accel_at_points(
            tuple(jax.device_put(f, sh) for f in field), pos)
        np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-12, atol=1e-13)


def test_the_sharded_field_is_the_analytic_field():
    """ShardedParticleBridge.accel_field on a (2,2,2) mesh against the exact
    -grad phi of a single cosine mode, 64^3 periodic: within 1%, the same
    standard as test_poisson_multigrid_multigpu (the 7-point stencil's
    discretisation floor at N=64)."""
    if jax.device_count() < 8:
        import pytest
        pytest.skip("needs 8 devices")
    from dataclasses import replace
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.gravity.particle_mesh_shardmap import ShardedParticleBridge
    N, amp, G = 64, 1.0, 1.0
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        ORDER=1, LIMITER=1, GRAVITATIONAL_G=G, POISSON_BOUNDARY=0)
    mesh = Mesh(np.array(jax.devices()[:8]).reshape(2, 2, 2), ("x", "y", "z"))
    x = (np.arange(N) + 0.5) / N
    k = 2.0 * np.pi
    X = np.broadcast_to(x[:, None, None], (N, N, N))
    rho = jax.device_put(jnp.asarray(amp * np.cos(k * X)), NamedSharding(mesh, P("x", "y", "z")))
    gx, gy, gz = ShardedParticleBridge(replace(cfg, MESH_SHAPE=(2, 2, 2)), mesh).accel_field(rho)
    # phi = -(4 pi G amp / k^2) cos(k x), so -dphi/dx = -(4 pi G amp / k) sin(k x)
    exact = -(4.0 * np.pi * G * amp / k) * np.sin(k * X)
    err = np.max(np.abs(np.asarray(gx) - exact)) / np.max(np.abs(exact))
    assert err < 1e-2, f"sharded bridge field off the analytic one by {err:.3e}"
    # no transverse field: zero up to the solver's own noise (measured 2.6e-8 of
    # a field of 2.0, one FMG cycle per level)
    side = max(np.max(np.abs(np.asarray(gy))), np.max(np.abs(np.asarray(gz))))
    assert side < 1e-6 * np.max(np.abs(exact)), side
