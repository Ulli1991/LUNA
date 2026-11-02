"""The two 3D kernels must do the SAME arithmetic at a shard seam.

Before 2026-09-18 the non-fused sharded path built the seam flux from a
reconstructed state on one side and the RAW cell-centred halo on the other
(muscl_hancock.py:161-163). That is first order in the one cell layer next to
each shard face, and -- worse -- the two shards sharing the face computed
DIFFERENT fluxes for it, so the flux sum did not telescope and mass/energy
drifted. The fused kernel never had this, because its 2-ghost pad lets it
reconstruct both sides.

`wide` gives the non-fused path the same 2-ghost pad and the same construction,
so the two kernels now agree and both conserve exactly. See
docs/HYDRO_KERNEL_PATHS.md for the full branch matrix.

The conservation test is the sharp one: on a PERIODIC box every face is
interior, so sum(U) is invariant to machine precision if and only if every face
flux is counted once with the same value on both sides. A non-telescoping seam
shows up there and nowhere else. GPU test; needs >= 2 devices for a real seam.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp



from luna.hydro.conversions import prim_to_cons                      # noqa: E402
from luna.time_integration.integrators import create_muscl_hancock_modular_step  # noqa: E402
from luna.utils.halo_exchange_shardmap import make_halo_shardmap     # noqa: E402

N, GAMMA, L = 32, 5.0 / 3.0, 1.0
DX = L / N


def _state():
    """A smooth but fully 3D periodic state: no symmetry to hide an asymmetry."""
    x = (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    k = 2.0 * np.pi / L
    rho = 1.0 + 0.3 * np.sin(k * X) * np.cos(k * Y) * np.cos(k * Z)
    vx = 0.2 * np.cos(k * X) * np.sin(k * Y)
    vy = 0.2 * np.sin(k * Z) * np.cos(k * X)
    vz = 0.2 * np.cos(k * Y) * np.sin(k * Z)
    P = 1.0 + 0.2 * np.cos(k * X) * np.cos(k * Y) * np.sin(k * Z)
    W = jnp.asarray(np.stack([rho, vx, vy, vz, P]))
    return prim_to_cons(W, GAMMA)


def _stepper(mesh, fused):
    pad = make_halo_shardmap(mesh, 3)
    inv = 1.0 / DX
    return create_muscl_hancock_modular_step(
        pad, GAMMA, 1, 0, DX, DX, DX, inv, inv, inv,
        0,                       # BOUNDARY 0 = periodic
        3, 0.4, riemann_solver="hllc", use_shardmap=True, mesh=mesh, fused=fused)[1]


def _mesh(ndev):
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:ndev]).reshape(ndev, 1, 1), ("x", "y", "z"))


def _shard(U, mesh):
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(U, NamedSharding(mesh, P(None, "x", "y", "z")))


@pytest.mark.parametrize("fused", [False, True])
def test_periodic_box_conserves_across_the_seam(fused):
    """sum(U) is invariant iff every face flux telescopes, seams included."""
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices for a real shard seam")
    mesh = _mesh(2)
    U = _shard(_state(), mesh)
    step = _stepper(mesh, fused)
    # Normalise by sum|U_k|, NOT by sum(U_k): on a periodic box the momentum
    # sums are ~0 by construction, so dividing by them turns round-off into a
    # 100% "drift" and the test fails on every kernel (first run, 2026-09-18).
    before = np.asarray([float(jnp.sum(U[k])) for k in range(5)])
    scale = np.asarray([float(jnp.sum(jnp.abs(U[k]))) for k in range(5)])
    Un = U
    for _ in range(5):
        Un = step(Un, 1e-3)
    after = np.asarray([float(jnp.sum(Un[k])) for k in range(5)])
    drift = np.abs(after - before) / np.maximum(scale, 1e-30)
    names = ("rho", "mom_x", "mom_y", "mom_z", "E")
    print(f"[seam fused={fused}] drift over 5 steps: "
          + "  ".join(f"{n}={d:.2e}" for n, d in zip(names, drift)))
    assert drift.max() < 1e-13, (
        f"seam flux does not telescope: {dict(zip(names, drift))}")


def test_fused_and_unfused_agree_pointwise_on_a_seam():
    """Same arithmetic, so the two kernels must agree to round-off everywhere.

    Sums can hide a redistribution; this compares cell by cell, which is what
    actually settles whether the seam treatments match.
    """
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices for a real shard seam")
    mesh = _mesh(2)
    U = _shard(_state(), mesh)
    a, b = _stepper(mesh, False), _stepper(mesh, True)
    Ua, Ub = U, U
    for _ in range(5):
        Ua, Ub = a(Ua, 1e-3), b(Ub, 1e-3)
    Ua, Ub = np.asarray(Ua), np.asarray(Ub)
    err = np.max(np.abs(Ua - Ub)) / np.max(np.abs(Ub))
    print(f"[seam] fused vs unfused pointwise max rel diff = {err:.3e}")
    assert err < 1e-10, f"kernels disagree at the seam: {err:.3e}"


@pytest.mark.parametrize("dim", [1, 2])
def test_low_dim_periodic_is_decomposition_independent(dim):
    """1D/2D periodic under shard_map wrapped a shard onto its OWN opposite
    edge (muscl_hancock.py:239-243, 262-266), so a 2-device run solved a
    different problem than a 1-device run.

    NOTE conservation is the WRONG probe for this bug: the self-wrap uses the
    same state pair at both ends, so F[0] == F[N] and the sum telescopes
    exactly while being wrong. Decomposition independence is the probe.
    """
    if jax.device_count() < 2:
        pytest.skip("needs >= 2 devices for a real shard seam")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.hydro.conversions import prim_to_cons as p2c
    axes = ("x", "y", "z")[:dim]
    x = (np.arange(N) + 0.5) * DX
    g = np.meshgrid(*([x] * dim), indexing="ij")
    k = 2.0 * np.pi / L
    rho = 1.0 + 0.3 * np.sin(k * g[0])
    for a_ in range(1, dim):
        rho = rho * (1.0 + 0.2 * np.cos(k * g[a_]))
    W = jnp.asarray(np.stack([rho] + [0.2 * np.cos(k * g[a_]) for a_ in range(dim)]
                             + [1.0 + 0.2 * np.cos(k * g[0])]))
    U0 = p2c(W, GAMMA)
    inv = 1.0 / DX

    def run(ndev):
        devs = np.array(jax.devices()[:ndev]).reshape((ndev,) + (1,) * (dim - 1))
        mesh = Mesh(devs, axes)
        U = jax.device_put(U0, NamedSharding(mesh, P(None, *axes)))
        step = create_muscl_hancock_modular_step(
            make_halo_shardmap(mesh, dim), GAMMA, 1, 0, DX, DX, DX, inv, inv, inv,
            0, dim, 0.4, riemann_solver="hllc", use_shardmap=True, mesh=mesh)[1]
        for _ in range(5):
            U = step(U, 1e-3)
        return np.asarray(U)

    one, two = run(1), run(2)
    err = np.max(np.abs(one - two)) / np.max(np.abs(one))
    print(f"[seam {dim}D periodic] 1-device vs 2-device max rel diff = {err:.3e}")
    assert err < 1e-12, f"{dim}D result depends on the decomposition: {err:.3e}"


def test_single_device_is_unaffected():
    """One device has no seam, so the wide pad must not change the answer."""
    mesh = _mesh(1)
    U = _shard(_state(), mesh)
    step = _stepper(mesh, False)
    Un = U
    for _ in range(5):
        Un = step(Un, 1e-3)
    before = float(jnp.sum(U[0]))
    after = float(jnp.sum(np.asarray(Un)[0]))
    assert abs(after - before) / abs(before) < 1e-13
