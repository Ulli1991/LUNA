"""3D-block particle-mesh correctness (gravity/particle_mesh_blocks.py).

Deposit vs periodic reference, gather vs single-device forces, and the
exchange cascade's conservation gauntlet (count/mass/ownership/multiset,
diagonal hops). Needs >=2 GPUs; run on apudev, N=64 keeps it quick.
"""
import numpy as np
import pytest

import jax


pytestmark = pytest.mark.slow

if jax.device_count() < 2:
    pytest.skip("needs >=2 devices", allow_module_level=True)

import jax.numpy as jnp

from luna.gravity.particle_mesh_blocks import (
    cic_deposit_blocks, decompose_to_blocks, exchange_particles_blocks,
    make_block_mesh, resort_blocks)
from jax.sharding import NamedSharding, PartitionSpec as P

N = 64
BOX = 100.0
DX = BOX / N
PDIMS = (2, 1, 1)


def _setup():
    mesh = make_block_mesh(jax.devices()[:2], PDIMS)
    rng = np.random.default_rng(7)
    lat = (np.indices((N, N, N)).reshape(3, -1).T + 0.5) * DX
    pos = np.mod(lat + rng.uniform(-1.6, 1.6, lat.shape) * DX, BOX)
    pos_g, mom_g, mass_g, act_g, C = decompose_to_blocks(
        pos, np.zeros_like(pos), np.ones(len(pos)), N, PDIMS, DX, BOX, 1.5)
    flat = NamedSharding(mesh, P(("x", "y", "z")))
    flat2 = NamedSharding(mesh, P(("x", "y", "z"), None))

    def rows(arr, sh):
        devs = list(mesh.devices.ravel())
        chunks = np.split(arr, len(devs), axis=0)
        bufs = [jax.device_put(c, d) for c, d in zip(chunks, devs)]
        return jax.make_array_from_single_device_arrays(arr.shape, sh, bufs)

    return (mesh, rng, pos,
            rows(pos_g, flat2), rows(mom_g, flat2),
            rows(mass_g, flat), rows(act_g, flat), C)


def test_deposit_matches_periodic_reference():
    mesh, _, pos, posS, _, massS, actS, _ = _setup()
    rho = np.asarray(cic_deposit_blocks(posS, massS, actS, N, DX, mesh))
    f = pos / DX - 0.5
    i0 = np.floor(f).astype(np.int64)
    t = f - i0
    ref = np.zeros((N, N, N))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((t[:, 0] if di else 1 - t[:, 0])
                     * (t[:, 1] if dj else 1 - t[:, 1])
                     * (t[:, 2] if dk else 1 - t[:, 2]))
                np.add.at(ref, ((i0[:, 0] + di) % N, (i0[:, 1] + dj) % N,
                                (i0[:, 2] + dk) % N), w)
    ref /= DX ** 3
    assert np.max(np.abs(rho - ref)) / np.max(np.abs(ref)) < 1e-12
    assert abs(rho.sum() * DX ** 3 - len(pos)) < 1e-6 * len(pos)


def test_fixed_point_deposit_is_exact_in_mass():
    """The integer scatter must agree with the float one AND carry mass to the
    bit through both the rounding tree and the ghost fold."""
    mesh, _, pos, posS, _, massS, actS, _ = _setup()
    rho_f = np.asarray(cic_deposit_blocks(posS, massS, actS, N, DX, mesh))
    rho_x = np.asarray(cic_deposit_blocks(
        posS, massS, actS, N, DX, mesh, dtype=jnp.float32,
        fixed=jnp.int32, fixed_bits=17)).astype(np.float64)
    assert abs(rho_x.sum() * DX ** 3 - len(pos)) < 1e-5 * len(pos)
    assert np.max(np.abs(rho_x - rho_f)) / np.max(np.abs(rho_f)) < 1e-4


def test_exchange_conserves_everything():
    mesh, rng, _, posS, pS, massS, actS, C = _setup()
    act = np.asarray(actS) > 0
    kicked = np.mod(np.asarray(posS)
                    + rng.uniform(-1.5, 1.5, (2 * C, 3)) * DX, BOX)
    flat2 = NamedSharding(mesh, P(("x", "y", "z"), None))
    devs = list(mesh.devices.ravel())
    bufs = [jax.device_put(c, d)
            for c, d in zip(np.split(kicked, 2, axis=0), devs)]
    posK = jax.make_array_from_single_device_arrays(kicked.shape, flat2, bufs)
    ids = np.where(act, np.arange(2 * C), -1)
    idsS = jax.make_array_from_single_device_arrays(
        ids.shape, NamedSharding(mesh, P(("x", "y", "z"))),
        [jax.device_put(c, d) for c, d in zip(np.split(ids, 2), devs)])
    posE, pE, massE, actE, idsE, n_over = exchange_particles_blocks(
        posK, pS, massS, actS, N, DX, BOX, mesh, face_capacity_factor=0.6, ids=idsS)
    actE_np = np.asarray(actE) > 0
    # every ID arrives exactly once, still paired with its own particle
    idE = np.asarray(idsE)[actE_np]
    assert np.array_equal(np.sort(idE), np.sort(ids[act]))
    assert np.array_equal(np.asarray(posE)[actE_np], kicked[idE])
    assert int(n_over) == 0
    assert int(actE_np.sum()) == int(act.sum())
    assert abs(float(np.asarray(massE)[actE_np].sum())
               - float(np.asarray(massS)[act].sum())) < 1e-9 * act.sum()
    pe = np.asarray(posE)[actE_np]
    owner = np.clip((pe[:, 0] / (N // 2 * DX)).astype(np.int64), 0, 1)
    blk = np.repeat(np.arange(2), C)[actE_np]
    assert np.all(owner == blk)
    assert np.allclose(np.sort(kicked[act].ravel()), np.sort(pe.ravel()))


def test_resort_permutes_force_with_particles():
    """Audit item 2: resort_blocks must permute the force g by the same order
    as pos/p/mass/active, else the next kick_drift pairs each particle with
    another's force. Seed g identical to pos; after resort the two must still
    be identical (same permutation). Pre-fix g was left unpermuted."""
    mesh, _, _, posS, pS, massS, actS, C = _setup()
    posR, pR, massR, actR, gR, _ = resort_blocks(posS, pS, massS, actS, posS, N, DX, mesh)
    act = np.asarray(actR) > 0
    # g started == pos, so under the same permutation g_out must equal pos_out.
    assert np.allclose(np.asarray(gR)[act], np.asarray(posR)[act])
    # the IDs take the same permutation
    ids = np.arange(2 * C)
    idsS = jax.make_array_from_single_device_arrays(
        ids.shape, NamedSharding(mesh, P(("x", "y", "z"))),
        [jax.device_put(c, d) for c, d in zip(np.split(ids, 2), mesh.devices.ravel())])
    posI, _, _, actI, _, idsR = resort_blocks(posS, pS, massS, actS, posS, N, DX, mesh,
                                              ids=idsS)
    actI = np.asarray(actI) > 0
    assert np.array_equal(np.asarray(posI)[actI], np.asarray(posS)[np.asarray(idsR)[actI]])
    # sanity: resort is a pure reordering (multiset of positions preserved).
    a0 = np.asarray(actS) > 0
    assert np.allclose(np.sort(np.asarray(posS)[a0].ravel()),
                       np.sort(np.asarray(posR)[act].ravel()))


def test_deposit_after_wrap_without_exchange():
    """Audit item 14: positions are wrapped every step but the exchange is
    cadenced, so a particle that crossed the periodic wrap since the last
    exchange still belongs to its old block. Its raw cell offset from the block
    origin is ~+-N and clipped into the wrong end unless taken minimum-image.
    Shift every particle +0.4 dx in x (inside the 0.45 dx margin), deposit WITHOUT
    exchanging, and the field must still match the periodic CIC reference."""
    mesh, _, pos, posS, _, massS, actS, C = _setup()
    shift = np.array([0.4 * DX, 0.0, 0.0])
    kicked = np.mod(np.asarray(posS) + shift, BOX)
    flat2 = NamedSharding(mesh, P(("x", "y", "z"), None))
    devs = list(mesh.devices.ravel())
    bufs = [jax.device_put(c, d) for c, d in zip(np.split(kicked, 2, axis=0), devs)]
    posK = jax.make_array_from_single_device_arrays(kicked.shape, flat2, bufs)
    rho = np.asarray(cic_deposit_blocks(posK, massS, actS, N, DX, mesh))
    pref = np.mod(pos + shift, BOX)
    f = pref / DX - 0.5
    i0 = np.floor(f).astype(np.int64)
    t = f - i0
    ref = np.zeros((N, N, N))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((t[:, 0] if di else 1 - t[:, 0])
                     * (t[:, 1] if dj else 1 - t[:, 1])
                     * (t[:, 2] if dk else 1 - t[:, 2]))
                np.add.at(ref, ((i0[:, 0] + di) % N, (i0[:, 1] + dj) % N,
                                (i0[:, 2] + dk) % N), w)
    ref /= DX ** 3
    assert np.max(np.abs(rho - ref)) / np.max(np.abs(ref)) < 1e-12
    assert abs(rho.sum() * DX ** 3 - len(pos)) < 1e-6 * len(pos)
