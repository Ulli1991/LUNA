"""The fused M1 sweep kernel (radiation.m1_pallas) against the XLA sweep it
replaces: bitwise, for the split step on one device with one and two
groups, and through the radiation stage on a (2,2,2) mesh, where each block
runs it after the halo exchange. On a CPU the kernel runs in Pallas's
interpreter, which rounds differently from XLA's fused CPU code (2.5e-13
relative at most, measured): there it is compared to round-off. On a GPU the
compiled Triton kernel matches one sweep to one unit in the last place, in
the flux only (the floor rounds differently, 3.8e-16 relative, measured);
over the six sweeps of a step that reaches the energy too.
The interpreter cannot run inside a shard_map, so the mesh test needs GPUs."""
from dataclasses import replace

import numpy as np
import pytest

import jax

import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

from luna.radiation import m1, m1_pallas  # noqa: E402
from tests.test_radiation_coupled import _gas, _mesh, _rad_cfg  # noqa: E402

N = 16
CHAT = 7.0


def _field(groups, seed=0, n=N):
    rng = np.random.default_rng(seed)
    E = np.exp(rng.normal(size=(groups, n, n, n)))
    F = 0.6 * CHAT * E[:, None] * rng.uniform(-0.5, 0.5, size=(groups, 3, n, n, n))
    E[:, 5:7, :, :] *= 1e-40                        # a front: the floor and minmod earn their keep
    return jnp.asarray(np.concatenate([E[:, None], F], axis=1))


def _split(R, kernel, monkeypatch, dt=0.3 / N / CHAT):
    monkeypatch.setattr(m1_pallas, "KERNEL", kernel)
    return np.asarray(jax.jit(lambda R: m1.m1_split_step(R, dt, 1.0 / N, chat=CHAT))(R))


@pytest.mark.parametrize("groups", [1, 2])
def test_the_fused_split_step_is_the_xla_one(groups, monkeypatch):
    R = _field(groups)
    a = _split(R, "xla", monkeypatch)
    b = _split(R, "pallas", monkeypatch)
    assert np.all(a[:, 0] >= 0.0)
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-14 * np.abs(a).max())


def _padded_sweeps(R, dt, dx, boundary):
    """The fused split step as it was: an XLA pad along d before every sweep."""
    for d in m1_pallas.SWEEPS:
        Rp = jax.vmap(lambda r: m1._pad_axis_local(r, 2, boundary, d))(R)
        R = jnp.stack([m1_pallas.sweep(Rp[k], 0.5 * dt, dx, d, chat=CHAT,
                                       interpret=m1_pallas.interpret_mode()) for k in range(R.shape[0])])
    return R


@pytest.mark.parametrize("boundary", [0, 1])
@pytest.mark.parametrize("once", [False, True])
def test_the_unpadded_split_step_is_the_padded_sweeps(boundary, once, monkeypatch):
    """No pad on x and y (an edge or wrapped plane index in the kernel), in-place z ghosts, and
    the predictor once per cell (two kernels): bitwise the padded fused sweeps, two groups, a
    grid whose axes differ."""
    monkeypatch.setattr(m1_pallas, "KERNEL", "pallas")
    monkeypatch.setattr(m1_pallas, "PREDICT_ONCE_MAX_CELLS", 2 ** 40 if once else 0)
    shape = (8, 16, 32)
    rng = np.random.default_rng(7)
    E = np.exp(rng.normal(size=(2,) + shape))
    E[:, :, 5:7] *= 1e-40
    F = 0.6 * CHAT * E[:, None] * rng.uniform(-0.5, 0.5, size=(2, 3) + shape)
    R = jnp.asarray(np.concatenate([E[:, None], F], axis=1))
    dt, dx = 0.3 / 32 / CHAT, 1.0 / 32
    a = np.asarray(jax.jit(lambda R: _padded_sweeps(R, dt, dx, boundary))(R))
    b = np.asarray(jax.jit(lambda R: m1.m1_split_step(R, dt, dx, chat=CHAT, boundary=boundary))(R))
    np.testing.assert_array_equal(b, a)


@pytest.mark.skipif(jax.default_backend() == "cpu" or jax.device_count() < 2,
                    reason="the compiled kernel inside a shard_map: two GPUs")
def test_the_fused_stage_on_a_mesh_is_the_fused_stage_on_one_device(monkeypatch):
    """Sharded along x, the x sweeps take the halo and y and z the unpadded path: bitwise one device."""
    from luna.sources.radiation import setup_radiation
    cfg = replace(_rad_cfg(N), rad_groups=1, rad_ionization=True, rad_E0=0.0,
                  rad_kappa_a=0.0, rad_kappa_s=0.0, rad_source=0.0, rad_sigma=50.0,
                  rad_alpha_B=0.3, rad_m_H=1.0, rad_photon_rate=20.0,
                  rad_cs2_neutral=0.01, rad_cs2_ionized=1.0, rad_source_radius=0.1,
                  gamma=5.0 / 3.0, rad_substeps=2, rad_kernel="pallas")
    monkeypatch.setattr(m1_pallas, "KERNEL", "auto")      # restored after: the stage sets it
    out = []
    for mesh_shape in ((1, 1, 1), (2, 1, 1)):
        sh = NamedSharding(_mesh(mesh_shape), P(None, "x", "y", "z"))
        rad = setup_radiation(cfg, sh, is_main=False)
        U = jax.device_put(jnp.asarray(_gas(N)), sh)
        c = rad["R0"]
        for _ in range(2):
            U, c = rad["before"](U, c, 2e-2)
            U, c = rad["after"](U, c, 2e-2)
        out.append([np.asarray(U)] + [np.asarray(q) for q in c])
    assert float(np.max(out[0][2])) > 0.1, "nothing was ionised"
    for a, b in zip(*out):
        np.testing.assert_array_equal(b, a)


@pytest.mark.skipif(jax.default_backend() == "cpu" or jax.device_count() < 2,
                    reason="the compiled kernel inside a shard_map: two GPUs")
def test_the_fused_stage_on_a_mesh_is_the_xla_one(monkeypatch):
    from luna.sources.radiation import setup_radiation
    cfg = replace(_rad_cfg(N), rad_groups=1, rad_ionization=True, rad_E0=0.0,
                  rad_kappa_a=0.0, rad_kappa_s=0.0, rad_source=0.0, rad_sigma=50.0,
                  rad_alpha_B=0.3, rad_m_H=1.0, rad_photon_rate=20.0,
                  rad_cs2_neutral=0.01, rad_cs2_ionized=1.0, rad_source_radius=0.1,
                  gamma=5.0 / 3.0, rad_substeps=2)
    many = NamedSharding(_mesh((2, 1, 1)), P(None, "x", "y", "z"))
    monkeypatch.setattr(m1_pallas, "KERNEL", "auto")      # restored after: the stage sets it
    out = {}
    for kernel in ("xla", "pallas"):
        rad = setup_radiation(replace(cfg, rad_kernel=kernel), many, is_main=False)
        assert m1_pallas.KERNEL == kernel
        U = jax.device_put(jnp.asarray(_gas(N)), many)
        c = rad["R0"]
        for _ in range(2):
            U, c = rad["before"](U, c, 2e-2)
            U, c = rad["after"](U, c, 2e-2)
        out[kernel] = (np.asarray(U), [np.asarray(q) for q in c])
    assert float(np.max(out["xla"][1][1])) > 0.1, "nothing was ionised"
    a, b = out["xla"][0], out["pallas"][0]
    np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-15 * np.abs(a).max())
    for a, b in zip(out["xla"][1], out["pallas"][1]):
        np.testing.assert_allclose(b, a, rtol=1e-12, atol=1e-15 * max(np.abs(a).max(), 1e-300))


@pytest.mark.skipif(jax.default_backend() not in ("gpu", "rocm", "cuda"),
                    reason="the compiled kernel needs a GPU")
def test_on_a_gpu_auto_picks_the_kernel_and_it_is_the_xla_sweep(monkeypatch):
    R = _field(1, n=64)
    monkeypatch.setattr(m1_pallas, "KERNEL", "auto")
    assert m1_pallas.use_pallas(R)
    dt = 0.3 / 64 / CHAT
    b = np.asarray(jax.jit(lambda R: m1.m1_split_step(R, dt, 1.0 / 64, chat=CHAT))(R))
    monkeypatch.setattr(m1_pallas, "KERNEL", "xla")
    a = np.asarray(jax.jit(lambda R: m1.m1_split_step(R, dt, 1.0 / 64, chat=CHAT))(R))
    np.testing.assert_allclose(b, a, rtol=1e-13, atol=1e-15 * np.abs(a).max())
