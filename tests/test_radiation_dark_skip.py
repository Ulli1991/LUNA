"""The radiation half step skips its M1 substeps while nothing shines, no
light is left and nothing is ionised (sources.radiation.SKIP_DARK). It must
be exact: the radiation carry bitwise the substeps it skips, on one device
and on a (2,2,2) mesh, for the ionising path and the plain M1 one; and once
a source shines the substeps run. Two differences, both round-off: the
implicit ionisation solve returns x = 1.1e-16 rather than 0 without any
light (the skip keeps the exact 0), and the prescribed pressure comes from
its own small program rather than the substep one, which rounds the energy
differently (1e-14 relative, measured)."""
from dataclasses import replace

import numpy as np
import pytest

import jax

import jax.numpy as jnp  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

import luna.sources.radiation as radmod  # noqa: E402
from luna.sources.radiation import setup_radiation  # noqa: E402
from tests.test_radiation_coupled import _gas, _mesh, _rad_cfg  # noqa: E402

N = 16


def _ion_cfg(rate):
    return replace(_rad_cfg(N), rad_groups=1, rad_ionization=True, rad_E0=0.0,
                   rad_kappa_a=0.0, rad_kappa_s=0.0, rad_source=0.0, rad_sigma=50.0,
                   rad_alpha_B=0.3, rad_m_H=1.0, rad_photon_rate=rate,
                   rad_cs2_neutral=0.01, rad_cs2_ionized=1.0, rad_source_radius=0.1,
                   gamma=5.0 / 3.0, rad_substeps=4)


def _sharding(mesh_shape):
    return NamedSharding(_mesh(mesh_shape), P(None, "x", "y", "z"))


def _run(cfg, sh, U, skip, monkeypatch, steps=3, h=2e-2):
    monkeypatch.setattr(radmod, "SKIP_DARK", skip)
    rad = setup_radiation(cfg, sh, is_main=False)
    U = jax.device_put(jnp.asarray(U), sh)
    c = rad["R0"]
    for _ in range(steps):
        U, c = rad["before"](U, c, h)
        U, c = rad["after"](U, c, h)
    return np.asarray(U), jax.tree_util.tree_map(np.asarray, c)


def _meshes():
    shapes = [(1, 1, 1)]
    if jax.device_count() >= 8:
        shapes.append((2, 2, 2))
    return shapes


@pytest.mark.parametrize("mesh_shape", _meshes())
def test_a_dark_ionising_step_is_the_substeps(mesh_shape, monkeypatch):
    cfg, sh = _ion_cfg(0.0), _sharding(mesh_shape)
    U0 = _gas(N)
    a = _run(cfg, sh, U0, True, monkeypatch)
    b = _run(cfg, sh, U0, False, monkeypatch)
    np.testing.assert_array_equal(a[0][:4], b[0][:4])
    np.testing.assert_allclose(a[0][4], b[0][4], rtol=1e-13, atol=0.0)
    (Ra, xa, wa), (Rb, xb, wb) = a[1], b[1]
    np.testing.assert_array_equal(Ra, Rb)
    np.testing.assert_array_equal(wa, wb)
    assert np.all(xa == 0.0) and float(np.max(xb)) <= 1e-15
    # and the gas did get the prescribed pressure: the step is not a no-op
    assert not np.array_equal(a[0][4], U0[4])


@pytest.mark.parametrize("mesh_shape", _meshes())
def test_a_dark_m1_step_is_bitwise_the_substeps(mesh_shape, monkeypatch):
    cfg, sh = replace(_rad_cfg(N), rad_E0=0.0, rad_source=0.0), _sharding(mesh_shape)
    U0 = _gas(N)
    a = _run(cfg, sh, U0, True, monkeypatch)
    b = _run(cfg, sh, U0, False, monkeypatch)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_a_shining_source_is_not_skipped(monkeypatch):
    cfg, sh = _ion_cfg(20.0), _sharding((1, 1, 1))
    U0 = _gas(N)
    a = _run(cfg, sh, U0, True, monkeypatch)
    b = _run(cfg, sh, U0, False, monkeypatch)
    assert float(np.max(a[1][1])) > 0.5, "nothing was ionised"
    np.testing.assert_array_equal(a[0], b[0])
    for x, y in zip(a[1], b[1]):
        np.testing.assert_array_equal(x, y)
