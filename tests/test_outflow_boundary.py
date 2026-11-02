"""Diode outflow (BOUNDARY=4) in 3-D: gas leaves through a face and nothing
comes back, and a shard seam changes nothing.

The 3-D diode used to take the narrow (one-ghost) path: at a shard seam the
two sides built the face flux from different states, so the sums did not
telescope and a 2-shard run differed from one device at the seam (audit M13).
It now takes the wide path with a two-layer diode ghost. docs/WALL_M3.md.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp


from jax.sharding import Mesh, NamedSharding, PartitionSpec as P   # noqa: E402

from luna.hydro.conversions import prim_to_cons                      # noqa: E402
from luna.time_integration.integrators import create_muscl_hancock_modular_step  # noqa: E402
from luna.utils.halo_exchange_shardmap import make_halo_shardmap     # noqa: E402

N, GAMMA, L = 32, 5.0 / 3.0, 1.0
DX = L / N


def _blob(vx=1.0):
    """A dense blob at x = 0.7 moving +x through a light, still ambient."""
    x = (np.arange(N) + 0.5) * DX
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    b = np.exp(-((X - 0.7) ** 2 + (Y - 0.5) ** 2 + (Z - 0.45) ** 2) / (2 * 0.08 ** 2))
    rho = 0.1 + b
    W = np.stack([rho, vx * b / (0.1 + b), 0.1 * b * np.sin(6 * Y), 0.05 * b, 0.1 + 0.0 * X])
    return prim_to_cons(jnp.asarray(W), GAMMA)


def _stepper(mesh, boundary):
    pad = make_halo_shardmap(mesh, 3)
    inv = 1.0 / DX
    return create_muscl_hancock_modular_step(
        pad, GAMMA, 1, 0, DX, DX, DX, inv, inv, inv, boundary, 3, 0.4,
        riemann_solver="hllc", use_shardmap=True, mesh=mesh)


def _mesh(shape):
    n = int(np.prod(shape))
    if jax.device_count() < n:
        pytest.skip(f"needs {n} devices")
    return Mesh(np.array(jax.devices()[:n]).reshape(shape), ("x", "y", "z"))


def _run(shape, boundary, n_steps):
    mesh = _mesh(shape)
    dt_fn, step = _stepper(mesh, boundary)
    U = jax.device_put(_blob(), NamedSharding(mesh, P(None, "x", "y", "z")))
    masses = [float(jnp.sum(U[0]))]
    for _ in range(n_steps):
        U = step(U, float(dt_fn(U)))
        masses.append(float(jnp.sum(U[0])))
    return np.asarray(U), np.array(masses) * DX ** 3


@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 2, 2)])
def test_diode_sharded_equals_one_device(shape):
    one, _ = _run((1, 1, 1), 4, 20)
    sh, _ = _run(shape, 4, 20)
    d = np.max(np.abs(sh - one) / (np.abs(one).max(axis=(1, 2, 3), keepdims=True)))
    print(f"diode {shape} vs one device: {d:.2e}")
    assert d < 1e-12


def test_diode_gas_leaves_and_does_not_come_back():
    """The blob crosses the +x face: the box loses its mass, step by step,
    and none of it reappears at -x (the periodic box brings it round)."""
    U, m = _run((2, 1, 1), 4, 120)
    Up, mp = _run((2, 1, 1), 0, 120)
    lost = m[0] - m[-1]
    rho_lo = U[0, :4].max()               # the -x face layers
    rho_lo_p = Up[0, :4].max()
    print(f"diode: lost {lost:.4f} of {m[0]:.4f}; max step gain {np.max(np.diff(m)):.2e}; "
          f"-x face rho {rho_lo:.4f} (periodic {rho_lo_p:.4f})")
    assert lost > 0.3 * (m[0] - 0.1)       # most of the blob's own mass is gone
    assert np.all(np.diff(m) <= 1e-14 * m[0])   # never gains mass
    assert rho_lo < 0.1 + 1e-3             # nothing arrived at the far face
    assert rho_lo_p > 0.1 + 1e-2           # (and in a periodic box it does)
    assert abs(mp[-1] - mp[0]) < 1e-12 * mp[0]
    from tests._helpers import check_conserved
    check_conserved("outflow periodic twin", np.asarray(_blob()), Up)     # mass, energy, momentum to round-off
