"""The dimensionally split MUSCL-Hancock M1 transport (rad_scheme =
"hancock_split") against the same 3D gates as the unsplit scheme
(test_radiation_m1_3d.py), at its own, 6x longer, step.

What the split gives up is exact invariance under exchanging axes: x y z z y x
is not z y x x y z. What it keeps exactly is mirror symmetry along each axis,
which is gated here to round-off. Measured 2026-09-24 on the diffusion
problem (kappa = 120, t = 0.1), spread against analytic: +10.1% at 96^3,
+2.0% at 128^3, -3.7% at 192^3 (unsplit: +3.0, -2.0, -5.6); anisotropy
2.8e-7, 9.4e-8, 2.2e-8. So the gate is at 128^3, with the unsplit bounds.
"""
import numpy as np
import pytest

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation.m1 import m1_damp, m1_step                # noqa: E402
from tests.test_radiation_m1_3d import CHAT, _blob, _centroid, _grid  # noqa: E402

pytestmark = pytest.mark.fast

CFL = 0.4
SCHEME = "hancock_split"


def _dt(dx):
    """The split step: each of the six sweeps at dt/2 <= cfl dx / c."""
    return 2.0 * CFL * dx / CHAT


def _run(R, t, dx, kappa=None):
    dt = _dt(dx)
    steps = int(round(t / dt))

    def one(r):
        r = m1_step(r, dt, dx, chat=CHAT, boundary=1, order=2, scheme=SCHEME)
        if kappa is not None:
            r = m1_damp(r, dt, chat=CHAT, kappa=kappa)
        return r
    step = jax.jit(one)
    for _ in range(steps):
        R = step(R)
    return R, dt * steps


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_beam_streams_along_each_axis_at_the_light_speed(axis):
    n = 48
    dx = 1.0 / n
    R, _ = _blob(np.eye(3)[axis], n)
    c0 = _centroid(np.asarray(R[0, 0]))
    R, t = _run(R, 60 * CFL * dx / (3.0 * CHAT), dx)
    moved = _centroid(np.asarray(R[0, 0])) - c0
    assert abs(moved[axis] - CHAT * t) < 2.0 * dx
    for other in range(3):
        if other != axis:
            assert abs(moved[other]) < 0.2 * dx


def test_a_diagonal_beam_keeps_its_direction_and_its_local_flux():
    n = 48
    dx = 1.0 / n
    R, d = _blob((1.0, 1.0, 1.0), n)
    c0 = _centroid(np.asarray(R[0, 0]))
    R, t = _run(R, 60 * CFL * dx / (3.0 * CHAT), dx)
    moved = _centroid(np.asarray(R[0, 0])) - c0
    dist = float(np.linalg.norm(moved))
    assert float(moved @ d / dist) > 0.999
    assert abs(dist - CHAT * t) < 2.0 * np.sqrt(3.0) * dx
    E, F = np.asarray(R[0, 0]), np.asarray(R[0, 1:])
    core = E > 0.1 * E.max()
    f = np.sqrt((F ** 2).sum(axis=0))[core] / (CHAT * E[core])
    assert f.min() > 0.95
    nhat = F[:, core] / np.sqrt((F[:, core] ** 2).sum(axis=0))
    assert (nhat * d[:, None]).sum(axis=0).min() > 0.999


def test_the_diffusion_limit_is_at_the_analytic_rate_and_isotropic():
    n = 128
    dx = 1.0 / n
    kappa, sigma0 = 120.0, 0.10
    X, Y, Z = _grid(n)
    R = np.zeros((1, 4, n, n, n))
    R[0, 0] = np.exp(-0.5 * ((X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2) / sigma0 ** 2)
    R, t = _run(jnp.asarray(R), 0.1, dx, kappa=kappa)
    E = np.asarray(R[0, 0])
    x = (np.arange(n) + 0.5) / n
    w = E / E.sum()
    var = []
    for keep in ((1, 2), (0, 2), (0, 1)):
        p = w.sum(axis=keep)
        mu = float((p * x).sum())
        var.append(float((p * (x - mu) ** 2).sum()))
    spread = float(np.mean(var)) - sigma0 ** 2
    expect = 2.0 * CHAT / (3.0 * kappa) * t
    assert abs(spread - expect) / expect < 0.10, f"{(spread - expect) / expect * 100:+.1f}%"
    assert (max(var) - min(var)) / np.mean(var) < 1e-6, f"anisotropic: {var}"


def test_mirror_symmetry_is_exact():
    """x y z z y x sweeps treat +x and -x alike: an isotropic packet stays
    mirror-symmetric along each axis to round-off (exchanging axes it does
    not, which is the price of splitting)."""
    n = 32
    dx = 1.0 / n
    X, Y, Z = _grid(n)
    R = np.zeros((1, 4, n, n, n))
    R[0, 0] = np.exp(-0.5 * ((X - .5) ** 2 + (Y - .5) ** 2 + (Z - .5) ** 2) / 0.06 ** 2)
    R, _ = _run(jnp.asarray(R), 42 * CFL * dx / (3.0 * CHAT), dx)
    E = np.asarray(R[0, 0])
    for a in range(3):
        np.testing.assert_allclose(E, np.flip(E, a), rtol=1e-12, atol=1e-15)
    assert float(E.max()) < 1.0
