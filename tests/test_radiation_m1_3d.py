"""The M1 transport in three dimensions, against what it should do.

The analytic gates in test_radiation_m1.py are one-dimensional, and the
coupled gates assert conservation, which a mis-wired axis or a wrong
off-diagonal of the pressure tensor would survive: energy is conserved
whichever direction you move it in. These are the checks that are not
blind to direction.

A free-streaming beam has |F| = c E, so the Eddington factor is 1 and the
pressure tensor is E n n: the packet translates at c along n and does not
spread. That is the strongest statement about the closure the scheme makes,
and it is the one tested here, along each axis and along the diagonal, where
every off-diagonal term of the tensor is in play.
"""

import numpy as np
import pytest


import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation.m1 import m1_damp, m1_step                # noqa: E402
from tests._helpers import check  # noqa: E402

pytestmark = pytest.mark.fast

CHAT = 1.0
CFL = 0.4
N = 48
M1_ISO_E = 1e-5         # measured 7.17e-6 (CPU, A100): the front's foot crosses the faces by step 40


def _grid(n=N):
    x = (np.arange(n) + 0.5) / n
    return np.meshgrid(x, x, x, indexing="ij")


def _blob(direction, n=N, sigma=0.08, centre=(0.3, 0.3, 0.3)):
    """A Gaussian packet of radiation streaming along `direction` at c."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    X, Y, Z = _grid(n)
    r2 = ((X - centre[0]) ** 2 + (Y - centre[1]) ** 2 + (Z - centre[2]) ** 2)
    E = np.exp(-0.5 * r2 / sigma ** 2)
    R = np.zeros((1, 4, n, n, n))
    R[0, 0] = E
    for k in range(3):
        R[0, 1 + k] = CHAT * E * d[k]          # |F| = c E: free streaming
    return jnp.asarray(R), d


def _centroid(E):
    """Energy-weighted centre, in units of the box."""
    n = E.shape[0]
    x = (np.arange(n) + 0.5) / n
    w = E / E.sum()
    return np.array([float((w.sum(axis=(1, 2)) * x).sum()),
                     float((w.sum(axis=(0, 2)) * x).sum()),
                     float((w.sum(axis=(0, 1)) * x).sum())])


def _run(R, steps, dx, kappa=None):
    def one(r):
        r = m1_step(r, CFL * dx / (3.0 * CHAT), dx, chat=CHAT, boundary=1, order=2)
        if kappa is not None:
            r = m1_damp(r, CFL * dx / (3.0 * CHAT), chat=CHAT, kappa=kappa)
        return r
    step = jax.jit(one)
    for _ in range(steps):
        R = step(R)
    return R


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_beam_streams_along_each_axis_at_the_light_speed(axis):
    """One test per axis, because an axis swapped in the flux or in the
    pressure row is invisible to any conservation check."""
    n = N
    dx = 1.0 / n
    dt = CFL * dx / (3.0 * CHAT)
    steps = 60
    d = np.eye(3)[axis]
    R, _ = _blob(d, n)
    c0 = _centroid(np.asarray(R[0, 0]))
    R = _run(R, steps, dx)
    c1 = _centroid(np.asarray(R[0, 0]))
    moved = c1 - c0
    expect = CHAT * dt * steps
    assert abs(moved[axis] - expect) < 2.0 * dx, (
        f"axis {axis}: moved {moved[axis]:.5f}, expected {expect:.5f}")
    for other in range(3):
        if other != axis:
            assert abs(moved[other]) < 0.2 * dx, (
                f"axis {axis}: leaked {moved[other]:.2e} along {other}")


def test_a_diagonal_beam_keeps_its_direction():
    """Along (1,1,1) every off-diagonal P_ij carries flux. A sign or index
    error there tilts the packet away from the diagonal, which the per-axis
    tests above cannot see."""
    n = N
    dx = 1.0 / n
    dt = CFL * dx / (3.0 * CHAT)
    steps = 60
    R, d = _blob((1.0, 1.0, 1.0), n)
    c0 = _centroid(np.asarray(R[0, 0]))
    R = _run(R, steps, dx)
    moved = _centroid(np.asarray(R[0, 0])) - c0
    dist = float(np.linalg.norm(moved))
    expect = CHAT * dt * steps
    cos = float(moved @ d / max(dist, 1e-30))
    assert cos > 0.999, f"direction off the diagonal: cos = {cos:.6f}"
    assert abs(dist - expect) < 2.0 * np.sqrt(3.0) * dx, (
        f"travelled {dist:.5f}, expected {expect:.5f}")

    # Where the beam is, the LOCAL flux must still be c E along d. The
    # centroid alone does not see this: the energy is advected by F_x, so
    # dropping the off-diagonal P_xy leaves the centroid almost right while
    # F_y quietly decays. Measured: with the off-diagonals removed every
    # other gate in this file still passed, and this one fails.
    E = np.asarray(R[0, 0])
    F = np.asarray(R[0, 1:])
    core = E > 0.1 * E.max()
    f = np.sqrt((F ** 2).sum(axis=0))[core] / (CHAT * E[core])
    assert f.min() > 0.95, (
        f"free streaming lost: min |F|/(cE) = {f.min():.4f} in the beam")
    nhat = F[:, core] / np.sqrt((F[:, core] ** 2).sum(axis=0))
    cos_local = (nhat * d[:, None]).sum(axis=0)
    assert cos_local.min() > 0.999, (
        f"local flux tilted off the diagonal: min cos = {cos_local.min():.6f}")


def test_the_diffusion_limit_is_isotropic_and_at_the_analytic_rate():
    """Optically thick and starting isotropic: the packet must spread by
    2 D t with D = c/(3 kappa), equally in x, y and z.

    The quantity gated is the SPREAD, sigma^2 - sigma0^2, not the total
    variance: the total is dominated by the initial width and would hide a
    large error in the part the physics decides.

    The grid has to resolve the diffusion length or the scheme's own
    diffusion dominates. Measured at t = 0.1, spread against analytic
    (2026-09-21): at kappa = 120, +38.4% at 48^3, +3.0% at 96^3, -5.6% at
    192^3; at kappa = 480, +197.7%, +46.6%, +9.9%. It converges, which is
    what makes it discretization and not a bug -- and the anisotropy is
    0.000% at every one of those six points, which is the part of this that
    is about three dimensions rather than about resolution.
    """
    n = 96
    dx = 1.0 / n
    dt = CFL * dx / (3.0 * CHAT)
    steps = 72                                   # t = 0.1
    kappa = 120.0
    D = CHAT / (3.0 * kappa)
    sigma0 = 0.10
    X, Y, Z = _grid(n)
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    R = np.zeros((1, 4, n, n, n))
    R[0, 0] = np.exp(-0.5 * r2 / sigma0 ** 2)
    R = _run(jnp.asarray(R), steps, dx, kappa=kappa)

    E = np.asarray(R[0, 0])
    x = (np.arange(n) + 0.5) / n
    w = E / E.sum()
    var = []
    for keep in ((1, 2), (0, 2), (0, 1)):
        p = w.sum(axis=keep)
        mean = float((p * x).sum())
        var.append(float((p * (x - mean) ** 2).sum()))
    spread = float(np.mean(var)) - sigma0 ** 2
    expect = 2.0 * D * dt * steps
    assert abs(spread - expect) / expect < 0.10, (
        f"spread {spread:.5e}, diffusion predicts {expect:.5e} "
        f"({(spread - expect) / expect * 100:+.1f}%)")
    # the 3D claim: no axis is special, to far tighter than the rate is known
    assert (max(var) - min(var)) / np.mean(var) < 1e-6, f"anisotropic: {var}"


def test_an_isotropic_source_stays_isotropic():
    """E only, no flux, no opacity: the M1 closure at f = 0 is 1/3 delta_ij,
    so the packet must expand as a sphere. Any directional bias in the
    reconstruction or the GLF flux breaks the symmetry."""
    n = 32
    dx = 1.0 / n
    X, Y, Z = _grid(n)
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    R = np.zeros((1, 4, n, n, n))
    R[0, 0] = np.exp(-0.5 * r2 / 0.06 ** 2)
    R0 = jnp.asarray(R)
    R = _run(R0, 40, dx)
    E = np.asarray(R[0, 0])
    # sum E: what crossed the zero-gradient faces; sum F = 0 by the mirror symmetry
    check("m1 isotropic source |d sum E| / sum E", abs(E.sum() / float(jnp.sum(R0[0, 0])) - 1.0), M1_ISO_E)
    check("m1 isotropic source |sum F| / (c sum E)",
          float(np.abs(np.asarray(R[0, 1:]).sum(axis=(1, 2, 3))).max()) / (CHAT * E.sum()), 1e-13)
    # the three permutations of the same field must agree cell for cell
    assert np.allclose(E, np.transpose(E, (1, 0, 2)), rtol=1e-10, atol=1e-12)
    assert np.allclose(E, np.transpose(E, (2, 1, 0)), rtol=1e-10, atol=1e-12)
    assert np.allclose(E, np.transpose(E, (0, 2, 1)), rtol=1e-10, atol=1e-12)
    assert float(E.max()) < 1.0 and float(E.sum()) > 0.0      # it did spread
