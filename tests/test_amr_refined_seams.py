"""Regression for the refined-level periodic-neighbour fix.

A silent bug built refined levels with ``find_face_neighbors(periodic=False)``, so
fine blocks at periodic / parent-crossing seams could not see their same-level
fine neighbour and fell back to the coarse underlay -> 1st-order seams (2D/3D
convergence degraded from 2 to ~1). The fix passes the domain periodicity to all
levels.

Invariant pinned here: a FULL-refine hierarchy (whole periodic domain refined one
level), seeded with the same fine data as an equivalent UNIFORM block grid, must
advance BIT-EXACTLY like that uniform grid -- interiors AND every seam, including
the periodic wrap and cross-parent seams.
"""
import numpy as np
import jax
import jax.numpy as jnp

from luna.amr.config import AMRConfig
from luna.amr import driver

GAMMA = 5.0 / 3.0


def _smooth_U(n, ndim):
    """Smooth periodic primitive->conservative field (density bump, small v)."""
    x = (np.arange(n) + 0.5) / n
    k = 2 * np.pi
    rho = 1.0 + 0.2 * np.sin(k * x)
    u = 0.1 * np.cos(k * x)
    p = 1.0 + 0.1 * np.sin(k * x)
    if ndim == 2:
        b = lambda a: np.broadcast_to(a[:, None], (n, n))
        E = p / (GAMMA - 1) + 0.5 * rho * u * u
        return np.stack([b(rho), b(rho * u), np.zeros((n, n)), b(E)], axis=0)
    b = lambda a: np.broadcast_to(a[:, None, None], (n, n, n))
    E = p / (GAMMA - 1) + 0.5 * rho * u * u
    z = np.zeros((n, n, n))
    return np.stack([b(rho), b(rho * u), z, z, b(E)], axis=0)


def _assemble(level, nblk, bs, ndim):
    na = int(level.n_active)
    data = np.asarray(jax.device_get(level.data))[:na]
    cc = np.asarray(jax.device_get(level.coords))[:na]
    nv = data.shape[1]
    out = np.full((nv,) + (nblk * bs,) * ndim, np.nan)
    for i in range(na):
        sl = tuple(slice(cc[i, d] * bs, (cc[i, d] + 1) * bs) for d in range(ndim))
        out[(slice(None),) + sl] = data[i]
    return out


def _run_full_refine_vs_uniform(ndim, base, bs):
    neff = base * 2
    dt = jnp.asarray(0.2 * (1.0 / neff))

    # uniform 2N block grid (max_levels=0), true fine field
    cfgU = AMRConfig(ndim=ndim, box=(1.0,) * ndim, base_grid=(neff,) * ndim,
                     block_shape=(bs,) * ndim, max_levels=0, boundary=0, gamma=GAMMA)
    hU = driver.build_hierarchy(jnp.asarray(_smooth_U(neff, ndim)), cfgU,
                                thresholds=[0.0], criterion="static", periodic=True, pad_to=1)
    U = _assemble(driver.advance(hU, dt, cfgU, gamma=GAMMA, limiter=1)[0], neff // bs, bs, ndim)

    # full-refine base N -> L1 2N-eff, fine level overwritten with the SAME field
    cfgA = AMRConfig(ndim=ndim, box=(1.0,) * ndim, base_grid=(base,) * ndim,
                     block_shape=(bs,) * ndim, max_levels=1, boundary=0, gamma=GAMMA,
                     static_region=((0.0, 1.0),) * ndim)
    hA = driver.build_hierarchy(jnp.asarray(_smooth_U(base, ndim)), cfgA,
                                thresholds=[0.0], criterion="static", periodic=True, pad_to=1)
    fine = hA[1]
    ref = _smooth_U(neff, ndim)                       # true fine field, dense
    d = np.array(jax.device_get(fine.data))
    cc = np.asarray(jax.device_get(fine.coords))
    for i in range(int(fine.n_active)):
        sl = tuple(slice(cc[i, dd] * bs, (cc[i, dd] + 1) * bs) for dd in range(ndim))
        d[i] = ref[(slice(None),) + sl]
    hA = (hA[0], fine._replace(data=jnp.asarray(d))) + tuple(hA[2:])
    A = _assemble(driver.advance(hA, dt, cfgA, gamma=GAMMA, limiter=1)[1], neff // bs, bs, ndim)
    return float(np.nanmax(np.abs(U[0] - A[0])))


def test_full_refine_matches_uniform_2d():
    """2D: refined-level advance == uniform, bit-exact (periodic seams incl wrap)."""
    assert _run_full_refine_vs_uniform(2, base=16, bs=8) < 1e-13


def test_full_refine_matches_uniform_3d():
    """3D: same invariant in the production dimensionality."""
    assert _run_full_refine_vs_uniform(3, base=8, bs=8) < 1e-13
