"""Reflecting walls (BOUNDARY=2) conserve mass and energy to round-off, 1D-3D."""
import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, check_conserved, check_mirror, evolve

# BOUNDARY=2 once paired a raw ghost with the reconstructed state at the wall (|dM|/M up to 4e-3
# here); now two mirrored ghosts on the wide path.
SHAPES = {1: (256,), 2: (32, 32), 3: (16, 16, 16)}
PROBLEM = {1: "Woodward-Colella", 2: "Planar-Riemann-2D", 3: "Sedov-Blast-3D"}



def blast(shape, gamma=1.4, centre=(0.2, 0.3, 0.25)):
    """At rest, a Gaussian over-pressure and over-density, by default off centre near the lower walls."""
    x = np.meshgrid(*[(np.arange(n) + 0.5) / n for n in shape], indexing="ij")
    g = np.exp(-sum((xi - c) ** 2 for xi, c in zip(x, centre)) / 0.01)
    rho, p = 1.0 + 0.5 * g, 1.0 + 10.0 * g
    return np.stack([rho, *[np.zeros(shape)] * len(shape), p / (gamma - 1.0)])


def wall_cfg(sdim):
    """MC, HLLC, FP64, reflecting walls on the unit box of SHAPES[sdim]."""
    shape = SHAPES[sdim]
    kw = dict(zip(("Nx", "Ny", "Nz"), shape)) | dict(zip(("Lx", "Ly", "Lz"), (1.0,) * sdim))
    return accuracy_cfg(test_problem=PROBLEM[sdim], SPATIAL_DIMENSION=sdim, BOUNDARY=2, gamma=1.4,
                        LIMITER=1, **kw)


@pytest.mark.parametrize("sdim", (1, 2, 3))
def test_walls_conserve(sdim):
    """To t = 0.2: the blast has reflected off the near walls."""
    U0 = blast(SHAPES[sdim])
    U, _ = evolve(wall_cfg(sdim), U0, 0.2)
    check_conserved(f"{sdim}D wall", U0, U, momentum=False)     # the walls push: P has a budget


@pytest.mark.parametrize("sdim", (1, 2, 3))
def test_centred_blast_is_symmetric(sdim):
    """Centred, the walls push equally on both sides: P stays 0, and the state mirror-symmetric."""
    U0 = blast(SHAPES[sdim], centre=(0.5, 0.5, 0.5))
    U, _ = evolve(wall_cfg(sdim), U0, 0.2)
    check_conserved(f"{sdim}D centred", U0, U)
    for d in range(sdim):      # round-off, not bitwise: XLA's fused HLLC arithmetic (measured <= 6.0e-15)
        check_mirror(f"{sdim}D centred", U, d, tol=1e-13)


@pytest.mark.parametrize("sdim", (1, 2, 3))
def test_walls_sharded_equal_one_device(sdim):
    """Two shards per axis equal one device to round-off: the wall ghosts mirror on edge shards only."""
    import jax
    if jax.device_count() < 2 ** sdim:
        pytest.skip(f"needs {2 ** sdim} devices")
    U0 = blast(SHAPES[sdim])
    one, _ = evolve(wall_cfg(sdim), U0, 0.2)
    sh, _ = evolve(wall_cfg(sdim), U0, 0.2, mesh_shape=(2,) * sdim)
    check(f"{sdim}D wall sharded vs one device", float(np.abs(sh - one).max() / np.abs(one).max()), 1e-12)
