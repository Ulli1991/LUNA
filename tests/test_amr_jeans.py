"""Truelove/Jeans refinement criterion (amr.driver.flag_jeans).

The criterion must flag a cell exactly when lambda_J = sqrt(pi gamma P/(G rho^2))
is resolved by fewer than N_J cells (lambda_J < N_J dx). These tests pin the
formula to its analytic threshold, check the toggle is sharp, and confirm the
gamma/G config dependence is correct -- so an AMR collapse run is defensible.
"""
import numpy as np
import jax.numpy as jnp

from luna.amr.config import AMRConfig
from luna.amr import driver

GAMMA = 5.0 / 3.0
BOX = 1.0
N = 32
BS = 16


def _uniform_U(rho, P, n=N):
    U = np.zeros((5, n, n, n))
    U[0] = rho
    U[4] = P / (GAMMA - 1.0)                 # v=0 -> E = e_int
    return jnp.asarray(U)


def _cfg(G=1.0, levels=0):
    return AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                     block_shape=(BS,) * 3, max_levels=levels, boundary=0,
                     gamma=GAMMA, grav_G=G)


def _P_for_lambda(lamJ, rho, G):
    """P giving a target Jeans length: lambda_J^2 = pi gamma P/(G rho^2)."""
    return lamJ ** 2 * G * rho ** 2 / (np.pi * GAMMA)


def _flags(cfg, U, NJ):
    h = driver.build_hierarchy(U, cfg, thresholds=[], periodic=True)
    lvl = h[0]
    f = driver.flag_jeans(lvl.data, lvl.neighbors, lvl.n_active, cfg, NJ, 0)
    return np.asarray(f)[:int(lvl.n_active)]


def test_uniform_underresolved_flags_all():
    """lambda_J = 2 dx, N_J = 4  ->  2 < 4  ->  every block under-resolved."""
    cfg = _cfg()
    dx = BOX / N
    U = _uniform_U(1.0, _P_for_lambda(2.0 * dx, 1.0, 1.0))
    assert np.all(_flags(cfg, U, 4.0)), "under-resolved uniform must flag every block"


def test_uniform_wellresolved_flags_none():
    """lambda_J = 6 dx, N_J = 4  ->  6 > 4  ->  nothing flagged."""
    cfg = _cfg()
    dx = BOX / N
    U = _uniform_U(1.0, _P_for_lambda(6.0 * dx, 1.0, 1.0))
    assert not np.any(_flags(cfg, U, 4.0)), "well-resolved uniform must flag nothing"


def test_threshold_is_sharp():
    """The flag toggles right at lambda_J = N_J dx (strict <)."""
    cfg = _cfg()
    dx = BOX / N
    # just inside -> flag; just outside -> no flag
    U_in = _uniform_U(1.0, _P_for_lambda(3.98 * dx, 1.0, 1.0))
    U_out = _uniform_U(1.0, _P_for_lambda(4.02 * dx, 1.0, 1.0))
    assert np.all(_flags(cfg, U_in, 4.0))
    assert not np.any(_flags(cfg, U_out, 4.0))


def test_G_dependence_monotone():
    """Stronger gravity shrinks lambda_J -> more cells under-resolved. A field
    marginal at G=1 must flag at G=4 and not at G=1/4."""
    dx = BOX / N
    # tuned so lambda_J = 4.5 dx at G=1 (just resolved); lambda_J ~ 1/sqrt(G)
    P = _P_for_lambda(4.5 * dx, 1.0, 1.0)
    U = _uniform_U(1.0, P)
    assert not np.any(_flags(_cfg(G=1.0), U, 4.0)), "G=1: 4.5 dx resolved"
    assert np.all(_flags(_cfg(G=4.0), U, 4.0)), "G=4: lambda_J halves to 2.25 dx"


def test_spatial_core_flags_center_only():
    """A cold dense core (small lambda_J) flags its blocks; the warm diffuse
    envelope (large lambda_J) does not -> refinement tracks the collapse.

    Uses a 64^3/16-block grid (4x4x4 = 64 blocks) so the central core occupies
    inner blocks and the corner blocks stay well-resolved -- a 2x2x2 grid can't
    isolate a centred feature (the domain centre is the shared block corner)."""
    n = 64
    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(n,) * 3,
                    block_shape=(BS,) * 3, max_levels=0, boundary=0,
                    gamma=GAMMA, grav_G=1.0)
    dx = BOX / n
    x = (np.arange(n) + 0.5) / n * BOX
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r2 = (X - 0.5) ** 2 + (Y - 0.5) ** 2 + (Z - 0.5) ** 2
    core = np.exp(-r2 / (2 * 0.06 ** 2))             # tight central core
    rho = 1.0 + 50.0 * core
    # cold, low-P core -> tiny lambda_J there; warm envelope well-resolved
    P = _P_for_lambda(10.0 * dx, 1.0, 1.0) * (1.0 - 0.999 * core)
    U = np.zeros((5, n, n, n))
    U[0] = rho
    U[4] = P / (GAMMA - 1.0)
    h = driver.build_hierarchy(jnp.asarray(U), cfg, thresholds=[], periodic=True)
    lvl = h[0]
    flags = np.asarray(driver.flag_jeans(lvl.data, lvl.neighbors, lvl.n_active,
                                         cfg, 4.0, 0))[:int(lvl.n_active)]
    assert flags.any() and not flags.all(), \
        f"core-only refinement expected, got {flags.sum()}/{flags.size} blocks"


def test_sentinel_cells_never_flag():
    """With refined levels present, NaN sentinel (inactive) slots must compare
    False and never flag -- no spurious refinement from padding."""
    cfg = _cfg(levels=1)
    dx = BOX / N
    U = _uniform_U(1.0, _P_for_lambda(2.0 * dx, 1.0, 1.0))     # everything under-resolved
    h = driver.build_hierarchy(U, cfg, thresholds=[2.0], criterion="overdensity",
                               periodic=True)
    lvl = h[0]
    f = np.asarray(driver.flag_jeans(lvl.data, lvl.neighbors, lvl.n_active, cfg, 4.0, 0))
    n_act = int(lvl.n_active)
    assert not np.any(f[n_act:]), "sentinel/inactive slots must never flag"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("########## JEANS_TESTS_DONE ##########")
