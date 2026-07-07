"""Conservative prolongation for child-init (amr.geometry).

Both limiters must conserve (child pair-mean = parent, so restrict undoes
prolong); the unlimited CENTERED variant must additionally be exact on a linear
field (2nd order at smooth extrema, where minmod clips to 1st).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from amr.geometry import prolong_linear_cons, prolong_centered_cons, restrict


def test_prolong_conservative_both_limiters():
    """restrict(prolong(d)) == d exactly for minmod and centered (mass/mom/E)."""
    rng = np.random.default_rng(0)
    d = jnp.asarray(rng.standard_normal((5, 8, 8, 8)))
    for prol in (prolong_linear_cons, prolong_centered_cons):
        rt = np.asarray(jax.device_get(restrict(prol(d, 3), 3)))
        assert np.max(np.abs(rt - np.asarray(d))) < 1e-12


def test_centered_prolong_exact_on_linear_interior():
    """Centered (unlimited) prolongation reproduces a smooth linear field to
    machine precision in the block interior; minmod does NOT at the extrema."""
    n = 8
    ax = np.arange(n)
    # linear ramp along axis 0 (broadcast over the other two), one variable
    coarse = np.broadcast_to((1.0 + 0.5 * ax)[None, :, None, None], (1, n, n, n)).astype(float)
    fine = np.asarray(jax.device_get(prolong_centered_cons(jnp.asarray(coarse), 3)))
    # analytic linear value at fine-cell centers along axis 0 (child of cell i at i-/+0.25)
    xf = np.repeat(ax, 2) + np.tile([-0.25, 0.25], n)
    exact = (1.0 + 0.5 * xf)
    interior = slice(2, -2)                          # drop the one-sided block edges
    got = fine[0, :, 0, 0][interior]                 # constant in y,z -> any line
    assert np.max(np.abs(got - exact[interior])) < 1e-12


def test_prolong_restrict_shapes():
    """Round-trip preserves shape (ratio-2)."""
    d = jnp.zeros((5, 4, 4, 4))
    assert prolong_linear_cons(d, 3).shape == (5, 8, 8, 8)
    assert restrict(prolong_centered_cons(d, 3), 3).shape == (5, 4, 4, 4)
