"""LUNA's vectorised port of Torch's IMF sampler.

Same distribution as Torch's one-star-at-a-time loop (a two-sample KS test
against the real `imf_sample`; no Torch clone at $TORCH_SRC fails the test, never skips), the mass
asked for, stars inside their bins, and Torch's merge of the small stars
reproduced. Host numpy only.
"""
import inspect
import os
import sys

import numpy as np
import pytest

from luna.species.imf import (collect_small_stars_mass, poisson_counts,
                              sample_stellar_mass)

TORCH_SRC = os.environ.get("TORCH_SRC", "")      # a Torch clone's src/ (stock Torch main 744bcae suffices)
KW = dict(num_bins=10, min_samp_mass=0.08, max_samp_mass=120.0)


def test_the_queue_carries_the_mass_asked_for_in_its_bins():
    rng = np.random.default_rng(1)
    m = sample_stellar_mass(rng, 1.0e4, **KW)
    assert 0.95e4 < m.sum() < 1.1e4                  # Poisson counts: a few percent
    assert m.min() >= 0.08 and m.max() <= 120.0
    n, bins, lam, _ = poisson_counts(np.random.default_rng(1), 1.0e4, 0.08, 120.0, 10)
    assert abs(n.sum() - lam.sum()) < 5 * np.sqrt(lam.sum())


def test_the_small_stars_merge_as_torch_merges_them():
    m = np.array([3.0, 0.2, 0.3, 0.6, 5.0, 0.4, 0.1])
    out = collect_small_stars_mass(m.copy(), m_small=1.0)
    np.testing.assert_allclose(out.sum(), m.sum())
    assert (out[:2] == [3.0, 5.0]).all()             # the big ones first, untouched
    assert (out[2:] >= 1.0).sum() >= len(out[2:]) - 1  # at most one merged star short of m_small


def _torch_imf_sample():
    """Torch's imf_sample from $TORCH_SRC, else the test fails naming what is missing."""
    if not os.path.isfile(os.path.join(TORCH_SRC, "imf_sample.py")):
        pytest.fail(f"no Torch clone at $TORCH_SRC={TORCH_SRC!r} (needs imf_sample.py)")
    sys.path.insert(0, TORCH_SRC)
    import imf_sample
    # the function torch_sf.queue_stars calls, with the keywords it passes; the port mirrors it
    params = list(inspect.signature(imf_sample.sample_stellar_mass).parameters)
    assert params == ["sample_imf_mass", "num_bins", "min_samp_mass", "max_samp_mass", "sum_small", "m_small"], params
    return imf_sample


def _ks_against_torch(**port_kw):
    """(KS statistic, p-value, mean difference / its standard error): 3 x 1e4 Msun from Torch and from the port."""
    from scipy import stats
    imf_sample = _torch_imf_sample()
    np.random.seed(3)
    ref = np.concatenate([imf_sample.sample_stellar_mass(1.0e4, sum_small=False, **KW) for _ in range(3)])
    rng = np.random.default_rng(3)
    got = np.concatenate([sample_stellar_mass(rng, 1.0e4, sum_small=False, **{**KW, **port_kw}) for _ in range(3)])
    ks = stats.ks_2samp(ref, got)
    # the means against their own standard error: a 120 Msun tail makes the mean of 5e4 stars uncertain to ~0.013,
    # so a fixed tolerance is not a test
    se = np.sqrt(ref.var() / len(ref) + got.var() / len(got))
    print(f"MEASURED imf port {port_kw or 'as Torch'} vs Torch: n {len(got)} / {len(ref)}, KS D {ks.statistic:.4g} "
          f"p {ks.pvalue:.4g}, mean diff / se {(got.mean() - ref.mean()) / se:.3f}")
    return ks.statistic, ks.pvalue, (got.mean() - ref.mean()) / se


def test_the_port_draws_torchs_distribution():
    """Two-sample KS (p > 0.01) and mean (< 4 se) against Torch's own sample_stellar_mass, same arguments."""
    _, p, z = _ks_against_torch()
    assert p > 0.01, p
    assert abs(z) < 4.0, z


def test_the_ks_check_catches_a_wrong_lower_mass():
    """Teeth: the port drawing from 0.1 Msun instead of Torch's 0.08 fails the same KS check."""
    _, p, _ = _ks_against_torch(min_samp_mass=0.1)
    assert not p > 0.01, p
