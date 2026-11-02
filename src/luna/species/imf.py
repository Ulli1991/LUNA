"""Torch's Kroupa (2001) IMF sampler, vectorised, quirks kept (docs/TORCH.md)."""
import numpy as np


def _kroupa(m, a):
    """Kroupa segments as Torch writes them (piecewise, unnormalised by `a`)."""
    m = np.asarray(m, dtype=np.float64)
    k = np.where(m < 0.08, a * m ** -0.3,
                 np.where(m < 0.5, a * 0.08 * m ** -1.3, a * 0.08 * 0.5 * m ** -2.3))
    return np.where(m < 0.001, 0.0, k)


def _mkroupa(m, a):
    return np.asarray(m) * _kroupa(m, a)


def _integrate(f, lo, hi, a, n=4001):
    """Integral of `f` on [lo, hi] over a log grid split at the Kroupa breaks."""
    edges = [lo] + [b for b in (0.08, 0.5) if lo < b < hi] + [hi]
    total = 0.0
    for x0, x1 in zip(edges[:-1], edges[1:]):
        x = np.geomspace(x0, x1, n)
        total += np.trapezoid(f(x, a), x)
    return total


def poisson_counts(rng, sink_mass, m_min, m_max, num_bins):
    """(n_stars per bin, bin edges, lam, norm), as `sample_stars_poisson`."""
    norm = 1.0 / _integrate(_kroupa, m_min, m_max, 1.0)
    bins = np.logspace(np.log10(m_min), np.log10(m_max), num_bins + 1)
    avg_mass = _integrate(_mkroupa, m_min, m_max, norm)
    mass_per_bin = np.array([_integrate(_mkroupa, bins[i], bins[i + 1], norm)
                             / _integrate(_kroupa, bins[i], bins[i + 1], norm)
                             for i in range(num_bins)])
    frac_per_bin = np.array([_integrate(_mkroupa, bins[i], bins[i + 1], norm) / avg_mass
                             for i in range(num_bins)])
    lam = sink_mass * frac_per_bin / mass_per_bin
    return rng.poisson(lam), bins, lam, norm


def collect_small_stars_mass(masses, m_small=1.0):
    """Torch's merge of the stars below `m_small`, unchanged."""
    small = masses[masses < m_small]
    masses = masses[masses >= m_small]
    b = 0
    if len(small) > 1:
        while small[-1] < m_small and len(small[b:]) > 1:
            small[b] = small[b] + small[-1]
            small = np.delete(small, -1)
            if len(small[b:]) > 1 and small[b] >= m_small:
                b += 1
        if small[-1] < m_small:
            small[-2] = small[-2] + small[-1]
            small = np.delete(small, -1)
    return np.append(masses, small)


def sample_stellar_mass(rng, sample_imf_mass, num_bins=10, min_samp_mass=1.0,
                        max_samp_mass=150.0, sum_small=False, m_small=1.0):
    """Stellar masses adding up to about `sample_imf_mass`, Torch's way."""
    n_stars, bins, _, norm = poisson_counts(rng, sample_imf_mass, min_samp_mass,
                                            max_samp_mass, num_bins)
    out = []
    for i, n in enumerate(n_stars):
        got = []
        need = int(n)
        while need > 0:
            k = max(4 * need, 64)
            m = rng.uniform(bins[i], bins[i + 1], k)
            r = rng.uniform(size=k)
            ok = m[_mkroupa(m, norm) / r > 1.0]
            got.append(ok[:need])
            need -= len(ok[:need])
        if got:
            out.append(np.concatenate(got))
    masses = np.concatenate(out) if out else np.zeros(0)
    if sum_small:
        masses = collect_small_stars_mass(masses, m_small)
    rng.shuffle(masses)
    return masses
