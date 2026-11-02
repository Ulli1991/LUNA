"""The cloud's turbulent field has the shell spectrum it is asked for.

The cloud's `problem.turb_slope` is the exponent of the shell energy spectrum, E(k) ~ k^slope.
Before the fix the code gave per-mode power k^slope (a flat E(k) for the
default -2); both checks below fail on that formula. Tolerances are about
3 sigma of the seed-to-seed scatter measured over 8 seeds (slope: 0.029 at
64^3, 0.014 at 128^3; fraction: 0.033-0.040). Pure numpy.
"""
import numpy as np
import pytest

from luna.ic_generation.turbulent_sphere_3d import _turbulent_field

SLOPE_TOL = 0.1       # fitted slope vs requested, k = 4 .. N/4
FRAC_TOL = 0.12       # energy fraction at k <= k_cloud vs its expected value
K_CLOUD = 2           # L / (2 R) of the study cloud: 8 pc box, 2 pc radius


def _kshell(N):
    kf = np.fft.fftfreq(N) * N
    k = np.sqrt(kf[:, None, None] ** 2 + kf[None, :, None] ** 2 + kf[None, None, :] ** 2)
    return k, np.rint(k).astype(int)


def _shell_spectrum(v):
    """E(k) summed over integer shells rint(|k|) = k, all three components."""
    _, ki = _kshell(v.shape[1])
    return sum(np.bincount(ki.ravel(), weights=(np.abs(np.fft.fftn(c)) ** 2).ravel())
               for c in v)


def _expected_fraction(N, slope, kc):
    """Expected energy fraction in shells <= kc: every mode carries k^(slope-2)."""
    k, ki = _kshell(N)
    k[0, 0, 0] = 1.0
    w = k ** (slope - 2.0)
    w[0, 0, 0] = 0.0
    return w[ki <= kc].sum() / w.sum()


@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("slope", [-2.0, -5.0 / 3.0])
def test_shell_spectrum_slope(N, slope):
    E = _shell_spectrum(_turbulent_field(N, slope, seed=1))
    k = np.arange(4, N // 4 + 1)
    fitted = np.polyfit(np.log(k), np.log(E[k]), 1)[0]
    assert abs(fitted - slope) < SLOPE_TOL, f"E(k) ~ k^{fitted:.3f}, asked k^{slope:.3f}"


@pytest.mark.parametrize("N", [64, 128])
@pytest.mark.parametrize("slope", [-2.0, -5.0 / 3.0])
def test_energy_fraction_on_cloud_scales(N, slope):
    E = _shell_spectrum(_turbulent_field(N, slope, seed=1))
    frac = E[:K_CLOUD + 1].sum() / E.sum()
    want = _expected_fraction(N, slope, K_CLOUD)
    assert abs(frac - want) < FRAC_TOL, f"fraction at k<={K_CLOUD}: {frac:.3f}, expected {want:.3f}"


def test_unit_rms_and_no_bulk_motion():
    v = _turbulent_field(64, -2.0, seed=3)
    assert np.sqrt(np.mean(np.sum(v ** 2, axis=0))) == pytest.approx(1.0, rel=1e-12)
    for c in v:
        assert abs(c.mean()) < 1e-12
