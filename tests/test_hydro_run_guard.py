"""The accuracy-run drivers raise when a run stops short of its target time; they never return as if it finished."""
import numpy as np
import pytest

from tests._helpers import RunAborted, _build_cfg_for_test, _run_short, accuracy_cfg, evolve



def sod(n=64):
    """A 1D Sod config and its conserved IC on n cells."""
    cfg = accuracy_cfg(test_problem="Sod-Shock", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0, BOUNDARY=1, gamma=1.4)
    x = (np.arange(n) + 0.5) / n
    rho, p = np.where(x < 0.5, 1.0, 0.125), np.where(x < 0.5, 1.0, 0.1)
    return cfg, np.stack([rho, 0.0 * rho, p / 0.4])


def test_evolve_reaches_the_target():
    """A healthy run returns the state at t and a plausible step count."""
    cfg, U0 = sod()
    U, n = evolve(cfg, U0, 0.1)
    assert np.isfinite(U).all() and 15 <= n <= 20, n          # measured 17 steps


def test_evolve_raises_at_the_step_cap():
    """The step cap raises RunAborted; it does not return the state short of t."""
    cfg, U0 = sod()
    with pytest.raises(RunAborted, match="stopped at t="):
        evolve(cfg, U0, 0.1, max_steps=5)


def test_evolve_raises_on_a_nan_state():
    """A NaN state (NaN dt) raises RunAborted instead of returning."""
    cfg, U0 = sod()
    U0[0, 10] = np.nan
    with pytest.raises(RunAborted):
        evolve(cfg, U0, 0.1)


def test_evolve_raises_when_devices_are_missing():
    """A mesh with more devices than are visible raises; it does not skip."""
    import jax
    cfg, U0 = sod()
    try:
        evolve(cfg, U0, 0.1, mesh_shape=(len(jax.devices()) + 1,))
    except RunAborted as e:
        assert "devices" in str(e)
    except BaseException as e:                  # a pytest skip is a BaseException; it must not happen here
        pytest.fail(f"evolve did not raise RunAborted: {type(e).__name__}: {e}")
    else:
        pytest.fail("evolve returned with too few devices")


def test_run_short_raises_on_a_nan_state():
    """_run_short raises RunAborted on a NaN state instead of returning early."""
    cfg = _build_cfg_for_test(test_problem="Soundwave", SPATIAL_DIMENSION=1, BOUNDARY=0, Lx=1.0, Nx=32,
                              gamma=5.0 / 3.0, ORDER=1, LIMITER=1, t_end=0.1)
    U = np.stack([np.ones(32), np.zeros(32), np.full(32, 0.9)])
    U[0, 3] = np.nan
    with pytest.raises(RunAborted):
        _run_short(cfg, U)
