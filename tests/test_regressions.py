"""LUNA regression tests.

Small, fast tests that assert behaviour rather than just plot it. Run with:

    pytest tests/test_regressions.py -v

These tests exist to fail CI when a refactor breaks a known-good behaviour.
They are deliberately small (Nx ≤ 64, t_end < 1 sec wall time) so they can
run on a CPU laptop or a single GPU in seconds. The full convergence /
shock-tube suites live in scripts/validation/plots/* and produce figures
instead of asserting.
"""

import pytest
pytestmark = pytest.mark.slow


import jax
import numpy as np
import pytest

from tests._helpers import _build_cfg_for_test, _run_short, check, check_conserved, check_tube_budget

# One-period sound wave below, amplitude 1e-4 * SOUND_SCALE; L1(rho) per unit 1e-4 amplitude at Nx = 32-256.
SOUND_SCALE = 0.01
SOUND_L1_BOUND = (7.0e-7, 1.7e-7, 3.8e-8, 9.0e-9)   # ~1.3x measured 5.37e-7, 1.29e-7, 2.90e-8, 6.94e-9
SOUND_ORDER = (2.0, 2.4)                           # measured 2.10, -0.1 .. +0.3


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sod_1d_runs_no_nan():
    """1D Sod shock tube — must reach t_end with finite state and the
    pressure jump going the right direction (P_left > P_right preserved)."""
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock",
        SPATIAL_DIMENSION=1,
        BOUNDARY=1,
        Nx=128,
        Lx=1.0,
        gamma=1.4,
        ORDER=1,
        LIMITER=0,
        CFL=0.4,
        t_end=0.2,
    )
    from luna.ic_generation.ic import initial_conditions
    U0 = np.asarray(initial_conditions(cfg))
    U = _run_short(cfg, U0)
    assert np.all(np.isfinite(U)), "Sod produced NaN/Inf"
    # zero-gradient ends, the waves inside: every total changes by t (F(W_L) - F(W_R))
    check_tube_budget("sod 1d", U0, U, (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.2, 1.0 / 128, 1.4)
    rho = U[0]
    mom = U[1]
    E = U[2]
    KE = 0.5 * mom * mom / rho
    P = (cfg.gamma - 1.0) * (E - KE)
    # Left state should still have higher pressure than right state.
    assert P[10] > P[-10], (
        f"Sod pressure profile inverted: P_L={float(P[10]):.3f} " f"P_R={float(P[-10]):.3f}"
    )
    # And there must be a non-trivial density contrast.
    assert (rho.max() / rho.min()) > 4.0, "Sod density contrast collapsed"


def test_sedov_3d_runs_no_nan():
    """3D Sedov blast must reach t_end, conserve mass to ~machine precision
    (no source terms, periodic BC), and produce a shocked region with peak
    density > background."""
    cfg = _build_cfg_for_test(
        test_problem="Sedov-Blast-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=1,
        Nx=32,
        Ny=32,
        Nz=32,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=0,
        t_end=0.05,
    )
    U_init = np.asarray(
        jax.device_get(
            __import__("luna.ic_generation.ic", fromlist=["initial_conditions"]).initial_conditions(cfg)
        )
    )
    mass0 = U_init[0].sum()
    U = _run_short(cfg)
    assert np.all(np.isfinite(U)), "Sedov produced NaN/Inf"
    mass_final = U[0].sum()
    rel_mass_err = abs(mass_final - mass0) / mass0
    # Outflow BC + interior source — should still be conservative to ~ε.
    assert rel_mass_err < 1e-4, f"Sedov mass not conserved: rel err {rel_mass_err:.3e}"
    check_conserved("sedov 3d regression", U_init, U)   # the faces see the ambient at rest: round-off
    # Shocked region should produce overdense cells.
    assert U[0].max() > 1.5 * U[0].mean(), "Sedov shock peak missing"


def test_soundwave_1d_second_order_convergence():
    """One period of a linear sound wave (the registered IC scaled by SOUND_SCALE): L1(rho - rho_init) and its order."""
    # at amp 1e-4 the amp^2 steepening floors the error near 256 cells; at 1e-6 it sits far below
    from luna.ic_generation.ic import initial_conditions
    g = 5.0 / 3.0
    base = dict(
        test_problem="Soundwave",
        SPATIAL_DIMENSION=1,
        BOUNDARY=0,
        Lx=1.0,
        gamma=g,
        ORDER=1,
        LIMITER=1,
        t_end=1.0,  # one full acoustic period
    )
    Nxs = [32, 64, 128, 256]
    L1s = []
    for Nx in Nxs:
        cfg = _build_cfg_for_test(**base, Nx=Nx)
        U = np.asarray(initial_conditions(cfg), dtype=np.float64)
        W = np.stack([U[0], U[1] / U[0], (g - 1.0) * (U[2] - 0.5 * U[1] ** 2 / U[0])])
        Wb = W.mean(axis=1, keepdims=True)
        W = Wb + SOUND_SCALE * (W - Wb)
        U_init = np.stack([W[0], W[0] * W[1], W[2] / (g - 1.0) + 0.5 * W[0] * W[1] ** 2])
        U_end = _run_short(cfg, U_init)
        L1 = float(np.mean(np.abs(U_end[0] - U_init[0]))) / SOUND_SCALE
        L1s.append(L1)
    for Nx, L1, bound in zip(Nxs, L1s, SOUND_L1_BOUND):
        check(f"soundwave MC Nx={Nx} L1", L1, bound)
    slope = -np.polyfit(np.log(Nxs), np.log(L1s), 1)[0]
    check("soundwave MC L1 order 32-256", slope, SOUND_ORDER[0], lower=True)
    check("soundwave MC L1 order 32-256", slope, SOUND_ORDER[1])


def test_kh_2d_no_nan_short():
    """KH instability at low res should evolve for 200 steps without NaN —
    catches regressions in the 2D PLM / HLLC / halo-exchange path."""
    cfg = _build_cfg_for_test(
        test_problem="Kelvin-Helmholtz",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,
        Nx=64,
        Ny=64,
        Lx=2.0,
        Ly=2.0,
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=1,
        t_end=0.5,
    )
    U_init = np.asarray(
        __import__("luna.ic_generation.ic", fromlist=["initial_conditions"]).initial_conditions(cfg)
    )
    mass0 = U_init[0].sum()
    U = _run_short(cfg)
    assert np.all(np.isfinite(U)), "KH NaN'd"
    mass_rel = abs(U[0].sum() - mass0) / mass0
    assert mass_rel < 1e-6, f"KH mass conservation broke: rel err {mass_rel:.3e}"


def test_soundwave_low_mach_no_nan():
    """Low-Mach soundwave (small amplitude perturbation) must not NaN even
    with HLLC — small-amplitude tests catch numerical-noise blow-up."""
    cfg = _build_cfg_for_test(
        test_problem="Soundwave",
        SPATIAL_DIMENSION=1,
        BOUNDARY=0,
        Nx=64,
        Lx=1.0,
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=1,
        CFL=0.4,
        t_end=0.1,
    )
    U = _run_short(cfg)
    assert np.all(np.isfinite(U)), "Low-Mach soundwave NaN'd"


def test_square_advection_2d_no_nan():
    """2D square advection at Nx=64 — must reach t_end without NaN and
    preserve mass to machine precision (uniform-flow advection, periodic)."""
    cfg = _build_cfg_for_test(
        test_problem="Square-Advected",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,
        Nx=64,
        Ny=64,
        Lx=1.0,
        Ly=1.0,
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=1,
        t_end=0.5,
    )
    U_init = np.asarray(
        __import__("luna.ic_generation.ic", fromlist=["initial_conditions"]).initial_conditions(cfg)
    )
    mass0 = U_init[0].sum()
    U = _run_short(cfg)
    assert np.all(np.isfinite(U)), "Square advection NaN'd"
    mass_rel = abs(U[0].sum() - mass0) / mass0
    assert mass_rel < 1e-10, f"Square advection mass conservation broke: rel err {mass_rel:.3e}"


def test_sod_3d_no_nan():
    """3D Sod shock should run without NaN at low res — exercises 3D PLM
    + 3D HLLC + 3D halo path together."""
    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock-3D",
        SPATIAL_DIMENSION=3,
        BOUNDARY=1,
        Nx=32,
        Ny=32,
        Nz=32,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        gamma=1.4,
        ORDER=1,
        LIMITER=0,
        t_end=0.1,
    )
    U = _run_short(cfg)
    assert np.all(np.isfinite(U)), "Sod-3D NaN'd"
    # Density contrast should still be present.
    assert (U[0].max() / U[0].min()) > 4.0, "Sod-3D density contrast collapsed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
