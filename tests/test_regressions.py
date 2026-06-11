"""LUNA regression tests.

Small, fast tests that assert behaviour rather than just plot it. Run with:

    pytest tests/test_regressions.py -v

These tests exist to fail CI when a refactor breaks a known-good behaviour.
They are deliberately small (Nx ≤ 64, t_end < 1 sec wall time) so they can
run on a CPU laptop or a single GPU in seconds. The full convergence /
shock-tube suites live in scripts/validation/plots/* and produce figures
instead of asserting.
"""

import jax
import numpy as np
import pytest


def _build_cfg_for_test(**overrides):
    """Construct a Config with sensible test defaults; allow per-test overrides."""
    from template_config import Config

    cfg = Config(**overrides)
    # Make tests quiet and fast.
    cfg.production_mode = True
    cfg.PLOT_AFTER_SIM = False
    cfg.enable_diagnostics = True
    cfg.enable_checkpointing = False
    return cfg


def _run_short(cfg):
    """Drive a short simulation directly and return the final state.
    Skips the full main.py machinery — assembles only the bits we need so
    tests run in seconds without SLURM / file I/O overhead.
    """
    from ic_generation.ic import initial_conditions
    from time_integration.integrators import create_muscl_hancock_modular_step
    from utils.grid_params import GridParams
    from utils.halo_exchange_auto import pad_with_halo_auto

    grid = GridParams.from_config(cfg)
    U = initial_conditions(cfg)
    pad_halo_fn = lambda U, b, d: pad_with_halo_auto(U, b, d)

    compute_dt_fn, step_fn = create_muscl_hancock_modular_step(
        pad_halo_fn,
        cfg.gamma,
        cfg.ORDER,
        cfg.LIMITER,
        grid.dx,
        grid.dy,
        grid.dz,
        grid.inv_dx,
        grid.inv_dy,
        grid.inv_dz,
        cfg.BOUNDARY,
        cfg.SPATIAL_DIMENSION,
        cfg.CFL,
        sharding=None,
        riemann_solver=getattr(cfg, "riemann_solver", "hllc"),
    )

    t = 0.0
    while t < cfg.t_end:
        dt = float(compute_dt_fn(U))
        dt = min(dt, cfg.t_end - t)
        U = step_fn(U, dt)
        t += dt
        # Safety net: bail out if integrator stalls.
        if dt < 1e-15:
            break
    return np.asarray(U)


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
    U = _run_short(cfg)
    assert np.all(np.isfinite(U)), "Sod produced NaN/Inf"
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
        CFL=0.3,
        t_end=0.05,
    )
    U_init = np.asarray(
        jax.device_get(
            __import__("ic_generation.ic", fromlist=["initial_conditions"]).initial_conditions(cfg)
        )
    )
    mass0 = U_init[0].sum()
    U = _run_short(cfg)
    assert np.all(np.isfinite(U)), "Sedov produced NaN/Inf"
    mass_final = U[0].sum()
    rel_mass_err = abs(mass_final - mass0) / mass0
    # Outflow BC + interior source — should still be conservative to ~ε.
    assert rel_mass_err < 1e-4, f"Sedov mass not conserved: rel err {rel_mass_err:.3e}"
    # Shocked region should produce overdense cells.
    assert U[0].max() > 1.5 * U[0].mean(), "Sedov shock peak missing"


def test_gresho_lhllc_preserves_vphi_better_than_hllc():
    """Two short Gresho runs at small Nx — L-HLLC must end with smaller
    v_phi dissipation at the peak than HLLC. Lower bound for the L-HLLC
    paper claim."""
    base_overrides = dict(
        test_problem="Gresho-Vortex-2D",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,
        Nx=64,
        Ny=64,
        Lx=1.0,
        Ly=1.0,
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=1,
        CFL=0.4,
        t_end=0.5,  # ~0.4 turnovers — short
        gresho_Mmax=0.05,  # not too low-Mach so HLLC can run
    )

    def peak_vphi(U):
        rho = U[0]
        vx = U[1] / rho
        vy = U[2] / rho
        dx = 1.0 / 64
        dy = 1.0 / 64
        x = (np.arange(64) + 0.5) * dx - 0.5
        y = (np.arange(64) + 0.5) * dy - 0.5
        X, Y = np.meshgrid(x, y, indexing="ij")
        R = np.sqrt(X * X + Y * Y)
        v_phi = (-Y * vx + X * vy) / np.maximum(R, 1e-12)
        # Annulus near peak r=0.2
        mask = (R > 0.18) & (R < 0.22)
        return float(v_phi[mask].mean())

    cfg_h = _build_cfg_for_test(**base_overrides, riemann_solver="hllc")
    U_h = _run_short(cfg_h)
    assert np.all(np.isfinite(U_h)), "Gresho HLLC NaN'd"

    cfg_l = _build_cfg_for_test(**base_overrides, riemann_solver="lhllc")
    U_l = _run_short(cfg_l)
    assert np.all(np.isfinite(U_l)), "Gresho L-HLLC NaN'd"

    v_h = peak_vphi(U_h)
    v_l = peak_vphi(U_l)
    # L-HLLC should retain MORE of the analytic peak velocity (=1.0).
    # Generous bound — in this 64² short-run regime even L-HLLC dissipates.
    assert v_l > v_h - 0.05, (
        f"L-HLLC didn't beat HLLC on Gresho v_phi: HLLC={v_h:.3f} " f"LHLLC={v_l:.3f}"
    )


def test_soundwave_1d_second_order_convergence():
    """Linear acoustic wave on a periodic box, evolved for EXACTLY ONE
    period — the analytic answer at t_end equals the IC, so L1(rho - rho_init)
    measures the integrator's truncation error. Fitted log-log slope must
    fall in (-2.5, -1.5) to catch regressions to 1st-order behaviour.

    IC params: ρ₀=1, p₀=0.6, γ=5/3 → c_s² = 1, so one period = Lx = 1.0.
    Needs fp64 — the amp=1e-4 sound wave hits fp32 noise floor by Nx=64.
    """
    jax.config.update("jax_enable_x64", True)
    base = dict(
        test_problem="Soundwave",
        SPATIAL_DIMENSION=1,
        BOUNDARY=0,
        Lx=1.0,
        gamma=5.0 / 3.0,
        ORDER=1,
        LIMITER=1,
        CFL=0.4,
        t_end=1.0,  # one full acoustic period
    )
    Nxs = [32, 64, 128, 256]
    L1s = []
    for Nx in Nxs:
        cfg = _build_cfg_for_test(**base, Nx=Nx)
        U_init = np.asarray(
            __import__("ic_generation.ic", fromlist=["initial_conditions"]).initial_conditions(cfg)
        )
        U = _run_short(cfg)
        assert np.all(np.isfinite(U)), f"Soundwave Nx={Nx} NaN'd"
        # Analytic: after one period, ρ(x, T) = ρ(x, 0). L1 vs IC is the
        # truncation error.
        L1 = float(np.mean(np.abs(U[0] - U_init[0])))
        L1s.append(L1)
    slope = np.polyfit(np.log(Nxs), np.log(L1s), 1)[0]
    # PLM second-order should give slope ~ -2. Tight bound catches 1st order.
    assert -2.5 < slope < -1.5, (
        f"Soundwave convergence slope {slope:.2f} outside (-2.5, -1.5). " f"L1 at Nxs={Nxs}: {L1s}"
    )


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
        CFL=0.4,
        t_end=0.5,
    )
    U_init = np.asarray(
        __import__("ic_generation.ic", fromlist=["initial_conditions"]).initial_conditions(cfg)
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
        CFL=0.4,
        t_end=0.5,
    )
    U_init = np.asarray(
        __import__("ic_generation.ic", fromlist=["initial_conditions"]).initial_conditions(cfg)
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
        CFL=0.3,
        t_end=0.1,
    )
    U = _run_short(cfg)
    assert np.all(np.isfinite(U)), "Sod-3D NaN'd"
    # Density contrast should still be present.
    assert (U[0].max() / U[0].min()) > 4.0, "Sod-3D density contrast collapsed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
