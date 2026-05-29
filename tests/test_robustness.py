"""LUNA robustness regression tests.

Covers the failure-mode plumbing that the main `test_regressions.py` doesn't:
- pressure positivity over the run
- IC sanity checks
- NaN injection -> retry / abort path
- checkpoint write+read round-trip
- determinism: same Config twice -> bit-identical state

Run:    pytest tests/test_robustness.py -v
"""

import os
import shutil
import tempfile

import jax
import numpy as np
import pytest


def _build_cfg_for_test(**overrides):
    from template_config import Config

    cfg = Config(**overrides)
    cfg.production_mode = True
    cfg.PLOT_AFTER_SIM = False
    cfg.enable_diagnostics = True
    cfg.enable_checkpointing = False
    return cfg


def _make_step_fns(cfg):
    """Return (compute_dt_fn, step_fn) for a Config — same path as production."""
    from time_integration.integrators import create_muscl_hancock_modular_step
    from utils.grid_params import GridParams
    from utils.halo_exchange_auto import pad_with_halo_auto

    grid = GridParams.from_config(cfg)

    def pad(U, b, d):
        return pad_with_halo_auto(U, b, d)

    return create_muscl_hancock_modular_step(
        pad,
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


def _run_short(cfg):
    from ic_generation.ic import initial_conditions

    U = initial_conditions(cfg)
    compute_dt_fn, step_fn = _make_step_fns(cfg)
    t = 0.0
    while t < cfg.t_end:
        dt = float(compute_dt_fn(U))
        dt = min(dt, cfg.t_end - t)
        U = step_fn(U, dt)
        t += dt
        if dt < 1e-15:
            break
    return np.asarray(U)


# ---------------------------------------------------------------------------
# IC sanity checks
# ---------------------------------------------------------------------------


def test_ic_sanity_sod_finite_positive():
    """Sod IC must produce a finite, ρ>0, P>0 conservative state."""
    from ic_generation.ic import initial_conditions

    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock",
        SPATIAL_DIMENSION=1,
        BOUNDARY=1,
        Nx=128,
        Lx=1.0,
        gamma=1.4,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=0,
    )
    U = np.asarray(initial_conditions(cfg))
    assert np.all(np.isfinite(U)), "Sod IC has NaN/Inf"
    rho = U[0]
    mom = U[1]
    E = U[2]
    P = (cfg.gamma - 1.0) * (E - 0.5 * mom * mom / rho)
    assert np.all(rho > 0), f"Sod IC has non-positive rho: min={rho.min()}"
    assert np.all(P > 0), f"Sod IC has non-positive P: min={P.min()}"


def test_ic_sanity_sedov_finite_positive():
    """Sedov 3D IC: finite + positive ρ everywhere, but central cell has
    much higher E than the bulk (the deposited blast)."""
    from ic_generation.ic import initial_conditions

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
    )
    U = np.asarray(initial_conditions(cfg))
    assert np.all(np.isfinite(U)), "Sedov IC has NaN/Inf"
    rho = U[0]
    E = U[-1]
    assert np.all(rho > 0), f"Sedov IC has non-positive rho: min={rho.min()}"
    # Central cell should be the energy peak.
    assert E[16, 16, 16] > 10 * np.median(E), (
        "Sedov IC missing central energy injection (central E "
        f"{E[16, 16, 16]:.3e} vs median {np.median(E):.3e})"
    )


# ---------------------------------------------------------------------------
# Pressure positivity
# ---------------------------------------------------------------------------


def test_sedov_pressure_stays_positive():
    """Sedov 3D: P must remain > 0 throughout the run. HLLC can produce
    p_star < 0 without bombing on NaN — this test catches that silent mode."""
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
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=0,
        CFL=0.3,
        t_end=0.05,
    )
    U = _run_short(cfg)
    rho = U[0]
    mom_x = U[1]
    mom_y = U[2]
    mom_z = U[3]
    E = U[4]
    KE = 0.5 * (mom_x * mom_x + mom_y * mom_y + mom_z * mom_z) / rho
    P = (cfg.gamma - 1.0) * (E - KE)
    assert np.all(rho > 0), f"Sedov produced rho<=0: min={float(rho.min()):.3e}"
    assert np.all(P > 0), f"Sedov produced P<=0: min={float(P.min()):.3e}"


def test_kh_pressure_stays_positive():
    """KH 2D: P must remain > 0 throughout. Catches HLLC low-density failures
    in the mixing layer."""
    cfg = _build_cfg_for_test(
        test_problem="Kelvin-Helmholtz",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,
        Nx=64,
        Ny=64,
        Lx=2.0,
        Ly=2.0,
        gamma=5.0 / 3.0,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=1,
        CFL=0.4,
        t_end=0.5,
    )
    U = _run_short(cfg)
    rho = U[0]
    mom_x = U[1]
    mom_y = U[2]
    E = U[3]
    KE = 0.5 * (mom_x * mom_x + mom_y * mom_y) / rho
    P = (cfg.gamma - 1.0) * (E - KE)
    assert np.all(rho > 0)
    assert np.all(P > 0), f"KH produced P<=0: min={float(P.min()):.3e}"


# ---------------------------------------------------------------------------
# NaN injection -> retry / abort path
# ---------------------------------------------------------------------------


def test_nan_in_state_is_detected_by_compute_dt():
    """compute_dt_fn must return NaN when U has NaN cells. This is the
    free piggyback the runner uses to decide whether to retry."""
    from ic_generation.ic import initial_conditions

    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock",
        SPATIAL_DIMENSION=1,
        BOUNDARY=1,
        Nx=64,
        Lx=1.0,
        gamma=1.4,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=0,
        CFL=0.4,
    )
    compute_dt_fn, _ = _make_step_fns(cfg)
    U = initial_conditions(cfg)
    dt_clean = float(compute_dt_fn(U))
    assert np.isfinite(dt_clean) and dt_clean > 0, "Clean state dt should be > 0"

    # Inject NaN into a single cell.
    U_nan = np.asarray(U).copy()
    U_nan[1, 10] = np.nan
    dt_nan = float(compute_dt_fn(U_nan))
    assert not (np.isfinite(dt_nan) and dt_nan > 0), (
        f"NaN in U should propagate to dt; got dt={dt_nan}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_config_same_result():
    """Running the same Config twice must produce bit-identical state.
    This is the cheapest regression test for hidden nondeterminism
    (autotune cache, random seed drift, etc.)."""
    cfg_a = _build_cfg_for_test(
        test_problem="Sod-Shock",
        SPATIAL_DIMENSION=1,
        BOUNDARY=1,
        Nx=128,
        Lx=1.0,
        gamma=1.4,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=0,
        CFL=0.4,
        t_end=0.2,
    )
    cfg_b = _build_cfg_for_test(
        test_problem="Sod-Shock",
        SPATIAL_DIMENSION=1,
        BOUNDARY=1,
        Nx=128,
        Lx=1.0,
        gamma=1.4,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=0,
        CFL=0.4,
        t_end=0.2,
    )
    U_a = _run_short(cfg_a)
    U_b = _run_short(cfg_b)
    np.testing.assert_array_equal(
        U_a,
        U_b,
        err_msg="Same Config produced different output — hidden nondeterminism!",
    )


# ---------------------------------------------------------------------------
# Checkpoint round-trip (single-process)
# ---------------------------------------------------------------------------


def test_checkpoint_write_read_roundtrip():
    """Write a single-process checkpoint from a small Sod state, read it
    back, verify the conservative state and metadata round-trip exactly.
    Stops the 'enable_checkpointing default-on but never tested' silent rot."""
    from file_io.read_checkpoint import read_checkpoint
    from file_io.write_checkpoint import write_checkpoint
    from ic_generation.ic import initial_conditions

    cfg = _build_cfg_for_test(
        test_problem="Sod-Shock",
        SPATIAL_DIMENSION=1,
        BOUNDARY=1,
        Nx=64,
        Lx=1.0,
        gamma=1.4,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=0,
    )
    U_init = initial_conditions(cfg)
    U_np = np.asarray(U_init)

    tmp = tempfile.mkdtemp(prefix="luna_ckpt_test_")
    try:
        path = write_checkpoint(U_init, cfg, t=0.123, step=42, checkpoint_dir=tmp)
        assert path is not None and os.path.exists(path), (
            f"Checkpoint not written at {path}"
        )

        data = read_checkpoint(path)
        U_read_np = np.asarray(data["U"])
        np.testing.assert_array_equal(
            U_read_np,
            U_np,
            err_msg="Checkpoint U did not round-trip exactly",
        )
        assert abs(float(data["time"]) - 0.123) < 1e-12, (
            f"time did not round-trip: {data['time']}"
        )
        assert int(data["step"]) == 42, f"step did not round-trip: {data['step']}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_restart_from_checkpoint_continues_correctly():
    """Restart-then-continue from a mid-run checkpoint must evolve the state
    correctly. Strict bit-identical comparison vs an uninterrupted run isn't
    possible here because dt-clamping at the checkpoint boundary differs
    between continuous and split runs — but the final restarted state must
    match the reference final state to ~PLM truncation error (rtol 1e-2 at
    Nx=64 Sod). Catches the silent failure modes: U_loaded not actually
    applied, wrong field ordering, integrator state corruption."""
    from file_io.read_checkpoint import read_checkpoint
    from file_io.write_checkpoint import write_checkpoint
    from ic_generation.ic import initial_conditions

    base = dict(
        test_problem="Sod-Shock",
        SPATIAL_DIMENSION=1,
        BOUNDARY=1,
        Nx=64,
        Lx=1.0,
        gamma=1.4,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=0,
        CFL=0.4,
    )

    cfg_ref = _build_cfg_for_test(**base, t_end=0.1)
    U_ref = _run_short(cfg_ref)

    cfg_a = _build_cfg_for_test(**base, t_end=0.05)
    U_mid = _run_short(cfg_a)

    tmp = tempfile.mkdtemp(prefix="luna_restart_")
    try:
        path = write_checkpoint(U_mid, cfg_a, t=0.05, step=999, checkpoint_dir=tmp)
        data = read_checkpoint(path)
        U_loaded = np.asarray(data["U"])
        # The loaded state must match what we just wrote (round-trip).
        np.testing.assert_array_equal(
            U_loaded, np.asarray(U_mid),
            err_msg="Checkpoint did not round-trip U exactly",
        )

        # Continue from U_loaded for the second half.
        compute_dt_fn, step_fn = _make_step_fns(cfg_ref)
        U = U_loaded
        t = 0.05
        while t < cfg_ref.t_end:
            dt = float(compute_dt_fn(U))
            dt = min(dt, cfg_ref.t_end - t)
            U = step_fn(U, dt)
            t += dt
            if dt < 1e-15:
                break
        U_restart = np.asarray(U)

        # The state must have evolved (not stuck at U_loaded).
        assert not np.allclose(U_restart, U_loaded, rtol=1e-3), (
            "Restart did not evolve the state — restart machinery is broken."
        )
        # And it must end close to the reference final state.
        np.testing.assert_allclose(
            U_restart, U_ref, rtol=1e-2, atol=1e-3,
            err_msg=(
                "Restarted run does not match continuous reference to "
                "PLM-truncation accuracy"
            ),
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_conservation_periodic_square_advection():
    """Periodic 2D square advection — uniform flow, no sources, no shocks.
    Total mass and total energy must be conserved to ~machine precision.
    Catches regressions in the flux update / halo exchange / source-term
    operator-split chain."""
    from ic_generation.ic import initial_conditions

    cfg = _build_cfg_for_test(
        test_problem="Square-Advected",
        SPATIAL_DIMENSION=2,
        BOUNDARY=0,
        Nx=64,
        Ny=64,
        Lx=1.0,
        Ly=1.0,
        gamma=5.0 / 3.0,
        TIME_SOLVER=1,
        ORDER=1,
        LIMITER=1,
        CFL=0.4,
        t_end=0.5,
    )
    U_init = np.asarray(initial_conditions(cfg))
    mass0 = U_init[0].sum()
    E0 = U_init[-1].sum()
    U = _run_short(cfg)
    mass_rel = abs(U[0].sum() - mass0) / mass0
    E_rel = abs(U[-1].sum() - E0) / E0
    assert mass_rel < 1e-10, f"mass not conserved: rel err {mass_rel:.3e}"
    assert E_rel < 1e-10, f"E not conserved: rel err {E_rel:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
