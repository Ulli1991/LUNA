"""Coupled gas+DM Zel'dovich pancake — the gate for the coupled runner
(setup/runner_cosmo_hydro.py), which had no tests at all.

Both species follow the same analytic Zel'dovich map pre-caustic, so one run
checks the shared-potential coupling end to end: gas density profile, DM
positions, and mass conservation for both species. The DM metric is ORDER-FREE
(sorted x within the 1D flow): the sharded paths' exchange/compaction reorders
particles, so the per-particle Lagrangian metric of run_pancake_coupled.py
only works single-device.

Gas runs with LUNA_HYDRO_FUSED=1 (the seam-conservative kernel) so gas mass is
a tight gate rather than tracking the known width-1 leak. Conservation is
measured end-vs-initial state (the IC's mean rho~ is 1 only to O(dx^2)).

Two variants: the dense (replicated-gravity) loop, and the grid-sharded slab
loop on >=2 devices. When the blocks coupled path lands it gains a third
variant with the SAME gates. GPU test; run via pytest on apudev.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("LUNA_HYDRO_FUSED", "1")   # before solver construction

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.getcwd())

pytestmark = pytest.mark.slow

from scripts.cosmo.run_pancake import analytic, build_cfg
from scripts.cosmo.run_pancake_coupled import dm_zeldovich_ic

# Gates: the 64^3 thresholds run_pancake_coupled.py has used since the coupled
# runner landed (gas rho~ L1 vs analytic; DM sorted-x L1 vs the Zel'dovich
# displacement amplitude). Mass: fused gas kernel conserves to ~1e-12 (measured
# on the pancake ledger, 2026-07-28); DM mass is a sum of an unchanging array.
L1_GAS = 1.0e-2
L1_DM = 2.0e-2
GAS_REL = 1.0e-10


def _cfg():
    from dataclasses import replace
    return replace(build_cfg(), output_dir="snapshots/test_pancake_coupled",
                   production_mode=True, PLOT_AFTER_SIM=False)


def _setup(cfg, mesh):
    """U0 + solver on `mesh` — the same construction main.run's dense path and
    the music slab branch use, kept here so the tests hold on to U0."""
    from setup import (create_solver_distributed,
                       initialize_simulation_distributed)
    from utils.grid_params import GridParams
    U0, _, _ = initialize_simulation_distributed(cfg, mesh)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, _ = create_solver_distributed(cfg, grid, mesh)
    return U0, compute_dt_fn, step_fn


def _gas_mass(U, cfg):
    return float(jnp.sum(U[0])) * (cfg.Lx / cfg.Nx) ** 3


def _gas_l1(U, a, cfg):
    rho_gas = np.asarray(U)[0, :, 0, 0]
    _, rho_ana, _ = analytic(a, cfg)
    return float(np.mean(np.abs(rho_gas - rho_ana)) / np.mean(rho_ana))


def _dm_sorted_x_l1(pos, a, cfg, n_expect):
    """Order-free DM metric: the flow is 1D and every (y,z) column carries the
    same analytic x-multiset, so compare the sorted x's of ALL particles to the
    sorted analytic map (each q replicated Ny*Nz times)."""
    Nx, box = cfg.Nx, cfg.Lx
    dx = box / Nx
    k = 2 * np.pi / box
    A_end = a / cfg.zel_a_caustic
    q = (np.arange(Nx) + 0.5) * dx
    x_ana = np.mod(q - A_end * np.sin(k * q) / k, box)
    x_ana = np.sort(np.tile(x_ana, Nx * Nx))
    xs = np.sort(pos[:, 0])
    assert xs.size == n_expect
    ampl = A_end / k
    return float(np.mean(np.abs(xs - x_ana)) / ampl)


def test_coupled_pancake_dense():
    """Dense loop: replicated PM gravity + gas grid."""
    from setup import create_distributed_mesh
    from setup.runner_cosmo_hydro import run_simulation_cosmo_hydro

    cfg = _cfg()
    dm_pos, dm_p, dm_mass = dm_zeldovich_ic(cfg)
    mesh, _ = create_distributed_mesh(cfg)
    U0, compute_dt_fn, step_fn = _setup(cfg, mesh)
    m_gas0 = _gas_mass(U0, cfg)
    U, pos, p, a, step = run_simulation_cosmo_hydro(
        U0, cfg, step_fn, compute_dt_fn,
        jnp.asarray(dm_pos), jnp.asarray(dm_p), jnp.asarray(dm_mass))

    assert abs(_gas_mass(U, cfg) - m_gas0) < GAS_REL * m_gas0
    assert _gas_l1(U, a, cfg) < L1_GAS
    assert _dm_sorted_x_l1(np.asarray(pos), a, cfg, dm_pos.shape[0]) < L1_DM
    assert step > 50                                   # actually integrated


def test_coupled_pancake_blocks():
    """Blocks coupled loop (_run_blocks_hydro): gas + DM on ONE 3D block mesh,
    warm unscaled-psi solve. pdims (2,1,1) = the same decomposition as the
    slab variant, so the gates are directly comparable."""
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    from gravity.particle_mesh_blocks import make_block_mesh
    from setup.runner_cosmo_hydro import _run_blocks_hydro

    cfg = _cfg()
    dm_pos, dm_p, dm_mass = dm_zeldovich_ic(cfg)
    mesh = make_block_mesh(jax.devices()[:2], (2, 1, 1))
    U0, compute_dt_fn, step_fn = _setup(cfg, mesh)
    m_gas0 = _gas_mass(U0, cfg)
    U, posS, pS, a, step = _run_blocks_hydro(
        U0, cfg, step_fn, compute_dt_fn, mesh, dm_pos, dm_p, dm_mass)

    assert abs(_gas_mass(U, cfg) - m_gas0) < GAS_REL * m_gas0
    assert _gas_l1(U, a, cfg) < L1_GAS
    posS = np.asarray(posS)
    act = np.any(posS != 0.0, axis=1)
    assert int(act.sum()) == dm_pos.shape[0]           # count conserved
    assert _dm_sorted_x_l1(posS[act], a, cfg, dm_pos.shape[0]) < L1_DM
    assert step > 50


def test_coupled_pancake_slab():
    """Grid-sharded slab loop (_run_slab_hydro): joint solve on the sharded
    grid, DM in per-slab buffers. Pins the current multi-GPU coupled path so
    the blocks path must reproduce it."""
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    from gravity.particle_mesh_shardmap import make_slab_mesh
    from setup.runner_cosmo_hydro import _run_slab_hydro

    cfg = _cfg()
    dm_pos, dm_p, dm_mass = dm_zeldovich_ic(cfg)
    mesh = make_slab_mesh(jax.devices())
    U0, compute_dt_fn, step_fn = _setup(cfg, mesh)
    m_gas0 = _gas_mass(U0, cfg)
    U, posS, pS, a, step = _run_slab_hydro(
        U0, cfg, step_fn, compute_dt_fn, mesh, dm_pos, dm_p, dm_mass)

    assert abs(_gas_mass(U, cfg) - m_gas0) < GAS_REL * m_gas0
    assert _gas_l1(U, a, cfg) < L1_GAS
    # padded slab buffers: active rows are the not-exactly-zero ones (real
    # particles never land on (0,0,0) exactly; padding is exactly zero)
    posS = np.asarray(posS)
    act = np.any(posS != 0.0, axis=1)
    assert int(act.sum()) == dm_pos.shape[0]           # count conserved
    assert _dm_sorted_x_l1(posS[act], a, cfg, dm_pos.shape[0]) < L1_DM
    assert step > 50
