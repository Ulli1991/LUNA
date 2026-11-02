"""Coupled gas+DM Zel'dovich pancake: the gate for the coupled run
(assemble + evolve with a dark-matter species), on the two gravity backends.

Both species follow the same analytic Zel'dovich map pre-caustic, so one run
checks the shared-potential coupling end to end: gas density profile, DM
positions, and mass conservation for both species. The DM metric is ORDER-FREE
(sorted x within the 1D flow): the block exchange reorders particles.

Gas runs with hydro_fused=True (the seam-conservative kernel) so gas mass is
a tight gate. Conservation is measured end-vs-initial state (the IC's mean
rho~ is 1 only to O(dx^2)).

Two variants with the SAME gates: the local backend (one device, whole grid)
and the sharded backend on a (2,1,1) block mesh. GPU test; run on apudev.
"""

import numpy as np
import pytest

import jax

import jax.numpy as jnp


pytestmark = pytest.mark.slow

from scripts.cosmo.run_pancake import analytic, build_cfg
from scripts.cosmo.run_pancake_coupled import dm_zeldovich_ic, run_coupled

# Gates: the 64^3 thresholds run_pancake_coupled.py has used since the coupled
# runner landed (gas rho~ L1 vs analytic; DM sorted-x L1 vs the Zel'dovich
# displacement amplitude). Mass: fused gas kernel conserves to ~1e-12 (measured
# on the pancake ledger, 2026-07-28); DM mass is a sum of an unchanging array.
L1_GAS = 1.0e-2
L1_DM = 2.0e-2
GAS_REL = 1.0e-10


def _cfg(**kw):
    from dataclasses import replace
    # MC's Hancock faces at the accretion shocks either side of the pancake (cold infall onto it, from a ~ 0.95,
    # step 115) go to p <= 0 (audit C8): first order there, counted and checked in _check_faces
    return replace(build_cfg(), output_dir="snapshots/test_pancake_coupled",
                   PLOT_AFTER_SIM=False, allow_first_order_faces=True,
                   hydro_fused=True, **kw)   # seam-conservative kernel: GAS_REL gate


def _check_faces(text, cfg):
    """The first-order faces (C8) of a planar flow: x faces only, whole (y, z) planes, near the pancake at x = 0.

    The IC and the flow are uniform in y and z, so every face decision repeats over a plane: the count is a
    positive multiple of Ny Nz, and each listed cell lies within PLANES cells of x = 0 (periodic)."""
    import re
    total = re.findall(r"\[hydro\] run total: (\d+) first-order faces", text)
    assert total, "no first-order face: the opt-in is no longer needed"
    n = int(total[-1])
    print(f"FACES coupled pancake: {n} first-order faces")
    assert n > 0 and n % (cfg.Ny * cfg.Nz) == 0, n
    for where in re.findall(r"first-order faces \(C8\), (.*)$", text, re.M):
        assert not re.search(r"[yz][+-]", where), where
        for i in re.findall(r"\((\d+), \d+, \d+\)", where):
            assert min(int(i), cfg.Nx - 1 - int(i)) < PLANES, where


PLANES = 6      # the accretion shocks sit 2-3 cells from x = 0 at a = 0.95-1 (A100, 64^3; 2x margin)


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
    A_end = a / cfg.problem.a_caustic
    q = (np.arange(Nx) + 0.5) * dx
    x_ana = np.mod(q - A_end * np.sin(k * q) / k, box)
    x_ana = np.sort(np.tile(x_ana, Nx * Nx))
    xs = np.sort(pos[:, 0])
    assert xs.size == n_expect
    ampl = A_end / k
    return float(np.mean(np.abs(xs - x_ana)) / ampl)


def _check(cfg, U0, state, n_dm):
    dm = state.species["dm"]
    pos = np.asarray(dm.pos)
    act = np.asarray(dm.active) > 0
    assert abs(_gas_mass(state.U, cfg) - _gas_mass(U0, cfg)) < GAS_REL * _gas_mass(U0, cfg)
    assert _gas_l1(state.U, state.a, cfg) < L1_GAS
    assert int(act.sum()) == n_dm                       # count conserved
    assert _dm_sorted_x_l1(pos[act], state.a, cfg, n_dm) < L1_DM
    assert state.step > 50                              # actually integrated


def test_coupled_pancake_local(capsys):
    """One device, the whole grid: the local backend."""
    cfg = _cfg()
    dm_pos, dm_p, dm_mass = dm_zeldovich_ic(cfg)
    U0, state = run_coupled(cfg, dm_pos, dm_p, dm_mass, devices=jax.devices()[:1])
    _check_faces(capsys.readouterr().out, cfg)
    _check(cfg, U0, state, dm_pos.shape[0])


def test_coupled_pancake_blocks(capsys):
    """The sharded backend on a (2,1,1) block mesh: same gates."""
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    cfg = _cfg()
    dm_pos, dm_p, dm_mass = dm_zeldovich_ic(cfg)
    U0, state = run_coupled(cfg, dm_pos, dm_p, dm_mass, devices=jax.devices()[:2],
                            pdims=(2, 1, 1))
    _check_faces(capsys.readouterr().out, cfg)
    _check(cfg, U0, state, dm_pos.shape[0])
