"""Cosmological restart, part B (coupled blocks loop): restart from part A's
step-40 checkpoint and run to a_end; the final state must match part A's
uninterrupted run. The warm Poisson carry is not checkpointed (a restart
starts from a cold FMG solve), so agreement is to the FMG floor, not
bitwise. GPU test, >=2 devices; run test_cosmo_restart_a.py first."""
import os
import sys

import numpy as np
import pytest

import jax


sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.slow

from test_cosmo_restart_a import CK, dm_x_profile

TOL = 1.0e-4                      # L1/mean of the gas density and of the DM x-profile


def test_coupled_blocks_restart_matches(tmp_path):
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    ref = os.path.join(CK, "rho_ref.npy")
    ck40 = os.path.join(CK, "cosmo_000040.0.hdf5")
    if not (os.path.exists(ref) and os.path.exists(ck40)):
        pytest.skip("run test_cosmo_restart_a.py first")
    from _music_growing_mode import BOX, NP, run_loop
    a_ref, step_ref = open(os.path.join(CK, "ref.txt")).read().split()
    U, pos, a_f, step = run_loop(os.path.join(CK, "ic.hdf5"), str(tmp_path / "out"),
                                 restart=True, restart_file=ck40, checkpoint_dir=CK,
                                 cosmo_checkpoint_every=0)
    assert step == int(step_ref), (step, step_ref)            # continued, not restarted
    assert abs(a_f - float(a_ref)) < 1e-12
    rho_ref = np.load(ref)
    l1 = np.mean(np.abs(U[0] - rho_ref)) / np.mean(rho_ref)
    dmx = dm_x_profile(pos, BOX, NP)
    dmx_ref = np.load(os.path.join(CK, "dmx_ref.npy"))
    l1_dm = np.mean(np.abs(dmx - dmx_ref)) / np.mean(dmx_ref)
    print(f"[restart] gas L1/mean={l1:.3e}  DM x-profile L1/mean={l1_dm:.3e}")
    assert pos.shape[0] == NP ** 3
    assert l1 < TOL, f"restarted gas density differs: L1/mean {l1:.3e}"
    assert l1_dm < TOL, f"restarted DM x-profile differs: L1/mean {l1_dm:.3e}"
