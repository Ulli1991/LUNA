"""Cosmological restart, part A (coupled blocks loop): run the MUSIC growing
mode to a_end writing a checkpoint every 20 steps; keep the checkpoints and
the final gas density / DM x-profile for part B, which restarts from the
step-40 checkpoint in its own process (one coupled loop per process on ROCm).
GPU test, >=2 devices."""
import os
import shutil
import sys

import numpy as np
import pytest

import jax


sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.slow

SCRATCH = os.environ.get("LUNA_TEST_SCRATCH", "snapshots/test_music_growing_mode")
CK = os.path.join(SCRATCH, "restart_ck")


def dm_x_profile(pos, box, n):
    f = np.mod(pos[:, 0], box) / (box / n) - 0.5
    i0 = np.floor(f).astype(int)
    w1 = f - i0
    nx = np.zeros(n)
    np.add.at(nx, np.mod(i0, n), 1.0 - w1)
    np.add.at(nx, np.mod(i0 + 1, n), w1)
    return nx


def test_coupled_blocks_run_with_checkpoints(tmp_path):
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    from _music_growing_mode import BOX, NP, growing_mode_ic, run_loop
    shutil.rmtree(CK, ignore_errors=True)
    os.makedirs(CK, exist_ok=True)
    ic = os.path.join(CK, "ic.hdf5")
    growing_mode_ic(ic)
    U, pos, a_f, step = run_loop(ic, str(tmp_path / "out"),
                                 cosmo_checkpoint_every=20, checkpoint_dir=CK,
                                 max_checkpoints=0)
    cks = sorted(f for f in os.listdir(CK) if f.startswith("cosmo_"))
    assert any(f.startswith("cosmo_000040.") for f in cks), cks
    np.save(os.path.join(CK, "rho_ref.npy"), U[0])
    np.save(os.path.join(CK, "dmx_ref.npy"), dm_x_profile(pos, BOX, NP))
    with open(os.path.join(CK, "ref.txt"), "w") as fh:
        fh.write(f"{a_f!r} {step}\n")
