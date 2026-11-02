"""DM-only restart, part A (blocks loop through main.run): the growth test's
growing mode with a checkpoint every 10 steps; keep the checkpoints and the
final sorted particle x for part B (its own process). GPU test."""
import glob
import os
import shutil
import sys

import numpy as np
import pytest

import jax


sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.slow

SCRATCH = os.environ.get("LUNA_TEST_SCRATCH", "snapshots/test_music_growing_mode")
CK = os.path.join(SCRATCH, "dm_restart_ck")


def dm_run(out, ic, **kw):
    from dataclasses import replace

    from luna import main
    from luna.template_config import Config
    from test_cosmo_dm_growth import A_END, NP
    cfg = replace(Config(), run="dm", CosmologicalIntegration=True, music_ic_path=ic,
                  Nx=NP, a_end=A_END, dloga_max=0.02, output_dir=out,
                  clear_output_dir=True, PLOT_AFTER_SIM=False,
                  decomposition="blocks", checkpoint_dir=CK, **kw)
    main.run(cfg)
    import h5py
    snap = sorted(glob.glob(os.path.join(out, "snap_a*.hdf5")))[-1]
    with h5py.File(snap, "r") as f:
        a_f = float(f["Header"].attrs["Time"])
        pos = np.asarray(f["PartType1"]["Coordinates"], np.float64)
    return a_f, np.sort(pos[:, 0])


def test_dm_blocks_run_with_checkpoints(tmp_path):
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    from test_cosmo_dm_growth import _growing_mode_ic
    shutil.rmtree(CK, ignore_errors=True)
    os.makedirs(CK, exist_ok=True)
    ic = os.path.join(CK, "ic.hdf5")
    _growing_mode_ic(ic)
    a_f, xs = dm_run(str(tmp_path / "out"), ic, cosmo_checkpoint_every=10, max_checkpoints=0)
    cks = sorted(f for f in os.listdir(CK) if f.startswith("cosmo_"))
    assert any(f.startswith("cosmo_000020.") for f in cks), cks
    np.save(os.path.join(CK, "xs_ref.npy"), xs)
    with open(os.path.join(CK, "ref.txt"), "w") as fh:
        fh.write(repr(a_f))
