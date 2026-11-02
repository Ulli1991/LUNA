"""DM-only restart, part B (blocks loop through main.run): restart from part
A's step-20 checkpoint; the final sorted particle x must match part A's
uninterrupted run (order-free: the blocks loop reorders particles). The
warm Poisson carry is recomputed cold at the restart, so agreement is to
the FMG floor. GPU test; run test_cosmo_dm_restart_a.py first."""
import os
import sys

import numpy as np
import pytest

import jax


sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.slow

from test_cosmo_dm_restart_a import CK, dm_run


def test_dm_blocks_restart_matches(tmp_path):
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    from test_cosmo_dm_growth import BOX, NP
    ck20 = os.path.join(CK, "cosmo_000020.0.hdf5")
    ref = os.path.join(CK, "xs_ref.npy")
    if not (os.path.exists(ck20) and os.path.exists(ref)):
        pytest.skip("run test_cosmo_dm_restart_a.py first")
    a_f, xs = dm_run(str(tmp_path / "out"), os.path.join(CK, "ic.hdf5"),
                     restart=True, restart_file=ck20, cosmo_checkpoint_every=0)
    a_ref = float(open(os.path.join(CK, "ref.txt")).read())
    xs_ref = np.load(ref)
    assert abs(a_f - a_ref) < 1e-12
    assert xs.shape == xs_ref.shape
    dx = BOX / NP
    err = np.max(np.abs(xs - xs_ref)) / dx
    print(f"[dm restart] max |dx_sorted| / cell = {err:.3e}")
    assert err < 1e-3, f"restarted DM positions differ by {err:.3e} cells"
