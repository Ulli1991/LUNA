"""Coupled gas+DM MUSIC run through the BLOCK loop on a layout that is not an
x-split, against the same run on an x-only split. On (n,1,1) the particle-
buffer spec P(('x','y','z')) and the grid spec P('x','y','z') coincide, so a
mix-up between them passes every x-split test and only shows on a real 3D
layout (the 128^3 (2,2,1) A/B produced a (64,256,128) grid, 2026-09-18).

The layout comes from the device count: (1,2,1) on 2 devices, (1,2,4) on 8.
The x-only reference runs in this file, so the comparison never depends on
another file having run. Needs >=2 devices (one device has no split).
"""
import os
import sys

import numpy as np
import pytest

import jax


sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.slow


def split_layouts(n):
    """(layout under test with x unsplit and y split, x-only reference) on n devices."""
    return ((1, 2, n // 2) if n % 2 == 0 else (1, n, 1)), (n, 1, 1)


def test_split_layouts():
    assert split_layouts(2) == ((1, 2, 1), (2, 1, 1))
    assert split_layouts(8) == ((1, 2, 4), (8, 1, 1))
    assert split_layouts(3) == ((1, 3, 1), (3, 1, 1))


def test_music_gas_dm_growing_mode_blocks_y_split(tmp_path):
    n = jax.device_count()
    if n < 2:
        pytest.skip("one device has no y-split layout")
    from _music_growing_mode import BOX, NP, check_growing_mode, growing_mode_ic, run_loop
    split, ref = split_layouts(n)
    # growth gate (gas + DM), particle IDs and the final snapshot on the y-split
    U, pos = check_growing_mode(tmp_path, loop="blocks_y", dm_pdims=split)
    assert U.shape == (5, NP, NP, NP)                       # the grid keeps its shape
    # the same run on the x-only split must agree pointwise
    ic = str(tmp_path / "ic_ref.hdf5")
    growing_mode_ic(ic)
    U_ref, pos_ref, _, _ = run_loop(ic, str(tmp_path / "out_ref"), dm_pdims=ref)
    rho, rho_ref = U[0], U_ref[0]
    l1 = np.mean(np.abs(rho - rho_ref)) / np.mean(rho_ref)
    d = (pos - pos_ref + 0.5 * BOX) % BOX - 0.5 * BOX       # same particle, by ID
    dpos = np.max(np.abs(d)) / (BOX / NP)
    print(f"[blocks {split} vs {ref}] gas density L1/mean = {l1:.3e}, "
          f"DM max |dx| = {dpos:.3e} cells")
    assert l1 < 1e-6, f"decomposition-dependent gas: {l1:.3e}"
    # measured 2.8e-7 cells on CPU-8 (the solves converge per layout); a layout
    # bug misplaces particles by a good fraction of a cell
    assert dpos < 1e-5, f"decomposition-dependent DM positions: {dpos:.3e} cells"
