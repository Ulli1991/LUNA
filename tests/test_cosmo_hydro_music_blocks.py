"""Coupled gas+DM MUSIC run through the 3D block loop, gated against linear
growth (tests/_music_growing_mode.py). GPU test, >=2 devices.

This was the gate under which the block loop replaced the grid-sharded slab
twin, now deleted: measured 2026-09-17 (2 MI300A, rocm 7.2 / jax 0.10.0 and
rocm 7.0 / jax 0.6) blocks gas 0.987 / DM 0.984 x linear against slab 0.944 /
0.942, the two loops agreeing to 2.8e-3 (1.0e-4 at 128^3, audit sec.16).
"""
import os
import sys

import pytest

import jax


sys.path.insert(0, os.path.dirname(__file__))

pytestmark = pytest.mark.slow


def test_music_gas_dm_growing_mode_blocks(tmp_path):
    if jax.device_count() < 2:
        pytest.skip("needs >=2 devices")
    from _music_growing_mode import check_growing_mode
    check_growing_mode(tmp_path)
