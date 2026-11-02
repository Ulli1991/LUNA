"""exact_block_capacity sizes a one-time assembly buffer by counting, not guessing.

dm_capacity_factor exists because DM particles drift out of their own block
between steps and are lost if the buffer cannot hold the arrivals. Gas has no
such growth: it is meshed onto the grid once at IC time and then lives on the
meshblock, so its buffer is exactly countable. The heuristic 1.3 it replaced
missed the real l9 IC on a (2,2,2) mesh by 9151 of 134M particles and killed
the run (job 11815523, 2026-09-18).
"""

import numpy as np
import pytest

import jax



from luna.gravity.particle_mesh_blocks import exact_block_capacity  # noqa: E402

N, BOX = 32, 1.0
DX = BOX / N


def _owner_counts(pos, pdims):
    """Reference: the same block index decompose_to_blocks uses."""
    ndev = int(np.prod(pdims))
    bw = [(N // pdims[a]) * DX for a in range(3)]
    b = [np.clip((np.mod(pos[:, a], BOX) / bw[a]).astype(np.int64), 0, pdims[a] - 1)
         for a in range(3)]
    owner = (b[0] * pdims[1] + b[1]) * pdims[2] + b[2]
    return np.bincount(owner, minlength=ndev)


@pytest.mark.parametrize("pdims", [(1, 1, 1), (2, 1, 1), (2, 2, 1), (2, 2, 2)])
def test_capacity_equals_the_fullest_block(pdims):
    if jax.process_count() != 1:
        pytest.skip("single-process reference")
    rng = np.random.default_rng(7)
    pos = rng.random((5000, 3)) * BOX
    C = exact_block_capacity(pos, N, pdims, DX, BOX)
    counts = _owner_counts(pos, pdims)
    # one process holds every row, so its own pre-exchange occupancy bounds C
    assert C == max(int(counts.max()), pos.shape[0])


def test_capacity_covers_a_clustered_block():
    """A buffer sized from a uniform-IC guess is what the l9 run overflowed."""
    if jax.process_count() != 1:
        pytest.skip("single-process reference")
    rng = np.random.default_rng(11)
    pdims = (2, 2, 2)
    uniform = rng.random((8000, 3)) * BOX
    clump = np.clip(rng.normal(0.1 * BOX, 0.02 * BOX, (8000, 3)), 0, BOX - 1e-9)
    pos = np.concatenate([uniform, clump])
    counts = _owner_counts(pos, pdims)
    assert counts.max() > 1.5 * pos.shape[0] / 8, "test IC is not clustered enough"
    assert exact_block_capacity(pos, N, pdims, DX, BOX) >= int(counts.max())
