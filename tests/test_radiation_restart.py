"""Can a radiative run be resumed?

It could not: `setup_radiation` refused `cfg.restart` outright, because a
checkpoint carried U and the OU driver state and nothing else, so a resumed
run would have started again from `rad_E0` without a word. For a benchmark
that is an annoyance; for a cloud collapsing across queue slots it decides
whether the run is finishable.

The test that matters is the round trip -- write the carry, read it back,
compare -- because every cheaper check passes on a file full of zeros.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from luna.file_io.radiation_checkpoint import (                # noqa: E402
    load_radiation_checkpoint, pack, unpack, write_radiation_checkpoint)

pytestmark = pytest.mark.fast

N = 8


def _cfg(tmp):
    from dataclasses import replace
    from luna.template_config import Config
    return replace(Config(), SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N,
                   Lx=1.0, Ly=1.0, Lz=1.0, BOUNDARY=0, enable_x64=True,
                   radiation_enabled=True, rad_groups=2, rad_chat=1.0,
                   rad_E0=1.0, checkpoint_dir=str(tmp))


def _mesh():
    from luna.utils.distributed import create_device_mesh
    m, _ = create_device_mesh(jax.device_count(), 3)
    return m


def _carry(ionization, seed=0, moments=4):
    rng = np.random.default_rng(seed)
    R = jnp.asarray(rng.normal(size=(2, moments, N, N, N)))
    if not ionization:
        return R
    x = jnp.asarray(rng.uniform(size=(N, N, N)))
    w = jnp.asarray(rng.uniform(size=(N, N, N)))
    return (R, x, w)


@pytest.mark.parametrize("moments", [4, 10])
def test_pack_unpack_is_the_identity(moments):
    """Flattening to U's rank is the whole trick; if it is not reversible the
    restart is wrong in a way no file check would catch. M1 (4) and M2 (10)."""
    for ion in (False, True):
        c = _carry(ion, moments=moments)
        flat, groups, got_ion, got_m = pack(c)
        assert got_ion is ion and groups == 2 and got_m == moments
        back = unpack(flat, groups, got_ion, got_m)
        if ion:
            for a, b in zip(c, back):
                assert float(jnp.max(jnp.abs(a - b))) == 0.0
        else:
            assert float(jnp.max(jnp.abs(c - back))) == 0.0


@pytest.mark.parametrize("moments", [4, 10])
@pytest.mark.parametrize("ionization", [False, True])
def test_the_field_survives_the_round_trip(tmp_path, ionization, moments):
    """Write it, read it back, compare every element. Exactly, not nearly:
    it is the same float64 going to disk and coming back. The moment count
    (M1 4, M2 10) comes back from the file, not from the config."""
    cfg = _cfg(tmp_path)
    mesh = _mesh()
    carry = _carry(ionization, seed=1, moments=moments)

    sharding = NamedSharding(mesh, P(None, "x", "y", "z"))
    flat, _, _, _ = pack(carry)
    flat = jax.device_put(flat, sharding)
    packed = unpack(flat, 2, ionization, moments)

    path = write_radiation_checkpoint(packed, cfg, 0.25, 7, str(tmp_path), mesh)
    assert path is not None and os.path.exists(path)

    back = load_radiation_checkpoint(str(tmp_path), 7, mesh, 3)
    assert back is not None
    pairs = zip(packed, back) if ionization else [(packed, back)]
    for a, b in pairs:
        assert float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b)))) == 0.0


def test_a_missing_set_is_reported_not_guessed(tmp_path):
    """No file for that step must return None, so the caller can raise. The
    failure to avoid is resuming from rad_E0 while looking healthy."""
    assert load_radiation_checkpoint(str(tmp_path), 999, _mesh(), 3) is None


def test_the_config_allows_a_radiative_restart(tmp_path):
    """The rule that used to forbid this is gone; the check now lives in
    setup_radiation, which can tell a present field from an absent one."""
    from dataclasses import replace
    replace(_cfg(tmp_path), restart=True).validate()


def test_setup_refuses_a_restart_with_no_field(tmp_path):
    """The teeth: restarting radiation with no stored field must STOP, not
    quietly begin again from rad_E0. This is the failure the old blanket
    refusal existed to prevent, and it still cannot happen."""
    from dataclasses import replace
    from luna.sources.radiation import setup_radiation
    cfg = replace(_cfg(tmp_path), restart=True)
    sharding = NamedSharding(_mesh(), P(None, "x", "y", "z"))
    with pytest.raises(FileNotFoundError, match="radiation file set"):
        setup_radiation(cfg, sharding, is_main=False)
