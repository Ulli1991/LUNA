"""The layout choice lives in one function.

A run on one device holds the whole grid (`dense`); any particle run on more
than one device is the block mesh (`blocks`); AMR has its own; naming
`decomposition="blocks"` puts the block layout on one device. A change to any of
these fails here rather than silently changing how a production run is laid
out.
"""

import pytest


from luna.setup.mesh import AMR_BLOCKS, BLOCKS, DENSE, block_dims, decomposition  # noqa: E402


def _cfg(**kw):
    from dataclasses import replace

    from luna.template_config import Config
    return replace(Config(), **kw)


CASES = [
    # (what it is, cfg kwargs, devices, processes, expected)
    ("plain hydro, one device", {}, 1, 1, DENSE),
    ("plain hydro, many devices", {}, 8, 1, DENSE),
    ("AMR wins over everything", {"AMR": True, "run": "gas+dm"}, 8, 1, AMR_BLOCKS),
    ("DM on one device", {"run": "dm", "Nx": 256}, 1, 1, DENSE),
    ("DM, several devices", {"run": "dm", "Nx": 256}, 8, 1, BLOCKS),
    ("DM, multi-process", {"run": "dm", "Nx": 256}, 8, 4, BLOCKS),
    ("DM, blocks named on one device",
     {"run": "dm", "Nx": 256, "decomposition": "blocks"}, 1, 1, BLOCKS),
    ("coupled, one device", {"run": "gas+dm"}, 1, 1, DENSE),
    ("coupled, several devices", {"run": "gas+dm"}, 2, 1, BLOCKS),
    ("coupled, blocks named on one device",
     {"run": "gas+dm", "decomposition": "blocks"}, 1, 1, BLOCKS),
]


@pytest.mark.parametrize("what,kw,nd,npr,expect",
                         CASES, ids=[c[0] for c in CASES])
def test_layout_matches_the_runners(what, kw, nd, npr, expect):
    got = decomposition(_cfg(**kw), n_devices=nd, n_processes=npr)
    assert got == expect, f"{what}: got {got}, expected {expect}"


def test_naming_one_overrides_the_derivation():
    """A run script can say the layout outright in one line."""
    cfg = _cfg(run="dm", Nx=256, decomposition=BLOCKS)
    assert decomposition(cfg, n_devices=1, n_processes=1) == BLOCKS


def test_a_typo_is_refused_not_guessed():
    with pytest.raises(ValueError, match="expected 'auto'"):
        decomposition(_cfg(decomposition="block"), n_devices=1, n_processes=1)


def test_dm_pdims_must_use_every_device():
    """A named block layout for another device count is refused by name, not
    left to a reshape error deep in the mesh build."""
    cfg = _cfg(run="gas+dm", dm_pdims=(1, 2, 1))
    assert block_dims(cfg, n_devices=2) == (1, 2, 1)
    with pytest.raises(ValueError, match=r"dm_pdims \(1, 2, 1\) needs 2 devices, found 8"):
        block_dims(cfg, n_devices=8)
    with pytest.raises(ValueError, match="needs 8 devices, found 1"):
        block_dims(_cfg(dm_pdims=(2, 2, 2)), n_devices=1)
    assert block_dims(_cfg(), n_devices=8) == (2, 2, 2)
