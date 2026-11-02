"""The run kind is decided in one place, and contradictions are refused.

Five booleans used to pick the run kind between them, resolved by whichever
dispatch branch in main.run sat first: dark matter + AMR silently ran dark matter
and ignored AMR. These cases pin what each combination means now.
"""

import pytest


from luna.setup.run_kind import COUPLED, DM, GAS, check_modifiers, run_kind  # noqa: E402


def _cfg(**kw):
    from dataclasses import replace

    from luna.template_config import Config
    return replace(Config(), **kw)


KIND_CASES = [
    ("plain hydro", {}, GAS),
    ("gas, comoving", {"CosmologicalIntegration": True}, GAS),
    ("gas, refined", {"AMR": True}, GAS),
    ("gas, adjoint", {"autodiff_mode": True}, GAS),
    ("dark matter only", {"run": "dm"}, DM),
    ("coupled", {"run": "gas+dm"}, COUPLED),
    ("coupled, refined", {"run": "gas+dm", "AMR": True}, COUPLED),
    ("gas named", {"run": "gas"}, GAS),
]


@pytest.mark.parametrize("what,kw,expect", KIND_CASES, ids=[c[0] for c in KIND_CASES])
def test_kind(what, kw, expect):
    assert run_kind(_cfg(**kw)) == expect, what
    assert check_modifiers(_cfg(**kw)) == expect, what


REFUSED = [
    ("dm + AMR: ran dm, ignored AMR", {"run": "dm", "AMR": True}, "no dark-matter-only"),
    ("dm + adjoint: ran dm, ignored adjoint", {"run": "dm", "autodiff_mode": True}, "dense gas loop only"),
    ("coupled + adjoint", {"run": "gas+dm", "autodiff_mode": True}, "dense gas loop only"),
    ("AMR + adjoint: ran AMR", {"AMR": True, "autodiff_mode": True}, "no refinement"),
    ("comoving + adjoint: ran adjoint, DROPPED comoving",
     {"CosmologicalIntegration": True, "autodiff_mode": True}, "proper time only"),
    ("typo", {"run": "gas+DM"}, "expected one of"),
]


@pytest.mark.parametrize("what,kw,msg", REFUSED, ids=[c[0] for c in REFUSED])
def test_refused(what, kw, msg):
    with pytest.raises(ValueError, match=msg):
        check_modifiers(_cfg(**kw))


def test_validate_carries_the_refusal():
    """The config's own validate() surfaces it, so a run script fails at
    startup and not after allocating a node."""
    cfg = _cfg(run="dm", AMR=True, music_ic_path="ic.hdf5")
    with pytest.raises(ValueError, match="no dark-matter-only"):
        cfg.validate()
