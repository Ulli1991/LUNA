"""The M1 substep count follows the step it is asked to take.

`rad_substeps` used to be the count every half step ran, however short the
step: it is what dt_max is sized for, and a run whose hydro step sits well
below dt_max (the 256^3 radiation cloud: 3-5x below, 16 substeps, 97% of
its step in M1) paid for all of them. Now a half of `h` runs
ceil(h / (cfl dx / (ndim c_hat))) substeps, capped at rad_substeps: the same
CFL constraint, and the same program whenever every substep is needed.
"""
import numpy as np



from tests.test_radiation_coupled import _gas, _mesh, _rad, _rad_cfg  # noqa: E402

N = 16


def _stage(n_sub):
    from jax.sharding import NamedSharding, PartitionSpec as P
    from luna.sources.radiation import setup_radiation
    one = NamedSharding(_mesh((1, 1, 1)), P(None, "x", "y", "z"))
    from dataclasses import replace
    return setup_radiation(replace(_rad_cfg(N), rad_substeps=n_sub), one, is_main=False)


def test_the_count_is_the_cfl_need_capped_at_the_configured_one():
    rad = _stage(8)
    half_max = 0.5 * rad["dt_max"]                 # a step at dt_max: all 8
    assert rad["count"](half_max) == 8
    assert rad["count"](half_max / 4) == 2
    assert rad["count"](half_max / 8 * 1.001) == 2  # just over one substep's worth
    assert rad["count"](1e-12) == 1
    assert rad["count"](10.0 * half_max) == 8      # never above the cap


def test_a_short_step_runs_the_program_of_its_own_count():
    """At a step needing 2 substeps, a stage capped at 8 gives the 2-substep
    program's answer, bitwise (the teeth check forces the old always-the-cap
    count on this test and it goes red)."""
    capped, two = _stage(8), _stage(2)
    h = 0.5 * capped["dt_max"] / 4                 # needs 2 of the 8
    assert capped["count"](h) == 2
    U, R = _gas(N), _rad(N)
    Uc, Rc = capped["before"](U, R, h)
    U2, R2 = two["before"](U, R, h)
    np.testing.assert_array_equal(np.asarray(Uc), np.asarray(U2))
    np.testing.assert_array_equal(np.asarray(Rc), np.asarray(R2))


def test_a_step_at_dt_max_still_runs_every_substep():
    capped, fixed = _stage(4), _stage(4)
    h = 0.5 * capped["dt_max"]
    assert capped["count"](h) == 4
    U, R = _gas(N), _rad(N)
    a = capped["after"](U, R, h)
    b = fixed["after"](U, R, h)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_dispatching_a_substep_ahead_changes_nothing(monkeypatch):
    """The next substep is queued before the last one ends up to LAG_MAX_CELLS: scheduling, not arithmetic."""
    import luna.sources.radiation as radmod
    U, R = _gas(N), _rad(N)
    out = []
    for cells in (0, radmod.LAG_MAX_CELLS):
        monkeypatch.setattr(radmod, "LAG_MAX_CELLS", cells)
        rad = _stage(4)
        h = 0.5 * rad["dt_max"]
        out.append(rad["after"](*rad["before"](U, R, h), h))
    for x, y in zip(*out):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
