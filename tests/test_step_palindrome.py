"""One step is a palindrome around the hydro update by construction.

Every operator-split stage's `before` runs in list order, the hydro, then
every `after` in REVERSED order, and the projections (not dt-scaled) run once
at the end. The runner used to spell the mirror out by hand, and its cheap
sources were applied in the same order on both sides; here the order is
structural, so adding a stage cannot break it. Pure Python: no JAX.
"""


from luna.setup.evolve import step                      # noqa: E402
from luna.setup.state import State                      # noqa: E402


class _Log:
    def __init__(self):
        self.calls = []


class _Stage:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def before(self, state, h, dt):
        self.log.calls.append((self.name, "before", h, dt))
        return state.update(U=state.U + [self.name + "<"])

    def after(self, state, h, dt):
        self.log.calls.append((self.name, "after", h, dt))
        return state.update(U=state.U + [self.name + ">"])


class _Hydro:
    def __init__(self, log):
        self.log = log

    def step(self, U, dt):
        self.log.calls.append(("hydro", "step", dt, dt))
        return U + ["hydro"]


class _Clock:
    def scale(self, t):
        return 1.0


class _Physics:
    def __init__(self, stages, hydro, projections):
        self.stages, self.hydro, self.projections, self.clock = stages, hydro, projections, _Clock()


def test_before_in_order_after_reversed_projections_last():
    log = _Log()
    a, b, c = (_Stage(n, log) for n in "abc")
    seen = []
    phys = _Physics([a, b, c], _Hydro(log),
                    [lambda U, a_: seen.append("p1") or U + ["p1"],
                     lambda U, a_: seen.append("p2") or U + ["p2"]])
    out = step(State(U=[], t=0.0, step=3), 0.5, 0.5, phys)
    assert out.U == ["a<", "b<", "c<", "hydro", "c>", "b>", "a>", "p1", "p2"]
    assert [(n, w) for n, w, _, _ in log.calls] == [
        ("a", "before"), ("b", "before"), ("c", "before"), ("hydro", "step"),
        ("c", "after"), ("b", "after"), ("a", "after")]
    assert all(h == 0.25 and dt == 0.5 for n, w, h, dt in log.calls if n != "hydro")
    assert out.step == 4 and out.t == 0.5


def test_a_run_without_gas_still_mirrors():
    log = _Log()
    a, b = (_Stage(n, log) for n in "ab")
    out = step(State(U=[], t=1.0), 0.2, 1.2, _Physics([a, b], None, []))
    assert out.U == ["a<", "b<", "b>", "a>"]
