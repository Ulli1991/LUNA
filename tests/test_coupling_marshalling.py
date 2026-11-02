"""Does what the interface DECLARES match what runs in the worker?

AMUSE calls an implementation with keyword arguments taken from the
`@legacy_function` declaration, so a method whose parameters are spelled
differently from its declaration is handed a keyword it does not have. Inside
an AMUSE worker that surfaces as a UserWarning -- "mismatch in python function
specification(?)" -- and the call quietly does nothing.

Nothing that calls `LunaWorker` directly can see this, which is exactly how it
got in: thirty-two generated methods, every one of them wrong across the
boundary and right in every unit test.

Needs amuse, not a running worker: the specifications are built at import.
"""
import inspect

import pytest

amuse_rfi = pytest.importorskip("amuse.rfi.core")

from amuse_luna.interface import LunaImplementation, LunaInterface  # noqa: E402


# AMUSE answers these itself inside the worker loop; they are declared on the
# interface but are not the code's to implement.
FRAMEWORK = ("_stop_worker", "get_working_directory", "set_working_directory")


def _declared():
    """Every legacy function on the interface, with its parameter names."""
    for name in dir(LunaInterface):
        if name in FRAMEWORK or name.startswith("internal__"):
            continue
        obj = getattr(LunaInterface, name, None)
        if isinstance(obj, amuse_rfi.legacy_function):
            yield name, [p.name for p in obj.specification.parameters]


def test_every_declaration_has_an_implementation():
    missing = [n for n, _ in _declared() if not hasattr(LunaImplementation, n)]
    assert missing == [], "declared but not implemented: {}".format(missing)


def test_every_implementation_takes_the_parameters_it_declares():
    wrong = []
    for name, declared in _declared():
        impl = getattr(LunaImplementation, name, None)
        if impl is None:
            continue
        taken = list(inspect.signature(impl).parameters)[1:]      # drop self
        if taken != declared:
            wrong.append((name, declared, taken))
    assert wrong == [], "\n".join(
        "{}: declares {} but takes {}".format(*w) for w in wrong)
