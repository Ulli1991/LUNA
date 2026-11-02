"""AMD_SERIALIZE_COPY=3 is set exactly when the run is on ROCm and the variable is unset,
before the backend starts (luna/setup/jax_init.py); the platform check is mocked."""
import os

import pytest

from luna.setup import jax_init

KEY = "AMD_SERIALIZE_COPY"


@pytest.fixture
def world(monkeypatch):
    """set_(rocm, started) mocks the platform and backend checks; the variable starts unset."""
    monkeypatch.delenv(KEY, raising=False)
    from jax._src import xla_bridge

    def set_(rocm=True, started=False):
        monkeypatch.setattr(jax_init, "_on_rocm", lambda: rocm)
        monkeypatch.setattr(xla_bridge, "backends_are_initialized", lambda: started)
    return set_


def _run():
    lines = []
    got = jax_init.serialize_rocm_copies(lambda *a, **k: lines.append(" ".join(map(str, a))))
    return got, lines


def test_set_on_rocm(world):
    world()
    got, lines = _run()
    assert got and os.environ[KEY] == "3"
    assert len(lines) == 1 and KEY in lines[0]


def test_not_set_off_rocm(world):
    world(rocm=False)
    got, lines = _run()
    assert not got and KEY not in os.environ and not lines


@pytest.mark.parametrize("value", ["0", "", "2"])
def test_a_value_already_set_wins(world, monkeypatch, value):
    world()
    monkeypatch.setenv(KEY, value)
    got, lines = _run()
    assert not got and os.environ[KEY] == value and not lines


def test_too_late_after_the_backend_started(world):
    world(started=True)
    got, lines = _run()
    assert not got and KEY not in os.environ
    assert len(lines) == 1 and "already started" in lines[0]


def test_initialize_sets_it_first(world, monkeypatch):
    """initialize_jax_distributed calls it before anything else touches JAX."""
    calls = []
    monkeypatch.setattr(jax_init, "serialize_rocm_copies", lambda *a: calls.append("copy"))
    monkeypatch.setattr(jax_init.jax.distributed, "initialize",
                        lambda **k: calls.append("init") or (_ for _ in ()).throw(ValueError("x")))
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    monkeypatch.setattr(jax_init.jax.config, "update", lambda *a, **k: None)
    jax_init.initialize_jax_distributed()
    assert calls[:2] == ["copy", "init"]


@pytest.mark.parametrize("plats,plugin,want", [
    ("rocm", False, True),
    ("rocm,cpu", False, True),
    ("cpu", True, False),
    ("cuda", True, False),
    ("", True, True),                     # unset: the installed plugin decides
    ("", False, False),
    ("gpu", True, True),
])
def test_rocm_detection_needs_no_backend(monkeypatch, plats, plugin, want):
    import importlib.metadata as md

    class Dist:
        def __init__(self, name):
            self.metadata = {"Name": name}
    names = ["numpy", "jax"] + (["jax-rocm7-pjrt"] if plugin else [])
    monkeypatch.setattr(md, "distributions", lambda: [Dist(n) for n in names])
    monkeypatch.setenv("JAX_PLATFORMS", plats)
    assert jax_init._on_rocm() is want
