"""Shared pytest configuration for the LUNA test suite.

Pins a CPU backend with enough virtual devices for the sharded / shard_map
tests *before* JAX initialises, so the whole suite runs on any machine with no
GPU and a deterministic device count. Multi-GPU tests that need a real
multi-process launch are auto-skipped unless ``LUNA_MULTIGPU_TEST=1``.

Usage:
    pytest                     # full suite (CPU, 8 virtual devices)
    pytest -m "not slow"       # the fast tier (a measured wall-clock budget)
    LUNA_MULTIGPU_TEST=1 ...   # also run the distributed multi-GPU tests
    python tools/run_tests.py  # the same tiers, one process per file, report
"""
import os
import sys

# Must be set before JAX is imported anywhere in the test process.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

# The repo root, for `tools.run_tests` below -- NOT for `luna`, which is
# installed (`pip install -e .`) and must be imported like any other
# package. Every test file used to carry its own copy of this line.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# src/ last, for amuse_luna only: the wheel ships luna/ alone, and an installed
# luna still comes first on the path.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def _fast_files():
    """The fast tier as tools/run_tests.py computes it: the cheapest files by
    MEASURED cost (tests/tiers.json) that fit the budget. Marking by filename
    instead put every test_amr_* in the slow tier whatever it cost, which left
    push-time refinement coverage at zero while the cheapest six AMR suites
    together cost less than a minute."""
    try:
        from tools.run_tests import select
        budget = float(os.environ.get("LUNA_TEST_BUDGET_S", 120))
        return {os.path.basename(f) for f in select("fast", budget)}
    except Exception:
        return set()


def pytest_collection_modifyitems(config, items):
    """Mark everything outside the fast tier `slow`, and skip
    `@pytest.mark.multigpu` unless explicitly enabled."""
    import pytest

    fast = _fast_files()
    slow = pytest.mark.slow
    for item in items:
        if os.path.basename(str(item.fspath)) not in fast:
            item.add_marker(slow)

    if os.environ.get("LUNA_MULTIGPU_TEST"):
        return
    skip = pytest.mark.skip(
        reason="needs a multi-process launch; set LUNA_MULTIGPU_TEST=1 to run")
    for item in items:
        if "multigpu" in item.keywords:
            item.add_marker(skip)
