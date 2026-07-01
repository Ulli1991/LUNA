"""Shared pytest configuration for the LUNA test suite.

Pins a CPU backend with enough virtual devices for the sharded / shard_map
tests *before* JAX initialises, so the whole suite runs on any machine with no
GPU and a deterministic device count. Multi-GPU tests that need a real
multi-process launch are auto-skipped unless ``LUNA_MULTIGPU_TEST=1``.

Usage:
    pytest                     # full suite (CPU, 8 virtual devices)
    pytest -m "not slow"       # fast smoke subset (seconds)
    LUNA_MULTIGPU_TEST=1 ...    # also run the distributed multi-GPU tests
"""
import os

# Must be set before JAX is imported anywhere in the test process.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")


def pytest_collection_modifyitems(config, items):
    """Auto-mark the heavy block-AMR tests `slow`, and auto-skip
    `@pytest.mark.multigpu` tests unless explicitly enabled.

    The `test_amr_*` suite runs real multi-level block sims / shard_map and is
    minutes-long — it belongs in the full suite, not the fast CI smoke tier
    (`pytest -m "not slow"`). Marking here keeps the marker in one place.
    """
    import pytest

    slow = pytest.mark.slow
    for item in items:
        if "test_amr_" in item.nodeid:
            item.add_marker(slow)

    if os.environ.get("LUNA_MULTIGPU_TEST"):
        return
    skip = pytest.mark.skip(
        reason="needs a multi-process launch; set LUNA_MULTIGPU_TEST=1 to run")
    for item in items:
        if "multigpu" in item.keywords:
            item.add_marker(skip)
