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
    """Auto-skip `@pytest.mark.multigpu` tests unless explicitly enabled."""
    import pytest

    if os.environ.get("LUNA_MULTIGPU_TEST"):
        return
    skip = pytest.mark.skip(
        reason="needs a multi-process launch; set LUNA_MULTIGPU_TEST=1 to run")
    for item in items:
        if "multigpu" in item.keywords:
            item.add_marker(skip)
