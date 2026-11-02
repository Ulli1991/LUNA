"""One precision mode per test: @pytest.mark.precision("fp64" | "fp32" | "mixed"), default fp64.

fp64: x64 on; fp32: x64 off; mixed: x64 on with hydro_recon32 (FP32 reconstruction, FP64 fluxes and
state). The session starts in fp64, so module-scoped fixtures see a defined precision; a marked test
switches and is restored after. tests/_helpers.evolve and _run_short refuse a cfg that disagrees."""
import pytest

MODES = ("fp64", "fp32", "mixed")


def pytest_configure(config):
    config.addinivalue_line("markers", "precision(mode): run the test in fp64 (default), fp32 or mixed")
    import jax
    from luna.setup.jax_init import with_host_platform
    with_host_platform()                  # JAX_PLATFORMS=cuda -> cuda,cpu (the AMR face report's callback)
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def precision(request):
    """The test's precision mode, set for its duration."""
    import jax

    from tests import _helpers
    marker = request.node.get_closest_marker("precision")
    mode = marker.args[0] if marker else "fp64"
    if mode not in MODES:
        raise ValueError(f"precision {mode!r}: expected one of {MODES}")
    old, old_mode = jax.config.jax_enable_x64, _helpers.PRECISION
    jax.config.update("jax_enable_x64", mode != "fp32")
    _helpers.PRECISION = mode
    yield mode
    jax.config.update("jax_enable_x64", old)
    _helpers.PRECISION = old_mode


@pytest.fixture(autouse=True)
def amr_faces_reported():
    """An AMR face report (audit C8, setup/face_report.AMR_EVENTS) no test consumed fails the test: the driver
    functions refuse such a face as NaN, and a test that allows them reads and asserts its count."""
    from luna.setup import face_report
    face_report.AMR_EVENTS.clear()
    yield
    import sys
    if "jax" in sys.modules:
        sys.modules["jax"].effects_barrier()
    if face_report.AMR_EVENTS:
        ev = list(face_report.AMR_EVENTS)
        face_report.AMR_EVENTS.clear()
        pytest.fail(f"unconsumed AMR face reports (audit C8): {sum(e['n'] for e in ev)} faces on levels "
                    f"{sorted({e['level'] for e in ev})}")
