"""Once the state owns the initial arrays, nothing else may hold them: the
initial gas grid and each stage's initial carry are released, and a second
initial state cannot be built from the released carries by accident."""
import os

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")


from tests._helpers import _build_cfg_for_test  # noqa: E402


def _cfg():
    return _build_cfg_for_test(
        test_problem="Soundwave-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
        Nx=8, Ny=8, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0, gamma=1.4,
        ORDER=1, LIMITER=1, SELF_GRAVITY=False,
        radiation_enabled=True, rad_chat=10.0, rad_E0=1e-3, rad_substeps=1)


def test_the_initial_arrays_are_released_to_the_state():
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, release_initial
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind

    cfg = resolve(_cfg())
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, is_main=False)
    physics = assemble(cfg, mesh, kind, layout, init, is_main=False)
    state = initial_state(init, physics)
    rad = next(s for s in physics.stages if s.name == "radiation")
    assert init.U is state.U and rad.carry0 is state.sources["radiation"]

    release_initial(init, physics)
    assert init.U is None and rad.carry0 is None
    assert state.U is not None and state.sources["radiation"] is not None
    with pytest.raises(RuntimeError, match="released"):
        initial_state(init, physics)


def test_the_worker_state_holds_the_only_copy():
    from amuse_luna.stepper import build
    st = build(_cfg(), is_main=False)
    rad = next(s for s in st.physics.stages if s.name == "radiation")
    assert rad.carry0 is None and st._s.sources["radiation"] is not None
