"""The run log states what ran: the hydro kernel (xla or pallas), the real
domain decomposition, the grid's used dimensions only, and a throughput that
leaves the compile out instead of an average that rounds to 0.00."""
import re

import numpy as np
import pytest

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("shape, name", [
    ((1, 1, 1), "none (one device)"),
    ((4,), "slab (cut along x)"),
    ((8, 1, 1), "slab (cut along x)"),
    ((1, 2, 1), "slab (cut along y)"),
    ((2, 2, 1), "pencil (cut along x, y)"),       # printed "slab (1D)" before
    ((2, 1, 2), "pencil (cut along x, z)"),       # printed "block" before
    ((2, 2), "block (cut along x, y)"),
    ((2, 2, 2), "block (cut along x, y, z)"),
])
def test_decomposition_name(shape, name):
    from luna.setup.mesh import decomposition_name
    assert decomposition_name(shape) == name


@pytest.mark.parametrize("kernel", ["xla", "pallas", "auto"])
def test_the_banner_names_the_kernel_that_runs(capsys, kernel):
    """The banner's kernel is the one traced into the step: a pallas_call iff it says pallas."""
    import jax
    from tests._helpers import _build_cfg_for_test, _make_step_fns
    from luna.hydro.conversions import prim_to_cons
    cfg = _build_cfg_for_test(test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3, BOUNDARY=0,
                              Nx=16, Ny=16, Nz=16, hydro_kernel=kernel)
    capsys.readouterr()
    _, step = _make_step_fns(cfg)
    said = re.search(r"MUSCL-Hancock \(HLLC, (\w+) kernel, shard_map\)", capsys.readouterr().out)
    assert said, "no integrator banner"
    W = np.stack([np.ones((16,) * 3), *np.zeros((3, 16, 16, 16)), np.ones((16,) * 3)])
    U = prim_to_cons(jax.numpy.asarray(W), cfg.gamma)
    ran = "pallas" if "pallas_call" in str(jax.make_jaxpr(step)(U, 1e-3)) else "xla"
    assert said.group(1) == ran
    if kernel != "auto":
        assert ran == kernel


def test_the_run_report_shows_used_dims_and_a_real_throughput(tmp_path, capsys):
    import jax
    from dataclasses import replace
    from luna.setup.jax_init import initialize_jax_distributed
    from luna.setup.mesh import decomposition_name
    from luna.template_config import Config
    cfg = replace(Config(), test_problem="Sod-Shock", SPATIAL_DIMENSION=1, Nx=64, Ny=1024, Nz=1024,
                  t_end=0.2, max_steps=40, output_cadence=5, write_snapshots=False,
                  PLOT_AFTER_SIM=False, output_dir=str(tmp_path / "run"), enable_x64=True)
    initialize_jax_distributed(enable_x64=cfg.enable_x64)
    from luna import main
    capsys.readouterr()
    res = main.run(cfg)
    log = capsys.readouterr().out
    assert re.search(r"^Global grid: 64$", log, re.M), "a 1D run shows Nx only"
    grid = tuple(int(v) for v in re.search(r"^Process grid: \(([\d, ]+)\)", log, re.M).group(1)
                 .rstrip(",").split(","))
    assert int(np.prod(grid)) == jax.device_count()
    assert f"Actual decomposition: {decomposition_name(grid)}\n" in log
    assert "Average performance" not in log
    m = re.search(r"Performance after step (\d+) \(no compile, output included\): ([\d.e+-]+) "
                  r"Mcells/s \(global grid\), ([\d.e+-]+) ms/step over (\d+) steps", log)
    assert m, "no performance line"
    first, rate, ms, n = int(m[1]), float(m[2]), float(m[3]), int(m[4])
    assert first == 1 and first + n == res.n_steps         # from the first perf line to the end
    assert rate > 0.0 and ms > 0.0
    assert rate == pytest.approx(64 / (ms * 1e-3) / 1e6, rel=2e-3)   # 4 significant digits each
