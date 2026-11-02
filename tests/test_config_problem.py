"""cfg.problem: a test problem's own parameters live in its IC module's Params.

A dotted key on the command line (`problem.amp=0.01`) reaches the IC; a flat
name that moved (`jeans_amp`) fails in a config file and on the command line
with its new key; the AMUSE worker translates the old flat keys instead.
"""
import textwrap
from dataclasses import replace

import numpy as np
import pytest

pytestmark = pytest.mark.fast

JEANS = ["test_problem=Jeans-Instability-3D", "SPATIAL_DIMENSION=3", "Nx=8", "Ny=4", "Nz=4",
         "BOUNDARY=0"]


@pytest.fixture
def plain(tmp_path):
    """A config file holding the template's Config, built (an import alone is refused)."""
    f = tmp_path / "plain.py"
    f.write_text("from luna.template_config import Config\nConfig = Config()\n")
    return str(f)


def _jeans_amplitude(cfg):
    """The density amplitude of the IC; cell centres sample the cosine's peak at cos(pi / Nx)."""
    from luna.ic_generation.ic import initial_conditions
    rho = np.asarray(initial_conditions(cfg), dtype=np.float64)[0]
    return (rho.max() - rho.min()) / (2.0 * cfg.problem.rho0 * np.cos(np.pi / cfg.Nx))


def test_a_dotted_key_reaches_the_ic(plain):
    """The amplitude measured in the IC is the one on the command line, not the default."""
    from luna.ic_generation.jeans3d import Params
    from luna.setup.cli import configure
    cfg = configure([plain] + JEANS + ["problem.amp=0.01", "problem.rho0=2.0"])
    assert cfg.problem == Params(amp=0.01, rho0=2.0)
    assert _jeans_amplitude(cfg) == pytest.approx(0.01, rel=1e-5)
    assert Params().amp != 0.01


def test_a_dotted_key_in_a_run_script_command_line():
    from luna.setup.cli import script_args, apply
    from luna.template_config import Config

    def config(nx=8):
        return replace(Config(), test_problem="Gresho-Vortex-2D", SPATIAL_DIMENSION=2,
                       Nx=nx, Ny=nx)
    params, over = script_args(config, ["nx=16", "problem.Mmax=0.1"])
    cfg = apply(config(**params), over)
    assert cfg.Nx == 16 and cfg.problem.Mmax == 0.1


def test_a_dotted_key_the_problem_does_not_have_is_refused(plain):
    from luna.setup.cli import configure
    with pytest.raises(ValueError, match="problem.ampl: Jeans-Instability-3D has no parameter "
                                         "'ampl' \\(did you mean problem.amp"):
        configure([plain] + JEANS + ["problem.ampl=0.01"])
    with pytest.raises(ValueError, match="test_problem='Sod-Shock' has no parameters"):
        configure([plain, "problem.amp=0.01"])


def test_a_retired_flat_key_names_its_dotted_key(tmp_path, plain):
    from luna.setup.cli import configure, load_config
    with pytest.raises(ValueError, match="'jeans_amp' is retired: now problem.amp "
                                         "\\(test_problem 'Jeans-Instability-3D'\\)"):
        configure([plain] + JEANS + ["jeans_amp=0.01"])
    with pytest.raises(ValueError, match="'cc85_mu' is retired: now cooling_mu"):
        configure([plain, "cc85_mu=0.6"])
    f = tmp_path / "old_run.py"
    f.write_text(textwrap.dedent("""
        from dataclasses import replace
        from luna.template_config import Config
        Config = replace(Config(), test_problem="Turbulent-Sphere-3D", cloud_mach=4.0)
    """))
    with pytest.raises(ValueError, match="'cloud_mach' is retired: now problem.mach"):
        load_config(str(f))


def test_a_config_file_sets_the_params(tmp_path):
    from luna.setup.cli import load_config
    f = tmp_path / "run.py"
    f.write_text(textwrap.dedent("""
        from dataclasses import replace
        from luna.ic_generation import jeans3d
        from luna.template_config import Config
        Config = replace(Config(), test_problem="Jeans-Instability-3D", SPATIAL_DIMENSION=3,
                         Nx=8, Ny=4, Nz=4, BOUNDARY=0, problem=jeans3d.Params(amp=0.02))
    """))
    cfg = load_config(str(f))
    assert _jeans_amplitude(cfg) == pytest.approx(0.02, rel=1e-5)


def test_the_params_follow_test_problem():
    """Changing test_problem swaps a default Params; a changed one is refused."""
    from luna.ic_generation import gresho_vortex, jeans3d
    from luna.template_config import Config
    cfg = replace(Config(), test_problem="Jeans-Instability-3D")
    assert cfg.problem == jeans3d.Params()
    assert replace(cfg, test_problem="Gresho-Vortex-2D").problem == gresho_vortex.Params()
    assert replace(cfg, test_problem="Sod-Shock").problem is None
    changed = replace(cfg, problem=jeans3d.Params(amp=0.5))
    with pytest.raises(ValueError, match="is not test_problem='Gresho-Vortex-2D'"):
        replace(changed, test_problem="Gresho-Vortex-2D")


def test_the_worker_translates_old_flat_keys():
    """Torch scripts pass flat keys; the worker maps them to their new place."""
    from luna.setup.cli import apply
    from luna.setup.user_config import translate
    from luna.template_config import Config
    kv = translate({"test_problem": "Turbulent-Sphere-3D", "cloud_mach": 4.0,
                    "cloud_mu": 2.3, "cloud_seed": 3, "Nx": 16})
    assert kv == {"test_problem": "Turbulent-Sphere-3D", "problem.mach": 4.0,
                  "sink_star_mu": 2.3, "seed": 3, "Nx": 16}
    cfg = apply(Config(), kv)
    assert cfg.problem.mach == 4.0 and cfg.sink_star_mu == 2.3 and cfg.seed == 3
