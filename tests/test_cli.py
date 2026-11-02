"""The command line (setup/cli.py): typed field=value overrides, a config
file by path, and a run script's own parameters. No devices needed."""
import dataclasses
import sys
import textwrap

import pytest

pytestmark = pytest.mark.fast



def test_config_is_frozen():
    from luna.template_config import Config
    cfg = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.Nx = 3
    assert dataclasses.replace(cfg, Nx=3).Nx == 3 and cfg.Nx != 3


def test_overrides_are_typed_by_the_field():
    from luna.setup.cli import overrides
    o = overrides({"Nx": "512", "CFL": "0.3", "enable_x64": "1", "AMR": "false",
                   "MESH_SHAPE": "2,2,4", "cosmo_snapshot_a": "0.5,1.0",
                   "amr_thresholds": "0.1,0.2", "restart_file": "none",
                   "riemann_solver": "hll", "POISSON_N_LEVELS": "5"})
    assert o == {"Nx": 512, "CFL": 0.3, "enable_x64": True, "AMR": False,
                 "MESH_SHAPE": (2, 2, 4), "cosmo_snapshot_a": (0.5, 1.0),
                 "amr_thresholds": [0.1, 0.2], "restart_file": None,
                 "riemann_solver": "hll", "POISSON_N_LEVELS": 5}
    assert type(o["Nx"]) is int and type(o["CFL"]) is float


def test_bad_values_and_unknown_fields_are_refused():
    from luna.setup.cli import overrides
    with pytest.raises(ValueError, match="bool"):
        overrides({"enable_x64": "maybe"})
    with pytest.raises(ValueError):
        overrides({"Nx": "12.5"})
    with pytest.raises(ValueError, match="did you mean.*enable_x64"):
        overrides({"enable_x46": "1"})
    with pytest.raises(ValueError, match="cannot be set"):
        overrides({"units": "cgs"})


def test_split_args_takes_one_config_file():
    from luna.setup.cli import split_args
    assert split_args(["a.py", "Nx=4", "t_end=0.1"]) == ("a.py", {"Nx": "4", "t_end": "0.1"})
    assert split_args(["Nx=4"]) == (None, {"Nx": "4"})
    with pytest.raises(ValueError, match="two config files"):
        split_args(["a.py", "b.py"])


def test_load_config_from_a_file_and_configure(tmp_path):
    from luna.setup.cli import configure, load_config
    from luna.template_config import Config
    f = tmp_path / "my_run.py"
    f.write_text(textwrap.dedent("""
        from dataclasses import dataclass
        @dataclass
        class Config:
            test_problem: str = "Sod-Shock"
            Nx: int = 48
    """))
    with pytest.warns(UserWarning, match="missing from config.py"):
        cfg = load_config(str(f))
    assert isinstance(cfg, Config) and cfg.Nx == 48 and cfg.CFL == Config().CFL
    with pytest.warns(UserWarning):
        cfg = configure([str(f), "Nx=96", "CFL=0.2"])
    assert cfg.Nx == 96 and cfg.CFL == 0.2 and cfg.test_problem == "Sod-Shock"
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.py"))


def test_a_config_file_may_export_a_built_config(tmp_path):
    """`cp template_config.py config.py` leaves a Config CLASS behind; a file
    may also write `Config = replace(Config(), ...)` -- an instance. Both have
    to load, or the two styles disagree about what a config file is."""
    from luna.setup.cli import configure, load_config
    from luna.template_config import Config
    f = tmp_path / "built.py"
    f.write_text(textwrap.dedent("""
        from dataclasses import replace
        from luna.template_config import Config
        Config = replace(Config(), Nx=48, CFL=0.11)
    """))
    cfg = load_config(str(f))
    assert isinstance(cfg, Config) and cfg.Nx == 48 and cfg.CFL == 0.11
    assert configure([str(f), "Nx=96"]).Nx == 96      # overrides still win


RUN_SCRIPT = """
    from dataclasses import replace
    from luna.template_config import Config

    def config(nx=64, snap=0.5):
        return replace(Config(), test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2,
                       Nx=nx, Ny=nx, Lx=2.0, Ly=2.0, time_bet_snaps=snap)

    def report(res):
        print("REPORT", res)

    TASK = "the file's task"
"""


def test_main_py_on_a_run_script_gets_its_problem(tmp_path):
    """`python main.py runs/kh_run.py` once ran the template's Sod problem: the
    run script imports the template Config and builds its run in config(), and
    the loader took the import. Its config() is the run, its arguments come off
    the command line, and its TASK and report are what main.py hands on."""
    from luna.setup.cli import configure, configure_file, load_config
    from luna.template_config import Config
    assert Config().test_problem != "Kelvin-Helmholtz"      # else this proves nothing
    f = tmp_path / "kh_like_run.py"
    f.write_text(textwrap.dedent(RUN_SCRIPT))
    cfg = load_config(str(f))
    assert (cfg.test_problem, cfg.SPATIAL_DIMENSION, cfg.Nx, cfg.Ny) == ("Kelvin-Helmholtz", 2, 64, 64)
    cfg, task, report = configure_file([str(f), "nx=32", "snap=0.25", "CFL=0.2"])
    assert (cfg.test_problem, cfg.Nx, cfg.Ny, cfg.time_bet_snaps, cfg.CFL) == (
        "Kelvin-Helmholtz", 32, 32, 0.25, 0.2)
    assert task == "the file's task" and report.__name__ == "report"
    assert configure([str(f), "Nx=48"]).Nx == 48            # a field still overrides
    with pytest.raises(ValueError, match="did you mean nx"):
        configure([str(f), "nxx=32"])


@pytest.mark.parametrize("body, match", [
    ("from luna.template_config import Config\n", "neither config\\(\\) nor a Config of its own"),
    ("x = 1\n", "neither config\\(\\) nor a Config of its own"),
    ("from dataclasses import replace\nfrom luna.template_config import Config\n"
     "Config = replace(Config(), Nx=8)\ndef config():\n    return Config\n", "both config\\(\\)"),
])
def test_a_file_that_is_not_a_run_is_refused(tmp_path, body, match):
    """A file whose only Config is the imported template, or with no Config at
    all, is refused, not run as the template's problem; so is one defining the
    run twice."""
    from luna.setup.cli import configure_file, load_config
    f = tmp_path / "not_a_run.py"
    f.write_text(body)
    with pytest.raises((AttributeError, ValueError), match=match):
        load_config(str(f))
    with pytest.raises((AttributeError, ValueError), match=match):
        configure_file([str(f)])


def test_a_run_script_with_its_own_main_runs_through_it(tmp_path):
    """ic_inversion_run picks its task from its parameters in main(): main.py
    calls that main() with the rest of the line, and no Config is built
    without it (that would run config() under another task)."""
    from luna.setup.cli import configure_file, own_main
    f = tmp_path / "own_main_run.py"
    f.write_text(textwrap.dedent("""
        from luna.template_config import Config
        def config(nx=8):
            return Config()
        def main(argv=None):
            return ("ran", argv)
    """))
    entry = own_main(["nx=16", str(f), "CFL=0.2"])
    assert entry() == ("ran", ["nx=16", "CFL=0.2"])
    with pytest.raises(ValueError, match="its own main"):
        configure_file([str(f)])
    g = tmp_path / "plain_run.py"
    g.write_text(textwrap.dedent(RUN_SCRIPT))
    assert own_main([str(g)]) is None and own_main(["Nx=8"]) is None


def test_load_config_without_a_file_imports_config_py(monkeypatch):
    from luna.setup.cli import load_config
    monkeypatch.setitem(sys.modules, "config", None)     # import config -> ImportError
    with pytest.raises(ImportError, match=r"cp src/luna/template_config\.py config\.py"):
        load_config()


def test_a_stale_config_file_is_told_what_moved(tmp_path):
    """A config.py written before the luna/ namespace imports `units`; the
    error must name the replacement rather than just the missing module."""
    from luna.setup.cli import load_config
    f = tmp_path / "old_style.py"
    f.write_text("from units import UnitSystem\n")
    with pytest.raises(ModuleNotFoundError, match=r"luna\.units"):
        load_config(str(f))


def test_script_args_split_parameters_from_fields():
    from luna.setup.cli import script_args, script_main

    def config(nx=64, t_end_myr=10.0, sinks=False, tag="a"):
        from dataclasses import replace
        from luna.template_config import Config
        return replace(Config(), Nx=nx, t_end=t_end_myr, sinks_enabled=sinks,
                       output_dir=tag)

    params, over = script_args(config, ["nx=128", "sinks=1", "tag=b", "CFL=0.3", "Ny=8"])
    assert params == {"nx": 128, "sinks": True, "tag": "b"}
    assert over == {"CFL": 0.3, "Ny": 8}
    with pytest.raises(ValueError, match="did you mean.*t_end_myr"):
        script_args(config, ["t_end_my=3"])

    # the config a script_main would run: parameters first, then field overrides
    launched = {}
    from luna.setup import cli
    orig = cli.launch
    cli.launch = lambda cfg, banner=False: launched.setdefault("cfg", cfg)
    try:
        script_main(config, ["nx=32", "CFL=0.25"])
    finally:
        cli.launch = orig
    assert launched["cfg"].Nx == 32 and launched["cfg"].CFL == 0.25


def test_resolve_fills_a_problems_derived_grid():
    from dataclasses import replace
    from luna.ic_generation.ic import resolve
    from luna.template_config import Config
    cfg = replace(Config(), test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3,
                  Nx=100, Ny=100, Nz=100, Lx=1.0)
    r = resolve(cfg)
    assert (r.Ny, r.Nz) == (10, 10) and r.Ly == pytest.approx(0.1)
    assert cfg.Ny == 100                                     # the input is untouched
    assert resolve(replace(cfg, test_problem="Sedov-Blast-3D")).Ny == 100
