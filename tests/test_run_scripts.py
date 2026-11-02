"""Every run script builds a Config that validates, from its own defaults and
through the command-line parser, without touching a device. A script that
runs at import time, mutates a config, or names a field that no longer exists
fails here."""
import glob
import importlib
import os

import pytest

pytestmark = pytest.mark.fast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# solver benchmarks that build no Config (they take their parameters through
# luna.setup.cli.args); every other run script must expose config()
NO_CONFIG = {"scripts.examples.runs.poisson_accuracy_run"}


def _shipped(*patterns):
    """The run scripts the REPOSITORY ships, as module names. Tracked files
    only: a local experiment under scripts/ is the author's business, and this
    is exactly the set CI sees. (Measured the hard way: right after the luna/
    namespace landed, this gate went red on a gitignored script that exists in
    one checkout and no other.)"""
    import subprocess
    try:
        r = subprocess.run(["git", "ls-files", *patterns], cwd=ROOT,
                           capture_output=True, text=True, timeout=30)
        files = r.stdout.split() if r.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        files = []
    if not files:                      # no git (a source tarball): fall back
        files = [os.path.relpath(f, ROOT)
                 for pat in patterns for f in glob.glob(os.path.join(ROOT, pat))]
    return sorted(f[:-3].replace("/", ".").replace(os.sep, ".") for f in files)


RUN_SCRIPTS = [m for m in _shipped("scripts/validation/runs/*_run.py",
                                   "scripts/examples/*/*_run.py",
                                   "scripts/cosmo/run_cosmo_*.py")
               if m not in NO_CONFIG]

# the one script parameter that reaches a Config field of another type
ARGS = {"scripts.examples.autodiff.ic_inversion_run": ["problem=cooling", "nx=16"],
        "scripts.validation.runs.evrard_collapse_run": ["AMR=1", "amr_levels=2"],
        "scripts.validation.runs.m82_wind_run": ["cool=0", "enable_x64=0"],
        "scripts.validation.runs.sedov3d_run": ["nx=64", "MESH_SHAPE=2,1,1", "enable_x64=1"]}


@pytest.mark.parametrize("name", RUN_SCRIPTS)
def test_script_config_validates(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    mod = importlib.import_module(name)
    assert callable(getattr(mod, "config", None)), f"{name} has no config()"
    from luna.template_config import Config
    cfg = mod.config()
    assert isinstance(cfg, Config)
    cfg.validate()


@pytest.mark.parametrize("name", RUN_SCRIPTS)
def test_script_takes_argv(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    from dataclasses import replace
    from luna.setup.cli import script_args
    mod = importlib.import_module(name)
    argv = ARGS.get(name, ["CFL=0.25"])
    params, over = script_args(mod.config, argv)
    cfg = replace(mod.config(**params), **over)
    cfg.validate()
    for k, v in over.items():
        assert getattr(cfg, k) == v


def test_sedov_scaling_point_is_what_the_ladder_submits():
    from luna.setup.cli import script_args
    from scripts.validation.runs import sedov3d_run
    params, over = script_args(sedov3d_run.config,
                               ["nx=1024", "MESH_SHAPE=2,2,2", "enable_x64=1",
                                "max_steps=30", "write_snapshots=0"])
    from dataclasses import replace
    cfg = replace(sedov3d_run.config(**params), **over)
    assert cfg.Nx == 1024 and cfg.MESH_SHAPE == (2, 2, 2) and cfg.enable_x64
    assert cfg.max_steps == 30 and cfg.write_snapshots is False


def test_the_gate_ignores_an_untracked_script():
    """A local experiment dropped into scripts/ must not turn the suite red:
    it exists in one checkout and no other. Exactly that happened the day the
    luna/ namespace landed, on a gitignored script carrying pre-move imports."""
    probe = os.path.join(ROOT, "scripts", "validation", "runs", "_probe_run.py")
    assert not os.path.exists(probe), "leftover probe from an earlier run"
    open(probe, "w").write("def config():\n    raise AssertionError('never called')\n")
    try:
        shipped = _shipped("scripts/validation/runs/*_run.py")
        assert shipped, "no tracked run scripts found at all"
        assert any(m.endswith("sedov3d_run") for m in shipped)
        assert "scripts.validation.runs._probe_run" not in shipped
    finally:
        os.remove(probe)


def test_every_run_script_has_a_two_step_case():
    """A shipped run script with no smoke case would be config-checked only; one stale case is an error."""
    from tests._run_script_cases import CASES
    shipped = set(RUN_SCRIPTS) | NO_CONFIG
    assert shipped - set(CASES) == set(), "run scripts with no two-step case"
    assert set(CASES) - shipped == set(), "smoke cases for scripts that are not shipped"
