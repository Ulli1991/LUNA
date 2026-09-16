import importlib
import sys


def test_run_pancake_import_does_not_require_repo_config(monkeypatch):
    monkeypatch.setitem(sys.modules, "config", None)
    sys.modules.pop("main", None)
    sys.modules.pop("scripts.cosmo.run_pancake", None)

    mod = importlib.import_module("scripts.cosmo.run_pancake")

    assert callable(mod.build_cfg)
    assert callable(mod.analytic)
