"""The adjoint experiments are tasks in scripts/examples/autodiff/: every name
Config.autodiff_task accepts resolves to one, and a run script's command line
hands the task its own options. No device work."""
import inspect
import os

import pytest

pytestmark = pytest.mark.fast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS = ("eta_recovery", "ic_inversion", "luna_emergence", "corrector_training")


@pytest.mark.parametrize("name", TASKS)
def test_every_task_name_is_an_example(name, monkeypatch):
    monkeypatch.chdir(ROOT)
    from dataclasses import replace
    from luna.autodiff.rollout import example_task
    from luna.template_config import Config
    replace(Config(), autodiff_task=name).validate()
    task = example_task(name)
    assert list(inspect.signature(task).parameters)[:4] == ["init", "cfg", "physics", "is_main"]


def test_an_unknown_task_says_where_tasks_live():
    from luna.autodiff.rollout import example_task
    with pytest.raises(ModuleNotFoundError, match="scripts/examples/autodiff/no_such_task.py"):
        example_task("no_such_task")


def test_task_options_come_off_the_command_line():
    from luna.setup.cli import task_options

    def task(init, cfg, physics, is_main, *, movie="", movie_every=2, hostloop=False):
        return movie, movie_every, hostloop

    bound, rest = task_options(task, ["nx=64", "movie=out", "movie_every=5",
                                      "hostloop=1", "CFL=0.3"])
    assert rest == ["nx=64", "CFL=0.3"]
    assert bound(None, None, None, True) == ("out", 5, True)
    same, rest = task_options(task, ["nx=64"])
    assert same is task and rest == ["nx=64"]


def test_script_main_runs_the_task_it_is_given(monkeypatch):
    from luna.setup import cli

    def config(nx=16):
        from dataclasses import replace
        from luna.template_config import Config
        return replace(Config(), Nx=nx, autodiff_mode=True)

    def task(init, cfg, physics, is_main, *, forward_only=False):
        return forward_only

    seen = {}
    monkeypatch.setattr(cli, "launch", lambda cfg, banner=False, task=None:
                        seen.update(cfg=cfg, task=task))
    cli.script_main(config, ["nx=32", "forward_only=1", "CFL=0.25"], task=task)
    assert seen["cfg"].Nx == 32 and seen["cfg"].CFL == 0.25
    assert seen["task"](None, None, None, True) is True


def test_main_py_hands_the_example_task_its_options(tmp_path, monkeypatch):
    """The experiments' parameters left Config for the tasks' options:
    `python main.py cfg.py n_iter=7` binds them to cfg.autodiff_task's task,
    and outside autodiff_mode such a name is refused, not dropped."""
    monkeypatch.chdir(ROOT)
    from luna.setup.cli import configure_run
    f = tmp_path / "ad.py"
    f.write_text("from dataclasses import replace\n"
                 "from luna.template_config import Config\n"
                 "Config = replace(Config(), autodiff_mode=True, autodiff_task='ic_inversion')\n")
    cfg, task = configure_run([str(f), "Nx=16", "n_iter=7", "field=energy"])
    assert cfg.Nx == 16 and task.keywords == {"n_iter": 7, "field": "energy"}
    assert configure_run([str(f), "Nx=16"])[1] is None
    with pytest.raises(ValueError, match="'n_itr' is not a Config field"):
        configure_run([str(f), "n_itr=7"])
    with pytest.raises(ValueError, match="'n_iter' is not a Config field"):
        configure_run([str(f), "autodiff_mode=0", "n_iter=7"])
    with pytest.raises(ValueError, match="'autodiff_n_iter' is retired: a task option now"):
        configure_run([str(f), "autodiff_n_iter=7"])


@pytest.mark.parametrize("name,argv,want", [
    ("blast_emergence_run", ["seeds=20", "image=t.png"],
     {"driver": "blast", "n_iter": 200, "seeds": 20, "image": "t.png"}),
    ("turb_emergence_run", ["mach=2.5", "n_iter=3"],
     {"driver": "turbulence", "cfl": 0.2, "lr": 0.15, "n_iter": 3, "mach": 2.5}),
    ("turb_autodiff_run", ["n_iter=3"], {"n_iter": 3, "field": "density"}),
    ("kh_corrector_run", ["n_train=5"], {"n_iter": 100, "n_train": 5}),
])
def test_run_scripts_keep_their_parameters_as_task_options(name, argv, want, monkeypatch):
    """The command lines the run scripts documented still work: the names that
    were config() parameters feeding autodiff_* fields are the task's options."""
    import importlib
    monkeypatch.chdir(ROOT)
    from luna.setup.cli import task_options
    mod = importlib.import_module(f"scripts.examples.autodiff.{name}")
    bound, rest = task_options(mod.TASK, argv + ["nx=16"])
    assert rest == ["nx=16"]
    assert {k: bound.keywords[k] for k in want} == want    # partials flatten
