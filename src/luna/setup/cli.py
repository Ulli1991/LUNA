"""The command line: an optional config file, then field=value overrides.

    python main.py                              config.py in the cwd
    python main.py runs/kh.py Nx=2048           that file's config() or Config, then overrides
    python -m scripts.validation.runs.sedov3d_run nx=512 MESH_SHAPE=2,2,4 enable_x64=1

A run script's own parameters (the arguments of its `config(...)`) and the
Config fields share the one syntax; each value is parsed by the type of what
it sets: `1/0/true/false` for a bool, `2,2,4` for a tuple, `none` for None.
The problem's own parameters are `problem.<name>=value` (cfg.problem).
"""
import difflib
import importlib.util
import inspect
import typing
from dataclasses import MISSING, fields, replace

NONE_WORDS = ("none", "null")
TRUE_WORDS = ("1", "true", "yes", "on")
FALSE_WORDS = ("0", "false", "no", "off")
PROBLEM = "problem."


def _kind(annotation, default):
    """The Python type a value is parsed as, from the field's annotation, else
    from its default. Optional[X] and `X = None` both parse as X."""
    ann = annotation
    if isinstance(ann, str):                       # postponed annotations
        ann = {"int": int, "float": float, "bool": bool, "str": str,
               "tuple": tuple, "list": list}.get(ann, ann)
    if typing.get_origin(ann) is typing.Union:
        inner = [a for a in typing.get_args(ann) if a is not type(None)]
        ann = inner[0] if len(inner) == 1 else ann
    if typing.get_origin(ann) in (tuple, list):
        ann = typing.get_origin(ann)
    if ann in (int, float, bool, str, tuple, list):
        return ann
    if ann is inspect.Parameter.empty or ann is None:
        return type(default) if default is not None else str
    return ann


def parse_value(text, kind, name=""):
    """`text` as a value of type `kind`; `none` is None for any kind."""
    if text.strip().lower() in NONE_WORDS:
        return None
    if kind is bool:
        t = text.strip().lower()
        if t in TRUE_WORDS:
            return True
        if t in FALSE_WORDS:
            return False
        raise ValueError(f"{name}={text!r}: a bool is 1/0, true/false, yes/no")
    if kind is int:
        return int(text)
    if kind is float:
        return float(text)
    if kind is str:
        return text
    if kind in (tuple, list):
        items = [p.strip() for p in text.split(",") if p.strip()]
        try:
            vals = [int(p) for p in items]
        except ValueError:
            vals = [float(p) for p in items]
        return kind(vals)
    raise ValueError(f"{name}: a {getattr(kind, '__name__', kind)} cannot be set "
                     "from the command line; build it in a run script")


def split_args(argv):
    """`argv` as (config path or None, {name: text}). A token without `=` is the
    config file; at most one."""
    path, kv = None, {}
    for tok in argv:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip()] = v
        elif path is None:
            path = tok
        else:
            raise ValueError(f"two config files given: {path!r} and {tok!r}; "
                             "overrides are name=value")
    return path, kv


def config_fields(cls=None):
    """{name: (annotation, default)} of the Config dataclass."""
    if cls is None:
        from luna.template_config import Config as cls
    return {f.name: (f.type, f.default if f.default is not MISSING else None)
            for f in fields(cls)}


def _unknown(name, known):
    from luna.setup.user_config import retired
    if retired(name):
        return ValueError(retired(name))
    close = difflib.get_close_matches(name, known, n=3)
    hint = f" (did you mean {', '.join(close)}?)" if close else ""
    return ValueError(f"{name!r} is not a Config field or a run-script parameter{hint}")


def overrides(kv, cls=None):
    """{field: value} from {field: text}, typed by the Config field; `problem.x` stays text for apply()."""
    spec = config_fields(cls)
    out = {}
    for name, text in kv.items():
        if name.startswith(PROBLEM):
            out[name] = text
            continue
        if name == "problem":
            raise ValueError("problem=...: set one parameter as problem.<name>=value")
        if name not in spec:
            raise _unknown(name, list(spec))
        ann, default = spec[name]
        out[name] = parse_value(text, _kind(ann, default), name)
    return out


def apply(cfg, over):
    """cfg with `over` set: the Config fields first, then the `problem.<name>` keys on cfg.problem."""
    cfg = replace(cfg, **{k: v for k, v in over.items() if not k.startswith(PROBLEM)})
    sub = {k[len(PROBLEM):]: v for k, v in over.items() if k.startswith(PROBLEM)}
    if not sub:
        return cfg
    if cfg.problem is None:
        raise ValueError(f"test_problem={cfg.test_problem!r} has no parameters to set: "
                         + ", ".join(PROBLEM + k for k in sub))
    spec = config_fields(type(cfg.problem))
    vals = {}
    for name, v in sub.items():
        if name not in spec:
            close = difflib.get_close_matches(name, list(spec), n=3)
            hint = f" (did you mean {', '.join(PROBLEM + c for c in close)}?)" if close else ""
            raise ValueError(f"{PROBLEM}{name}: {cfg.test_problem} has no parameter {name!r}{hint}")
        ann, default = spec[name]
        vals[name] = parse_value(v, _kind(ann, default), PROBLEM + name) if isinstance(v, str) else v
    return replace(cfg, problem=replace(cfg.problem, **vals))


def _retired_keyword(err, path):
    """Raise the retired-field message if `err` is Config() refusing one."""
    import re
    from luna.setup.user_config import retired
    m = re.search(r"unexpected keyword argument '(\w+)'", str(err))
    if m and retired(m.group(1)):
        raise ValueError(f"{path}: {retired(m.group(1))}") from err


def _exec_file(path):
    """The module of the config or run file at `path`, executed."""
    import os
    spec = importlib.util.spec_from_file_location("luna_user_config", path)
    if spec is None or not os.path.isfile(path):
        raise FileNotFoundError(f"config file {path!r} not found")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except TypeError as e:
        _retired_keyword(e, path)
        raise
    except ModuleNotFoundError as e:
        moved = {"units", "template_config", "hydro", "gravity", "setup", "utils",
                 "amr", "cosmology", "file_io", "ic_generation", "sources",
                 "species", "radiation", "time_integration", "main"}
        top = (e.name or "").split(".")[0]
        if top in moved:
            raise ModuleNotFoundError(
                f"{path} imports '{e.name}', which moved under the luna "
                f"namespace: write 'luna.{e.name}' instead (the library is one "
                "package now, so a wheel claims no top-level name)") from e
        raise
    return mod


def _defined_here(obj, mod):
    """Whether `obj` (a function or class) was defined in `mod`, not imported into it."""
    return getattr(obj, "__module__", None) == mod.__name__


def run_file(path):
    """(module, config function or None, own Config or None) of the file at `path`.

    A run file defines its run in exactly one of two ways: `def config(...)`
    returning a Config (the run scripts under scripts/; its arguments are
    command-line parameters), or a Config of its own -- a class written in
    the file (`cp src/luna/template_config.py my_run.py`) or a built one
    (`Config = replace(Config(), ...)`). A Config class merely imported, as
    `from luna.template_config import Config` in every run script, is the
    template and not this run, so it never counts: a file with neither, or
    both, is refused rather than silently run as the template's problem."""
    mod = _exec_file(path)
    fn = getattr(mod, "config", None)
    fn = fn if inspect.isfunction(fn) else None
    own = getattr(mod, "Config", None)
    if isinstance(own, type) and not _defined_here(own, mod):
        own = None
    if fn is not None and own is not None:
        raise ValueError(f"{path} defines both config() and its own Config; "
                         "keep one, main.py will not guess which is the run")
    if fn is None and own is None:
        raise AttributeError(
            f"{path} defines neither config() nor a Config of its own (a Config "
            "imported from luna.template_config is the template, not a run): write "
            "`def config(...)` returning a Config, or `Config = replace(Config(), ...)`")
    return mod, fn, (own() if isinstance(own, type) else own)


def own_main(argv):
    """The run file's own `main(argv)`, bound to the rest of `argv`, or None.
    A run script whose task depends on its parameters (ic_inversion_run,
    m82_autodiff_run) builds the run in main(); main.py hands it over whole,
    as `python -m` would run it, rather than run config() with another task."""
    path, _ = split_args(argv)
    if path is None:
        return None
    mod, fn, _ = run_file(path)
    entry = getattr(mod, "main", None)
    if fn is None or not inspect.isfunction(entry) or not _defined_here(entry, mod):
        return None
    import functools
    rest = list(argv)
    rest.remove(path)
    return functools.partial(entry, rest)


def load_config(path=None):
    """The user's Config: from the file at `path` (its config() with default
    arguments, or its own Config; see run_file), else `import config` from the
    cwd (the gitignored config.py). Either is projected onto the template so
    every field exists and validate() runs (luna/setup/user_config.py)."""
    from luna.setup.user_config import from_user_config
    if path is not None:
        _, fn, own = run_file(path)
        return from_user_config(fn() if fn is not None else own)
    try:
        from config import Config
    except TypeError as e:
        _retired_keyword(e, "config.py")
        raise
    except ModuleNotFoundError as e:
        if (e.name or "") != "config":
            raise ModuleNotFoundError(
                f"config.py imports '{e.name}', which moved under the luna "
                f"namespace: write 'luna.{e.name}' instead") from e
        raise ImportError("no config.py found: create one with "
                          "`cp src/luna/template_config.py config.py`, or name a "
                          "config file: python main.py my_run.py") from e
    # config.py may export the Config CLASS (a copy of the template) or an
    # already built one (`Config = replace(Config(), ...)`). Accept both.
    return from_user_config(Config() if isinstance(Config, type) else Config)


def configure(argv):
    """`python main.py [config.py] [field=value ...]` as a Config."""
    return configure_file(argv)[0]


def configure_run(argv):
    """`python main.py [config.py] [name=value ...]` as (Config, task or None)."""
    return configure_file(argv)[:2]


def configure_file(argv):
    """`python main.py [config.py] [name=value ...]` as (Config, task or None,
    report or None). A run file's config() takes its own arguments off the
    command line first, as `script_main` does; its `TASK` is the autodiff task
    and its `report(result)` runs after, as in its `script_main(config,
    report=report, task=TASK)`. A remaining name that is not a Config field is
    an option of the autodiff task (the file's TASK, else cfg.autodiff_task),
    bound to it; outside autodiff_mode it is refused."""
    from luna.setup.user_config import from_user_config
    path, kv = split_args(argv)
    mod = fn = None
    if path is None:
        base = load_config(None)
    else:
        if own_main(argv) is not None:
            raise ValueError(f"{path} builds its run in its own main(); main.py runs that, "
                             "a Config alone would drop its task")
        mod, fn, own = run_file(path)
        params = {}
        if fn is not None:
            sig = inspect.signature(fn)
            for name in [k for k in kv if k in sig.parameters]:
                p = sig.parameters[name]
                params[name] = parse_value(kv.pop(name), _kind(p.annotation, p.default), name)
        base = from_user_config(fn(**params) if fn is not None else own)
    spec = config_fields()
    ours = {k: v for k, v in kv.items() if k in spec or k.startswith(PROBLEM)}
    rest = [f"{k}={v}" for k, v in kv.items() if k not in ours]
    cfg = apply(base, overrides(ours))
    task = getattr(mod, "TASK", None) if fn is not None else None
    report = getattr(mod, "report", None) if fn is not None else None
    if rest and cfg.autodiff_mode:
        if task is None:
            from luna.autodiff.rollout import example_task
            task = example_task(cfg.autodiff_task)
        task, rest = task_options(task, rest)
    if rest:
        known = list(spec) + (list(inspect.signature(fn).parameters) if fn else [])
        raise _unknown(rest[0].split("=", 1)[0], known)
    return cfg, task, report


def launch(cfg, banner=False, task=None, U0=None, ic=None):
    """Initialise JAX (distributed under SLURM, x64 as the config says), then
    main.run(cfg, task, U0, ic), as `python main.py` does. Returns its RunResult.
    U0 / ic: a user initial state (luna/ic_generation/user_ic.py). For run
    scripts (`script_main`) and `luna.launch`."""
    from luna.setup.jax_init import initialize_jax_distributed
    initialize_jax_distributed(enable_x64=cfg.enable_x64)
    if banner:
        import jax
        if jax.process_index() == 0:
            from luna.utils.logo import print_LUNA_banner
            print_LUNA_banner()
    from luna.main import run
    return run(cfg, task, U0=U0, ic=ic)


def args(argv=None, **defaults):
    """A benchmark script's parameters from `name=value` argv, typed by their
    defaults: `args(nx=64, reps=5)` -> {"nx": ..., "reps": ...}. For a script
    with no Config; one that builds a Config uses `script_main`."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    _, kv = split_args(argv)
    out = dict(defaults)
    for name, text in kv.items():
        if name not in defaults:
            raise _unknown(name, list(defaults))
        out[name] = parse_value(text, _kind(inspect.Parameter.empty, defaults[name]), name)
    return out


def script_args(config_fn, argv):
    """`argv` split for a run script: (its parameters, the Config overrides).
    A name that is an argument of `config_fn` is a script parameter, parsed
    by the type of its default; anything else must be a Config field."""
    _, kv = split_args(argv)
    sig = inspect.signature(config_fn)
    params, rest = {}, {}
    for name, text in kv.items():
        if name in sig.parameters:
            p = sig.parameters[name]
            params[name] = parse_value(text, _kind(p.annotation, p.default), name)
        else:
            rest[name] = text
    spec = config_fields()
    for name in rest:
        if name not in spec and not name.startswith(PROBLEM):
            raise _unknown(name, list(sig.parameters) + list(spec))
    return params, overrides(rest)


def task_options(task, argv):
    """`argv` split for an adjoint task: (the task with its options bound, the
    rest). The options are its keyword-only arguments, typed by their defaults."""
    import functools
    opts = {n: p for n, p in inspect.signature(task).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY}
    picked, rest = {}, []
    for tok in argv:
        name = tok.split("=", 1)[0].strip()
        if "=" in tok and name in opts:
            picked[name] = parse_value(tok.split("=", 1)[1],
                                       _kind(inspect.Parameter.empty, opts[name].default), name)
        else:
            rest.append(tok)
    return (functools.partial(task, **picked) if picked else task), rest


def script_main(config_fn, argv=None, report=None, task=None):
    """A run script's main: parse the command line, build the config, run.
    `report(result)` prints whatever the script measures. With `task` (an
    autodiff_mode run), the task's keyword options come off the command line
    too and main.run hands the prepared run to it."""
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if task is not None:
        task, argv = task_options(task, argv)
    params, over = script_args(config_fn, argv)
    cfg = apply(config_fn(**params), over)
    res = launch(cfg) if task is None else launch(cfg, task=task)
    if report is not None:
        import jax
        if jax.process_index() == 0:
            report(res)
    return res
