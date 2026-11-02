"""LUNA: a differentiable JAX hydrodynamics code.

    from luna import Config, run, launch
    res = launch(replace(Config(), test_problem="Sedov-Blast-3D", Nx=128))

`run(cfg)` is the entry point (mesh, initial conditions, physics, loop);
`launch(cfg)` initialises JAX distributed from the config first, which is what
the command line and the run scripts call. Your own initial state:
`launch(cfg, U0=U)` (conserved, cfg's shape) or `launch(cfg, ic=f)` with
`f(x, y, z, cfg)` per shard; `conserved(rho, v, p, gamma)` builds U
(luna/ic_generation/user_ic.py, docs/ADDING_A_PROBLEM.md). Imports are lazy: `import luna`
costs nothing until one of these is touched, so a config file can import
`luna.units` without pulling in JAX.
"""
__all__ = ["Config", "run", "launch", "main", "conserved", "__version__"]


def __getattr__(name):
    if name == "Config":
        from luna.template_config import Config
        return Config
    if name in ("run", "main"):
        import luna.main as m
        return getattr(m, name)
    if name == "__version__":
        from importlib.metadata import version
        return version("luna-hydro")      # the installed distribution's version (pyproject.toml)
    if name == "conserved":
        from luna.ic_generation.user_ic import conserved
        return conserved
    if name == "launch":
        from luna.setup.cli import launch
        return launch
    raise AttributeError(f"module 'luna' has no attribute {name!r}")
