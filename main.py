"""`python main.py [config.py] [field=value ...]` -- the HPC entry point.

A launcher, not a module: the code lives in src/luna/main.py, and this file
exists so the repo-root workflow is unchanged. It imports `luna`, which means
the package has to be installed -- `pip install -e .` once per environment.
That is the point of the src layout: there is no `luna/` beside this file for
an import to pick up by accident, so what runs is always what was installed.

See src/luna/setup/cli.py for the command line and docs/TUTORIAL.md for the
walk-through.
"""
from luna.main import main

if __name__ == "__main__":
    main()
