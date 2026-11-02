---
name: Bug report
about: LUNA crashed, refused a run it should accept, or gave a wrong result
labels: bug
---

**What happened**
What you ran and what went wrong. For a wrong result: what you expected and the reference
(exact solution, paper, other code) it disagrees with.

**To reproduce**
The run file (or its non-default fields) and the exact command:

```
luna my_run.py Nx=256 ...
```

**Output**
The end of the log, including the error message or traceback (and `run.json` if a run started).

**Environment**
- LUNA version or commit (`python -c "import luna; print(luna.__version__)"`, or `git rev-parse HEAD`):
- jax / jaxlib version and backend (`python -c "import jax; print(jax.__version__, jax.devices())"`):
- Hardware (CPU / GPU model and count) and precision (`enable_x64`):
- Python version and OS:
- Number of processes and `MESH_SHAPE`, if more than one device:
