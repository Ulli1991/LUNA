# LUNA

Differentiable astrophysical hydrodynamics in JAX (no MHD yet), one CPU to thousands of GPUs (4096 on JUPITER).

## Standing rules

- When in doubt, check the code; cite `file:line`; find the fix before calling it fixed.
- Refuse, don't fall back: `src/luna/setup/run_kind.py:31`.
- Refactors are bitwise; N shards equal one device: `tests/test_hydro_hard_sedov_offaxis.py:269`.
- Quiet is not correct: NaN dt becomes -inf, `src/luna/time_integration/integrators.py:27`.
- Measure before claiming: hardware, size, precision (`docs/history/CODE_NOTES.md`).
- One-line docstrings: `src/luna/time_integration/integrators.py:12`.
- Commits: one short plain sentence (~50 characters), no prefix; details go in the PR or docs.

## Where to read next

- Running: `scripts/validation/CLAUDE.md`, `docs/TUTORIAL.md`, `docs/CONFIG.md`.
- A new module: `src/luna/CLAUDE.md`, `docs/ARCHITECTURE.md`.
- Kernel performance, autodiff: `src/luna/hydro/CLAUDE.md`, `docs/AUTODIFF.md`.
- Your machine, your preferences: `CLAUDE.local.md` (not committed).
- `docs/history/` is dated; the code and `docs/` win.

## Commands

```bash
pip install -e ".[test]"                    # jax 0.10.x, CPU
python tools/run_tests.py                   # fast tier, ~2 min, as CI
python tools/run_tests.py --devices 2       # + halo paths
ruff check .
python main.py my_run.py Nx=256             # run, overriding fields
```

No jax 0.10 wheel for your machine: `tools/docker/Dockerfile`.
