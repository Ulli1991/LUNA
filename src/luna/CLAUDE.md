# Developing in luna/

A step: stages' `before` in order, `U = step(U, dt)`, `after` in reverse, projections
(`docs/ARCHITECTURE.md`).

## Standing rules

- When in doubt, check the code; cite `file:line`; find the fix before calling it fixed.
- Refuse, don't fall back: `src/luna/setup/run_kind.py:31`.
- Refactors are bitwise; N shards equal one device: `tests/test_hydro_hard_sedov_offaxis.py:269`.
- Quiet is not correct: NaN dt becomes -inf, `src/luna/time_integration/integrators.py:27`.
- Measure before claiming: hardware, size, precision (`docs/history/CODE_NOTES.md`).
- One-line docstrings: `src/luna/time_integration/integrators.py:12`; stories go to `docs/history/CODE_NOTES.md`.
- Don't sort imports (environment setup precedes JAX).

## Plugging in

- Source term: `setup_<name>(cfg, ...)` in `src/luna/sources/`, None when off; a stage from
  `src/luna/sources/stages.py`; appended in `src/luna/setup/assemble.py:build_stages` (order is step order).
- Test problem: an IC module with `Params` (`src/luna/ic_generation/gresho_vortex.py:11`), an entry
  in `_IC_REGISTRY` (`src/luna/ic_generation/ic.py`).
- Config field: `src/luna/template_config.py` + one-line comment, checks in `Config.validate()`,
  then `python scripts/gen_config_md.py`.
- New test file: `python tools/run_tests.py --tier full --update-costs`.
- Sharded data: local slices, jit arguments not closures, `src/luna/utils/mesh_reduce.py` on every rank.
- Energy is `U[-1]`, momenta `U[1:1+ndim]`; never `U[4]`.

## MHD (branch `mhd`)

UCT-HLLD (Mignone & Del Zanna 2021). Layout, energy always last:
1D `rho, mx, my, mz, Bx, By, Bz, E`; 2D adds `Bfx, Bfy`, 3D `Bfx, Bfy, Bfz` before E;
each cell stores its lower face. Plumbing → `U[4]`/`U[1:-1]` sites onto a layout
descriptor (bitwise) → `src/luna/mhd/` kernel (to be written) on the wide path. Stages opt in; others refuse MHD.
Formulas from the paper or PLUTO's `ct_emf`, never from memory.
