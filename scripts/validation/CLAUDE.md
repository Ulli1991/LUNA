# Running LUNA

Runs deliver results, not library changes. A refused setting is reported, not worked around.

## Standing rules

- When in doubt, check the code; cite `file:line`; find the fix before calling it fixed.
- Refuse, don't fall back: `src/luna/setup/run_kind.py:31`.
- Refactors are bitwise; N shards equal one device: `tests/test_hydro_hard_sedov_offaxis.py:269`.
- Quiet is not correct: NaN dt becomes -inf, `src/luna/time_integration/integrators.py:27`.
- Measure before claiming: hardware, size, precision (`docs/history/CODE_NOTES.md`).
- Compare against a reference (`src/luna/hydro/exact_riemann.py`, `src/luna/hydro/exact_sedov.py`, `scripts/validation/metrics.py`), not by eye.

## How

- A test problem: `python -m scripts.validation.runs.kh_run nx=512 MESH_SHAPE=2,2,2`.
- An own setup: copy `src/luna/template_config.py`, `python main.py my_run.py`;
  fields as `name=value`, problem parameters as `problem.<name>=value` (`docs/CONFIG.md`).
- Problems: the keys of `_IC_REGISTRY`, `src/luna/ic_generation/ic.py`.
- Clusters: one process per GPU under `srun`, `submit/examples/`; the multi-process rules
  in `docs/RUNNING.md`.
- Where time goes: `timing_every=N`, `tools/step_breakdown.py`.
- Output: HDF5 in `output_dir`, read with `luna.file_io` (`docs/TUTORIAL.md` section 4);
  `run.json` records settings and code revision. Figures: `scripts/validation/plots/`, after the matching `runs/` script.
