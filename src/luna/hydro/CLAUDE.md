# The hydro kernel

The hot path. Draft of 2026-10-06, checked against the code that day; review pending.

## Standing rules

- When in doubt, check the code; cite `file:line`; find the fix before calling it fixed.
- Refuse, don't fall back: `src/luna/setup/run_kind.py:31`.
- Refactors are bitwise; N shards equal one device: `tests/test_hydro_hard_sedov_offaxis.py:269`.
- Quiet is not correct: NaN dt becomes -inf, `src/luna/time_integration/integrators.py:27`.
- Measure: hardware, precision, tile, mesh, path, median after warm-up. Throughput is
  tile-dependent (fused vs unfused +4.6% at 256^3, -3.2% at 384^3), never generalise.
- Every path stays differentiable; the autodiff tests stay green.

## Paths

- Resolved in `src/luna/time_integration/integrators.py:_hydro_paths`; the table is
  `docs/HYDRO_KERNEL_PATHS.md`, updated in the same commit as the selection.
- wide (2-ghost): seams telescope. Narrow: first order, non-telescoping at seams.
- Pallas `src/luna/hydro/muscl_hancock_pallas.py`: 1-D/2-D/3-D, minmod/MC/van Leer, HLLC or HLL, wide;
  FP64, FP32, recon32 (3-D). Per-cell code is XLA's (`slopes.slope_limiter`, `jacobian.primitive_jacobian_rows_*`,
  the `fluxes` cores): bitwise with XLA under Triton. `unsupported()` names what it cannot run (autodiff,
  ORDER=0, hydro_fused, narrow path, AMR via `Config.validate`); `select('pallas', reason)` raises, never
  falls back; 'auto' takes XLA there. On NVIDIA without scalars, 2-D/3-D are two kernels (predictor once
  per cell, then faces); 1-D, AMD and scalars fused.
- recon32 excludes fused (`_hydro_paths`).

## Traps

- Barriers in `src/luna/hydro/muscl_hancock.py:_muscl_hancock_3d`: after-x and the sweep, 25.3 temporaries of N^3;
  dU-only 46.3; +after-y 23.3 at -3.8%; after-y only 37.3 at +4.8% (MI300A FP64 512^3). On an A100 the after-y
  and W_half barriers cost 5.5% in FP64, FP32 and recon32 at the same peak memory (FP32 512^3 30.2 GiB, 2026-10-10),
  so they are gone; none at all is +36% in recon32 at 384^3 but +18% memory, and OOMs FP32 at 512^3.
  Re-measure if touched.
- Pallas launch shape: on an A100, Triton's default num_warps=4 on the 1x4x64 tile was 2.9x slower in
  FP64 than 8 (bitwise equal); `_launch` sets one cell per thread on NVIDIA only, untested on MI300A
  (`docs/history/CODE_NOTES.md`, A100 tuning). Sweep tile x num_warps per GPU before quoting Pallas.
- 1-D/2-D FP64 Pallas is 0.8-1.05x XLA on an A100 (van Leer slowest); the gain is FP32 (1.3-1.6x).
- CUDA, FP64, 512^3 tile: field 4's offset is 2^32 and wraps to field 0, corrupting
  PLM y-fluxes (`docs/history/JUPITER.md`).
- Audit C8: a predicted face state with rho <= 0 or p <= 0 (MC near strong shocks) is refused by
  default: it becomes NaN and the runner stops naming step, t and cell (`src/luna/setup/face_report.py`).
  `allow_first_order_faces=True` takes the first-order (cell-centre) state there and logs each such step
  (`src/luna/hydro/slopes.py:face_rule`, XLA and Pallas). The step's face report (count per shard, first 8
  cells and their faces) is an extra output, `integrators._face_report`; physical faces are bitwise either
  way. AMR reports through `amr/hydro.report_amr_faces`; Pallas costs +1.2 % at 384^3
  (`docs/history/CODE_NOTES.md`).

## Measuring

`tools/bench_hydro_roofline.py` · `LUNA_FLOP_COUNT=1` · `timing_every=N` ·
`tools/step_breakdown.py` · `tools/fusediff.py` (does fusing change the answer).

## Gradients

`tests/test_autodiff_identities.py` (transpose, conservation, a corrupted adjoint must
fail) · `tests/test_autodiff_validation.py` (finite differences, floor 6.2e-7) ·
`tests/test_autodiff_sharded.py` (2 and 4 devices). Minmod's tangent at a flat state is first
order; `limiter_tangent_eps` smooths it (`docs/AUTODIFF.md`).
