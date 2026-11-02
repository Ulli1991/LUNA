# Hydro kernel paths: which branch runs, and what it is correct for

The MUSCL-Hancock step has five interacting switches (`spatial_dim`, `order`,
`boundary`, `hydro_fused`, `hydro_recon32`) plus one implicit one (single
device vs `shard_map`). Nothing in the code states which combinations are
valid, so this file does. Written 2026-09-18 from
`luna/time_integration/integrators.py` and `luna/hydro/muscl_hancock.py`; update it when
either changes.

XLA is the default and the differentiable reference; Pallas (`hydro_kernel`) is the opt-in
production fast path that must agree with it (`docs/ARCHITECTURE.md`, two kernel paths).

## Where each switch is resolved

`luna/time_integration/integrators.py`:

| where | expression | effect |
|---|---|---|
| `create_muscl_hancock_modular_step` | `exact_wrap = (boundary == 0) and (not use_shardmap)` | single-device periodic gets the telescoping seam |
| `_hydro_paths` | `fused = fused and order >= 1 and spatial_dim == 3 and boundary in (0, 1)` | fused silently falls back outside 3D / BC 0-1 |
| `_hydro_paths` | `recon32 = recon32 and order >= 1 and spatial_dim == 3 and not fused` | mixed precision and fused are MUTUALLY EXCLUSIVE |
| `_step_inner` | `pad_halo_fn(W, boundary, spatial_dim, width=2 if fused or wide else 1)` | halo width follows `fused`/`wide` |

Two of these fail SILENTLY: asking for `hydro_fused=True` in 2D, or for
`hydro_recon32=True` together with `hydro_fused=True`, gives you the other
path with no warning.

From plain arrays, one device, no mesh (`pad_halo_fn=None` is the local `jnp.pad` halo,
`luna/utils/halo_local.py`, bitwise the one-device `shard_map` halo):
`dt_fn, step = create_muscl_hancock_modular_step(None, gamma, 1, 0, dx, dx, dx, 1/dx, 1/dx, 1/dx, 0, 3, 0.4); U = step(U, dt_fn(U))`

## What each path does at a face

"Telescopes" means the two shards sharing a face compute the *same* flux for
it, so the flux sum cancels exactly and mass/momentum/energy are conserved to
machine precision across the seam.

| path | interior face | seam face (shard/periodic) | order at seam | telescopes |
|---|---|---|---|---|
| `first_order_divergence` (order 0) | cell-centred pair | cell-centred pair | 1 | YES - piecewise-constant needs no reconstruction, so both sides agree by construction |
| 3D unfused, `exact_wrap` (1 device, periodic) | two reconstructed states | flux built ONCE from both reconstructed states, used at both ends | 2 | YES |
| **3D unfused, `shard_map`** | two reconstructed states | reconstructed state on one side vs **raw cell-centred halo** on the other (`_muscl_hancock_3d`, the `else` branch of `_sweep`) | **1** | **NO** |
| 3D fused (2-ghost) | two reconstructed states | reconstructs into the ghost ring, so both sides are reconstructed and both shards compute identical inputs | 2 | YES |
| **1D/2D, `boundary == 0`, `shard_map`** | two reconstructed states | this shard's **OWN opposite edge** (the 1D/2D branches of `muscl_hancock.py`) | - | **NO - WRONG RESULT**, the shard wraps onto itself instead of its neighbour |
| 1D/2D, `boundary != 0` | two reconstructed states | `W_pad` edge = the BC ghost | - | n/a, this is a physical boundary and the ghost carries the BC |

The two bold rows are audit items 3 and 4. Both are guarded by warnings in
`Config.validate()`, not by errors.

## Practical consequences

- **Every multi-GPU run to date used the unfused 3D path**, i.e. first order
  and non-telescoping in the one cell layer next to each shard face
  (~1.2% of cells at a 512^3 tile). For accuracy an O(dx) error on an O(dx)
  fraction still contributes O(dx^2) globally, so measured convergence order
  usually survives. The conservation violation is the part that does NOT
  cancel: it accumulates as a net drift. **Unmeasured - measure it before
  quoting a number.**
- **Mixed-precision numbers are necessarily unfused**, by line 119. Every
  recon32 rung on viper and JUPITER therefore carries the degraded seam, and
  we do not know what recon32 costs or gains WITH a conservative seam, because
  the two have never run together.
- **Throughput ladders are still valid as throughput.** The seam changes which
  arithmetic happens in one cell layer, not how much work is done. They do
  need a kernel label: measured 2026-09-18 at 8 GPUs, FP64, mesh (2,2,2),
  fused vs unfused is +4.6% at a 256^3 tile (7723 vs 7386 Mcell/s) and -3.2%
  at a 384^3 tile (7949 vs 8210). So "fused is ~4% slower" is wrong as a
  general claim; it is tile-dependent.
- **Separately from all of the above**, FP64 at a 512^3 tile is corrupted by
  the XLA uint32 offset wrap (`docs/history/JUPITER.md:44`, and sec.16 of
  `docs/history/AUDIT_2026-09-16.md` for the confirming census). That is an indexing bug,
  not a seam issue, and it is why no plain-vs-fused comparison exists at the
  production tile size. FP32 and recon32 escape it because their four-byte
  elements put the offset at 2^31.

## Open question

At 256^3 and 384^3 tiles the fused and unfused kernels produced **bitwise
identical** global sums after one Sedov step on a (2,2,2) mesh, with the blast
sitting on the eight-block corner where seam gradients are largest. That is
hard to reconcile with a seam treatment that differs, and a non-telescoping
flux is exactly what should perturb a global sum. The comparison was of sums
only; the decisive check is a POINTWISE max difference between the two states,
which the census does not compute. Do not cite the bitwise agreement as
evidence that the seam is harmless until that is run.

## Intended direction (not yet done)

Keep both kernels, but make their arithmetic consistent: give the unfused path
the same two-ghost pad so it reconstructs the ghost ring and builds the seam
flux from two reconstructed states, exactly as fused does. Halo width is
orthogonal to reconstruction precision, so this keeps recon32 available and
keeps 1D/2D and boundary types 2/4 on a path that is no longer degraded. The
alternative, one extra face-sized `ppermute` of the reconstructed face state
per axis, communicates the same amount and skips the duplicated ghost
reconstruction.

## 3-D diode (BOUNDARY 4) on the wide path (2026-09-28, branch wall-m3)

The 3-D diode now takes the wide path (and, on a GPU with minmod + HLLC, the
fused Pallas divergence). Its two-layer ghost is the zero-gradient one with
the normal velocity clamped against inflow in BOTH layers
(`halo_exchange_shardmap._fix_face(width=2)`). Before, it stayed on the narrow
path, so a shard seam built its flux from different states on the two sides
(audit 2026-09-26 M13): a 2-shard run differed from one device. Now 2 and 8
shards equal one device to 0 in `tests/test_outflow_boundary.py`. Periodic
and zero-gradient paths are untouched. 1-D/2-D diode stays narrow.

## Reflective (BOUNDARY 2) on the wide path (2026-10-06)

The narrow path paired the raw cell-centred ghost with the reconstructed,
predicted interior state at a wall face, so HLLC let mass through: |dM|/M
4.9e-4 (1D 256), 4.4e-3 (2D 32^2), 2.9e-3 (3D 16^3), and Woodward-Colella lost
2e-4 of its mass at 1024. BOUNDARY 2 now takes the wide path with two mirrored
ghosts (`halo_local`: `jnp.pad(mode="symmetric")`; `halo_exchange_shardmap`:
the edge slab flipped on boundary shards), normal velocity negated in both. The
wall face then pairs exact mirror states: mass, energy and momentum to
round-off, and N shards equal one device to 0 in 1D-3D
(`tests/test_hydro_wall_conservation.py`).

## Pallas divergence (`hydro_kernel`)

`luna/hydro/muscl_hancock_pallas.py` runs the wide (2-ghost) step in 1-D, 2-D or 3-D as Triton
kernels. Each program owns a tile of cells and computes, in registers, the limited slopes, the
half-step predictor at the cell and its 2 ndim neighbours, the 2 ndim face fluxes and dU. The two
cells sharing a face compute it from the same inputs with the same code, so the fluxes still
telescope. The per-cell code is XLA's: `slopes.slope_limiter` (minmod, MC, van Leer),
`jacobian.primitive_jacobian_rows_*` and `fluxes._hllc_core` / `_hll_core`, so under Triton the
kernels equal the XLA wide step **bitwise** in every dimension, limiter, solver and precision
(A100, 2026-10-12/14, `tests/test_hydro_pallas_matrix.py`); the CPU interpreter rounds a few cells
differently (XLA CPU contracts FMAs per expression), <= 2e-16 (FP64) / 6e-8 (FP32) of max|dU|.
Precisions: the state's dtype (FP64 or FP32) or, in 3-D only (as XLA), recon32 (slopes,
predictor and face states in FP32, cast up before the solver; fluxes and dU in FP64). Passive
scalars (HLLC only, their flux is upwind of the contact) in every dimension.

`hydro_kernel='pallas'` refuses, never falls back: `unsupported()` names what it cannot run,
`select()` raises `NotImplementedError` with it in the step factory, and `Config.validate()` says
the same before any work. Refused: autodiff (no VJP; the adjoint differentiates XLA), ORDER=0,
`hydro_fused=True` (XLA's fused kernel), `limiter_tangent_eps` (an autodiff tangent), recon32
outside 3-D, the narrow one-ghost path (BOUNDARY 4 in 1-D/2-D), HLL with passive scalars, and AMR
(its runner's hydro is XLA). `'auto'` takes Pallas on a GPU where it can and XLA elsewhere,
quietly. Before 2026-10-12 `'pallas'` silently ran XLA for 1-D, 2-D and MC/van Leer.

On NVIDIA (no passive scalars, 2-D and 3-D) `divergence` runs two kernels (`split=None` resolves
to that; `split=False` forces the fused one): pass A writes the predictor W_half once per cell on
the (N+2)^ndim region (its last tile shifted back to stay in bounds), pass B recomputes the slopes
from W_pad and does the faces and dU. Same arithmetic, bitwise under Triton. +ndim+2 fields of
(N+2)^ndim in the reconstruction dtype; 3-D full step 1.15-1.3x the fused kernel on an A100, Sedov
512^3 FP64 835 -> 1108 Mcell/s on one GPU; 2-D divergence 1.2x (FP64) / 1.04x (FP32) at 4096^2.
1-D stays fused: split is 0.93x (FP64) / 0.6x (FP32) at 2^24 cells (memory-bound). AMD and passive
scalars keep the fused kernel: unmeasured there. Tiles (`FUSED_TILE`, `SPLIT_TILE`) give one cell
per thread on NVIDIA (num_warps = cells / 32); every tile is bitwise the same.

Production step on one A100 (2026-10-14, Sedov-like, BOUNDARY 1, Mcell/s Pallas / XLA,
`tools/bench_hydro_roofline.py` with DIM, LIMITER; `docs/history/CODE_NOTES.md`):

| | FP64 minmod / MC / VL | FP32 minmod / MC / VL |
|---|---|---|
| 1-D 2^24 | 3237/3424, 3014/3372, 2886/3358 | 5927/4392, 6225/4488, 6016/4590 |
| 2-D 4096^2 | 2082/1985, 1910/1882, 1526/1955 | 4466/3050, 4065/3016, 4359/3026 |
| 2-D 8192^2 | 2067/1964, 1913/1888, 1618/1943 | 4593/2856, 4339/2817, 4299/2843 |
| 3-D 256^3 | 1194/735, 1064/723, 886/709 | 2334/1341, 2269/1363, 2547/1341 |

3-D recon32 256^3: 1414/906, 1411/901, 1417/902. In 1-D and 2-D FP64 Pallas is no faster than
XLA (0.8-1.05x; van Leer's FP64 division, recomputed for 3 slopes per axis in pass B, costs most);
its gain there is FP32 (1.3-1.6x) and memory.

## Opt-in robustness switch: `preserve_face_states` (2026-10-19)

Default False; off, the step traces no new op (bitwise main's, `docs/history/CODE_NOTES.md`).

- **`preserve_face_states`**: mini-ramses's positivity floors with `hydro_smallr`, `hydro_smallc` (code
  units): floored primitives (`conversions.cons_to_prim_floored`, also for the CFL dt), the face trace
  guard and the Riemann-input floors (`slopes.ramses_face`, switched by `slopes.face_guard_fn`). Every dense
  path: 3-D narrow/wide/fused, face mass fluxes, `domain_face_fluxes_3d`, 1-D/2-D, Pallas fused and split
  (`_face` with the cell centres `W(p)`, `W(q)`). U is not floored. Refused with AMR and ORDER=0.

## Unphysical face states (audit C8)

Every dense path (XLA 1-D/2-D/3-D, fused, wide, narrow; Pallas fused and split) applies
`slopes.face_rule` to each predicted face state: rho <= 0 or p <= 0 becomes NaN (default, the run stops,
`setup/face_report.py`) or, with `allow_first_order_faces=True`, the cell-centre state. The RAMSES floors
(`preserve_face_states`) come after it. With `face_report=True` the step also returns the per-shard count of
such face states, the count of floored ones, and the first 8 cells per shard (global indices); XLA records the
masks (`slopes.recording_faces`); Pallas writes an int8 bitmask per cell (bit 2 d: the +d face, 2 d + 1: the -d
face). The AMR face paths (`amr/hydro.block_face_fluxes`, replicated and sharded) apply the same rule
(`AMRConfig.allow_first_order_faces`) and send a report through `jax.debug.callback`, only on a call with such a
face; the AMR runner reads it after each chunk (`setup/face_report.flush_amr`). The differentiable rollout cannot
report: `Config.validate` refuses `allow_first_order_faces` with `autodiff_mode`, so a refused face there is a
NaN loss.
