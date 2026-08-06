# Gravity solver: scalability + precision plan

Goal (standing): **MG-Poisson gravity, speed AND accuracy, on 1 and N GPUs.**
Everything here is measured on the real l9 IC (512³, 134,217,728 particles),
gated on the δ_max trace, with the residual reported next to every timing.

Baselines to beat — all measured 2026-08-06, real l9, 512³, 1 GPU, all-FP32:

| | LUNA | Horowitz (MG) |
|---|---|---|
| full step | **34.6 ms** / resid 2.96e-3 | 43.3 ms / resid 4.96e-3 |
| deposit | 15.25 ms | 9.12 ms ← our 1.66× gap |
| solve | 7.14 ms | 9.50 ms |
| gradient | 1.95 ms | 1.98 ms |
| gather | 10.02 ms | 16.03 ms |

Multi-GPU, 512³, matched FP32: LUNA 33.6 → 27.4 ms (1→2 GPU, 61% eff);
Horowitz 135.1 → 74.3 ms (91% eff, but 3.1× penalty for going multi-device).

δ_max gate (8 steps): 1.8074, 1.8366, 1.8205, 1.8289, 1.8529, 1.9407,
2.0276, 2.0914. Any change moving these is wrong.

---

## Phase 0 — precondition (blocks everything else)

**P0.1 · R1 · fix deposit dtype casting + fuse the 8 scatters** — LOW
`particle_mesh_blocks.py:188-206` casts *after* computing `f`/`i0`/`t`, so the
FP32 path still streams 3.2 GB of FP64 positions and recomputes index
arithmetic 8×. Survey is explicit: *until this is fixed, no measurement
isolates the atomic path.* Verify `t` precision against an FP64 reference.

**P0.2 · contention-ceiling experiment** — ~20 lines, one srun
Time the real deposit, then the identical kernel with collision-free indices
(wrong answer, accepted). The delta upper-bounds **every** atomics-targeting
optimization at once. **This is the gate for Phase 2.**

---

## Phase 1 — scalability (mission-critical)

The particle-parallel path replicates the mesh: solve is 6.97 ms at 1 GPU
*and* at 2 GPUs, by construction. It cannot scale. The blocks path shards the
solve properly (V-cycle 18.1 → 12.9 → 5.7 ms at 1/2/8 GPUs = 3.2×; weak
scaling ~78%/doubling) but its full step is 62.5 vs 27.4 ms at 512³/2 GPU.
**The gap is in particle handling, not the solve.**

**P1.1 · decompose blocks vs pp stage-by-stage** *(in flight)*
Three hypotheses: padding overhead (`dm_capacity_factor=1.5` → 33% of every
buffer is placeholders still processed), particle ordering/locality
(`assemble_blocks_sharded` reorders; CIC-corner resort key only just fixed),
per-step glue (split jits + mid-step ROCm sync vs pp's single fused graph).
Must also resolve the **8× disagreement** for the same deposit function
across harnesses (15.25 vs 134.51 ms) — until then no deposit number is
trustworthy.

**P1.2 · close the blocks-path gap** — driven by P1.1 findings.

**P1.3 · find the crossover** where blocks beats pp. Needs 4+ GPUs.
viper cannot schedule this (282/298 allocated) → **JUPITER**.

**P1.4 · weak-scaling ladder on real ICs** — l9@1 GPU vs l10@8 GPU at
matched per-GPU load. Also JUPITER. Note: the earlier ladder used random and
sine fields, which violates the real-IC rule; it must be redone on l9/l10.

**P1.5 · coarse-level agglomeration** — measured only 4% from `agg_n` sweeps
at 8 GPUs (128 best), so this is *not* the lever it was assumed to be.
Coarse levels below 64³ are launch-latency bound (~0.07-0.10 ms/kernel flat,
5 levels + 0.497 ms coarsest ≈ 21% of the solve). Real fix is fewer, larger
kernels on the coarse ladder — deferred until P1.1/P1.2 land.

---

## Phase 2 — deposit (our named 1.66× weak spot)

Gated on P0.2. If the contention ceiling is small (<20% of deposit time),
**abandon all atomics-targeting work** and go straight to P2.3/P2.4.

**P2.1 · R3 · int32 fixed-point accumulation** — MEDIUM
Fastest option tested, and **bitwise reproducible**: FP32 deposit differs in
7% of cells under mere particle reordering; integer is identical every time.
Exact-mass construction conserves mass to the last bit. Prior art:
QuanTaichi (SIGGRAPH 2021) for MPM grid nodes, 1.4× — credit them; LUNA's
novelty is first use in cosmological PM + the exact-mass construction.

**P2.2 · R1b · `indices_are_sorted`** — **DO NOT IMPLEMENT** while CONTESTED.
Open XLA bug openxla/xla#22233 shows ~95× regression for scalar-slice
scatters (exactly CIC). Best case ~8× against a documented 95× downside.

**P2.3 · R2/R2b · cell-ordered storage + cheaper resort** — ordering is worth
a measured 4× on the deposit; the resort itself costs 9.4 ms.

**P2.4 · R7 · LDS privatization** — **argued down, do not pursue.** SC20
measured tuned privatization at **0.91× (a slowdown)** on hardware with
native float atomics; WarpX reports no shared-memory scheme ever beat plain
global atomics. **MI300A has native atomics.**

---

## Phase 3 — integer positions (R4)

**The single change that gives speed *and* accuracy.** VERIFIED in GADGET-4
(`src/data/dtypes.h:38`, `uint32_t MyIntPosType`, with uint64/uint128
variants), PKDGRAV3, CUBE; pmwd's `pmid`+`disp` is the JAX cousin.

Gains:
- **12 B/particle instead of 24** — 1.6 GB saved at 134M particles, and
  1.6 GB/step of traffic in every kernel that touches positions
- **uniform 5.82e-6 kpc/h resolution** — 335× finer than FP32 at the box
  corner, and unlike FP32 it does not degrade with distance from origin
- **exact periodic wrapping** by unsigned overflow (structurally immune to
  the CIC wrap bug class we hit); **exact minimum-image differences**
- **CIC index + weight by shift/mask** — measured *not slower*, marginally
  faster
- lossless reversible whole-box shifts (GADGET-4 `RANDOMIZE_DOMAINCENTER`),
  possibly interesting for a differentiable code

Reachability: pure JAX (`uint32` arithmetic, `>>`, `&`, wraparound add).
Watch `lax.bitcast_convert_type` vs `.astype` for signed reinterpretation.

Risk: touches the particle data model, IC generation, snapshot I/O and halo
exchange — the largest *engineering* item, though each piece is simple.
Drift quantization is unbiased and 4 orders below a cell; needs a
convergence check.

**P3.1 · cheaper 80% variant first** — pmwd's split: `int32` cell id +
`float32` sub-cell displacement (`pmwd/particles.py:31-42`). Gets the
uniform-precision benefit without touching the periodic-wrap machinery.
JaxPM ships this as `cic_paint_dx`. **Do this before the full uint32 model.**

**P3.2 · full uint32 positions** — only if P3.1 measures well.

---

## Phase 4 — config surface (R6)

**P4.1 · `gravity_precision = 'double' | 'mixed' | 'single'`**
Replaces the scattered switches: `cfg.dm_mixed_precision`, `cfg.dm_fp32`,
and the `LUNA_MG_MIXED` env var in `poisson_multigrid.py`.

**Real bug to fix here:** `cfg.dm_fp32` is wired **only** into
`runner_cosmo_hydro.py:468` — it is a **no-op in `runner_cosmo_dm.py`**. So
the all-FP32 mode that won the head-to-head (34.6 vs 45.0 ms, −23%, at
+0.8% residual) **is not reachable by config today.**

**P4.2 · integer-position config surface** — propose a minimal one, do not
invent a sprawl of flags.

---

## Phase 5 — integrator (see INTEGRATOR_SURVEY.md)

LUNA is pure PM with a **global** timestep: the most-accelerated particle
sets dt for all 134M. Measured ~7000 steps to z=0 at 512³ with dloga falling
to 2.8e-4, where the expansion cap alone would allow 0.02 — a ~70×
oversampling for most particles.

Hierarchical/individual timesteps are natural for TreePM (short-range force
is separable and sub-cyclable) but a pure PM code needs all particles
synchronised to build the density field. Candidates that *do* work for pure
PM: GADGET-4-style **integer time** (exact, no drift — cheap, matches the
integer-position philosophy), COLA/temporal-COLA, FastPM's modified kick/drift
factors preserving linear growth at large dt, higher-order symplectic
integrators. Ranked list in the survey.

---

## Ordering summary

```
P0.1 (R1 dtype fix)  ──┬──> P0.2 (ceiling)  ──> [gate] ──> Phase 2
                       └──> Phase 1 (scalability) ──> JUPITER for P1.3/P1.4
P3.1 (pmwd-style split) ──> P3.2 (full uint32)
P4.1 (gravity_precision) — independent, unblocks the measured 23% FP32 win
```

**Rules for every item:** real l9/l10 ICs only; δ_max gate before any timing
is quoted; residual next to every ms; one harness
(`horowitz_bench/h2h_l9_step.py` methodology); no perf conclusion from a
small or synthetic run; nothing committed without a passing gate.
