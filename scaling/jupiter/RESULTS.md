# JUPITER GH200 results

## Hydro weak ladder — 512³/GPU tile, Sedov 3D (jobs 1263583/84/85, 2026-08-06)

Aggregate Mcell/s (steady plateau window); parallel efficiency vs the
single-GPU plateau (fp32 2193, recon32 2104).

| run | GPUs | nodes | fp32 | eff | recon32 | eff |
|------|------|-------|-----------|-------|---------|-------|
| 512³ | 1 | — | 2,193 | 100% | 2,104 | 100% |
| 1024³ | 8 | 2 | 17,423 | 99.3% | 15,863 | 94.2% |
| 2048³ | 64 | 16 | 134,209 | 95.6% | 121,680 | 90.4% |
| 4096³ | 512 | 128 | 1,018,884 | 90.7% | 936,555 | 86.9% |
| 8192³ | 4096 | 1024 | 7,986,287 | 88.9% | 7,616,118 | 88.4% |

- **1.02 Tcell/s aggregate fp32 at 512 GPUs, 90.7% weak efficiency.**
- Cold shard_map compile at 512 processes: **15.9 s** (viper's >12 min
  2-GPU cold compile does not carry over to CUDA).
- fp64-plain 3-step gate ran CLEAN at 8/64/512 GPUs; it NEVER RAN at
  4096 (job 1264426 cancelled after the timed variants) (no non-finite dt;
  single-GPU corruption trips it within 1 step) → the sharded compiles do
  not appear to hit the uint32 wrap. Caveat: gate relies on the corruption
  producing negative density; a one-step density census at 8 GPUs would
  make it airtight before a timed fp64-plain leg.
- probe6 (1263463): one-step density census CLEAN for fused-512³, 
  recon32-512³, and plain-516³ fp64 — all three workarounds are *correct*,
  not just NaN-free.
- viper 8-GPU regression (task #12 there): no sign of it — 8-GPU run is
  the most efficient. ROCm/interconnect artifact, as suspected.

## Hydro strong ladder — fixed global grid (jobs 1265676/1265738-41, 2026-08-06)

Step-90 plateau, aggregate Mcell/s. Efficiency vs the smallest-GPU anchor of
each curve (anchors shared with the weak ladder / 1-GPU roofline). fp64-plain
only exists where the per-GPU tile is <=256³ (uint32-wrap bug at 512³ tiles).

| grid | GPUs | tile | fp32 | eff | recon32 | fp64 |
|-------|------|------|---------|------|---------|--------|
| 512³ | 1 | 512³ | 2,193 | 100% | 2,104 | — |
| 512³ | 8 | 256³ | 15,523 | 88.5% | 14,715 | 9,573 |
| 512³ | 64 | 128³ | 67,214 | 47.9% | 62,101 | 44,912 |
| 1024³ | 8 | 512³ | 17,423 | 100% | 15,863 | — |
| 1024³ | 64 | 256³ | 118,916 | 85.3% | 108,872 | 70,782 |
| 1024³ | 512 | 128³ | 515,292 | 46.2% | 454,051 | 273,486 |
| 2048³ | 64 | 512³ | 134,209 | 100% | 121,680 | — |
| 2048³ | 512 | 256³ | 930,811 | 86.7% | 848,056 | 509,825 |
| 2048³ | 4096 | 128³ | 1,274,088¹ | 14.8%¹ | 2,512,293 | 2,305,445 |
| 4096³ | 512 | 512³ | 1,018,884 | 100% | 936,555 | — |
| 4096³ | 4096 | 256³ | 6,646,843 | 81.5% | 6,456,586 | 4,166,588 |

- Gap fill (jobs 1405898-908, 2026-08-18, /dev/shm env): all fillable
  recon32/fp64 cells measured. Pattern consistent everywhere: recon32 at
  95-97% of fp32, fp64 at 57-63% on compute-bound (>=256³) tiles and
  converging with fp32/recon32 on latency-bound 128³ tiles. 4096³/4096
  fp64 = 4.17 Tcell/s, the largest fp64 point of the campaign.
- ¹ NOT CLEANLY MEASURED (diagnosed 2026-08-19): the fp32 value at
  2048³/4096 (1.27 Tcell/s) sits 2x BELOW the 2026-08-18 recon32/fp64 at
  the identical config (2.3-2.5 Tcell/s, mutually consistent and
  latency-bound as expected). The cause is not precision — the fp32 entry
  differs from the other two in three ways at once (slurm.1265767.out vs
  slurm.1405902/904.out):
    * job 1265767 was **CANCELLED AT 2026-08-06T19:15:46 DUE TO TIME
      LIMIT** — it never finished, and 1.27 is the last window of a dying
      job. Its sibling 1265768 (4096³/4096 fp32) was cancelled the same way.
    * it ran in the pre-`/dev/shm` environment; the reruns used `[shm-env]`.
    * it ran steps=100 against the reruns' steps=60.
  Also, the tabulated 1.27 is the LAST window, while recon32/fp64 use the
  second-to-last per this file's own stated convention; applying that
  convention consistently gives 1.70 Tcell/s (still below recon32, so the
  confound remains). No fp32 rerun exists (checked on JUPITER 2026-08-19;
  nothing above job 1405908). Treat the 14.8% efficiency point as an
  artifact, not a measured scaling collapse. A rerun (~32k core-h) would
  give the first internally consistent 128³-tile row — at present no
  precision at 4096 GPUs is directly comparable to any other. NOTE: all
  three legs at this config swing 1.5-5x between perf windows, so a single
  "plateau" number is weak for this row regardless.
- Clean knee at the tile size: ~81-88% strong efficiency per 8× GPU step
  while the tile is >=256³ (incl. 4096³→6.65 Tcell/s at 4096 GPUs, jobs
  1265767/68), dropping to ~46-48% of ideal at 128³ tiles at moderate rank
  counts and to 14.8% at 4096 ranks (2048³), where 128³-tile halo exchange
  is latency-dominated — consistent with the 1-GPU roofline's 128³ point
  sitting at 63% of plateau.
- First submission of these runs crashed in jax.distributed.initialize
  (300 s DEADLINE_EXCEEDED): resubmitted seconds after scancel, workers
  raced the COMPLETING epilogue of the cancelled jobs. Clean-queue rerun of
  the identical config succeeded. LUNA_JAX_INIT_TIMEOUT env added.

## Hydro weak ladder, plain fp64 — 516³/GPU tile (jobs 1265632-35, 2026-08-06)

Non-po2 tile dodges the XLA uint32-wrap bug (see roofline note below), giving
a true fp64-plain column; same conventions as the fp32/recon32 ladder.

| run | GPUs | nodes | fp64 | per-GPU | eff |
|-------|------|-------|---------|-------|-------|
| 516³ | 1 | — | 1,333 | 1,333 | 100% |
| 1032³ | 8 | 2 | 10,045 | 1,256 | 94.2% |
| 2064³ | 64 | 16 | 79,446 | 1,241 | 93.1% |
| 4128³ | 512 | 128 | 624,175 | 1,219 | 91.5% |

- 1-GPU plain-fp64 anchor 1,333 Mcell/s at 516³ (vs fused-512³ 1,223) —
  recon32 (2,104) is ~1.6× fp64 while carrying fp64 state.
- All four runs finished inside 5-min walltimes.

## Gravity slab path, l10/8-GPU: where the 2.3 s/step goes (2026-08-16)

Rerun of the l9 1-GPU / l10 8-GPU weak pair (jobs 1392364/65) with the updated
runner (dm_host_every=10, chunked per-rank IC read, snapshots off): l10 slab
step is **2285.7 ms median device** — identical to the 2026-08-07 run
(2284.9), so the slowness is device-side, not host overhead. Per-GPU load is
matched to the anchor (134M particles + 1024x1024x128 slab per GPU), so ideal
weak scaling would be ~28-47 ms/step.

The 2026-08-07 slab-kernel bench (job 1274048, slab_bench_nx1024_g8.log) had
already localized it — its per-kernel budget closes against the full step:

| kernel (1024^3, 8 GPU) | ms | cause |
|---|---|---|
| exchange | 1244.7 | ships the FULL capacity buffers (~12 GB/device/direction) to both ring neighbours every step; receiver masks. Movers are ~0.1% of particles — ~1000x excess ppermute volume, inter-node links saturate. |
| deposit | 983.6 | plain fp64 scatter-add over all C=201M slots (204 Mpart/s); single-device path does int32 fixed-point (11,717 Mpart/s measured, 57x). Ghost-plane comm is negligible (17 MB). |
| fmg_warm | 44.6 | fine — the solve is NOT the problem. |
| gather | 33.1 | fine |
| grad | 4.9 | fine |

Sum 2311 ms ~= measured 2286 ms. Fixes queued: (1) pack boundary-crossers
into small fixed-capacity send buffers before the ppermute (keep the cumsum
compaction + overflow counting); (2) reuse the fixed-point int32 deposit
inside the slab shard_map body. Teardown crash (np.asarray on non-addressable
sharded arrays with snapshots off, runner_cosmo_dm.py _run_slab) fixed
2026-08-16.

### Fixed (2026-08-16, jobs 1392517-19, 1392559-61): 2286 -> 119.8 ms (19x)

Three stacked fixes, each validated by the kernel bench + a full l10 run:

| l10 full step (median device) | ms |
|---|---|
| baseline | 2285.7 |
| + packed exchange (movers only, C//64 send buffers) | 1090.1 |
| + fixed-point int32 deposit (LUNA_DM_FIXED=1; knob was default-off) | 531.9 |
| + padding slots dropped from the scatter + dm_host_every in slab loop | **119.8** |

Post-fix kernel budget (bench, closes against the step): deposit_fx 22.8,
exchange 40.5, fmg_warm 20.1, gather 26.6, grad 2.4 = 112. The deposit's
last 400 ms was NOT atomics-vs-float: 67M inactive padding slots (all at
pos=0) serialized on ONE grid cell's atomic; routing them out of bounds with
mode="drop" also took the float deposit 982 -> 33.8. Exchange overflow and
IC-settle losses are counted and raise — never silent. Physics gates: l9
anchor 27.8 ms unchanged with identical delta_max track; l10 delta_max
matches at equal a (dm_host_every=10 refreshes dt every 10 steps -> 41 vs 43
steps to a=0.028, drift <1%/step as designed).

Weak scaling, matched 512^3-per-GPU load, REAL ICs (l9/l10 monofonIC):
27.8 ms (1 GPU) -> 119.8 ms (8 GPU) = 23% efficiency. Remaining gap is now
honest: exchange 40 ms (inter-node ring bandwidth; u32 positions R4 would
cut ~20%), and every local kernel within ~2x of its 1-GPU roofline.

### Second round (2026-08-16/17, jobs 1392710/11/37/42): 119.8 -> 100.2 ms

- cap 1.2 (from 1.5) + occupancy-triggered proactive regrow (90% of C ->
  1.3x realloc + one cached recompile, JAX-MD-style; abort stays as
  backstop): 119.8 -> 104.4 ms. No regrow needed at z>=34 (occ 84%).
- movers diagnostic: ~23k movers/device/step at l10 — the C//64 send buffer
  was ~100x oversized. LUNA_DM_SEND_DIV=1024: 104.4 -> 100.2 ms. Only 4 ms:
  the exchange's ~36 ms remainder is LOCAL pack/merge compute (reads the
  full 161M-row buffers to find 23k movers, ~40 GB HBM traffic/step), not
  wire.
- IC settle is a different regime: chunk/slab misalignment moves ~6.4M
  particles in the first exchange — the lean buffer overflowed and the
  settle particle-count check ABORTED the run cleanly (safety validated at
  scale). Settle now always uses C//8; cfg send_div applies to steps only.
- Slab-path overhead probe (l9, 2 GPU, NVLink): slab 42.6 ms vs replicated
  particle-parallel 39.2 -> the distributed data structure costs ~9%.

**Cumulative: 2285.7 -> 100.2 ms (22.8x); weak eff 27.8/100.2 = 28%.**
Budget: deposit ~19, exchange ~36, fmg_warm 20, gather ~22, grad 2.3.
Next lever (not yet implemented): amortized exchange — deposit/gather read
through a 1-cell ghost layer, so exchange every k steps is EXACT while
cumulative drift since last exchange < 1 cell (track max drift, force
exchange before the bound; ~0.01 cells/step at these z) -> ~32 ms back,
projected ~70 ms/step (~40%). Then: u32 positions (R4, credit GADGET-4 /
pmwd), overlap exchange with deposit, blocks decomposition for >=64 GPUs.

## Single-GPU gravity roofline — PM kernels (job 1268206, 2026-08-06)

MUSIC unigrid DM ICs (Eisenstein-Hu, 100 Mpc/h), production mixed precision
(particles/phi fp64, deposit/cycle fp32, fixed-point CIC). Median of 20 reps,
kernels timed in isolation via scripts/validation/runs/gravity_roofline.py.

| mesh | deposit Mpart/s | vcycle Mcell/s | fmg Mcell/s | gather Mpart/s | f_warm Mpart/s | f_cold Mpart/s |
|------|-------|-------|-------|-------|-------|-------|
| 32³  | 498   | 89    | 23    | 471   | 82    | 24    |
| 64³  | 2,602 | 648   | 147   | 2,145 | 594   | 141   |
| 128³ | 8,033 | 4,278 | 866   | 5,096 | 2,272 | 804   |
| 256³ | 11,044| 9,319 | 3,025 | 6,137 | 2,957 | 1,895 |
| 512³ | 11,717| 10,080| 4,357 | 6,096 | 2,961 | 2,106 |

f_warm/f_cold = fused full force step (deposit -> solve -> 4th-order gather),
warm = 1 V-cycle defect correction on the previous phi, cold = full FMG.
Warm/cold ratio: 1.4× at 512³, up to 4× at small meshes (job 1268322).

- Saturation >=256³, latency floor below 128³ — same knee as the hydro
  roofline. Warm V(1,1) beats cold FMG 2.3× at 512³ (13.3 vs 30.8 ms).
- 512³ full-step budget: deposit 11.5 + vcycle 13.3 + gather 22.0 ~= 47 ms
  for 134M particles; gather (4th-order grad + trilinear interp) is the
  largest single kernel.

## Single-GPU hydro roofline — Sedov 3D full step (job 1262589, 2026-08-06)

Steady-state plateau Mcell/s (second-to-last perf window; last window is
teardown-tainted, systematically ~10-20% low — do not use it, and do not use
the end-of-run "Average performance" line either). Snapshots off
(`LUNA_SEDOV_WRITE_SNAPSHOTS=0`), per-config XLA caches, preallocation on.

| N³    | fp32 | recon32 | fp64 |
|-------|------|---------|------|
| 32³   |   91 |      88 |   86 |
| 64³   |  638 |     608 |  524 |
| 128³  | 1371 |    1379 |  904 |
| 256³  | 2117 |    1978 | 1251 |
| 512³  | 2193 |    2104 | NaN¹ (fused: 1223) |
| 1024³ | OOM² | OOM²    | OOM² |

¹ Plain (non-fused) 512³ fp64 goes NaN at step 1 on CUDA — was viper's FP64
  anchor on ROCm. **Root-caused (jobs 1262764/832/904, 1263084/203/272):**
  XLA:GPU uint32 index-arithmetic overflow. The (5,512,512,512) fp64 state
  has field byte-stride 512³×8 = 2^30 exactly; field 4 sits at byte offset
  4×2^30 = 2^32, which wraps to 0 (= field 0) in the 32-bit shift-based
  indexing XLA emits for power-of-two strides. Evidence: corruption is a
  single wrong conservative flux on the +y face of the blast (mass
  telescopes exactly, linear in dt); follows the blast under jnp.roll;
  ORDER=0 (no PLM) is clean; 516³ fp64 (offset 4.4e9 > 2^32 but non-po2
  stride → 64-bit mul indexing) is clean; fp32 (offset 2^31) clean;
  256³ fp64 (2^29) clean; identical on jax 0.10.1 and 0.11.0; ROCm/MI300A
  unaffected. => any all-po2 fp64 5-field array ≥512³ is at risk,
  including 512³ per-GPU shards in sharded ladders — verify at 8 GPUs.
  Workarounds, all measured clean: recon32 (fp32 W_pad, halves offsets),
  LUNA_HYDRO_FUSED=1, or a non-po2 tile (508³/516³). Upstream JAX issue
  to be filed (minimal repro pending).
² 1024³ does not fit on one GH200 (96 GB HBM) in any mode: plain fp32 OOMs
  on a 20 GB alloc; fused materializes a single 196.6 GiB buffer. viper's
  MI300A had 128 GB unified. Single-GPU roofline therefore ends at 512³;
  1024³ strong-scaling ladders start at ≥2 GPUs (or 8 = first weak-scaling run).

Conclusions vs viper MI300A anchors:
- GH200 is still rising at 512³ (2117 → 2193 fp32); viper's knee was 256³.
  **The 512³ per-GPU ladder tile is confirmed** — it sits at saturation.
- fp64 256³: 1251 vs viper's 1.11 Gcell/s full-step anchor (+13%).
- recon32 runs at ~96% of fp32 throughput while carrying FP64 state —
  ~1.7× over fp64 — consistent with viper's ~2× claim.
- 32³ floor ~0.37 ms/step across precisions = host dispatch floor (one dt
  round-trip per step), irrelevant at ladder tile sizes.
