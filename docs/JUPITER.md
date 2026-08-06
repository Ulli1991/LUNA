# JUPITER (JSC) — GH200 campaign

Target machine: ~5800 nodes × 4 GH200 (96 GB HBM3e + 480 GB Grace LPDDR per
GPU, NVLink-C2C, NDR InfiniBand). User has an allocation; machine was
near-empty when checked — a rare window for wide runs.

**Why we are going:** the questions that decide whether LUNA scales need
4+ GPUs, and viper cannot schedule them (282/298 `apu` nodes allocated;
multi-node jobs sat pending >24 h). Specifically `GRAVITY_PLAN.md` P1.3
(blocks-vs-pp crossover) and P1.4 (weak-scaling ladder on real ICs).

Goal, unchanged: **fast on 1 GPU, fast on a few hundred, fast on thousands** —
speed *and* accuracy, measured on real ICs.

---

## Day one — environment

1. **CUDA JAX env**, clean-room, mirroring the viper `jax_env2` recipe
   (see `luna-viper-rocm-setup` memory). Do not reuse the ROCm wheels.
2. **XLA compile cache on scratch** — load-bearing, not a nicety. Measured on
   viper: cold 2-GPU `shard_map` compile >12 min, warm ~40 s; a cold 1024³
   gravity graph never finished in 12 h before the solve was split out.
   Per-config cache dirs (size, ndev, precision), pre-seeded **before** any
   timed run — every ladder rung is a new shape.
3. **sbatch pattern**: 4 GPUs/node, one process per GPU, distinct JAX
   coordinator port per srun (`20000 + JOB_ID % 40000`); `PYTHONUNBUFFERED=1`.
4. **Never** set `HIP_LAUNCH_BLOCKING` (ROCm-only anyway) or its CUDA
   equivalent globally — it serialized every launch and turned 1.7 s steps
   into 52 s at 1024³ on viper.

## Day one — quirk retests (CUDA vs ROCm)

These cost us days on viper. Verify each on GH200 before trusting any number:

| quirk | viper behaviour | retest |
|---|---|---|
| deposit padding contention | placeholder slots at one position → 91× slowdown; fixed by spreading target cells | does the pathology exist under CUDA at all? |
| dispatch-pileup wedge | unsynced eager multi-program steps wedged the queue; needed a mid-step sync | still needed? |
| multi-device `device_put` | global sharded `device_put` deadlocked → per-shard assembly everywhere | retest |
| auto-sharded `jnp.pad` | hung multi-process | retest |
| compile blowup | fused step graph: 2 h at 512³/2 GPU, never at 1024³ → fixed by splitting the solve out | does the split still matter under CUDA? |
| `LUNA_MG_CA` threshold | auto = on iff `process_count>1`; tuned for 2-GPU intra-node | re-tune: intra-node group is **4**, not 2 |

## Order of battle (user-specified)

1. **Roofline anchors first**, two clean subsystem benches, single GPU:
   - **Sedov 3D hydro-only** — `scripts/validation/runs/sedov3d_run.py`
     (`LUNA_SEDOV_NX`, `LUNA_MESH_SHAPE`, `LUNA_X64`, max-steps cap).
   - **Gravity components separately** — CIC deposit, one warm V-cycle, cold
     FMG. *Not* the full step (explicit user instruction).
2. **Then** weak + strong ladders for both, ≥256³ per-GPU tiles (small tiles
   ran at 7% peak on viper and polluted the curve).

## viper reference numbers (MI300A) — what to beat/compare

Real l9 IC (512³, 134,217,728 particles), 512³ mesh, gated on δ_max, all-FP32
unless noted:

| | 1 GPU | 2 GPU |
|---|---|---|
| **LUNA full step** | **34.6 ms** (resid 2.96e-3) | **27.4 ms** (61% eff) |
| LUNA production (FP64 carry) | 45.0 ms (resid 2.93e-3) | — |
| Horowitz JaxPM MG | 43.3 ms (resid 4.96e-3) | 74.3 ms (91% eff) |

Components, 1 GPU, all-FP32: deposit 15.25 (his 9.12 — **our 1.66× gap**),
solve 7.14 (his 9.50), gradient 1.95, gather 10.02 (his 16.03).

Solve alone, sharded blocks path: 18.1 → 12.9 → 5.7 ms at 1/2/8 GPUs (3.2×).
Component rooflines, 512³ 1 GPU: hydro 2194 Mcell/s FP64 / 2453 FP32;
V-cycle 7412 / 13556; deposit 3978 / 4659 Mpart/s (synthetic particles —
the real l9 IC in lattice order deposits ~2× faster).

δ_max gate (8 steps, l9 from z=50): 1.8074, 1.8366, 1.8205, 1.8289, 1.8529,
1.9407, 2.0276, 2.0914.

## The measurements JUPITER is for

**P1.3 — blocks-vs-pp crossover.** The particle-parallel path replicates the
mesh: its solve is 6.97 ms at 1 GPU *and* 2 GPUs, by construction — a hard
ceiling. The grid-sharded blocks path shards the solve properly but its full
step was 62.5 vs 27.4 ms at 512³/2 GPU (gap is in particle handling, not the
solve). Find where blocks wins. Needs 4/8/16 GPUs.

**P1.4 — weak scaling on real ICs.** l9 (512³ particles) at 1 GPU vs l10
(1024³ particles, `/viper/ptmp2/uli/movies/ics_arepo_l10_bary.hdf5`, 57 GB) at
8 GPUs, matched per-GPU load. **Note:** the earlier viper ladder used random
and single-mode sine fields — that violates the real-IC rule and must not be
carried over. Redo on l9/l10.

**Then** strong + weak ladders out to hundreds of GPUs, and the head-to-head
against JaxPM at scale (his multi-device path costs him 3.1× — 135.1 vs
43.3 ms at 1 GPU — so his 91% efficiency starts from far behind; whether it
catches us is an open, measurable question).

## Blockers / prerequisites

- **SSH channel from the user** — JUPITER auth is interactive; needs a
  persistent ControlMaster session on the user's side. This is the only hard
  blocker.
- **Chunked IC read** — each rank currently reads the *whole* IC.
  `assemble_blocks_sharded` must use `read_music_dm_ic(lo=, hi=)` per rank
  (the reader already supports hyperslabs). At l10 this OOM'd on viper
  (107 GB/process × 2 = 225 GB vs a 220 GB limit). **Required before any
  1024³-particle multi-node run.**
- **LUNA ≥4 GPUs needs the blocks-path unit bridge + δ_max gate built** for
  the head-to-head harness (`horowitz_bench/h2h_l9_mgpu.py` covers 1–2 GPU).
- ICs beyond l10 must be generated with MUSIC (l11 = 2048³, l12 = 4096³ —
  the user notes MUSIC handles up to 4096³; 4096³ ≈ 69 G particles ≈ 3.3 TB,
  needs a big-memory node and the chunked reader).

## Carry-over rules

- Judge MG scaling on the **warm** cycle (the production op), not cold FMG.
- Weak-scaling rungs need ≥256³ per-GPU tiles.
- Real ICs only (l9/l10/…); no synthetic fields for any perf conclusion.
- Every timing gated on δ_max; residual reported beside every ms.
- Measured numbers only — never extrapolate across hardware.
