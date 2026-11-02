# LUNA scaling design: 1 GPU → hundreds → thousands

Goal (2026-07-30): fast on 1 GPU, fast on a few hundred GPUs, fast on
thousands (JUPITER, ~5800 nodes x 4 GH200). This document fixes the
architecture from a literature survey of production codes and orders the
work. Measured LUNA anchors live in the scaling campaign notes; numbers
here from external papers are cited.

## What the field converged on (survey, 2026-07-30)

**Particles at extreme scale — HACC-style "overloading" wins.** The only
PM code family demonstrated at ~10^4 GPU tiles (CRK-HACC: 72,000 GCDs on
9,000 Frontier nodes, 4e12 particles, 92-95% scaling efficiency;
arXiv:2510.03557) uses 3D cuboid domains with a replicated ghost-particle
margin, refreshed every N steps by nearest-neighbor exchange only; zero
particle communication between refreshes. Per-step migration survives only
inside frameworks that make it local (Nyx/AMReX Redistribute). Never-migrate
with a whole-run margin (FlowPM) works only for short quasi-linear runs.

**Multigrid Poisson at scale is real, and warm-started MG is the winning
regime.** Nyx runs GMG-only gravity at up to 13,824 Summit GPUs
(arXiv:1301.4498, JOSS 3068). Fast(er)PM (arXiv:2607.10983) shows JAX
pencil-MG with nearest-neighbor halos beating distributed FFT 1.5-2.4x at
1024^3 on 8-128 A100s. The definitive cold-solve comparison (Gholami et
al., arXiv:1408.6497, 600B unknowns) still favors FFT for uniform periodic
sources; LUNA's regime is the warm-started/differentiable/memory-light one.
Decision (user, 2026-07-30): LUNA stays MG-pure — no FFT solver path.

**Deposit/gather — sorted order + atomics is the modern consensus.**
WarpX measured ~7.5x from per-cell sorting (arXiv:2101.12149; sorted
deposition saturates L2 streaming). Sparse fixed cadences (4-30 steps in
PIC; cosmology drifts allow 10-50). Warp-aggregated atomics are the
escape hatch (3-5x in MPM, arXiv:2111.00699). In XLA the atomic
scatter-add path is the only fast path; sorted `segment_sum` is the
deterministic alternative (relevant for autodiff). Row-major cell keys;
FP32 weights from cell-relative offsets are exact (mixed-precision SPH).
LUNA's measured 4-8x ordering cliff (deposit 12.6->50.4 ms, gather
9.4->76.9 ms at 512^3 shuffled) is this same effect; pmwd/JaxPM/DISCO-DJ
do not address ordering decay at all.

**Scalable GMG checklist (HPGMG, hypre-at-100k-cores, Reisner
arXiv:1803.02481, McCormick mixed-precision SISC 2021):** coarse-level
agglomeration + redundant coarse ladders (LUNA: shipped, measured 2.2x at
2 GPUs), communication-avoiding wide-halo smoothing (LUNA: shipped,
-24/-29% inter-node at 4/8 GPUs), FP32 defect-correction cycles (LUNA:
shipped, convergence-identical), fixed cycle counts with no per-cycle norm
collectives (LUNA: already true). Literature says do NOT bother with full
segmental refinement or additive cycles for a 7-point Poisson. New items
for thousands of GPUs: staged agglomeration (node-group gather before
global), Chebyshev smoother re-evaluated for its COMMUNICATION profile
(degree-k polynomial = k stencils per exchange, no red-black width
doubling; analytic bounds for the 7-point operator).

**Hydro at extreme scale — LUNA already has the winning pattern.** The
codes demonstrated at 24k-74k GPUs with >=76% weak efficiency (Cholla
82% @74,088 GCDs, AthenaK 80% @65,536, Parthenon 92% @73,728, Nyx) all
share: one fat FP64 domain per GPU (>=128^3, ideally 256^3+; throughput
knees below that — AthenaK 614->126 Mzc/s from 128^3 to 16^3 blocks),
fully device-resident state, fused kernels/packed launches, single-kernel
boundary packing (Parthenon: naive packing costs 82x, fused 3.5x),
bulk-synchronous stepping. LUNA = exactly this pattern, at 2.2 Gcell/s
FP64-state per GPU.
The dt allreduce is measured harmless to 10^4-10^5 GPUs (Cholla runs a
blocking per-step MIN at 74k GCDs; Frontier-E: ~2.8% in all global
reductions); the actual full-machine limiter is per-step runtime VARIANCE
(stragglers) — for JAX: fixed shapes, no in-loop recompiles, minimal host
dispatch. Ghost exchange done right costs 5-20% at full machine
(Parthenon 7-8%); the JAX-specific risk is packing granularity (verify
XLA fuses halo slicing), not transport. Wide halos for hydro measured at
only 1.04-1.15x (FDTD study) — MUSCL-Hancock's one-exchange-per-step is
already the communication-light end (JAX-Fluids pays 1-3 per RK stage).
Do NOT build temporal blocking/swept-rule (no production shock code uses
them). Mixed-precision hydro at scale is unexplored territory (GAMER-2's
FP32 default, 2.1-2.6x, is the lone precedent) — recon32 is a genuine
differentiator. No JAX code has published beyond 512 GPUs (JAX-Fluids
2.0); MFC-lineage ran 98% weak efficiency 64->9,200 GH200 on Alps — the
hardware supports a first-of-its-kind JAX full-machine run.

## LUNA's architecture (decided)

One 3D meshblock decomposition — the hydro's — carries everything:
gas fields, deposited DM density, potential, forces. Gravity solves on
the same mesh with the same halo machinery (agglomeration + CA + warm
+ mixed). DM particles are owned by block, stored cell-sorted
(row-major key, resort every ~10-50 steps, knob), deposited block-locally
with an overload margin; migration is a per-axis ring exchange executed
only at refresh cadence, with particle-count + mass assertions at every
refresh (historical particle-loss bug class). Correctness rule (HACC):
a particle's force is valid while its drift since last refresh is under
(margin - CIC support).

## Build schedule

Tier 1 — single GPU (mostly done, measured):
  [x] DM step 72.2 -> 46.3 ms/512^3 (fused KDK, mixed precision, warm +
      extrapolated seed, V(1,1))
  [x] hydro 120.3 -> 61.1 ms/512^3 (FP32 reconstruction, FP64 fluxes;
      e2e + convergence validated)
  [ ] resort cadence knob (literature-backed; protects the 4-8x)
  [ ] coupled gas+DM step audit (488 ms/step movie anchor, never audited)

Tier 2 — hundreds of GPUs:
  [ ] validate sharded MG on a true 3D mesh (job queued: (2,2,2), 512^3)
  [ ] particle refactor: 3D-block ownership + overload margins +
      cadence refresh (rewrites gravity/particle_mesh_shardmap.py)
  [ ] per-process chunked IC assembly (kills the host-OOM class)
  [ ] block-local sorted deposit/gather on the shared mesh
  [ ] viper ladders to 16-32 GPUs, then JUPITER O(100) — warm-cycle and
      full-step metrics, weak + strong

Tier 3 — thousands of GPUs (JUPITER):
  [ ] staged agglomeration (node-group first; switch level from measured
      time-per-level plateau)
  [ ] Chebyshev-CA smoother option (communication profile, not kernel speed)
  [ ] bootstrap/compile-at-scale engineering (4-digit process counts)
  [ ] snapshot/IO at scale
  [ ] full-machine ladders; hydro-only can go first (meshblock-native today)

Known environment hazards: ROCm multi-device device_put deadlock (slab
probe, 2026-07-30) — retest on CUDA before working around; sequential
sruns need distinct JAX coordinator ports; PYTHONUNBUFFERED=1 always.
