# LUNA performance improvement plan

Derived from a multi-agent investigation (6 discovery agents over the hot path,
65 candidate findings) on 2026-06-03, plus an inline scout of the MUSCL-Hancock
pipeline. Cross-corroborated findings — items independently surfaced by
multiple agents — are weighted highest.

## Motivation

| Test           | Hardware            | AthenaK | LUNA           | Gap        |
|----------------|---------------------|---------|----------------|------------|
| Sedov 1024³    | 8 vs 32 A100s       | 4h 25m  | ≥20h projected | ~4–5× wall + 4× GPUs |
| KH 8192²       | 2 nodes / 8 A100s   | 3h 28m  | ~4h 35m proj.  | ~30%       |

LUNA needs 4× the GPUs because workspace overflows 40 GB/A100 at 1024³ fp32,
and is slower per GPU on top of that — especially on Sedov, where the tiny
initial dt (~2.5e-10) means tens of thousands of tiny steps where per-step
Python/JAX dispatch dominates.

AthenaK runs equivalent or more expensive integrators (RK2 for KH, RK3 for
Sedov) than LUNA's MUSCL-Hancock, so the headline gap *understates* the
architectural difference.

## Findings, grouped into 5 workstreams

Numbers are rough speedup estimates from the discovery agents; treat as ranges.

### #1 — Remove the double host-device sync (Easy, big win)
- **Impact:** ~15–30% wall-clock at small dt.
- **Effort:** small. **Risk:** low.

Five of six discovery agents independently flagged this. The runner calls
`float(compute_dt_fn(...))` twice per step (pre-step for dt selection,
post-step for NaN guard). Each blocks GPU dispatch and forces a host transfer
plus Python branch.

Fixes:
- Delete the post-step `dt_check` entirely on healthy runs. NaN detection can
  be done on `dU` itself, or deferred to the next-step `compute_dt` (any NaN
  poisons everything anyway).
- Move dt selection into a `lax.scan` over K=10–50 steps. Snapshot/t_end
  clamps fire from inside the scan via `lax.cond`.

Files: `setup/runner_distributed.py:247–330`, `time_integration/integrators.py`.

### #2 — Delete the redundant `device_put` in the step (Trivial)
- **Impact:** ~2–8% (and helps unblock pipelining).
- **Effort:** delete 10 lines. **Risk:** none.

`with_sharding_constraint` at `integrators.py:130` already enforces sharding
inside the JIT. The `jax.device_put(U_new, sharding)` at
`integrators.py:192–202` just outside is redundant — possibly triggering a
no-op collective, definitely blocking dispatch.

### #3 — Cache W from `compute_dt` and reuse in the step
- **Impact:** ~15–25%.
- **Effort:** medium (refactor `compute_dt` to return `(dt, W)`).
- **Risk:** medium — touches several call sites.

`compute_dt(U)` already builds the full primitive array via `jnp.stack`
(~21 GB at 1024³ fp32), then the result is thrown away and `convert_to_prim(U)`
re-does it inside the step. Pass W from `compute_dt` to the step, halve the
`cons_to_prim` work.

Files: `hydro/conversions.py`, `time_integration/integrators.py:92`
(`convert_to_prim`), `setup/runner_distributed.py:252`.

Bonus: `prim_to_cons` is dead code — delete it.

### #4 — Kill the memory-hungry materializations (Bigger, highest ceiling)
- **Impact:** ~20–40% per-step + lets 1024³ fit on 8 GPUs instead of 32.
- **Effort:** medium-large.
- **Risk:** medium (bit-different fused output is possible).

This is the "very memory hungry" issue. Specific items the memory-discovery
agent quantified at 1024³ fp32:

- `jnp.stack([dWx, dWy, dWz])` at `muscl_hancock.py:186` → fresh 63 GB array
  just for `lax.scan` input. Delete the scan, unroll _x/_y/_z branches.
- `dU0 = jnp.zeros_like(W_half)` → 21 GB zero seed. Use the first branch's
  contribution as the seed, accumulate the other two.
- 6× `jnp.concatenate` (lines 153/154/167/168/181/182) → 6 fresh ~21 GB
  arrays per step. Replace with `dynamic_update_slice` into pre-allocated
  padded buffer.
- `jnp.pad(mode='wrap')` in `halo_exchange_auto.py` → another fresh 21 GB.
  Combine the halo with the slice writes.
- **L-HLLC 3D: ~172 GB per face eval** — worst single item. The L-HLLC
  kernel allocates 8 full-size stacked tensors. Rewrite face-by-face with
  scalar locals, not full-array stacks.
- `lax.switch` on the static limiter int (`muscl_hancock.py`) forces XLA to
  trace all 3 limiter branches. Use `if limiter == X` outside the JIT.

Files: `hydro/muscl_hancock.py`, `hydro/riemann_solvers/*.py`,
`utils/halo_exchange_auto.py`, `hydro/conversions.py:54` (trailing `jnp.stack`).

### #5 — Fuse the three step JITs into one (Medium)
- **Impact:** ~10–20% on its own; mostly enables #4 to bite.
- **Effort:** medium.
- **Risk:** medium.

`integrators.py:92,98,112` are three separate `@jax.jit`-decorated functions
(`convert_to_prim` / `apply_halo_exchange_prim` / `compute_flux_and_update_from_prim`).
XLA can't fuse across JIT boundaries, so the intermediate `W_pad` gets
materialized to HBM and re-read.

Fuse into one JIT. The halo exchange uses `with_sharding_constraint`-driven
collectives which JAX can still emit inside a larger JIT. This is the
structural change AthenaK already gets for free (one Kokkos kernel per
direction). The AthenaK-comparison agent flagged this as the single biggest
architectural gap.

## Open questions / things needing a profile run

Static analysis couldn't conclusively answer:

1. Does XLA actually lower `jnp.pad(mode='wrap')` on a `NamedSharding`'d
   array to a thin-slab halo collective, or does it do a full-volume shift?
2. On 32 GPUs, what mesh shape does `('x','y','z')` decompose to? If it
   falls to a 2D slab partition, halos in one direction are much fatter —
   may partly explain the Sedov gap.
3. Per-step source-term dispatch overhead (gravity / forcing / cooling /
   iso-projection each their own JIT) — three agents flagged this
   independently but per-step cost is unknown without a trace.

Run nsys on one 1024³ Sedov step (single node, fp32, no checkpointing) to
nail these down before doing the structural refactor.

## Recommended order

**#1 → #2 → #3 first** — low effort, easy wins, reduces noise in the profile.
Then nsys profile. Then tackle **#4 → #5** with the profile guiding which
subitems are worth the work.

## Benchmarking protocol

To compare LUNA vs AthenaK fairly:

- Use **AthenaK's RK2** (closest equivalent to MUSCL-Hancock cost). AthenaK
  has no `vl2` / van-Leer integrator option — only `rk1/2/3/4/imex*`.
- Reference numbers from May 2026 runs:
  - **KH 8192²** (LUNA MUSCL-Hancock vs AthenaK RK2): 2 nodes / 8 A100s,
    AthenaK 3h 28m, LUNA ~4h 35m projected. Gap ~30%.
  - **Sedov 1024³** (LUNA MUSCL-Hancock vs AthenaK RK3): 8 vs 32 A100s,
    AthenaK 4h 25m, LUNA times out at 20% in 4h. Need 24h walltime to finish.
    Re-run *after each batch of fixes* and compare.
- Track per-snapshot wall time and steps-per-snapshot.

## Status log

- **2026-06-03** — Plan written from workflow output. Sedov 1024³ 24h-walltime
  run (job 27679427) still pending. Once it completes and gives a baseline
  number, start implementing #1, #2, #3, then re-benchmark.
