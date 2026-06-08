# shard_map refactor plan — explicit per-shard memory + halo control

## Why this is the right next step

After the dispatch + JIT-fusion fixes (item #5 of `perf_plan.md`) landed,
the 32-GPU baseline showed per-task MaxRSS drop **4.3 GB → 2.5 GB (-42%)**
with perf nudged from 877 → 895 Mcells/s. Memory pressure scales linearly
with per-shard data, so the same fixes extrapolate to ~43 GB per shard at
8 GPUs — close to A100's 40 GB but not under.

The remaining headroom is locked behind XLA's auto-sharding. We saw the
compiler refuse to budge the working set below 71 GB regardless of
source-level slope+jacobian restructuring — its static analysis decides
the materialisation schedule and ignores Python-level scoping. The only
way to get explicit per-shard control is `jax.experimental.shard_map`:
each device runs the step on its **own** local (padded) shard, halos are
exchanged via explicit `jax.lax.ppermute`, and intermediates are bounded
by the local-shard footprint.

Expected outcome: 1024³ fp32 on 2 nodes / 8 A100s in a 2×2×2 mesh. Each
shard holds (5, 512, 512, 512) = 2.5 GB; with ~10× working-set multiplier
(typical of MUSCL-Hancock when XLA can fuse cleanly) → 25 GB per device.
That fits with room for halos + collective scratch.

## Architectural overview

Replace the current auto-sharded fused step:

```
@jax.jit
def time_step_muscl_hancock_modular(U, dt):
    W = cons_to_prim(U, gamma)                       # auto-sharded
    W_pad = pad_halo_fn(W, boundary, spatial_dim)    # jnp.pad → XLA collective
    dU = muscl_hancock_divergence(W_pad, dt, ...)
    U_new = U + dt * dU
    return jax.lax.with_sharding_constraint(U_new, sharding)
```

with a `shard_map`-wrapped version:

```
@partial(shard_map, mesh=mesh, in_specs=(P(None,'x','y','z'),), out_specs=P(None,'x','y','z'))
@partial(jax.jit, static_argnames=(...))
def time_step_per_shard(U_local, dt):
    # U_local is shape (5, nx_local, ny_local, nz_local) — one shard's worth.
    W_local = cons_to_prim(U_local, gamma)
    W_pad_local = halo_exchange_3d(W_local, boundary, mesh_axis_names)  # ppermute under the hood
    dU_local = muscl_hancock_divergence(W_pad_local, dt, ...)
    U_new_local = U_local + dt * dU_local
    return U_new_local
```

The body of `muscl_hancock_divergence` and its helpers are **unchanged**.
They already operate on a single padded array; they don't care whether
that array is the full domain or just one shard.

## Per-shard halo exchange design

### Periodic boundary, axis 'x' (mesh dim `nx_proc`):

```python
def halo_x_periodic(W_local, mesh_axis='x'):
    # W_local: (5, nx_local, ny_local, nz_local)
    # Send the left-most slab to the next-lower-x neighbour, receive from the
    # next-higher-x neighbour; symmetric for the right.
    left_slab  = W_local[:, 0:1, :, :]
    right_slab = W_local[:, -1:, :, :]
    # ppermute sends my left_slab to me-1 (mod nx_proc); I receive my right
    # ghost from me+1.
    right_ghost = jax.lax.ppermute(left_slab,  axis_name=mesh_axis,
                                   perm=[(i, (i-1) % nx_proc) for i in range(nx_proc)])
    left_ghost  = jax.lax.ppermute(right_slab, axis_name=mesh_axis,
                                   perm=[(i, (i+1) % nx_proc) for i in range(nx_proc)])
    return jnp.concatenate([left_ghost, W_local, right_ghost], axis=1)
```

Three of these calls (axes 1, 2, 3) wrap any 3D field. Output shape
`(5, nx_local+2, ny_local+2, nz_local+2)`. Each transfers only a 1-cell
slab — total bytes per axis per direction = `5 * 4 * ny_local * nz_local`
at fp32 — for a 2×2×2 mesh that's `5 * 4 * 512 * 512 = 5 MB`. Sub-ms over
NVLink/Infiniband.

### Zero-gradient and reflective:

Boundary shards (rank 0 in 'x', or `nx_proc - 1`) need position-aware
behavior. `jax.lax.axis_index('x')` returns the shard's index along the
'x' mesh axis from inside `shard_map`. Sketch:

```python
def halo_x_zerograd(W_local, mesh_axis='x'):
    nx_proc = mesh_axis_size(mesh_axis)
    idx = jax.lax.axis_index(mesh_axis)
    # Non-boundary shards: still need cross-shard data (the interior is split).
    left_ghost  = jax.lax.ppermute(W_local[:, -1:, :, :], ..., axis_name=mesh_axis,
                                   perm=[(i, i+1) for i in range(nx_proc-1)])  # only interior pairs
    right_ghost = jax.lax.ppermute(W_local[:, 0:1, :, :], ..., perm=[(i, i-1) for i in range(1, nx_proc)])
    # Boundary shards: use their own edge cell as the ghost (zero-grad).
    left_ghost  = jnp.where(idx == 0,         W_local[:, 0:1, :, :],  left_ghost)
    right_ghost = jnp.where(idx == nx_proc-1, W_local[:, -1:, :, :], right_ghost)
    return jnp.concatenate([left_ghost, W_local, right_ghost], axis=1)
```

Reflective: same shape as zero-gradient but boundary shards negate the
normal momentum component in their ghost cells. The non-shard
implementation already does this at `halo_exchange_auto.py:111-117`.

### 1D and 2D variants

Same pattern but only `'x'` or `('x','y')` are sharded — the un-sharded
axes can still use `jnp.pad(mode='wrap'/'edge')` (no collective needed)
since the data is full on each shard. Keep current path for 1D/2D, only
rewrite the 3D path.

## Boundary contract

`shard_map` requires every shard to take the same control flow (no
Python-level branching on data). All boundary handling must go through
`jnp.where` on `jax.lax.axis_index(...)` so each shard runs the same
HLO graph.

For reflective on `mesh_axis='x'`, the momentum-axis-0 component (`U[1]`
for conservative or `W[1]` for primitive) is the one to negate. Same code
as today, just gated on `idx`.

## Mesh-shape contract

`shard_map` needs an explicit `Mesh` object with named axes. Reuse the
existing `MESH_SHAPE`-driven mesh from `utils/distributed.py`. For 3D:
- 8 GPUs: `MESH_SHAPE=(2, 2, 2)`, axes `('x','y','z')`
- 16 GPUs: `(2, 2, 4)` or `(4, 2, 2)`
- 32 GPUs: `(4, 4, 2)` (proper 3D, vs the existing auto `(4, 8, 1)` slab)

Drop the "auto picks slab" branch entirely for 3D — once shard_map is in,
balanced 3D is always strictly better since the halo cost scales with
surface area.

## File-by-file changes

| File | Change |
|------|--------|
| `utils/halo_exchange_shardmap.py` | **NEW**. `halo_3d_periodic_shardmap(W_local, mesh)` + `_zerograd` + `_reflective`. Each is a pure function that takes a *local* shard and returns a *local* padded shard, with the ppermute calls inside. |
| `time_integration/integrators.py` | Wrap `time_step_muscl_hancock_modular` in `shard_map` if `cfg.USE_SHARDMAP` is true. The body (cons_to_prim → halo → MUSCL → update) stays — we just change how it's dispatched. |
| `setup/distributed_setup.py` | Pass the new `pad_halo_fn` (shard_map variant) when `USE_SHARDMAP` is set. Build the `Mesh` with explicit axis names matching what shard_map needs. |
| `setup/runner_distributed.py` | Make sure the input `U_sharded` already has the right `NamedSharding` so shard_map sees correctly-laid-out shards. Update compute_dt similarly (reduce via `jax.lax.pmin` inside shard_map). |
| `template_config.py` | Add `USE_SHARDMAP: bool = False` (opt-in, fall back to auto-sharding for the unconverted KH/Solar/etc. paths). |
| `time_integration/wb_integrator.py` | Same shard_map wrapping for the well-balanced path used by solar convection. Defer until the MUSCL path is validated. |

## Test strategy

1. **Unit-level halo test** (CPU, single host, fake mesh): `(2,2,2)` mesh
   of CPU devices, exchange a synthetic field, assert bit-equality vs
   the current `jnp.pad(mode='wrap')` reference. Catches off-by-ones
   before any GPU run.
2. **Single-GPU bit-equality test**: at `(1,1,1)` mesh, shard_map should
   produce bit-identical results to the current code. If not, we have a
   correctness regression.
3. **32-GPU regression smoke**: same Sedov 1024³ smoke we already run.
   Result must still produce snap_000 at t=0 bitwise-equal to the
   auto-sharded baseline (allowing 1e-6 fp32 noise from collective
   reduction order if necessary).
4. **8-GPU stretch goal**: only after 1–3 pass. Use `MESH_SHAPE=(2,2,2)`.
   Pass = step 100 reached without OOM, ETA reasonable.

## Phasing

| Phase | Scope | Estimated effort |
|-------|-------|------------------|
| 1 | Periodic-only 3D halo via shard_map, MUSCL path only. Tested at single-GPU + 32 GPUs. | ~1 day |
| 2 | Zero-gradient + reflective 3D halo. Tested with M82 (zero-grad) + Sedov (zero-grad) at 32 GPUs. | ~0.5 day |
| 3 | 8-GPU stretch run on Sedov 1024³. If it OOMs, profile + remove remaining bottlenecks. | ~0.5 day |
| 4 | WB integrator shard_map for solar convection. | ~0.5 day |

## Risks and open questions

1. **shard_map disables some XLA fusions** between shards. Per-shard
   compute is the same but XLA cannot see across the ppermute boundaries.
   Net perf at 32 GPUs may dip slightly. Acceptable trade for 8-GPU
   capability, but measure.
2. **ppermute requires the mesh axis to be the full mesh dimension** —
   we cannot partially-permute along a subset of devices. Already
   guaranteed by passing the full mesh in.
3. **Memory still bounded by working-set multiplier per shard**. If the
   multiplier is 14× rather than the hoped-for 10×, the 2×2×2 8-GPU run
   will still OOM. Mitigation: drop to fp16/bf16 for non-critical
   intermediates, or `jax.checkpoint` around the predictor. Out of scope
   for phase 1.
4. **WB integrator complexity**. The well-balanced path has an extra
   equilibrium-subtraction inside the JIT and uses `U_eq` (a static
   array). Refactor must thread `U_eq` through `shard_map` as a
   replicated input. Defer to phase 4.
5. **Cooling, source terms (CC85, OU forcing, gravity)**. These run in
   the runner *after* the hydro step. Each is its own JIT and currently
   relies on auto-sharding. They may need shard_map too if the
   per-shard footprint matters; for Sedov (no sources) we ignore.

## Open data needed

- A profile (nsys) of the 32-GPU smoke would tell us exactly which op is
  the working-set peak and whether the halo collective itself is the
  remaining limit. Defer to phase 3 if 8-GPU still doesn't fit.
