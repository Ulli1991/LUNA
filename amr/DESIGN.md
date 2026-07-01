# LUNA block-structured AMR — design note

Status: **active**, branch `amr`. This is the LUNA-native, dimension-generic
block-AMR layer. We cherry-pick ideas from JAX-AMR (vendored read-only under
`third_party/jaxamr/`, MIT) but own all the code here.

## Locked decisions

1. **Dimension-generic.** One `ndim`/axis-tuple code path serves 2D and 3D. No
   hand-unrolled per-axis code. **Validate 2D first**, then flip `ndim=3`.
2. **Cherry-pick, don't fork.** LUNA already owns the hard 3D physics
   (`hydro/muscl_hancock.py`, `gravity/poisson_multigrid.py`); AMR only supplies
   block bookkeeping. We re-implement that bookkeeping cleanly.
3. **Everything in JAX.** Pure `jax.numpy`, `jit`/`vmap`/`scan`; no host-numpy in
   the hot path, no Warp. Fixed shapes + masking (no ragged trees).
4. **Multi-device via `jax.shard_map`.** Blocks are the shardable unit; the AMR
   layer must stay shard-compatible from the start and reuse LUNA's slab/halo
   patterns (`gravity/particle_mesh_shardmap.py`, `gravity/poisson_bc_shardmap.py`).
   Note: use `jax.shard_map` (the `jax.experimental.shard_map` path is deprecated
   as of jax 0.8.0).
5. **Wired through `main.py`.** The AMR runner plugs into the existing entry
   point / config flow, not a separate script.
6. **Poisson coupling: level-by-level Dirichlet first** (RAMSES/Guillet–Teyssier
   style) reusing LUNA's existing Dirichlet/isolated multigrid path. The block
   layer MUST expose the coarse-fine primitives (covered-mask, C/F face
   enumeration, scalar restrict/prolong, halo fill from neighbours + interpolated
   coarse) so the **force-conserving composite (Martin–Colella)** scheme stays
   implementable later WITHOUT redesign. The MG choice is deliberately not
   locked by the data structure.

## Block model (clean, standard fixed-size blocks)

- Refinement ratio `r = 2` per level.
- **Every block has the same fixed cell shape** `block_shape` (e.g. `(16,16)` /
  `(16,16,16)`) at every level. Finer levels just have smaller `dx`.
- **Level 0** tiles the whole domain with `base_blocks = base_grid // block_shape`
  blocks per dim (must divide evenly); all level-0 blocks always exist.
- **Refine** a level-`L` block (flagged if any of its cells exceed the δ
  threshold, after a buffer-halo dilation) into `2**ndim` level-`(L+1)` children,
  each covering half the parent's physical width per dim. (We use whole-block
  refinement rather than upstream's intra-block sub-blocking: with a modest
  `block_shape` it is far simpler and is the natural shard unit.)
- **Block identity**: integer block-coords per level → **Morton code** (interleave
  `ndim` bits/level). O(N log N) sort/searchsorted for neighbour finding and for
  matching unaltered blocks across re-adaptation. (Upstream used an O(N²) pairwise
  compare — `third_party/jaxamr/amr.py:356`.)
- **NaN sentinel block**: the last slot of each level's padded block array is a
  NaN block (invalid-ghost source). All reductions are nan-aware (`nanmax`/`nansum`).
- **Ghost halo** width `g` matches LUNA hydro's stencil need. Fill order per face:
  same-level neighbour if present, else interpolate from parent (coarse) level,
  else physical BC.

## Layout

- Block data: `(n_blocks, n_vars, *block_shape)` — `n_vars=5` for 3D hydro
  `[ρ̃, ρ̃ṽx, ρ̃ṽy, ρ̃ṽz, Ẽ]`; scalar fields (`φ`, `ρ`) use `n_vars=1`.
- `BlockInfo` (per level): `morton` (block id), `coords` (int block-coords),
  `level`, `neighbor` `(n_blocks, 2*ndim)` (slot id or sentinel), plus parent
  slot for C/F fill. Immutable pytrees; config is a frozen `AMRConfig`, NOT
  global module state (upstream used globals via `set_amr`).

## Build order

1. **Geometry core** (`amr/geometry.py`): Morton encode/decode, block→physical
   coords, generic `prolong`/`restrict`, refinement masks (δ criterion + buffer).
   ← *current step, 2D-validated against upstream, plus 3D unit tests.*
2. Block topology + halo exchange (`amr/blocks.py`, `amr/halo.py`).
   - DONE: refinement block-mask, Morton neighbour finding (2*ndim faces,
     periodic option), child-block generation (2**ndim conservative children),
     same-level ghost fill. Morton dtype is x64-aware (uint64/21b normal fp64
     production; uint32 fallback under float32 — `geometry._morton_params`).
   - DONE (cont.): `AMRState`/`LevelData` pytree container + uniform<->block
     partition/assemble (`amr/state.py`); cross-block buffer dilation; "preserve
     unaltered blocks" via Morton set-diff (`blocks.match_blocks` /
     `preserve_unaltered`, O(N log N)).
   - DONE (step 2 crux): coarse-fine ghost interpolation (`halo.fill_ghost_from_coarse`)
     — fills all NaN ghosts (faces+corners) from a prolonged coarse underlay;
     physical BC falls out of the parent-field pad mode. First cut = dense parent
     (L=1 / base grid). FOLLOW-ONS: deeper-level sparse parent via gathered patch
     (same interpolation, gather instead of dense slice); linear (vs piecewise-
     constant) coarse prolong for gravity; hydro reflective sign-flips.
   **Step 2 COMPLETE for the integration proof (42 tests, 2D+3D).**
3. Hydro-on-blocks (graft LUNA MUSCL-Hancock per block + reflux). `amr/hydro.py`.
   - DONE (per-block advance): each block = a `shard_map` shard; reuse
     `muscl_hancock_divergence` *verbatim* via `vmap`, in LUNA's shard-seam mode
     (`exact_wrap=False` / `boundary != 0`) so the block-face flux comes from the
     ghost. Consumes a 1-ghost `W_pad` sliced from the wider block halo (inner
     ghost = neighbour edge cell); the slope/face-BC helpers read only axis-aligned
     face ghosts, so the halo's NaN corners are never touched. Validated to match
     a global-periodic-pad + tile-slice reference *exactly* (1e-10) in 2D, non-
     square 2D, and 3D — proves partition→Morton-neighbours→halo→solve→assemble.
     Refined-level advance with coarse-parent ghosts runs finite (2D). Also fixed
     a latent int32/int64 dtype bug in `fill_ghost_from_coarse` (dynamic_slice
     index dtype) exposed by real int32 `LevelData` coords.
   - DONE (flux variant): `muscl_hancock_face_fluxes` / `divergence_from_face_fluxes`
     in `hydro/muscl_hancock.py`, alongside the untouched production divergence
     path. Returns per-axis `(N_d+1)` face fluxes; reproduces
     `muscl_hancock_divergence(order=2)` bit-equal (1e-12, 2D+3D). The block-
     boundary faces are the coarse-fine interface fluxes. `amr.hydro.block_face_fluxes`
     vmaps it over blocks.
   - DONE (coarse-fine reflux): `amr/reflux.py` — Berger-Colella correction of
     the dense coarse cell just outside a fine patch by `(dt/dx_c)(<F_f> - F_c)`,
     with `<F_f>` the transverse-averaged fine subface flux. `coarse_fine_face_mask`
     (face has no same-level fine neighbour), `apply_reflux_dense`,
     `restrict_fine_into_coarse_dense`. Validated by a **mass-conservation** test
     that isolates reflux: monolithic (seam-free) dense coarse + ONE fine block,
     so the C/F interface is the only non-conservative face — total mass returns
     to machine precision WITH reflux and provably drifts WITHOUT it. First cut:
     dense coarse (L0↔L1), no time subcycling, periodic; per-slot Python loops
     unroll under jit (revisit for the hot path).
   - DONE (same-level seam consistency): `conservative_block_divergence` /
     `advance_blocks_conservative` in `amr/hydro.py`. Each block-block face shares
     ONE Riemann flux built from both blocks' boundary reconstructions (this
     block's `R_d` + neighbour's `L_d`, exchanged by gathering on the slot axis),
     so the conserved variables telescope across the seam. New solver primitives
     `muscl_hancock_boundary_states` + `riemann_face_flux`. Faces with no same-
     level neighbour keep the ghost-based flux (handled by reflux / BC).
     Validated: periodic 3×3 base level conserves mass AND energy to machine
     precision (ghost-based advance drifts). **Capstone**: dense coarse + a fine
     patch of 2**ndim sibling blocks conserves total mass only when BOTH the
     sibling seam fluxes and coarse-fine reflux are on (control with non-
     conservative fine siblings drifts). Note: a *fresh* prolong has zero boundary
     slopes (first two cells share a coarse parent → one-sided slope limits to 0),
     so the seam inconsistency is invisible on step one; tests use sampled fine
     data to expose it.
   **Step 3 (hydro-on-blocks) COMPLETE: per-block MUSCL-Hancock + same-level seam
   conservation + coarse-fine reflux, 2D+3D, conservation proven.**
   - FOLLOW-ONS (with the multi-level driver, not blocking step 4): level-0
     physical-BC halo for non-periodic base; sparse (deeper-level) coarse parent
     for reflux/ghosts; block-decomposed (vs dense) coarse reflux flux capture;
     time-subcycled multi-level integrator; replace reflux/restrict per-slot
     Python loops with a scatter on the hot path.
4. Multigrid-on-blocks (level-by-level Dirichlet, LUNA Poisson per level).
5. PM/DM on the finest covering block.
6. `shard_map` blocks across devices; wire through `main.py`.
