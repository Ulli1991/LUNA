# Multigrid self-gravity Poisson solver — implementation plan

## Motivation

LUNA currently supports *external* gravity (fixed Φ from a Miyamoto-Nagai
+ NFW potential for M82, optional `gravity_z` constant for solar
convection) via `setup/source_external_gravity.py`. There is no
*self*-gravity path where the gas density sources Φ through
`∇²Φ = 4πGρ` — required for the next class of physics applications
(gravity-driven turbulence, isolated disk + clump collapse, GMC
formation, the cosmological-zoom future).

Target: RAMSES-style multigrid V-cycle Poisson solver on the uniform
grid, integrated with the existing shard_map / unsplit MUSCL-Hancock
step. Multigrid is the right tool because:

- O(N) per V-cycle vs O(N log N) for FFT-Poisson, and FAR better than
  the O(N · κ) of Krylov methods at high resolution (κ ~ N for
  Poisson).
- Handles general boundary conditions (Dirichlet, Neumann, periodic,
  isolated) — FFT-Poisson is locked to periodic and can be extended only
  via the Hockney zero-padding trick.
- Matches the RAMSES algorithmic family LUNA is already aligned with on
  the hydro side (unsplit MUSCL-Hancock with Hancock predictor).

## Algorithm overview

Standard V-cycle on a uniform-grid hierarchy `L0 (finest, Nx=1024³)` →
`L1 (Nx/2)³` → `L2 (Nx/4)³` → ... → `Lk (coarsest)`.

```
def vcycle(φ, ρ, level):
    if level == coarsest:
        return coarsest_solve(φ, ρ)
    φ = smooth(φ, ρ, n_pre)         # red-black Gauss-Seidel × n_pre
    r = residual(φ, ρ)              # r = 4πGρ - ∇²φ
    e_coarse = vcycle(0, restrict(r), level + 1)
    φ = φ + prolong(e_coarse)
    φ = smooth(φ, ρ, n_post)
    return φ
```

Typical hyperparameters: `n_pre = n_post = 2`, V-cycles until residual
norm drops below `tol = 1e-6 * ||ρ||`.

### Smoother: red-black Gauss-Seidel

Standard 7-point Laplacian stencil

    ∇²φ_ijk ≈ (φ_{i±1,j,k} + φ_{i,j±1,k} + φ_{i,j,k±1} - 6 φ_ijk) / dx²

Two-color sweep:

```
for color in (RED, BLACK):
    halo_exchange(φ)
    φ_color = (sum_of_neighbours - 4πGρ * dx²) / 6      # only "color" cells
```

Compatible with shard_map: halo exchange uses the same per-axis
`ppermute` we wrote for hydro, just on the scalar potential field.

### Restriction (fine → coarse)

8-cell average per coarser cell:

```
ρ_coarse[I,J,K] = sum_{di,dj,dk in {0,1}} ρ_fine[2I+di, 2J+dj, 2K+dk] / 8
```

Implemented as a reshape + mean over 4D blocks — XLA-friendly,
sharding-safe as long as the fine mesh divides cleanly by 2 per shard.

### Prolongation (coarse → fine)

Trilinear interpolation. For each coarse cell at (I,J,K), fill 8 fine
cells (2I:2I+2, 2J:2J+2, 2K:2K+2) with weighted contributions from the
coarse neighbours. Simplest: nearest-neighbour first (zero-order), then
add a linear correction loop. Plan to start with nearest-neighbour and
upgrade.

### Coarsest-grid solve

When the per-shard grid drops below ~4³, we can't keep refining inside
the shard_map mesh. Options:

1. **Stop the V-cycle early** and crank up the smoother on the coarsest
   level (50–100 iterations of Jacobi). Cheap because it's tiny
   problem. Resolution: Nx=1024 with mesh (2,2,2) → coarsest before this
   wall is Nx=512/2 = 256 per shard / 2^L. At L=6, per-shard is 4 cells
   — practical floor.
2. **Gather to one rank** and solve serially (RAMSES-style "tier
   reduction"). Complex but needed for AMR; defer.
3. **Switch to FFT at the coarsest** (Hockney for isolated BC).

Start with (1), upgrade to (2) when AMR lands.

### Boundary conditions

| Mode | Method |
|------|--------|
| Periodic (`BOUNDARY=0`) | same `ppermute` wrap as hydro halo |
| Zero-gradient/Neumann (`BOUNDARY=1`) | reflect potential at boundary shards (no-flux: `Φ[-1] = Φ[0]`) |
| Reflective (`BOUNDARY=2`) | Neumann on Φ same as zero-grad |
| Isolated (new mode, `BOUNDARY=3`?) | Hockney zero-pad to 2N + FFT for the BC, or multipole expansion |

Isolated BC is the astrophysically important new case. Phase 2 — Phase
1 ships periodic + Neumann to cover driven turbulence and solar
convection.

### Coupling into the hydro step

Add a new source-term apply function `apply_self_gravity` in
`setup/source_self_gravity.py` that:

1. Solves `∇²Φ = 4πGρ` from the current ρ field
2. Computes `g = -∇Φ` (centered difference, halo-exchanged)
3. Applies the momentum kick: `ρv ← ρv + dt · ρ · g`
4. Applies the energy kick: `E ← E + dt · ρv · g`

Wires into the runner's existing source-term loop alongside cooling,
forcing, etc.

## Phasing

| Phase | Scope | Estimated effort |
|-------|-------|------------------|
| 1 | Periodic + Neumann V-cycle on uniform grid via shard_map. Validated on Jeans instability + uniform sphere. | ~1–2 weeks |
| 2 | Isolated (Hockney/multipole) BC. Validated on Plummer sphere, two-point-mass test. | ~1 week |
| 3 | FMG (Full Multi-Grid) for faster first-cycle convergence. | ~3 days |
| 4 | Solver hardening — adaptive cycle count, residual diagnostics, NaN guards. | ~3 days |
| 5 | Application: self-gravitating turbulence box, GMC collapse demo. | open-ended |

## File-by-file changes

| File | Change |
|------|--------|
| `gravity/poisson_multigrid.py` | **NEW**. V-cycle, restriction, prolongation, smoother. Pure functions, JAX-jit'd. |
| `gravity/poisson_bc.py` | **NEW**. Boundary helpers (periodic via ppermute, Neumann via axis_index + jnp.where). Largely a `BOUNDARY`-aware mirror of `utils/halo_exchange_shardmap.py`, on scalar fields. |
| `setup/source_self_gravity.py` | **NEW**. Wraps Poisson solver + gradient + momentum/energy kicks. Same API as `source_external_gravity.py`. |
| `setup/distributed_setup.py` | Wire `apply_self_gravity` into the source-term chain when `cfg.SELF_GRAVITY = True`. |
| `template_config.py` | `SELF_GRAVITY: bool = False`, `GRAVITATIONAL_G: float = ...`, `POISSON_TOL: float = 1e-6`, `POISSON_MAX_CYCLES: int = 50`, `POISSON_N_LEVELS: int = None` (auto). |
| `setup/runner_distributed.py` | Source-term order: existing forcing → self-gravity → cooling → isothermal projection. |
| `scripts/validation/runs/jeans_instability_run.py` | **NEW**. Linear Jeans test (growth rate vs analytic). Phase 1 acceptance. |
| `scripts/validation/runs/uniform_sphere_run.py` | **NEW**. Static uniform-density sphere → analytic interior+exterior Φ. Phase 1 acceptance. |

## Test strategy

1. **Unit: residual computation** — set Φ = sin(kx)sin(ky)sin(kz),
   verify ∇²Φ matches -(k²) Φ to 7-point-stencil truncation error.
2. **Unit: restrict + prolong** — verify restrict ∘ prolong is identity
   on a smooth field (to interpolation order).
3. **Unit: smoother convergence** — Gauss-Seidel on a high-frequency
   error mode damps geometrically with the expected rate.
4. **Integration: Jeans instability** — perturb a uniform medium with
   wavelength longer than λ_J, measure growth rate, compare against
   analytic linear theory. Phase 1 acceptance.
5. **Integration: uniform sphere** — static configuration, no hydro
   advance, just call Poisson solver; compare interior (parabolic) and
   exterior (1/r) potential. Phase 1 acceptance (with periodic BC,
   compare modulo zero-mode).
6. **Integration: two-blob orbit** — two finite-extent gas blobs at
   apoapsis, evolve, verify Keplerian-like orbit (phase 2, needs
   isolated BC for clean comparison).
7. **Multi-node**: all of the above must run unchanged at 32 GPUs and
   pass to the same tolerance as a single GPU. Standard shard_map
   regression.

## Risks and open questions

1. **Coarsest-level deadlock with shard_map**. If the per-shard grid
   drops below 1 cell during V-cycle recursion, restrict/prolong break.
   Mitigation: cap N_levels at the mesh-shape floor; smooth heavily on
   the coarsest level. RAMSES handles via rank-reduction — defer.

2. **Convergence at high resolution**. For 1024³ Poisson with periodic
   BC and tol=1e-6, expect 5–10 V-cycles per timestep. Wall cost: each
   V-cycle ~5–10% of a hydro step (smoothers are 7-point stencils on
   the same grid). Estimate: ~50% wall overhead per step. Worth it for
   self-gravity physics.

3. **Memory pressure**. The V-cycle needs Φ + ρ (input) + r (residual)
   + e (correction) + 2 levels of restrict scratch — about
   `4 + 1.5 * (1 + 1/8 + 1/64 + ...)` ~6 full-grid arrays at the
   finest. At 1024³ fp32 = 17 GB global = 2 GB per shard at 8 GPUs.
   Comfortably fits in the shard_map memory budget we already cleared.

4. **Hydro time-step coupling**. Poisson is elliptic — Φ is
   instantaneously determined by current ρ. Solver runs ONCE per hydro
   step (at the start). Conservation: ρv kick uses g = -∇Φ_n, energy
   kick uses ρv_{n+1/2} · g. Standard.

5. **Isolated BC nontrivial**. Hockney's zero-padding doubles the
   linear dimension (8× memory and FFT cost), pushing 1024³ → 2048³
   FFT. Could be prohibitive at 8 GPUs even after the hydro work. The
   multipole-expansion BC is harder to implement but cheaper at
   runtime. Phase 2 deals with this.

6. **AMR vs uniform grid**. RAMSES's multigrid is AMR-native (different
   levels live on different parts of the tree). LUNA's uniform grid
   means our V-cycle is the "easy" case — uniform refinement
   everywhere. If LUNA ever goes AMR, the Poisson solver needs major
   rework.

7. **G = ?**. Need to pin down units. M82 uses `units = UnitSystem(kpc,
   1e10 Msun, km/s)`, so G in code units = `G_cgs * UnitMass / (UnitLength
   * UnitVelocity²)`. Computed once at config time, baked into the
   Poisson source `4π G ρ`.

## Open data needed before phase 1

- Validation reference: an analytic Jeans growth-rate solution at our
  fiducial M_isothermal. Can derive from textbook (Binney & Tremaine).
- Single-GPU dev-loop strategy. Smoother + V-cycle correctness needs to
  be ironable on a small Nx (64³ or 128³) before we touch the multi-GPU
  path. Plan an interactive Jupyter or short single-GPU SLURM job for
  the iteration loop.
