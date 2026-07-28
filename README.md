# LUNA

**A differentiable astrophysical MHD code for turbulence, shocks, ISM, and galaxy evolution**

A high-performance JAX-based hydrodynamics simulation code for solving the Euler equations in 1D, 2D, and 3D with **single-GPU and multi-GPU support**.

**Highlights**

- Unsplit 2nd-order **MUSCL-Hancock** finite-volume scheme (PLM + Hancock predictor), HLLC / L-HLLC / HLL Riemann solvers
- **Multi-GPU / multi-node** via JAX `shard_map` with explicit `ppermute` halos — validated to 256 GPUs (FP64, 4096³), NVIDIA **and** AMD
- **End-to-end differentiable** through the production solver (forward + reverse mode), including the FMG-Poisson gravity adjoint
- Self-gravity (multigrid FMG, periodic / isolated Guillet-Teyssier BC), radiative cooling, source terms, sink particles
- Cosmology: super-comoving integration, PM dark matter, coupled gas+DM boxes from MUSIC ICs
- Block-structured **AMR** (dimension-generic, Berger-Oliger subcycling, machine-precision conservation)

The full **results & validation gallery** (shock tubes, KH, Sedov, scaling,
autodiff showcases, science applications) lives in
[docs/GALLERY.md](docs/GALLERY.md).

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Single-GPU Usage](#single-gpu-usage)
3. [Multi-GPU Usage](#multi-gpu-usage)
4. [Production usage](#production-usage)
5. [Performance](#performance)
6. [Block-AMR](#block-amr-experimental)
7. [Results gallery](docs/GALLERY.md)
8. [Citation](#citation) · [License](#license) · [Contributing](#contributing) · [Support](#support)

---

## Quick Start

### Setup

1. **Configuration**: Copy the template configuration file:
   ```bash
   cp template_config.py config.py
   ```
   Edit `config.py` to set your simulation parameters (grid size, test problem, etc.)

2. **Dependencies**: install the package (pulls the pinned runtime deps from
   `pyproject.toml`; Python ≥ 3.11):
   ```bash
   pip install -e .            # runtime (CPU JAX)
   pip install -e ".[plot]"    # + matplotlib, for the figure scripts
   ```
   For GPU runs, install the matching CUDA build of JAX instead of the CPU
   wheel, e.g. `pip install -U "jax[cuda12]==0.10.1"` (see the
   [JAX install guide](https://docs.jax.dev/en/latest/installation.html)).

### Run Simulation

```bash
# Single- or multi-GPU: main.py initializes JAX distributed automatically
# (single process uses all visible GPUs; under SLURM, srun spawns one
# process per GPU and they coordinate via JAX_COORDINATOR_PORT).
python main.py

# Validation problems run through the runner instead (see submit/ for
# SLURM templates):
python -m scripts.validation.runs.sedov3d_run
```

---

## Single-GPU Usage

### Configuration

`config.py` is a dataclass (copied from `template_config.py`); edit the field
defaults. The knobs that matter most:

- **Test problem**: `test_problem` — any name registered in
  `ic_generation/ic.py` (e.g. `"Sod-Shock"`, `"Kelvin-Helmholtz"`,
  `"Sedov-Blast-3D"`, `"Boss-Bodenheimer-3D"`, `"CC85-Wind-3D"`)
- **Dimension / grid**: `SPATIAL_DIMENSION`, `Nx/Ny/Nz`, `Lx/Ly/Lz`
- **Scheme** (time integration is always unsplit MUSCL-Hancock):
  - `ORDER` — 0 = first-order Godunov, 1 = PLM (2nd order, default)
  - `LIMITER` — 0 = MinMod, 1 = MonCen, 2 = VanLeer
  - `riemann_solver` — `"hllc"` (default), `"hll"` (more dissipative),
    `"lhllc"` (low-Mach fix for stratified/subsonic flows)
  - `CFL` — 0.3–0.5 for 3D unsplit; up to 0.8 for 1D
- **Boundaries**: `BOUNDARY` — 0 = periodic, 1 = outflow/zero-gradient,
  2 = reflective
- **Self-gravity**: `SELF_GRAVITY=True` + `GRAVITATIONAL_G`;
  `POISSON_BOUNDARY` — inherit (None), 0 = periodic (Jeans swindle),
  3 = isolated (multipole-Dirichlet FMG)

### Example Configuration

```python
# config.py — fields inside the Config dataclass
test_problem: str = "Kelvin-Helmholtz"
SPATIAL_DIMENSION: int = 2
Nx: int = 2048
Ny: int = 2048
Lx: float = 1.0
Ly: float = 1.0
t_end: float = 2.0
CFL: float = 0.5
ORDER: int = 1        # PLM
LIMITER: int = 1      # MonCen
BOUNDARY: int = 0     # periodic
```

### Run Simulation

```bash
python main.py
```

For parameter sweeps and validation problems, skip `config.py` and use the
programmatic runner, which applies overrides on top of the template and goes
through the same `main.run` entry path as `python main.py`:

```python
from scripts.validation.runner import sweep

# Vary a single field; output_dir_template gets the scan value.
sweep(
    base_overrides=dict(test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3),
    scan_field="Nx",                       # Ny, Nz default to the same value
    scan_values=[128, 256, 512],
    output_dir_template="snapshots/sedov_{}",
)
```

The per-problem run scripts under `scripts/validation/runs/` (invoked as
`python -m scripts.validation.runs.sedov3d_run`) wrap this for each test.

---

## Multi-GPU Usage

###  Overview

Multi-GPU runs use a single execution path: **shard_map** with explicit
`ppermute` halo exchanges. Whenever the device mesh spans more than one GPU,
the whole MUSCL-Hancock step (and the FMG-Poisson solve) runs per-shard, with
each shard trading one-cell boundary slabs with its face-neighbors via ring
shifts. This sidesteps the working-set blowup of compiler-inferred sharding
and is what the 1024³ production runs use. No flag is needed — it is selected
automatically from the mesh size.

Mesh decomposition is controlled by `DECOMPOSITION` ("auto"/"slab"/"block")
or pinned exactly with `MESH_SHAPE` (e.g. `(2, 2, 4)` for 16 GPUs on 1024³);
the run scripts also honour a `LUNA_MESH_SHAPE` env var (comma-separated,
e.g. `LUNA_MESH_SHAPE=2,2,4`).

### Quick Start

```bash
# Single node: JAX automatically uses all visible GPUs
python main.py

# Multi-node under SLURM: one process per GPU via srun (see submit/ templates)
srun python -u -m scripts.validation.runs.sedov3d_run
```

### Multi-process rules (hard-won)

- Build per-cell sharded arrays from local slices via
  `make_array_from_single_device_arrays` — a full-array `device_put`
  deadlocks silently multi-node.
- Pass sharded arrays as JIT *arguments*, never capture them in Python
  closures (XLA tries to inline them as constants and crashes).
- 1024³ float32 3D needs ≥16 GPUs with `MESH_SHAPE=(2, 2, 4)`; the MUSCL
  working set (~23 full-shard arrays) does not fit 8 A100-40GB.

---

## Production usage

For long-running production campaigns:

- **NaN abort guard**: the time loop checks for non-finite state after every step (free piggyback on `compute_dt`). On detection it retries the step with halved CFL up to **8 times** (i.e., `dt → dt/256`). If the state is still bad it dumps an emergency checkpoint (`cfg.enable_checkpointing` is default-on) and raises `RuntimeError` so SLURM exits non-zero. No more silent multi-hour NaN'd runs.
- **Periodic diagnostics**: every `cfg.diag_interval` steps the runner prints `min(rho), min(P), max(|v|), max(M)` and appends a row to `<output_dir>/history.csv` (Athena++-style `.hst` analog) for offline analysis.
- **Determinism**: validated submit scripts in `submit/` export `XLA_FLAGS="--xla_gpu_deterministic_ops=true"`. Together with `enable_x64=True` (set in run scripts) this gives bitwise-reproducible runs at fixed sharding.
- **Source terms** live in `sources/` (`ou_forcing`, `cc85`, `external_gravity`, `cooling`, `self_gravity`). Each returns a `{"apply": fn(U, dt)}` callable that the runner operator-splits after the hydro step. Adding a new source term means writing one `sources/X.py` and one line in `setup/runner_distributed.py`.
- **Multi-node**: the validated pattern (`submit/examples/example_production_3d_multinode.sh`) uses `--mem=500000`, `--constraint=gpu`, `--gres=gpu:4`, `JAX_COORDINATOR_PORT=12345`; see the "Multi-process rules" above for the sharded-array patterns that make it work.

---

## Performance

For production 3D the relevant numbers are from the shard_map path:

- **Sedov 1024³ float32**: runs on 16 A100-40GB (4 nodes × 4),
  `MESH_SHAPE=(2, 2, 4)`. 8 GPUs OOMs — the unsplit MUSCL working set needs
  the 16-GPU shard size.
- The explicit per-shard `ppermute` pipeline avoids the large scratch buffers
  that compiler-inferred 3D sharding materializes, giving a major step-time
  improvement at 1024³.
- Keep ≥256³ cells per GPU for good compute/halo ratio.

---

## Block-AMR (experimental)

A LUNA-native, dimension-generic block-structured AMR layer (`amr/`), wired
through `main.run` via `cfg.AMR=True` → `setup/runner_amr.py`. Everything is pure
JAX (jit/vmap/scan/shard_map).

**Validation — Kelvin-Helmholtz to t=5, AMR (128² base → 1024² effective) vs uniform 1024²:**

![KH: fixed grid vs block-AMR with refinement mesh overlaid](code_paper_figs/amr_kh_prototype.png)

Refinement mesh overlaid on the AMR panels (block edges colored by level). The AMR
refines the shear sheets while the flow is laminar (t=1), then **dynamically tracks
the vortices** as they roll up (t=2–3) and the mixing layer as it goes turbulent
(t=4–5). Fields match the uniform run through the linear/early-nonlinear phase;
late-time pointwise divergence is expected (KH turbulence is chaotic), with the
large-scale structures consistent.

**Validation — 3D Sedov-Taylor SN blast in physical units (AREPO: pc, km s⁻¹, M⊙; 10⁵¹ erg into an n=1 cm⁻³, 10⁴ K ISM), AMR (64³ base → 512³ effective) vs uniform 64³:**

![3D Sedov: fixed 64³ vs block-AMR with refinement mesh overlaid](code_paper_figs/amr_sedov3d_evolution.png)

Central z=Lz/2 density slice (log scale) at 2500 / 5000 / 7500 / 10000 yr. The uniform
64³ shock is a blocky Cartesian diamond; the AMR — identical base resolution and physics
— resolves a crisp spherical shell. The refinement mesh (block edges colored by level)
**hugs the shell**: the combined criterion (relative ∇P, OR relative ∇ρ gated by
overdensity) refines the shock and compressed shell while the smooth hot interior
de-refines back to the coarse levels as the remnant expands. Shell density jump ≈4× (the
γ=5/3 Sedov strong-shock limit), NaN-free, reflection-symmetric to <0.02%, and the full
64³→512³ run takes ~400 s on a single A100.

**Validation — self-gravitating collapse on AMR: Evrard spherical collapse, 64³ base → 512³ effective:**

![Evrard collapse: radial density profile (dense vs AMR) + refinement mesh](code_paper_figs/evrard_amr_t075.png)

Cold spherical cloud in free-fall self-gravity, at t = 0.75 (~0.68 t_ff). **Left:** the
spherically-averaged density profile — the AMR run (orange) matches the dense 64³ reference
(blue) from the envelope through the accretion-shock break at r≈0.12, then continues the
collapse profile **another decade inward** where the fixed 64³ hits its resolution floor
(core ρ≈236 vs 41). **Right:** the mesh — nested quasi-Lagrangian refinement (level 1 cyan,
2 yellow, 3 green) wraps the cloud and piles onto the collapsing core, ambient stays coarse.
This is the AMR hierarchy coupled to the **level-by-level Dirichlet multigrid Poisson solver**
(`amr/poisson.py`): L0 by the dense FMG, refined levels by within-block V-cycles with
linearly-prolonged Dirichlet ghosts — gravity + hydro + dynamic regridding all cooperating.
Validated on Jeans growth (dense 64³ +0.06%, AMR full chain +0.55% of the analytic σ).

**What works (2D + 3D, end-to-end through `main`):**
- Block layer: x64 Morton ids, conservative prolong/restrict, O(N log N) neighbour
  finding, NaN-sentinel padding.
- Hydro on blocks: per-block MUSCL-Hancock; same-level seam fluxes (mass & energy
  conserved to machine precision); coarse-fine reflux.
- Dynamic regridding on the live solution. `amr_criterion` is pluggable:
  `"density"` (|grad ρ|, the default), `"vorticity"` (|curl v|), `"pressure"`,
  `"pressure_density"`/`"prho"` (relative ∇P OR gated ∇ρ), `"overdensity"`,
  `"jeans"`, `"lohner"`, or `"static"`.
- Subcycling (Berger-Oliger): level L advances at `dt_base / 2**L`, with
  time-accumulated coarse-fine reflux default-on in both the synchronized and
  subcycled integrators.
- Multi-GPU via `jax.shard_map` (1-GPU-per-process): blocks sharded across devices.

**Config knobs** (`template_config.py`, the `amr_*` block): `AMR`, `amr_levels`,
`amr_block_shape`, `amr_criterion`, `amr_thresholds`, `amr_subcycle`,
`amr_shard`, `amr_regrid_interval`, `amr_block_bucket`, `amr_max_blocks`,
`amr_capacities`, `amr_rho_floor`, `amr_rho_floor_frac`, `amr_p_floor`,
`amr_dense_snapshots` — see the commented template for semantics.

**Benchmark — KH to t=5 on 4×A100 (measured Mcells/s):** at a saturating base
(1024²) the AMR compute runs at ~90% of the dense path (~750 vs ~830 Mcells/s);
the remaining overhead is the full-hierarchy regrid on the base-step cadence.

**Open / next:** cosmological hardening (DM coupling on AMR is Phase-2; dual-energy
on AMR not yet implemented); distributed-performance work at small per-GPU loads
(hybrid dense-L0 base level, `amr/hybrid.py`, in progress).


## Citation

If you use this code in your research, please cite:

```bibtex
@software{luna,
  author = {Steinwandel, Ulrich Philipp},
  title  = {LUNA: A Differentiable Astrophysical Hydrodynamics Code},
  year   = {2026},
  url    = {https://github.com/Ulli1991/LUNA}
}
```

---

## License

BSD 3-Clause License — see [LICENSE](LICENSE).

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## Support

For questions or issues:
- Open an issue on GitHub
- Check this documentation

---
