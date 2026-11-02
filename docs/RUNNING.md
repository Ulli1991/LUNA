# Running LUNA

Configuration, single- and multi-GPU runs, production settings and performance. The short version is the quick start in the [README](../README.md).

## Single-GPU Usage

### Configuration

`config.py` is a dataclass (copied from `src/luna/template_config.py`); edit the field
defaults. The knobs that matter most:

- **Test problem**: `test_problem` — any name registered in
  `luna/ic_generation/ic.py` (e.g. `"Sod-Shock"`, `"Kelvin-Helmholtz"`,
  `"Sedov-Blast-3D"`, `"Boss-Bodenheimer-3D"`, `"CC85-Wind-3D"`)
- **Dimension / grid**: `SPATIAL_DIMENSION`, `Nx/Ny/Nz`, `Lx/Ly/Lz`
- **Scheme** (time integration is always unsplit MUSCL-Hancock):
  - `ORDER` — 0 = first-order Godunov, 1 = PLM (2nd order, default)
  - `LIMITER` — 0 = MinMod, 1 = MonCen, 2 = VanLeer
  - `riemann_solver` — `"hllc"` (default), `"hll"` (more dissipative; 3D
    only, diagnostic)
  - `CFL` — the Courant number, 0.8 by default:
    dt = CFL / max over cells of sum_d (|v_d| + c)/dx_d, which holds the unsplit
    step's limit (1) in any dimension ([HYDRO_TIMESTEP.md](HYDRO_TIMESTEP.md))
- **Boundaries**: `BOUNDARY` — 0 = periodic, 1 = outflow/zero-gradient,
  2 = reflective (dense grid only; not implemented on AMR), 3 = isolated
  (self-gravity multipole BC), 4 = diode
- **Self-gravity**: `SELF_GRAVITY=True` + `GRAVITATIONAL_G`;
  `POISSON_BOUNDARY` — inherit (None), 0 = periodic (Jeans swindle),
  3 = isolated (multipole-Dirichlet FMG)
- **The problem's own parameters**: `problem`, the `Params` of the IC module
  `test_problem` names (`cfg.problem.amp` for the Jeans wave). `None` means its
  defaults; set one with `problem=jeans3d.Params(amp=0.01)` in a config file
  (`from luna.ic_generation import jeans3d`) or `problem.amp=0.01` on the command
  line. [CONFIG.md](CONFIG.md) lists every problem's parameters.

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
CFL: float = 0.8
ORDER: int = 1        # PLM
LIMITER: int = 1      # MonCen
BOUNDARY: int = 0     # periodic
```

### Run Simulation

```bash
python main.py
```

From Python, build the config in code and call the same entry path
(`luna.launch` initialises JAX, then `luna.main.run`):

```python
from dataclasses import replace
from luna import Config, launch

res = launch(replace(Config(), test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3,
                     Nx=128, Ny=128, Nz=128, output_dir="snapshots/sedov128"))
```

For resolution sweeps the programmatic runner applies overrides on top of the
template for every scan value:

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

Each run script under `scripts/validation/runs/` is one function,
`config(...)`, whose arguments are the problem's parameters (`nx`,
`t_end_myr`, ...), plus a `main` that parses `name=value` from the command
line and launches; its docstring lists the parameters. Config fields and
`problem.<name>` keys follow the script's own parameters on the same command line:
`python -m scripts.validation.runs.gresho_run nx=128 problem.Mmax=0.1`.

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

The device mesh is chosen from the grid and the device count, or pinned
exactly with `MESH_SHAPE` (e.g. `(2, 2, 4)` for 16 GPUs on 1024³);
on the command line that is `MESH_SHAPE=2,2,4` after `python main.py` or any
run script.

### Quick Start

```bash
# Single node: JAX automatically uses all visible GPUs
python main.py

# Multi-node under SLURM: one process per GPU via srun (see submit/ templates)
srun python -u -m scripts.validation.runs.sedov3d_run nx=1024 MESH_SHAPE=2,2,2
```

### Multi-process rules (hard-won)

- Build per-cell sharded arrays from local slices via
  `make_array_from_single_device_arrays` — a full-array `device_put`
  deadlocks silently multi-node.
- Pass sharded arrays as JIT *arguments*, never capture them in Python
  closures (XLA tries to inline them as constants and crashes).
- 1024³ float32 3D needs ≥16 GPUs with `MESH_SHAPE=(2, 2, 4)`; the MUSCL
  working set (~23 full-shard arrays) does not fit 8 A100-40GB.
- Reduce sharded arrays with `luna/utils/mesh_reduce.py` (`grid_reduce`,
  `flat_reduce`, `grid_mean`), explicit psum/pmax/pmin in a cached shard_map.
  Never call eager `jnp.sum`/`max`/`mean` on a sharded array between
  shard_map executables: the auto-built collective wedges the ROCm runtime.
  Call them on every rank, never under `if is_main_process:`.
- MUSIC runs use Arepo units (kpc/h, km/s, 1e10 Msun/h), in which H0 = 0.1
  for any h: the length unit already carries the h. Never set it to 0.1*h.

---

## Production usage

For long-running production campaigns:

- **NaN abort guard**: a non-finite state propagates into the next step's `compute_dt`, where `require_finite` raises `RuntimeError` with the failing step, time, and a remediation hint (e.g. "lower CFL") so SLURM exits non-zero. No silent multi-hour NaN'd runs; the newest `snapdir_*` in the output dir is the last good state.
- **Periodic diagnostics**: every `cfg.diag_interval` steps the runner prints `min(rho), min(P), max(|v|), max(M)` and appends a row to `<output_dir>/history.csv` (Athena++-style `.hst` analog) for offline analysis.
- **Determinism**: the NVIDIA example in `submit/examples/` exports `XLA_FLAGS="--xla_gpu_deterministic_ops=true"` (the AMD example does not; set it yourself). Together with `enable_x64=True` (set in run scripts) this gives bitwise-reproducible runs at fixed sharding.
- **Source terms** live in `luna/sources/` (`ou_forcing`, `cc85`, `external_gravity`, `cooling`, `particles`) and enter the step as stages with a `before` and an `after` half (`luna/sources/stages.py`), so every one is Strang-split around the hydro. Adding a new source term means writing one `luna/sources/X.py` and one line in `luna/setup/assemble.py`. Gravity and the particle species (`luna/species/`) are one stage around one Poisson solve per step (`luna/gravity/kdk.py`).
- **Multi-node**: the validated pattern (`submit/examples/example_production_3d_multinode.sh`) uses `--mem=500000`, `--constraint=gpu`, `--gres=gpu:4`, `JAX_COORDINATOR_PORT=12345`; see the "Multi-process rules" above for the sharded-array patterns that make it work.

---

## Checkpoints

- **Grid checkpoint (format v2)**: each process writes
  `checkpoint_{step:06d}.{pid}.hdf5` with every shard it addresses as
  `shards/<k>/U`, plus `start`/`stop` attributes for its global index range.
  A restart builds each device's block from the stored shards that overlap
  it, so it can run on a different device count or decomposition and no rank
  ever holds the global array. Version-1 files (one `conserved/U`) still
  load. `extra/<name>` holds the small replicated state a restart must
  reproduce: the OU forcing state, its RNG key, the cosmological `a`.
- **Sets and markers**: a checkpoint set is every file of one step (gas
  shards, `radiation_{step}.{pid}.hdf5`, a coupled run's
  `coupled_{step}.npz/json`). Each file is written as `.partial` and renamed,
  and rank 0 writes `checkpoint_{step}.complete` only after every rank's files
  are in place, so a killed job never leaves a truncated file under the real
  name. A restart uses the newest set with a marker (the newest step on disk
  if a directory has no markers at all). Rank 0 decides when a set is due
  (`due_everywhere`): the write is collective, and a rank deciding on its own
  clock would hang the others.
- **Cosmological runners** write `cosmo_{step:06d}.{pid}.hdf5`: the gas grid,
  the per-device particle buffers and (a, tau, step). The grid restores onto
  any decomposition; the particle buffers are `(ndev * C, ...)` capacity
  buffers, so a restart needs the same device count and `dm_capacity_factor`
  (a mismatch raises).

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

### Where a step's time goes

`timing_every = N` runs every N-th step part by part: after each part the
device is waited for, so each part's wall time is its own. The parts are
the dt estimate (`dt.hydro`, `dt.<stage>`), each stage's `before` and
`after` (`<stage>.before`, `<stage>.after`, by stage name), `hydro`,
`boundary_mass`, the projections (`project.<name>`), `gravity.check` and
`outputs` (diagnostics, progress, snapshot and checkpoint I/O when they
happen). The gravity backend's calls show as sub-parts of the stage half
that makes them (`gravity.after/gravity.solve`, `.../gravity.kick_gas`,
`.../gravity.source`, `.../gravity.grad`; the particle kicks, exchange and
resort for dark matter). `(between parts)` is the host work in between.

The main process prints a table (part, ms as the max over ranks, the min
when the ranks differ by more than 5 %, % of the step, and `> hydro` on
every part that costs more than the hydro step) and appends one JSON line
per timed step to `<output_dir>/step_timing.jsonl`; the line also carries
the average wall time of the untimed steps since the previous timed one.
A timed step computes bitwise the same state as an untimed one; it is
slower by the waits, so compare its parts with each other, and the step
rate with the untimed figure. Untimed steps run as before. The AMUSE
stepper (`amuse_luna/stepper.py`) honours the same field.

`tools/step_breakdown.py <config or run script> [name=value ...]` times
every step of a short run (`steps=10` after `warmup=3`, snapshots and
checkpoints off unless set) and writes the median table to
`<output_dir>/step_breakdown.json` (or `json=path`); a part that runs on
some steps only (a particle exchange, a snapshot) counts 0 on the others,
and every part that exceeds twice its median on some step is listed below
the table with those steps and its largest time. Given a
`step_timing.jsonl` instead of a config, it summarizes those records.

The same names label the work in profiles: `jax.named_scope` (inside a
traced step, as in the adjoint rollout), a `jax.profiler.TraceAnnotation`
on the host, and a roctx range when rocprofv3 is attached
(`rocprofv3 --marker-trace`; `LUNA_ROCTX=1` forces it).

