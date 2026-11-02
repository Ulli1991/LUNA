# Your first hour with LUNA

A single walk-through: run one shock-tube simulation, find the output, look at
it. Everything here uses the standard path described in
[ARCHITECTURE.md](ARCHITECTURE.md) — read that side by side.

## 1. Install

```bash
git clone https://github.com/Ulli1991/LUNA.git luna && cd luna
python -m venv venv && source venv/bin/activate
pip install -e ".[plot]"        # CPU JAX; fine for this tutorial
```

(For a GPU, install the matching `jax[cuda12]` / ROCm wheel instead — see the
README. Everything below is identical.)

## 2. Configure

Every knob of a run is a field of one dataclass. Make your local copy and set
five fields:

```bash
cp src/luna/template_config.py config.py
```

In `config.py` set:

```python
test_problem: str = "Sod-Shock"     # a registered IC (see src/luna/ic_generation/ic.py)
SPATIAL_DIMENSION: int = 1
Nx: int = 512
t_end: float = 0.2
PLOT_AFTER_SIM: bool = True         # auto-plot at the end (already the default)
```

Field-by-field documentation: [CONFIG.md](CONFIG.md).

## 3. Run

```bash
python main.py
```

`python main.py` reads `config.py` from the current directory. A config file
can also be named, with `field=value` overrides after it, which is how a
parameter sweep or an sbatch line changes one knob without editing the file:

```bash
python main.py my_sod.py Nx=1024 t_end=0.3
```

A name Config does not know is an error, in the file or on the command line
(it used to warn and run with the default); a field that was retired says
where its setting went.

What happens (follow along in `src/luna/main.py` — it is four calls): the mesh, the
initial conditions, the physics (`src/luna/setup/assemble.py`), then the loop
(`src/luna/setup/evolve.py`), which repeats
*pick a stable `dt` → one compiled MUSCL-Hancock step → write snapshots*.
The first step is slow — that's JAX compiling the step function; the rest are
fast. With `PLOT_AFTER_SIM` on, the density/velocity/pressure profiles are
plotted automatically when the run finishes.

## 4. Look at the output yourself

Snapshots are HDF5 directories under `cfg.output_dir` (default `snapshots/`),
one `snapdir_NNN/` per output time. Two ways in:

```python
# the bundled reader stitches multi-GPU shards for you:
from luna.file_io.read_snapshot import read_snapshot
snap = read_snapshot("snapshots/snapdir_004")
print(snap.keys())          # rho, vx, vy, vz, P, e_int, x, y, z, t, ...

# or raw h5py — fields live under the "cells" group:
import h5py, glob
with h5py.File(sorted(glob.glob("snapshots/snapdir_004/*.hdf5"))[0]) as f:
    rho = f["cells/density"][...]
```

Plot `rho` against position and you are looking at the classic Sod solution:
shock, contact discontinuity, rarefaction fan.

## 5. Where to go next

- Change `LIMITER` (0 minmod / 1 MC / 2 van Leer) and rerun — watch the
  shock sharpness change. That's the reconstruction section of
  `src/luna/hydro/muscl_hancock.py` in action.
- Go 3D: `test_problem="Sedov-Blast-3D"`, `SPATIAL_DIMENSION=3`, `Nx=Ny=Nz=64`.
- Multi-GPU: nothing to change — `python main.py` uses every visible GPU
  (see the Multi-GPU section of `docs/RUNNING.md` and `docs/ENV.md`).
- Physics add-ons: browse the `src/luna/sources/` package and the switches in
  [CONFIG.md](CONFIG.md) (self-gravity, cooling, forcing...).
- The validation problems in `scripts/validation/runs/` are ready-made
  configurations of everything above — each is one `config(...)` function of
  overrides on the same `Config` + `main.run` path you just used, run as
  `python main.py scripts/validation/runs/<name>.py nx=64` with its parameters
  and any Config field as `name=value`.
