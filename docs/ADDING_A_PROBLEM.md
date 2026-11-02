# Adding a problem

Four levels, from no code to new physics. Every code block below is a file that was run as shown
(CPU, FP64, LUNA `main` 2026-10-30); file:line references are to that revision.

1. Run an existing problem with your own numbers: a run file.
2. Start from an initial state of your own, without touching the library: `launch(cfg, U0=...)` or `ic=...`.
3. Add a problem to the library: an IC module, its `Params`, one registry line, a test.
4. Add physics (a source term): a stage in `build_stages`.

## 1. A run file with an existing problem

The registered problems are the keys of `_IC_REGISTRY` (`src/luna/ic_generation/ic.py:46`); their
own numbers are the `Params` dataclass of each IC module, listed per problem in
[CONFIG.md](CONFIG.md). A run file defines `config(...)` returning a `Config`; its arguments
become command-line parameters, every other `name=value` is a `Config` field, and
`problem.<name>=value` sets a field of `cfg.problem` (`src/luna/setup/cli.py:121`). Unknown names
are refused, with the closest matches.

```python
"""My Sedov run: python main.py my_run.py n=32 problem.E=2.0 t_end=0.02"""
from dataclasses import replace

from luna.ic_generation import sedov3d
from luna.template_config import Config


def config(n=64):
    """The run at n^3; n is a command-line parameter of this file."""
    return replace(Config(), test_problem="Sedov-Blast-3D", SPATIAL_DIMENSION=3,
                   Nx=n, Ny=n, Nz=n, BOUNDARY=1, enable_x64=True,
                   problem=sedov3d.Params(E=1.0, r_blast=1.24),
                   t_end=0.05, time_bet_snaps=0.025, PLOT_AFTER_SIM=False,
                   enable_checkpointing=False, output_dir="snapshots/my_sedov",
                   clear_output_dir=True)
```

```bash
python main.py my_run.py n=32 problem.E=2.0 t_end=0.02
```

The same Config runs from Python with `luna.launch(cfg)` (what `main.py` calls,
`src/luna/setup/cli.py:307`); the notebooks in [examples/notebooks/](../examples/notebooks/) do
that. A run file may instead hold a built Config (`Config = replace(Config(), ...)`, as
`blob_run.py` in section 3); a file with both or neither is refused (`src/luna/setup/cli.py:182`).

## 2. Your own initial state, no library change

`launch` takes a state of your own, in the run's own dimension, as the **conserved** variables
with energy last, the layout a registered IC returns: `(3, Nx)` (rho, rho vx, E) in 1D,
`(4, Nx, Ny)` (rho, rho vx, rho vy, E) in 2D, `(5, Nx, Ny, Nz)` in 3D
(`src/luna/ic_generation/user_ic.py`). Two forms:

- `launch(cfg, ic=f)`: `f(x, y, z, cfg)` gets the centres of the cells it is to fill,
  x = (i + 1/2) Lx / Nx from 0 (meshgrid arrays of those cells' shape; `y`, `z` are None below
  the dimension), and returns their state. With several processes each calls it for its own part
  of the grid only (the multi-process branch of `grid_from_ics`, `src/luna/setup/ics.py:193`), so
  this form scales to any grid.
- `launch(cfg, U0=U)`: the whole grid as one array; each process keeps its own part.

`luna.conserved(rho, (vx, ...), p, gamma)` turns primitives into that state. Either goes through
`initial_conditions` (`src/luna/ic_generation/ic.py`) on the path `python main.py` takes, sharded
like a registered problem; `test_problem` then only names the run (any name). A state of the
wrong shape (a 3D array for a 2D run, one cell too many, `ic` returning another shape),
non-finite values, or a config that brings its own state (`ic_dir`, a restart, AMR, disc
damping) are refused before the run starts.

**A 2D state as a function** (the first form):

```python
"""A user-defined initial state (2D: a hot spot next to a dense cloud) run through luna.launch."""
from dataclasses import replace

import numpy as np

from luna import Config, conserved, launch


def hot_spot_and_cloud(x, y, z, cfg):
    """(rho, rho vx, rho vy, E) on the cells with centres x, y (z is None in 2D)."""
    rho = 1.0 + 9.0 * (np.hypot(x - 0.65, y - 0.5) < 0.1)       # a cloud 10x denser
    p = 1.0 + 99.0 * (np.hypot(x - 0.3, y - 0.5) < 0.05)         # a hot spot
    return conserved(rho, (0.0, 0.0), p, cfg.gamma)              # at rest


cfg = replace(Config(), test_problem="Hot-Spot", SPATIAL_DIMENSION=2, Nx=64, Ny=64, Lx=1.0, Ly=1.0,
              BOUNDARY=0, gamma=5.0 / 3.0, enable_x64=True, t_end=0.05, time_bet_snaps=0.025,
              PLOT_AFTER_SIM=False, enable_checkpointing=False,
              output_dir="snapshots/my_ic_run", clear_output_dir=True)
res = launch(cfg, ic=hot_spot_and_cloud)
print(res.t, res.n_steps, np.asarray(res.extra["U"]).shape)
```

```bash
python my_ic.py          # prints 0.05 69 (4, 64, 64): t, steps, the 2D state
```

**A 1D state as one array** (the second form), and the refusal of a state that does not fit:

```python
"""A 1D state as one array (U0): Sod's left and right states, 256 cells."""
from dataclasses import replace

import numpy as np

from luna import Config, conserved, launch

n = 256
x = (np.arange(n) + 0.5) / n                                     # cell centres
left = x < 0.5
U = conserved(np.where(left, 1.0, 0.125), (0.0,), np.where(left, 1.0, 0.1), 1.4)   # (3, n)

cfg = replace(Config(), test_problem="My-Tube", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0, BOUNDARY=1,
              gamma=1.4, enable_x64=True, t_end=0.2, write_snapshots=False, PLOT_AFTER_SIM=False,
              enable_checkpointing=False, output_dir="snapshots/my_tube", clear_output_dir=True)
res = launch(cfg, U0=U)
print(res.t, res.n_steps, np.asarray(res.extra["U"]).shape)
launch(replace(cfg, SPATIAL_DIMENSION=2, Ny=4), U0=U)           # refused: U is 1D
```

```bash
python my_tube.py        # prints 0.2 140 (3, 256), then refuses the second run:
# ValueError: U0 has shape (3, 256); SPATIAL_DIMENSION=2, Nx=256, Ny=4 needs (4, 256, 4): (rho, rho vx, rho vy, E)
```

A state that is expensive to build (a 1024³ turbulent cloud) can be written once with
`write_ic(cfg, dir, U0=U)` (`src/luna/ic_generation/ic_file.py`) and read by any later run with
`cfg.ic_dir=dir`: `U.npy` in the same layout, 1D, 2D or 3D, and `meta.json`, a fingerprint of the
config (`test_problem`, grid, box, gamma, units) that the run checks, so a file written for
another grid is refused; each rank reads only its own slab.

[examples/notebooks/12_custom_problem.ipynb](../examples/notebooks/12_custom_problem.ipynb) does
the first form end to end with plots and the conservation check; notebooks 03-06 and 08-10 run
Toro's shock tubes, the 2D Sedov and Noh problems, linear waves, KH, the Jeans waves and the
top-hat collapse from states of their own with `U0`.

## 3. A new problem in the library

The pattern is `src/luna/CLAUDE.md` ("Plugging in"): an IC module with `Params`, an entry in
`_IC_REGISTRY`. The example: a Gaussian density blob at constant pressure, advected across a
periodic box, which comes back to its initial state after one crossing.

**The IC module**, `src/luna/ic_generation/gaussian_blob_2d.py`:

```python
"""A Gaussian density blob at constant pressure advected at (vx, vy): periodic, exact after one crossing."""
from dataclasses import dataclass

import jax.numpy as jnp

from ._grid import local_range


@dataclass(frozen=True)
class Params:
    """Gaussian-Blob-2D's parameters, cfg.problem."""
    amp: float = 1.0        # peak density excess over 1
    width: float = 0.1      # Gaussian sigma, units of Lx
    vx: float = 1.0
    vy: float = 1.0


def gaussian_blob_2d(cfg, x_slice=None, y_slice=None):
    """U = [rho, rho vx, rho vy, E] on this shard's cells: the blob at the box centre, p = 1."""
    pr = cfg.problem
    if not pr.width > 0.0:                    # refuse, don't fall back to a default
        raise ValueError(f"problem.width={pr.width}: Gaussian-Blob-2D needs a positive width")
    x0, x1 = local_range(cfg.Nx, x_slice)
    y0, y1 = local_range(cfg.Ny, y_slice)
    x = (jnp.arange(x0, x1) + 0.5) * cfg.Lx / cfg.Nx - 0.5 * cfg.Lx
    y = (jnp.arange(y0, y1) + 0.5) * cfg.Ly / cfg.Ny - 0.5 * cfg.Ly
    r2 = x[:, None] ** 2 + y[None, :] ** 2
    rho = 1.0 + pr.amp * jnp.exp(-r2 / (2.0 * (pr.width * cfg.Lx) ** 2))
    E = 1.0 / (cfg.gamma - 1.0) + 0.5 * rho * (pr.vx ** 2 + pr.vy ** 2)
    return jnp.stack([rho, rho * pr.vx, rho * pr.vy, E])
```

- `Params` is a frozen dataclass named `Params` in the IC module; `Config.__post_init__` fills
  `cfg.problem` with its defaults when it is left `None` (`src/luna/template_config.py:306`,
  `problem_params` at `src/luna/ic_generation/ic.py:101`), and the comment after each field
  becomes its description in CONFIG.md.
- The function takes the slices of its dimension (`x_slice`, `y_slice`, and `z_slice` in 3D) and
  builds **only this shard's cells**: with several processes each rank calls it for its own slab
  (`src/luna/setup/ics.py:193`). `local_range` turns a slice into a cell range (whole axis without
  one). Never build the full grid and cut it: that is the multi-node deadlock in
  `src/luna/CLAUDE.md` ("Sharded data").
- It returns the conserved state, energy last: `(3, Nx)` 1D, `(4, Nx, Ny)` 2D, `(5, Nx, Ny, Nz)` 3D.
- Bad parameters are refused with a message naming them, never silently replaced (the project's
  rule, `src/luna/setup/run_kind.py:31`). `Config.validate()` (`src/luna/template_config.py:314`)
  is where checks on `Config` fields go.
- Boundary conditions are not part of the problem: `BOUNDARY` is one code for every face, 0
  periodic, 1 zero-gradient, 2 reflecting, 4 diode outflow (`src/luna/template_config.py:26`). A
  problem that needs another boundary (an inflow face) needs a new boundary code in the halo
  (`src/luna/utils/halo_local.py`), not an IC trick.

**Register it**, in `src/luna/ic_generation/ic.py` (an import, and one entry: name ->
(function, dimension)):

```python
from .gaussian_blob_2d import gaussian_blob_2d
...
    "Gaussian-Blob-2D": (gaussian_blob_2d, 2),
```

A problem that needs comoving integration also goes into `_COSMOLOGICAL_PROBLEMS` (line 114), one
that sets its own grid (a thin tube) into `_RESOLVE_REGISTRY` (line 125).

**Run it**, with a parameter from the command line:

```python
"""The registered Gaussian blob: python main.py blob_run.py problem.amp=0.5 Nx=32 Ny=32"""
from dataclasses import replace

from luna.template_config import Config

Config = replace(Config(), test_problem="Gaussian-Blob-2D", SPATIAL_DIMENSION=2, Nx=64, Ny=64,
                 BOUNDARY=0, enable_x64=True, t_end=1.0, time_bet_snaps=0.5, PLOT_AFTER_SIM=False,
                 enable_checkpointing=False, output_dir="snapshots/blob", clear_output_dir=True)
```

```bash
python main.py blob_run.py problem.amp=0.5 Nx=32 Ny=32
python main.py blob_run.py problem.width=0 Nx=32 Ny=32    # refused: ValueError: problem.width=0.0: ...
```

**Document it**: `python scripts/gen_config_md.py` regenerates the tables of
[CONFIG.md](CONFIG.md) from `template_config.py` and every registered module that declares
`Params`; the new problem appears as

```
### Gaussian-Blob-2D (`luna/ic_generation/gaussian_blob_2d.py`)

| field | type | default | values |
|---|---|---|---|
| `amp` | `float` | `1.0` | peak density excess over 1 |
| `width` | `float` | `0.1` | Gaussian sigma, units of Lx |
| `vx` | `float` | `1.0` |  |
| `vy` | `float` | `1.0` |  |
```

(a field without a comment has an empty description: write one).

**Test it**, `tests/test_gaussian_blob_2d.py`. The standard (docs/TEST_PLAN.md, "Standards for
an accuracy test"): compare with an exact or published solution and say where it comes from;
bounds about 1.3 x the measured value, orders within -0.1 .. +0.3 of the measured one, with the
measurement written next to the bound; a test that shows the bound would catch a broken run
(teeth); FP64, HLLC, minmod unless the test is about something else. `tests/_helpers.py` has the
drivers: `accuracy_cfg` (the test configuration), `evolve` (the production step on a 1-device or
sharded mesh; it also checks mass, energy and momentum to round-off in a closed or periodic box),
`check` (prints `MEASURED name = value (<= bound)` and asserts).

```python
"""Gaussian-Blob-2D: after one diagonal crossing of the periodic box the exact solution is the initial state."""
import numpy as np

from tests._helpers import accuracy_cfg, check, evolve

NS = (32, 64)
# L1(rho) / amp after one crossing, CPU FP64 minmod HLLC, bound 1.3 x measured (measured 0.0187, 0.00627)
L1_BOUND = {32: 0.024, 64: 0.0082}
ORDER = (1.47, 1.87)      # fitted order over NS, measured 1.57; band -0.1 .. +0.3


def l1(n, frac=1.0):
    """L1(rho(T) - rho(0)) / amp at frac of one crossing T = Lx / vx."""
    from luna.ic_generation.ic import initial_conditions
    cfg = accuracy_cfg(test_problem="Gaussian-Blob-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, BOUNDARY=0)
    U0 = np.asarray(initial_conditions(cfg))
    U, _ = evolve(cfg, U0, frac * cfg.Lx / cfg.problem.vx)     # periodic: mass, energy, momentum checked too
    return float(np.mean(np.abs(U[0] - U0[0]))) / cfg.problem.amp


def test_blob_returns():
    """L1 at each N under its bound, and the order inside its band."""
    e = [l1(n) for n in NS]
    for n, v in zip(NS, e):
        check(f"blob N={n} L1(rho)/amp", v, L1_BOUND[n])
    order = -np.polyfit(np.log(NS), np.log(e), 1)[0]
    check("blob L1 order", order, ORDER[0], lower=True)
    check("blob L1 order", order, ORDER[1])


def test_bound_catches_a_short_run():
    """Teeth: stopped at 0.95 of the crossing, the blob is off its start and misses the bound."""
    assert l1(64, frac=0.95) > L1_BOUND[64]
```

The bounds above are 1.3 x the values measured with this exact file (CPU, 4 cores, FP64):
L1 = 0.0187 (32²), 0.00627 (64²), order 1.57; the run stopped at 0.95 of the crossing gives
L1 = 0.0346 at 64², 4 x the bound. Then time the new file for the tier split:
`python tools/run_tests.py --tier full --update-costs` and copy its line into `tests/tiers.json`.
Add a row to [CLAIMS.md](CLAIMS.md) only for a claim the README or docs make.

## 4. New physics: a stage

A step is: each stage's `before(state, h, dt)` in order, the hydro step, each stage's
`after(state, h, dt)` in reverse order, with `h = dt / 2` (Strang splitting;
`src/luna/setup/evolve.py:36`, docs/ARCHITECTURE.md). The pieces:

- `src/luna/sources/stages.py`: the stage classes. `CellLocal(name, apply)` (line 7) is the
  simplest: `apply(U, h)` is applied before and after the hydro; a stage may also define
  `dt_limit(state)` to bound the time step (`Wind`, line 66; `Radiation`, line 156).
- `src/luna/sources/<name>.py`: `setup_<name>(cfg, ...)` returns the stage's ingredients, or `None`
  when the feature is off; `setup_external_gravity` (`src/luna/sources/external_gravity.py:111`) is a
  short example.
- `src/luna/setup/assemble.py:119` `build_stages`: appends each stage; the order of the list is the
  step order.
- Its switch and parameters are `Config` fields (`src/luna/template_config.py`, one-line comment,
  checks in `validate()`), then `python scripts/gen_config_md.py`.

Some physics needs no code: an external potential is `external_potential` (docs/CONFIG.md),
cooling, OU forcing and self-gravity are switches. A stage takes part in the differentiable
rollout (`luna.autodiff.rollout.build_simulate`) as it is, if its update is written in JAX.
