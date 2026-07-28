# LUNA validation harness

End-to-end validation for LUNA. Driven from Python (not a SLURM submit
script) so resolution sweeps and figure assembly happen in one place.

## Layout

```
scripts/validation/
  runner.py            Programmatic LUNA driver (Config overrides + sweep)
  athenak_io.py        AthenaK .athdf loader + format introspection
  metrics.py           L1/L2 error, interpolation, radial-profile binning
  sedov_analytic.py    Sedov-Taylor self-similar solution (Kamm 2000 closed form)
  runs/                One run script per test problem; each builds a Config
                       and calls main.run (blast/turb emergence, BB79, CC85,
                       Evrard, Gresho, Jeans, KH, M82, Sedov, Sod-AMR, solar
                       convection, autodiff inversions, ...)
  plots/               One figure script per paper/README figure (~35; run the
                       matching runs/ script first where one exists)
```

Notable `plots/` entry points (see `ls scripts/validation/plots/` for the
full set):

```
soundwave_convergence.py       1D: L1/L2 vs Nx, demonstrates 2nd order
shock_panels.py                1D: 2x2 panels for Sod / Shu-Osher / WC
kh_grid.py                     2D KH: resolution series density grid
square_advection.py            2D: time progression at 1024^2
sedov3d_plot.py                3D Sedov: slices + radial profile vs analytic
amr_sedov_benchmark.py         AMR vs uniform vs Kamm-2000 analytic (L1, R_s, dM)
amr_sod_benchmark.py           AMR Sod across C/F interfaces vs exact Riemann
evrard_radial.py               Self-gravitating collapse radial profiles
scaling_plot_fp64.py           FP64 weak/strong scaling figure (MI300A / A100)
```

## Usage

Each plot script is a `python -m` entry point that:
1. Runs the required LUNA simulations via `runner.sweep` (or reads snapshots
   an earlier `runs/` script produced).
2. Loads the AthenaK reference (where applicable) via `athenak_io.load_1d`.
3. Computes metrics or extracts profiles via `metrics.*`.
4. Writes a PDF/PNG to `figures/` (paper copies live in `code_paper_figs/`).

```bash
# from the repo root
python -m scripts.validation.plots.soundwave_convergence
python -m scripts.validation.plots.shock_panels
python -m scripts.validation.plots.kh_grid
```

Multi-GPU runs require launching under SLURM (see `submit/examples/`),
pointing the wrapper at a `python -m scripts.validation.*` entry point
instead of `main.py`.

## Reference data

Place AthenaK reference snapshots under `reference/athenak/`:

```
reference/athenak/
  soundwave_ppm_8192.athdf
  sod_ppm_8192.athdf
  shu_osher_ppm_8192.athdf
  woodward_colella_ppm_8192.athdf
```
