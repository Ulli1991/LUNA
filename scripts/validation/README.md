# LUNA validation harness

End-to-end test suite for the modular MUSCL-Hancock integrator. Driven from
Python (not a SLURM submit script) so resolution sweeps and figure assembly
happen in one place.

## Layout

```
scripts/validation/
  runner.py            Programmatic LUNA driver (Config overrides + sweep)
  athenak_io.py        AthenaK .athdf loader + format introspection
  metrics.py           L1/L2 error, interpolation, radial-profile binning
  sedov_analytic.py    Sedov-Taylor self-similar solution (stub for now)
  plots/
    soundwave_convergence.py   1D: L1/L2 vs Nx, demonstrates 2nd order
    shock_panels.py            1D: 2x2 panels for Sod / Shu-Osher / WC
    kh_grid.py                 2D: 4-resolution × 5-time density grid
    square_advection.py        2D: time progression at Nx²=1024²
    sedov_radial.py            3D: spherically-averaged profiles vs analytic
    sedov_slice.py             3D: central-plane render of the blast
```

## Usage

Each plot script is a `python -m` entry point that:
1. Runs the required LUNA simulations via `runner.sweep` / `runner.run_one`.
2. Loads the AthenaK reference (where applicable) via `athenak_io.load_1d`.
3. Computes metrics or extracts profiles via `metrics.*`.
4. Writes a PDF to `figures/`.

```bash
# from the repo root
python -m scripts.validation.plots.soundwave_convergence
python -m scripts.validation.plots.shock_panels
python -m scripts.validation.plots.kh_grid
python -m scripts.validation.plots.square_advection
python -m scripts.validation.plots.sedov_radial
python -m scripts.validation.plots.sedov_slice
```

Multi-GPU runs require launching under SLURM with the usual
`submit_jax_*.sh` wrappers, just pointing them at one of the
`python -m scripts.validation.plots.*` entry points instead of `main.py`.

## Status

All plot scripts are in production use — figures appear in `figures/` and
are referenced from the top-level `README.md`. To regenerate after new
runs land on disk, just rerun the corresponding
`python -m scripts.validation.plots.*` entry point.

## Reference data

Place AthenaK reference snapshots under `reference/athenak/`:

```
reference/athenak/
  soundwave_ppm_8192.athdf
  sod_ppm_8192.athdf
  shu_osher_ppm_8192.athdf
  woodward_colella_ppm_8192.athdf
```
