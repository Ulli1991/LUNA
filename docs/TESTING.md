# Running the tests

```bash
python tools/run_tests.py                 # the fast tier, ~2 minutes
python tools/run_tests.py --tier full     # everything
pytest -m "not slow"                      # the same fast tier, one process
pytest tests/test_cli.py                  # one file
```

`tools/run_tests.py` gives each test **file** its own pytest process. On a GPU
a wedged runtime takes its process with it instead of the batch, a crash names
the file it happened in, and `--timeout-s` bounds a stall to that one file. It
writes `test_report.json`: one record per file, the counts, the host and the
Python version.

## Tiers are measured, not guessed

`tests/tiers.json` holds the measured seconds per file. The fast tier is the
cheapest files that fit `--budget-s` (120 s by default); everything else is
the slow tier. `pytest -m "not slow"` reads the same table through
`conftest.py`, so the two agree.

Re-time after adding or changing tests:

```bash
python tools/run_tests.py --tier full --update-costs
```

A file with no recorded cost counts as slow, so a new test cannot silently
inflate the fast tier. Tiering by filename (every `test_amr_*` slow) is what
left push-time refinement coverage at zero while the six cheapest refinement
suites together cost under a minute.

## Without a GPU

`conftest.py` pins a CPU backend with 8 virtual devices, so everything runs
anywhere. `--devices N` overrides that per run: `--devices 2` is the smallest
mesh that exercises a halo exchange, and is the cheapest way to cover the
sharded paths on a laptop.

```bash
python tools/run_tests.py --tier fast --devices 2
python tools/run_tests.py --tier full --lint      # + ruff check
```

`--lint` fails when ruff is missing instead of skipping, so a lint that is
configured but not installed cannot pass unnoticed.

## Silence is the failure mode

Every failure worth catching here has looked like nothing at all: blank pytest
output, a sweep killed on the wall clock, a node fault. So the check is on the
report, not on a log:

```bash
python tools/run_tests.py --check-report --max-age-h 30
```

It exits non-zero when the report is missing, stale or failing. Wire that one
line into whatever you already watch.

## Skips are counted, and judged across configs

A plugin (`tools/pytest_plugins/luna_outcomes.py`) records every test's
outcome, so a file is `pass`, `partial` (some tests skipped), `skip` (nothing
ran), `fail` or `timeout`; a fully skipped file is no longer a pass. Which
tests skip depends on the config (GPU tests skip on CPU), so the judgement is
over all standard configs at once:

```bash
python tools/run_tests.py --check-coverage gpu1.json gpu2.json dev8_full.json dev2_fast.json
```

It fails when a test is skipped in every report, or a test file is in none,
unless `tests/skip_allowlist.json` lists it with a written reason.
`submit/examples/nightly_tests.sbatch` writes one report per config.

## Continuous integration

`.github/workflows/tests.yml` runs three CPU jobs on every push and pull
request, all through the same runner so a laptop, CI and the nightly GPU job
select and report identically: the fast tier on three Python versions, the
fast tier again on two virtual devices (the smallest mesh with a halo
exchange), and `ruff check`. Each uploads its `test_report.json`.

What CI cannot cover is the slow tier and anything needing a real GPU or a
multi-process launch; that is what the nightly batch job below is for.

## Scheduling it

The runner is portable and knows nothing about any machine. How it is invoked
is per-site and disposable. `submit/examples/nightly_tests.sbatch` is one
example: a batch job that re-arms itself **at the start** of each run, so a
killed, stalled or never-started job cannot break the chain. A login crontab
cannot do that, since it lives on one login node and stops when that node
drains.

## What needs a real multi-GPU launch

`conftest.py` pins a CPU backend with 8 virtual devices, so the whole suite
runs anywhere. Tests marked `multigpu` need a real multi-process launch and
skip unless `LUNA_MULTIGPU_TEST=1`. The two-process checks that pytest cannot
start live in `scripts/validation/runs/*_2proc.py` and carry their `srun`
recipe in their docstring.

## Known failures

Three tests fail on the pinned stack and are tracked in
`docs/history/AUDIT_2026-09-16.md`:

| test | jax 0.10.0 | jax 0.10.2 |
|---|---|---|
| `test_sedov_gradients` | segfault | passes |
| `test_poisson_adjoint` | 2.8e-3 | 2.8e-3 |
| `test_poisson_bidiff` | failed | fixed (solver tightness) |

Only the first is a stack problem; the other two were found by running them on
both stacks and seeing the number not move. The pin stays at 0.10.1 until a
full suite runs on 0.10.2, which is not yet possible: the coupled MUSIC blocks
test takes 17 s on the pinned stack and did not finish in 600 s there.

## The two-device stall

One test process driving two GPUs sometimes wedges in a collective, with the
runtime reporting that it waited for two threads to join a rendezvous and one
arrived. Measured rate on the pinned stack: **2 stalls in 8 runs** of
`test_cosmo_hydro_music_blocks`, which passes on a rerun every time. It is not
a wrong answer and it is not in our code as far as anything shows; the runner
bounds it with `--timeout-s` so it costs one file instead of a batch. If a
suite run comes back with one or two stalls in the cosmology files, rerun
those files alone before believing them.

## Hydro accuracy and hard tests

`tests/test_hydro_accuracy_*.py` check the hydro solver against exact answers;
`tests/test_hydro_hard_*.py` run the problems that expose the known weak spots of
MUSCL/HLLC grid codes. All run the production step (minmod, HLLC unless HLL is named,
FP64, the default time step: RAMSES rule, CFL 0.8). Each bound is ~1.3x a measured value
(lower bounds: measured / 1.3; convergence orders: measured - 0.1 .. + 0.3), and the
measured value is written next to it. The `gpu` tier repeats each test at larger N.

| file | checks | CPU-8 bound (measured) |
|---|---|---|
| `accuracy_riemann` | HLLC/HLL face flux vs the exact Godunov flux (Toro 1-5), Toro's formulas, exact contacts, F(U,U) = F(U), positivity | flux error 0.066-0.4; round-off 1e-14 |
| `accuracy_shocktube` | Toro 1-5 at 100-400 cells: L1 of rho, u, p, order, front position; 3D Sod along x, y, z, xy: axes and 1 vs 2 devices alike | Sod L1 rho 0.012 (0.0088) |
| `accuracy_sedov` | 3D Sedov shock radius and interior profile vs the exact solution, 1 and 2 devices | 1.0 cell (0.81) |
| `accuracy_smooth` | sound wave, advected sound wave, Gresho, square and sine advection: L1 ladder and order | sound order 1.76-2.16 (1.86) |
| `accuracy_kh` | KH growth rate vs compressible linear theory | 0.039 (0.030) |
| `hard_carbuncle` | Quirk's duct, a strong shock off a wall, 2D Noh; HLL stays planar | duct row spread 5.5e-7 (4.4e-7) |
| `hard_galilean` | square, Sod, KH boosted by 0-100 c_s: L1 and the resolution penalty | Sod penalty at 100 c_s 11.7 (9.0) |
| `hard_low_mach` | Gresho at M = 0.1-0.001: KE loss and its M exponent | exponent >= -0.52 (-0.42) |
| `hard_vortex` | isentropic vortex over 1, 10, 100 periods: L1, peak, phase, order | order >= 1.87 (1.97) |
| `hard_well_balanced` | n = 1 polytrope under self-gravity at rest for 10 crossing times | drift 0.228 (0.203) |
| `hard_taylor_green` | Taylor-Green vs a spectral reference: KE loss, L2, enstrophy, M scaling | KE loss t = 2 0.47 (0.36) |

### Strict xfails for open bugs

Where LUNA fails the physics, the test says so instead of moving the bound: it is
`xfail(strict=True, raises=AssertionError)` with the bug as the reason, and the assertion
that fails is the one naming the bug. A different failure (a run that aborts, a wrong
profile) is a plain failure, not an xfail. The suite stays green, and a fix turns the test
into XPASS, which fails until the marker is removed, so a fix cannot land without someone
looking at it. Open now: the HLLC carbuncle (`hard_carbuncle`, the HLLC planarity and Noh
isotropy), a cold ambient around the polytrope whose pressure goes negative first
(`hard_well_balanced`, GPU, step by step), and the registered `Lane-Emden-3D` IC at n = 1
(rho = 0 at the surface).

### A run that stops short fails

`evolve` and `_run_short` in `tests/_helpers.py` raise `RunAborted` when a run stops before
its target time (step cap, NaN or non-positive dt, non-finite state) or when its mesh needs
more devices than are visible; nothing in these tests skips for lack of devices. Every run
prints a `RUN` line with the test id, t reached, t_end and the step count.
`tests/test_hydro_run_guard.py` checks each guard.

## GitHub runs the whole CPU suite; Raven runs the GPU tests

On every push to main and nightly, `.github/workflows/tests.yml` runs the full
tier in 12 shards on CPU-8 (`--tier full --shard I/12`, balanced by the
measured costs in `tests/tiers.json`; a pair `X_a.py`/`X_b.py` stays in one
shard) and the multidevice files in 4 shards on 2 devices. Files GitHub
cannot run are in `tests/ci_exclude.json`, each with its reason (today: the
chemistry files, whose tables live in `$LUNA_DATA`). The `coverage` job
merges each config's shards (`--merge-reports`) and runs `--check-coverage`
over them and the committed Raven GPU reports, so a GPU-only test is not
"skipped everywhere" when it ran on Raven.

The Raven reports are produced by hand or by a nightly job on Raven, one
config each (LUNA_DATA and LUNA_REFERENCE_SOLUTIONS set):

```bash
# 1 GPU, the full tier in shards (one SLURM job each, -p gpu1)
python tools/run_tests.py --tier full --shard I/8 --timeout-s 3600 --label raven_gpu1 --report gpu1-I.json
# 2 and 4 GPUs (-p gpu --gres=gpu:a100:2 / :4)
python tools/run_tests.py --tier multidevice --timeout-s 3600 --label raven_gpu2 --report gpu2-1.json
python tools/run_tests.py --tier multidevice --timeout-s 3600 --label raven_gpu4 --report gpu4-1.json
# merged without tails and committed
python tools/run_tests.py --merge-reports tests/reports/raven_gpu1.json gpu1-*.json
python tools/run_tests.py --merge-reports tests/reports/raven_gpu2.json gpu2-1.json
python tools/run_tests.py --merge-reports tests/reports/raven_gpu4.json gpu4-1.json
```

Each report records its commit; the coverage table prints it next to the
label, so a stale GPU report is visible.

The Raven reports are optional input: `coverage` reads whatever
`tests/reports/raven_*.json` exist. Without them every GPU-only test counts as
skipped everywhere and the check fails, which is the truth for that tree. The
first committed pair (raven_gpu1, raven_gpu2, from 88ed228) used a 600 s
per-file limit; 15 GPU files timed out at it, so their GPU tests still show as
skipped everywhere -- use 3600 s. Reports from ab90619 at 3600 s replace them (raven_gpu1, raven_gpu2); no
4-GPU report is committed (the jobs were cancelled in the queue; the nightly can add it later).

Files that fail on a GitHub config for a cited code or test bug are listed in
`tests/ci_known_failures_{cpu8,dev2}.json` with the cause (file:line) and run
with `--known-failures`: they show as KNWN with the cause printed, the job
stays green, and a listed file that passes fails the job, so the list cannot
outlive the bug. A collection error (e.g. missing data) counts as nothing run
in `--check-coverage`.
