# Spherical C/O explosion pilot

The central ignition and four-step burn/hydro/gravity smoke test pass on the Mac CPU.
This handoff prepares longer exploratory evolution on Raven. It does not establish a
self-sustaining detonation, converged explosion energy or reliable isotope yields.
The 512-cell model initializes successfully on the Intel Mac with JAX 0.4.38/FP64 and
adds 2.7940315e47 erg of thermal ignition energy. Profile export matches that checkpoint;
targeted analytic budget, profile-export and checkpoint/restart tests pass. This setup
has not been executed on a GPU or validated with Raven's JAX version.
[Local setup report](validation/stellar_explosion_setup_cpu.json).

## Model and run

`scripts/examples/stellar/white_dwarf_explosion.json` uses a 512-cell spherical C/O white
dwarf with central density 1e7 g/cm³, background temperature 1e7 K and an explicit ambient
density of 1e-4 g/cm³. The domain radius is 4.096e9 cm (dx=8e6 cm), leaving expansion room
beyond the initial stellar surface. A compact 6e7 cm thermal trigger extrapolates to
3e9 K at the origin. The unheated model remains the balancing reference. The source uses
aprox13, screen5 and no neutrino cooling; shock-burning suppression remains enabled.
The network runs only where T > 2e7 K and rho > 1e4 g/cm³ ([STELLAR_1D.md](STELLAR_1D.md)).
Set `CONFIG=/path/to/config.json` to run a variant configuration.

The fixed grid does not resolve the microscopic C/O reaction zone. The artificial trigger,
residual balancing and dilute surface response all require sensitivity studies. A quenched
or mesh-dependent front is a result to investigate, not permission to disable suppression
and call the resulting burn a validated detonation. Isotope masses are network diagnostics,
not detailed nucleosynthetic predictions.

After committing/pushing the changes locally and pulling them on Raven, submit from the
repository root in the prepared checkout:

```bash
sbatch --partition=YOUR_GPU_PARTITION submit/examples/stellar_explosion_raven.sbatch /path/to/helm_table.dat
```

Choose a GPU partition allowing two hours. The script follows the existing Raven module
setup (`gcc/13`, `cuda/12`, `~/jax-env`), requires SciPy, refuses a CPU backend and checks
the EOS table against the validated table hash. It writes `stellar-explosion-JOBID/` with
a software/device/source manifest, an initial checkpoint and checkpoints every 25 accepted
steps. Checkpoint messages appear in the Slurm log. Step exhaustion or repeated refusal
saves the latest accepted state before raising; scheduler termination may lose work since
the previous checkpoint. The default target is **0.1 s**, dt is capped at 1e-4 s, and the
total accepted-step limit is 200000. These are pilot controls, not convergence settings.

To continue the same model toward 2 s, use the original step-zero checkpoint as the
diagnostic baseline and a later checkpoint as the restart:

```bash
STOP_TIME=2 \
RESTART_CHECKPOINT=/path/to/stellar-explosion-OLDJOB/state_XXXXXXXX.h5 \
INITIAL_CHECKPOINT=/path/to/stellar-explosion-OLDJOB/state_00000000.h5 \
sbatch --export=ALL --partition=YOUR_GPU_PARTITION submit/examples/stellar_explosion_raven.sbatch /path/to/helm_table.dat
```

The continuation creates a new output directory and does not reapply ignition. Changing
the physics or mesh requires a new run; restart compatibility checks reject those changes.
If the original pilot has not reached 0.1 s, it can also be continued to that same target.

## Diagnostics

After normal completion or a Python run failure, the job analyzes available checkpoints
and writes `history.json` and `profiles.h5`. Its exit status still reports a failed run.
An external job kill may prevent analysis; run it afterwards with fresh output filenames:

```bash
PYTHONPATH=src python tools/analyze_stellar.py /path/to/run/state_*.h5 \
  --initial /path/to/original/state_00000000.h5 \
  --report history.json --profiles profiles.h5
```

The history contains remaining isotope masses, net C/O consumption, binding-energy
release corrected for boundary transport, thermal/kinetic/gravitational energies, mass
and energy budget errors, mass-weighted RMS speed, maximum speed and its density, and
radii enclosing 50%, 90% and 99% of the current grid mass. It retains the runner's retry,
limiter, suppression and boundary diagnostics. All totals include ambient material.

The energy check compares the change in gas plus gravitational energy with nuclear
binding release and net boundary input. The initial state already includes the thermal
trigger; its recorded energy is reported separately and is not counted as nuclear release.
Boundary energy includes gravitational potential transport. The analyzer currently refuses
neutrino-cooled runs because their integrated losses are not recorded by this runner.
Budget checks require mass error below 2e-12 and energy error below 1e-9, normalized by
initial mass and initial absolute gas plus gravitational energies respectively. Passing
these checks is bookkeeping evidence, not an explosion validation gate.

Each `step_XXXXXXXX` profile group stores radius, density, radial velocity, temperature,
13 mass fractions, enclosed mass, potential and specific thermal energy. Species order
is stored in the file metadata. The original checkpoints remain restartable.

Before interpreting a successful explosion, compare meshes and burning tolerances,
ignition radii/temperatures, ambient densities and domain sizes. Check that propagation
persists beyond the trigger, expansion is supported by the mass profiles, and boundary
losses are understood. Positive global gas-plus-gravity energy alone does not establish
that every shell is unbound. The unfinished He refinement and self-sustaining He/C/O
detonation tests remain separate requirements.
