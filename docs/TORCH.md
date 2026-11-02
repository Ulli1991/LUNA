# LUNA as Torch's hydro code

[Torch](https://github.com/torch-sf/torch) is a star-formation framework: it
samples an IMF, turns sink particles into stars, evolves them with SeBa,
integrates their mutual gravity with PeTar, and feeds their radiation and winds
back into the gas. It has always driven **FLASH** for the gas. This is LUNA
answering the same calls.

The point is not that LUNA replaces FLASH everywhere. It is that Torch's
physics — the IMF, the stellar evolution, the N-body — is worth keeping, and
the gas solver under it can be a JAX one.

## What Torch actually asks of a hydro code

Not what the name suggests. Torch's hydro code also hosts:

- the **star and sink particles** themselves, addressed by tag;
- **SeBa's state** for each star, parked between steps (relative mass and age,
  core and CO-core mass, stellar type, radius);
- each star's **emission properties** (ionizing and photoelectric photon rates
  and mean energies, the H and dust cross sections, wind mass-loss rate and
  terminal velocity);
- **both directions of the bridge kick**, gas→star and star→gas;
- the run's **control values** — current step, end time, step limit, output
  directory, and the file numbers Torch keys its own output off.

That is about sixty calls, and it is why `amuse_luna/interface.py` is not a
thin AMUSE hydro interface.

The awkward part: `torch_se.evolve_stars` writes the whole stellar set **every
step whenever `with_se` is on**, whether or not the corresponding feedback is
enabled. With LyC, photoelectric heating and winds all off it writes zeros —
but it writes them, and `torch_sf.add_particles_to_grav` reads them straight
back when a new star is handed over. So all of it has to exist even in a run
with no feedback. Those attributes are listed once in
`amuse_luna/particles.py` (`PARKED_ATTRS`) and their accessors are generated
from that table on both sides of the boundary.

**LUNA does not interpret any of it.** The particle store is a parking space.
The one exception is a new star's relative mass: zero means SeBa has never
written it, so it falls back to the ZAMS convention rather than handing SeBa a
zero-mass star — the same way Torch itself treats `stype == 0`.

## Units

Every call carries a unit, declared in `Luna.define_methods`. Without that
table each number crosses as a bare float while Torch does arithmetic against
quantities that have units. The units are FLASH's, because Torch was written
against FLASH; changing one does not raise, it silently rescales.

The AMUSE converter must describe the **hydro code's own** code units. Torch
hard-codes `1 cm / 1 g / 1 s`, which is the identity for FLASH and wrong for
anything else, so `USER['hydro_units']` overrides it. Read that triple off the
LUNA config rather than writing it twice.

## Running one

Stock Torch drives only FLASH. LUNA needs the `luna-hydro-code` branch of
[Ulli1991/torch](https://github.com/Ulli1991/torch/tree/luna-hydro-code), a fork
of Torch 2.2.2 that lets another hydro code be used (`USER['hydro_code'] = 'luna'`,
which imports `amuse_luna.interface`) and carries the fixes listed under "The
coupling side" below. The cloud runs used commit `f6fbe7f`. Point `$TORCH_ROOT` at
the clone; `torch_user_luna.py` puts it and its `src/` on the path.

Four files in the run directory:

| file | what it is |
|---|---|
| `luna_cloud.py` | the LUNA config: grid, cloud, gravity, sinks. Named by `$LUNA_CONFIG`. |
| `torch_user_luna.py` | Torch's user script: calls Torch's own `user_parameters()` and overrides what differs. |
| `flash.par` | Torch's parameters. `flash.par.turbsph_standard` from the Torch repo is the right starting point for a cloud. |
| `run.sh` | Torch reads its task count out of this with a regex. Bookkeeping, not a launcher. |

`$LUNA_CONFIG` exists because Torch's `initialize_workers` offers no hook
between constructing the hydro code and initialising it. Naming a config there
also tells LUNA it **owns the problem**, so it keeps the initial conditions the
config describes instead of the empty grid an orchestrator-driven worker
expects to be handed.

The `$LUNA_CONFIG` file is read as `main.py` reads one: the cloud's parameters
are `problem=turbulent_sphere_3d.Params(...)`, its mean molecular weight is
`sink_star_mu` and its seed is `seed`, and an old flat name (`cloud_mach`) fails
with its new key. Keys a script sets through `set_config_value` /
`set_config_string` may still use the old flat names for one release: the
worker translates them (`cloud_mach` -> `problem.mach`) and says so.

On viper the workers start with AMUSE's default MPI channel when Python itself runs
under `OMP_NUM_THREADS=8 mpirun -n 1 --bind-to none python torch_user_luna.py`, in a
single-task allocation. Without `--bind-to none` PeTar inherits a one-core binding
(109 s instead of 34 ms per bridge step with 30 stars); with several tasks the LUNA
worker reads `SLURM_NTASKS` and refuses to start. Plain `python` hangs at the first spawn
and `srun` fails (`MPI_ERR_UNKNOWN`). The sockets channel, set in an `amuserc`
(`scripts/coupling/tier1/amuserc`), also works.

## The cloud has to be barotropic

Not a matter of taste. A 15 K cloud at mach 8 carries a few percent of its
energy as heat, and a total-energy scheme recovers that few percent by
subtracting the kinetic part, so the first strong rarefaction drives it
negative and the run stops. Measured on this setup, 64^3, gravity and sinks
off, counting steps until the internal energy goes negative:

| setup | result |
|---|---|
| mach 8, `hllc` | step 1 |
| mach 8, `hll` | step 1 |
| mach 8, `lhllc` (solver since removed) | step 1 |
| mach 4, `hllc` | step 4 |
| mach 3, `hllc` | 80 steps, no failure |

The Riemann solver makes no difference and neither does CFL: 0.30, 0.15 and
0.10 all fail at the same physical time, and so does first-order
reconstruction. It is not a stability margin, it is the subtraction.

`barotropic_eos` fixes it, and it is what a cloud calculation should use
anyway: P becomes a function of density -- isothermal at `barotropic_T0_K`,
stiffening above `barotropic_rho_AD_cgs` where the first hydrostatic core
forms -- so negative pressure cannot arise. With it on, mach 8 runs, and with
gravity the cloud collapses from a mean density of 59.7 to 2.4e4 Msun/pc^3
inside one free-fall time.

A *supersonic adiabatic* run is the case with nothing to fall back on, and
LUNA has had both of the usual answers:

- **The floors are here and one gate away.** `sources/cosmo_floors.py` carries
  the whole stack -- buffered density floor, |v| ceiling, temperature ceiling,
  and `P >= max(p_floor, ct rho, cj rho^2)`, the temperature floor and the
  Truelove-Jeans floor. `floor_core` is layout-generic and elementwise, and
  only the coefficients carry the scale factor. But `assemble` gates it on
  `clock.comoving`, so a proper-time run cannot reach it. Opening that gate
  means deciding what `temp_coeff` and `jeans_coeff` are at a = 1, not writing
  a floor.
- **Dual energy was written and then lost.** Bryan+1995 with the eta switch,
  advecting the internal energy alongside the conserved state, in
  `setup/runner_cosmo_hydro.py` (added in 3a4756c, 2026-06-18). It went out in
  4848abe, "One run file", with the per-run runners it lived in; `DualEnergy`
  and `cosmo_dual_eta` left the config at the same time. No branch has it any
  more, so the reference is the git history. Re-adding it to the one hot path
  is real work rather than a gate: the internal energy has to be advected by
  the hydro step, not projected after it.

## PeTar's threads, or: the coupled run is not two runs

The single worst performance trap found so far, and it is invisible in
isolation. PeTar starts **48 OpenMP threads** by default. A coupled run gives
it 24 CPUs and JAX wants cores on the same node, so its force loop is
oversubscribed against a process that does not yield -- and it costs a
thousand times what it should:

| | grav evolve, per bridge step |
|---|---|
| default threads, alongside LUNA | 32.7 s (106 s on the first) |
| `OMP_NUM_THREADS=8` | **0.02 - 0.05 s** |

The same run went from 18 bridge steps to 87, and from 9 stars to 287.

It resists isolation completely. PeTar on its own -- the run's own stars, read
back from Torch's `stars0001.amuse` -- takes 0.15 s per step at every
changeover radius from 0.001 to 1.0 pc, with stellar radii, with a begin_time
offset, with masses pushed every step, with zero velocity dispersion, and over
thirty steps with no growth. Five hypotheses died against that wall before the
thread count came up, because every isolated test hands PeTar an idle node.

So: set `OMP_NUM_THREADS` in the job script. And when a coupled run is slow,
suspect the sharing before the physics.

## Where a coupled step actually goes

Torch tprints every phase with a timestamp, so a run log is already a trace --
`scripts/coupling/tier1/breakdown.py <log>` turns one into this. Measured at
64^3 with 109 stars, averaged over 74 steps:

| phase | s/step | % |
|---|---|---|
| **hydro (LUNA)** | **0.045** | **1.1%** |
| bridge kicks (four) | 0.252 | 6.3% |
| N-body (PeTar) | 0.007 | 0.2% |
| stellar evolution (SeBa) | 0.746 | 18.7% |
| **star formation (Torch)** | **1.680** | **42.1%** |
| unattributed (Torch Python) | 1.261 | 31.6% |
| total | 3.99 | |

Which is the case for GPU hydro and the limit of it in one table. The gas
update is one percent of a coupled step; about three quarters is Torch's
Python doing per-star and per-sink bookkeeping, and none of that scales with
the grid. Multiply the cells by 64 and the hydro column grows with them while
everything else stands still, so 256^3 is roughly where LUNA's speed starts to
matter to the coupled run instead of being hidden inside it.

If more is wanted after that, the work is not in the solver: it is amortising
the per-object Python -- longer bridge steps, or batching the star handling.

## Resolution, and what it costs

No LUNA run should go below a 256^3 base grid; smaller is for correctness and
turnaround. At 64^3 in this 8 pc box a 2 pc cloud is 32 cells across, the sink
accretion radius is a quarter of a parsec, and the Truelove threshold sits at
2.6x the mean cloud density -- so sinks form in the first turbulent peak
rather than in gas that has collapsed. At 256^3 it is 42x, and an H II region
around one 1e49 s^-1 star is 15 cells across instead of 3.7.

### The sink stage, and a measurement that was not one

An earlier version of this file said the sink stage was "97% of a 256^3 step"
and called it the barrier to production resolution. **That was compile time.**
The figure came from 10-step runs, where the sink stage's unrolled `n_max`
loop compiles for ~29 s while ten steps take 0.3 s -- so the number measured
the compile and wore a throughput label. Three conclusions drawn from those
runs were wrong, including a verdict that the fix below made things slower.

Measured properly -- 90 steps, the mean of six steady-state prints, 256^3,
FP64, one MI300A:

| | hydro+gravity | sinks, 1 slot | sinks, 32 slots |
|---|---|---|---|
| full-grid mask per slot | 27.8 ms | 33.2 ms | 89.3 ms |
| **(2R+1)^3 window per slot** | 27.8 ms | **31.5 ms** | **44.7 ms** |

So the sink stage costs 16.9 ms on top of 27.8 ms at 32 slots, not a second
and not 97% of anything. The window cuts that overhead 3.6x (61.5 -> 16.9 ms)
and, more importantly, scales: the full-grid version degrades 2.7x going from
1 slot to 32, the window 1.4x, and production wants many sinks.

The memory result stands on its own, because an allocation failure is not a
timing artefact: the full-grid version needs `sink_max_count x N^3`, and 256
slots at 256^3 asks for 98.6 GiB and dies. The window removes that scaling.

One behaviour change comes with it. The old kernel measured `X - pos`, a plain
Cartesian distance, so in a periodic box a sink near a face never accreted
across it -- gas one cell away through the boundary was treated as a box-width
away. The window wraps and measures to the nearest image, so it does. Interior
sinks agree to 1e-12; this is the only case where the two differ, and the old
answer was the wrong one.

After the window (and one batched accretion pass, windowed creation and the
warm-started gravity), timing each piece of the stage on its own found the
rest: 78% of the remaining sink cost was the sinks' analytic field on the
gas, computed twice per step (opening and closing kick) as `r2 ** -1.5`. In
FP64 that power is an exp and a log per cell per slot. Two changes, same
physics: `rsqrt(r2)` cubed (6x cheaper; FP64 round-off against a numpy
reference, 2.6e-16 of the field scale), and the field handed to the gas kick
as a program the kick evaluates inside itself, so the three field grids are
never written to memory. Stepper path, FP64, 256^3, 30 steady-state steps:

| | sinks off | 32 slots | sink overhead |
|---|---|---|---|
| before | 23.0 ms | 37.0 ms | 14.0 ms |
| **after** | 22.7 ms | **26.6 ms** | **3.9 ms** |

`main.py` with 32 slots went from 451.8 to 629.4 Mcells/s (sinks off: 729.8).

**Check the dtype of a profiling script.** `amuse_luna.stepper.build` does not
turn on x64; `main.py` and the AMUSE worker do (from `enable_x64` in the
config). A script that builds through the stepper without
`jax.config.update("jax_enable_x64", True)` profiles a float32 run, which
here halved the step and made an FP64-exact rsqrt look like a 2e-7 error.

**Read a short run at your peril.** At 256^3 a 10-step run spends ~0.3 s
stepping and 30-45 s compiling. Anything below a few dozen steps is measuring
XLA.

## Running on several GPUs, and what it took (2026-09-23)

The coupled run at 1024^3 needs 8 GPUs (4 viper nodes): the state is 40 GiB
and the hydro step's scratch about 200 GiB even after the kernel work below.
On 4 GPUs it is ~4 GiB short, measured (30 GiB resident + 46.4 scratch + 10
output against an 82.5 GiB device limit).

**The environment decides whether it runs at all.** LUNA's sharded gravity
hangs deterministically on the second step under jax 0.10.1 and 0.10.2 on
ROCm 7.2 (a collective that never completes; `HIP_LAUNCH_BLOCKING=1`
unhangs it on one node at -25% and hangs multi-node jobs outright). Under
jax 0.10.0 it runs, at 1245 Mcells/s on 2 GPUs for a 256^3 Sedov with
self-gravity. So every multi-GPU run uses `venv_jax_rocm7.2` (jax 0.10.0),
and the AMUSE worker for a multi-rank coupled run uses
`amuse_env010`, the same AMUSE build on jax 0.10.0.
Single-GPU runs are unaffected; the hydro-only weak-scaling ladder never
touched the collective in question.

**The worker as N ranks.** AMUSE's Python worker already broadcasts every
call over MPI when told to initialise MPI (`start_socket_mpi`): rank 0
speaks to Torch, every rank runs the call, JAX shards the grid across their
GPUs. `tier1/amuse-srun-exec` is the `mpiexec` stand-in that starts LUNA's
worker on every GPU of the job as one `srun` step (and execs the compiled
workers directly); `tier1/amuserc_mp` puts it in `[channel]` -- AMUSE reads
`mpiexec` from there only, so the per-code sections the old `amuserc` had
were never applied. Do not pass `--gpus-per-task=1`: RCCL's intra-node path
needs the peer GPU visible, and `SLURM_LOCALID` pinning picks the right one
(`setup/jax_init.py`; one task per node takes every GPU it sees).

**Sinks on the mesh** (`species/sinks_sharded.py`): the sink arrays are
replicated, the hooks are shard_map programs over each rank's block, the
window sums cross the mesh in a psum, the candidate for a new sink is a
pmin. Tested against the single-device kernels on a 2x2x2 mesh of CPU
devices, sinks on block faces and corners.

**The IC is written once** (`tools/write_ic.py`, `cfg.ic_dir`): the
turbulent cloud's velocity field is an FFT of the whole box, and building it
on every rank at 1024^3 took 165 GB of host memory per process -- on MI300A
the HBM the GPUs need. 43 GB on disk, 9 min to write, each rank reads its
own slab. Big runs must NOT set `XLA_PYTHON_CLIENT_PREALLOCATE=false`: with
it the allocator could not place a 33 GiB M1 request with 60 GiB free.

**The cloud's spectrum before 2026-09-30 was flat.** `problem.turb_slope` now
is the shell exponent, E(k) ~ k^slope (-2 Burgers). Every run before the fix
(frozen trees up to src_33, res2 included) had per-mode power k^slope: the
default -2 gave a flat E(k), 80 % of the kinetic energy at <= 8 cells at
256^3. Their -4 is today's -2. IC directories written before are refused.

**The cloud drifted before 2026-09-30 (cloud-drift).** The turbulent field
has zero mean over the box, but masked to the cloud it keeps a bulk velocity.
`problem.zero_drift` (default on) removes the cloud gas's mass-weighted mean
velocity and then renormalises the cloud's mass-weighted rms to the requested
one (sigma_v = problem.mach cs, or v0), on the top-hat and the Gaussian path
alike (neither did it before; Torch's turb-sphere.py does not either). The
Burgers study cloud (256^3, seed 1), r <= R, measured: before |v_com| = 0.96
km/s, 20 % of E_kin, v_rms 2.14 km/s (the box, not the cloud, was
normalised); after 0.001 km/s, v_rms 1.84 km/s (asked: 1.86), E_kin 3314
against 4509. Wall M3 (512^3) had 0.008 km/s, 2e-4 of E_kin. The net
angular momentum is kept: Torch keeps it, it moves no gas off centre, and it
is the largest-scale part of the turbulent field itself (rigid rotation
about the centre of mass, measured: 32 % of E_kin in the Burgers cloud, 3.9 %
in M3). `problem.zero_drift=False` rebuilds the old clouds; IC directories
written before are refused. `tests/test_cloud_drift.py`.

**Torch's calls, measured.** With the timers every worker call carries
(`LUNA calls:` per bridge step), 17 calls per bridge step cost ~100 ms
each with 0.000 s of work inside: AMUSE sets `TCP_NODELAY` on the
orchestrator's socket and not on the Python worker's. Set on our side,
three calls take 7 ms instead of 340. The star-formation check itself
(five calls per sink per step, and Torch's one-star-at-a-time IMF sampler
at 400 ms per 1e4 Msun queue) is served by `get_sink_state` (all sinks in
one call) and, opt-in through `USER['sf_in_hydro']`, by `form_stars`: the
same rule run in LUNA with the sampler vectorised (17 ms, same distribution,
`species/imf.py`). SeBa stays where it is.

The vectorised sampler is Torch's procedure: Poisson counts per logarithmic
mass bin of a Kroupa (2001) IMF, then rejection sampling in each bin with
Torch's acceptance rule (keep a uniform draw m when m·kroupa(m)·norm > r, r
uniform on [0, 1)), drawn in batches, then a shuffle; `sum_small` merges
stars below `m_small` as Torch does. It keeps Torch's quirks on purpose,
because it exists to be compared against Torch's sampler.

**M1 was the step.** 97% of a 1.5 s step at 256^3 was the radiation
stage: 16 fixed substeps per half at every timestep, 46 ms each. The count
follows the step now (3 per half at the cloud's 125 yr), the substeps run
as repeated one-substep calls (24 fields of scratch instead of 33), and the
closure's sqrt and divides run once per face state instead of once per
consumer (46 -> 31 ms). At the cloud's c_hat (150 km/s = 11.7 c_i) the
STARBENCH D-type front sits between the Spitzer and Hosokawa-Inutsuka
solutions (R/HI 0.987 at 96^3); Torch's own FERVENT ray tracing has no
reduced speed of light to compare against.

## Restarting a coupled run

Torch restarts the way FLASH does. `restart = .true.` and
`checkpointFileNumber = k` in `flash.par` make Torch load its own files of
that number from the hydro code's output directory (`all_masses{k}.pickle`,
the IMF queues; `rnd_state{k}.pickle`, its RNG) and take every star back from
the hydro code's particles -- `add_particles_to_grav` in its restart mode
reads the parked SeBa state and hands the set to PeTar and SeBa again. The
hydro code has to come up at that moment with everything it holds.

A **coupled checkpoint k** is therefore two things written together:

- LUNA's checkpoint set of that step -- the gas (`checkpoint_<step>.<rank>.hdf5`,
  one file per rank, any decomposition can read it) and the M1 carry
  (`radiation_<step>...`), through the same writers a standalone run uses;
- `coupled_<step>.npz` + `.json` with what only the worker knows: the sink
  species' arrays, the star store with every attribute Torch parked, the
  LUNA-side IMF queues and their RNG, the feedback totals, the number `k`.

It is written by `IO_out('chk')` whenever the snapshot tick
(`time_bet_snaps`) has moved on since the last one, which is the call Torch
already keys its own writing off: LUNA writes, reports the next number,
Torch sees it change and writes its files of the number before. So the
first call at the start writes 0000, as FLASH does, and every set is taken
at a bridge-step boundary where LUNA, PeTar and SeBa agree. The checkpoint
is also the run's gas output; the worker writes no separate snapshots.

To restart: `restart=True` in the LUNA config (`restart_file` names a
checkpoint file, otherwise the newest set is taken) and, in `flash.par`,
`restart = .true.` with `checkpointFileNumber = k` and `plotFileNumber = k+1`.
The worker checks the two against each other and refuses a mismatch: Torch
taking its stars from one moment and LUNA its gas from another is the
silent kind of wrong. What is not carried: the potential (the first step
after a restart solves cold) and the run's wall-clock timers.

A sink's tag is the slot it occupies in LUNA's carry, for life. Before,
the mirror numbered the active sinks in order every time Torch looked, so
a sink forming in a lower slot shifted every tag above it -- and both IMF
queues, Torch's `all_masses` and LUNA's, are keyed by tag.

## How LUNA's sinks work

Sinks (`species/sinks.py`) use hybrid gravity: the gas potential is the
run's one Poisson solve and sinks deposit nothing. Gas -> sink is the
CIC-interpolated gas field; sink -> gas is the softened point-mass field
added to the gas kick; sink -> sink is direct N-body, off
(`sink_sink=False`) under PeTar, which owns that force. Shapes are static:
`n_max` slots gated by `active`. The sink N-body is KDK around the hydro;
accretion and creation happen once at the end of the step on each sink's
(2R+1)^3 window, capped so it never wraps onto itself. A cell reached by two
sinks is split between them. Two kinds of gas moments feed sink-to-star
conversion: `accreted_gas_moments` reads exact running sums over the gas a
sink has taken in (what Torch's `make_stars_from_sinks` needs);
`sink_gas_moments` estimates from the gas inside r_acc now, for sinks that
were injected rather than grown.

## Sink rules: LUNA and FLASH (2026-09-28)

The res2 cloud turned ~70% of its 2000 Msun into stars within ~0.2 Myr of
t_ff, feedback or not; the Torch papers find 10-20%. LUNA's sinks followed
SEREN, FLASH-Torch's follow Federrath et al. (2010, ApJ 713, 269), and the two
differ in almost every check. Each FLASH rule is now a config field; the
defaults keep LUNA's rules bitwise (checked against merge-main on one device
and on a 2x2x2 mesh). A FLASH-like run sets all four:

    sink_accretion="federrath", sink_creation_checks="flash",
    sink_merging=True, sink_star_T_max=100.0

The FLASH side is read from Torch's `Particles_sinkCreateAccrete.F90`
(Couple_AMUSE, and Couple_AMUSE_Sinks_and_Stars for the cold-gas rule), not
only from the paper, where the two differ the code wins. FLASH's
`pt_sinkParticleMerging` is not in Torch's tree, so merging follows the paper.

| check | FLASH-Torch | LUNA's own (default) | LUNA, FLASH option |
|---|---|---|---|
| accretion: which gas | a cell's excess over rho_thr within r_acc, only if bound and infalling (below) | every cell's excess within r_acc, every step | as FLASH |
| accretion: bound | e = \|v-v_s\|^2/2 - G M_s (1/r - 1/r_acc) - (2pi/3) G rho_thr (r_acc^2 - r^2) < 0: the sink plus the gas left at rho_thr inside r_acc, zero at r_acc | none | the same with the sink's Plummer potential, (r^2+eps^2)^-1/2 - (r_acc^2+eps^2)^-1/2, which is the field the gas feels in LUNA |
| accretion: infall | (v-v_s).r_hat < 1e-5 c_s, c_s = sqrt(P/rho) | none | as FLASH |
| accretion: inner cell | r < max(0.2 r_acc, dx): no checks | -- | as FLASH |
| accretion: several sinks | the qualifying sink with the lowest e (most bound) | split evenly between the sinks that reach it | as FLASH; ties to the lower slot |
| accretion: sink position | moved to the centre of mass of sink + accreted gas | unchanged | as FLASH |
| creation: candidates | every cell above rho_thr | the global potential minimum only, one per step | the CREATE_K = 8 deepest cells above rho_thr that pass the potential test and are not within 2 r_acc of a sink, per step |
| creation: potential minimum | phi(centre) minimal over its control volume V (i^2+j^2+k^2 <= 9, \|i\|,\|j\|,\|k\| <= 2), to 1e-5 \|phi\| | global argmin | as FLASH |
| creation: converging flow | none of the 26 neighbours moves away faster than 1e-5 c_s | div v < 0 at the cell | as FLASH |
| creation: Jeans | 2 E_th <= E_grav over V, E_grav = sum rho (max_V phi - phi) dV | none | as FLASH (E_th = sum of internal energy; FLASH's 1.5 P for gamma 5/3) |
| creation: bound | E_th + E_kin <= E_grav, E_kin about V's centre of mass | E_kin + E_grav < 0 over the r_acc sphere, E_grav = sum rho phi dV (no E_th) | as FLASH |
| creation: proximity | no sink within 2 r_acc | none within creation_separation (4 dx) | as FLASH |
| creation: mass | born with no mass and no velocity at the cell centre; the accretion that follows in the same step gives it mass; one that gets none is removed | takes create_frac (0.9) of all gas in the r_acc sphere, above threshold or not | as FLASH |
| creation: neighbours | new sinks within 1.1 sqrt(3) dx merge at birth (to their mean position) | -- | the deepest keeps its cell centre, the others are dropped |
| merging | within r_acc of each other, approaching, bound (paper sec. 3.5) | none | mu \|dv\|^2/2 - G m_i m_j (r^2+eps^2)^-1/2 < 0; the lighter into the heavier at the centre of mass, mass, momentum and accreted-gas moments summed |
| stars from a sink | only while gas at T <= 100 K lies within 2 r_acc (Lewis et al. 2023 sec. 3.2 says "within the accretion radius"; the code uses 2 r_acc and at least one cold cell) | always | `sink_star_T_max`; T = sink_star_mu m_H (P/rho) / k_B; a blocked sink's mean_cs is NaN, which Torch's make_stars_from_sinks and `form_stars` both skip |

Order in a step, FLASH option: new sinks, then accretion by every sink, then
empty new sinks removed, then merging. LUNA's order is accretion, then
creation. The candidate cap CREATE_K is LUNA's: static shapes. A step with more
than 8 passing candidates leaves the rest to the next step, where they are
still candidates unless a new sink now sits within 2 r_acc.

What stays LUNA's: the gas feels the sinks' Plummer field (eps = 2 dx by
default; FLASH uses a spline kernel with softening radius r_acc), and the
sinks keep their own position between steps (Lewis et al. 2023 move theirs to
the potential minimum within r_acc every step). The creation density is
LUNA's Truelove density at every resolution.

Merging frees a slot. A tag is a slot, so a later sink in that slot inherits
Torch's `all_masses` queue of the merged one; it is a list of IMF draws, so the
statistics do not change, and the outflow path has always done the same.

Cost, measured with `tools/bench_sinks.py` at 512^3 on 2 MI300A (mesh 2x1x1,
lognormal gas, every live sink's window full), median of 10 (2026-09-28):

| sink `finish` (accrete, create, merge) | LUNA rules | FLASH rules |
|---|---|---|
| 32 slots, 32 live | 2.2 ms | 6.1 ms |
| 256 slots, 32 live | 2.5 ms | 6.6 ms |
| 256 slots, 256 live | 2.4 ms | 13.7 ms |

The first FLASH version ranked the candidates with a `top_k` over the whole
grid: 58 ms at 32 slots. `_lowest_k` sorts the row minima and then K rows
instead (same K cells, same order). For scale, fb_512's LUNA step is 186 ms
(median of 200 bridge steps, 32 slots). The slot count costs most in the
sinks' field on the gas, which the kick evaluates twice a step: 17.7 ms at 32
slots, 122 ms at 256 slots with 32 live, 136 ms with 256 live. Those three
were measured with HIP_LAUNCH_BLOCKING=1, because without it the isolated
field program hung in the ROCm runtime, so they are upper bounds. In this
measurement an empty chunk of 4 slots, which `lax.cond` skips, cost about
1.9 ms, nearly what a live one did (2.2 ms), perhaps because every launch was
serialised. So 256 slots at 512^3 is
not recommended until the real step has been timed. Compacting the live slots
and looping over only their chunks would remove the cost.

**What they did to the res2 cloud** (fb_256 vs fb_256_fed, 256^3, feedback
on, otherwise identical; one realisation each). SFE = stellar mass formed
(Torch's spawn log, initial masses, escaped stars included) / 2000 Msun:

| t [Myr] | 1.0 | 1.1 | 1.2 | 1.3 | 1.5 | 1.6 |
|---|---|---|---|---|---|---|
| LUNA rules | 0.113 | 0.328 | 0.624 | 0.695 | 0.721 | 0.721 |
| FLASH rules | 0.098 | 0.278 | 0.637 | 0.780 | 0.825 | 0.826 |

The FLASH rules do not lower the SFE here; it ends 0.83 against 0.72. Both
runs start ionising at ~1.0 Myr: Q >= 1e47/s at 0.996 Myr (LUNA) and 1.041 Myr
(FLASH). The 1e49/s mark comes at 1.182 and 1.185 Myr, when both clouds have
already formed ~1250 Msun of stars. The FLASH run's most massive star is
28 Msun and its Q stays at 1e49/s; LUNA's reaches 40+ Msun and 4e49/s. Each
is a single IMF draw. The sinks accreted 1700 Msun in total (sum of `macc`).
One accretion pass applied to the checkpointed state takes nothing under
federrath and 1.8-113 Msun under threshold. So the bound, infalling gas is
drained every step, and the rest reaches the sinks anyway a few steps later.
Sink tags seen by Torch: 19 in the FLASH run, which merges, and 28 of the
32 slots in LUNA's.

Tests: `tests/test_sink_flash_rules.py` (each check decides a case of its
own; conservation of mass, momentum and centre of mass; the 2x2x2 mesh against
one device) and `tests/test_coupling_sink_cold_gas.py`. Every test was shown to
fail with its condition broken.

## What is not done

- **No radiative feedback through Torch.** `with_lyc` ray-traces from the
  stars and asks the hydro code for an ionizing flux field, which is FLASH's
  path. LUNA's M1 does its own transport and is driven through
  `set_ionizing_sources` instead; wiring Torch's stellar photon rates into it
  is the next piece.
- **The thermal state.** With ionization on, LUNA prescribes the temperature
  rather than computing it (`two_temperature` overwrites `P` with
  `rho cs^2(x)`), which fights the cooling a cloud needs. `Config.validate`
  refuses the combination. See [RADIATION.md](RADIATION.md).
- **No FUV band**, so the photoelectric heating rate in `sources/cooling.py` is
  still a scalar constant rather than a function of the local flux.
- **AMR.** Torch's VorAMR calls (`get_leaf_indices`, `set_block_state`,
  `get_1blk_cell_coords`) are not implemented; LUNA runs unigrid here.

## Where the step still goes, and what is next (2026-09-25)

Measured at 512^3 on one MI300A, the 1024^3 run's per-GPU block. The XLA
programs run at full memory bandwidth but move many times their state, because
every intermediate is written to memory as a whole grid.

| piece | time | moved | the state |
|---|---|---|---|
| M1 substep, fused Pallas sweep (069b54b) | 115.5 ms | | was 363 ms |
| of which implicit ionisation | 39 ms | 32 GiB | 60 fixed iterations |
| hydro step | 106.6 ms | 272 GiB | U is 10 GiB |
| gravity, before + after the hydro | 36.6 + 69.6 ms | | |
| of which gas kick with the sinks' field (twice) | 36.9 ms | 27 GiB | kick alone 12.6 ms |
| of which warm multigrid | 20.9 ms | 22.8 GiB | 1.1 TB/s |

Done since (2026-09-25): the hydro divergence (0509559, 2.1x at 512^3, 1.87x
at 8 GPUs), and the kick with the sinks' field as one kernel (dfab9c7): 37.4
-> 4.1 ms at 512^3, with no live sink or four. The empty-sink-slot skip in the
XLA field (0509559) was predicted from the kick timed without the field and
bought only 37.4 -> 36.2 ms. Coupled, 256^3 on one GPU in the star phase, the
LUNA step went 0.065 -> 0.052 s (median of 200 steps, same config, 14 minutes
each).
The ionisation keeps 60 iterations: 30 leave a photon error of 4e-4 against
8e-8, and a log-space Newton step converges worse.

Left, measured and not done:

- Multigrid smoother in Pallas. The fine level dominates the warm solve (20.6
  ms at 512^3) and already runs within 1.7x of its memory floor (one RBGS
  iteration 3.21 ms against 1.94); a fused smoother would save ~12 ms a step,
  ~3% of a star-phase step.

Pallas on ROCm lowers through Triton only (Mosaic is CUDA-only), which JAX
deprecated in 0.11.0 (docs/history/PRECISION_SURVEY.md, R7); LUNA pins 0.10.0. Every such
kernel keeps its XLA path as the fallback.

## The coupling side (2026-09-25)

Once the solvers were fast, the bridge step around them dominated: at 256^3 in
the star phase, py-spy put the LUNA worker idle 57% of the time. In the worker:

- **b7ad949.** The particle store holds host arrays. Bridge fields are cached
  while their inputs are unchanged. Stars deposit shard-locally.
- **d3308b9.** The sink push-back is skipped when Torch changed no sink mass.
- **Result.** The same 12-minute apudev test went from 1076 to 4574 bridge
  steps (1028 -> 266 ms a step).

Outside LUNA, in stellar evolution:

- **SeBa's evolve is a fixed cost per star per call.** The stock worker formats
  a dump of every star to stderr on every evolve
  (`evolve_star_until_next_time`), and AMUSE discards it. A 0.1 Myr jump costs
  the same as a 100 yr step. For 2719 stars: 94.3 ms stock, 16.1 ms without that
  one line, and bitwise identical after 20 Myr through 43 supernovae. The run
  scripts point Torch at a worker built without it (`SEBA_WORKER`). In the
  coupled run, SeBa per SE call went 0.119 -> 0.036 s.
- **Torch's per-star loop in `stellar_evolution`** visited all stars to skip
  those below `min_feedback_mass`. It now visits the active ones only (Torch
  clone, branch luna-hydro-code).
- **SE is due only every `se_cadence`.** A star born in between gets its SeBa
  state alone (`newborn_properties`) instead of a full SE pass.

With all of it (src_29, 13 minutes on apudev, star phase, about 2700 stars),
the median bridge step without SE is about 100 ms. Of that:

| part | ms |
|---|---|
| LUNA's own step (hydro, M1, gravity) | 54 |
| gas field for the second kick, after the hydro | 14 |
| star-formation check, two sink syncs | 10 |
| waiting for PeTar | ~4 |
| the rest | ~17 |

An SE step adds about 70 ms: SeBa plus the copy back 43 ms, the loop 24 ms,
the push 4 ms. Over the last 300 steps the average is 123.9 ms.

## Where LUNA departs from FLASH-Torch on purpose

**Wind mass loading conserves mass.** FLASH-Torch (`GridInject_wind.F90`,
`wind_mass_load`) slows a wind to the speed that shocks to `wind_target_temp`
and multiplies its mass by the same factor, *adding* that mass to the grid.
The swept-up gas is created, not taken from the cells. LUNA
(`wind_target_temp_K > 0`) counts the gas already in the cells as the swept-up
mass: the grid gains exactly the wind's own mass and momentum. In the res2
fb_256 run FLASH's rule would have created about 12 Msun (0.5 % of the box)
next to the massive stars, where the sinks accrete; LUNA's mass budget closes.

**A wind adds no momentum of its own (2026-10-14).** The default wind
(`feedback.make_wind_injector`, and `inject_wind` behind the worker's
`wind_injection`) puts dm / n into each of the n cells whose centre lies within
the wind radius, at v_star + v_term e. The unit vectors e to the cell centres
cancel only when that set of cells is symmetric about the star, i.e. when the
star sits on a grid node. Anywhere else the radial part carried net momentum
dm v_term <e>: 0.6 % of sum dm v_term for the two stars of
`test_an_off_node_wind_adds_only_the_stars_momentum`, 3.5 % for a star on a
cell centre (round-off decides whether the cells at exactly r = 3 dx are in).
LUNA now uses e' = (e - <e>) sqrt(sum |e|^2 / sum |e - <e>|^2), the mean taken
over the deposit cells (`feedback._zero_net`, `_zero_net_grid`): sum e' = 0, so
the gas gains exactly sum dm v_star (measured 6e-17 against 7.4e-3), and
sum |e'|^2 = sum |e|^2, so the radial kinetic energy 0.5 dm v_term^2 is kept.
With `wind_add_heat` the energy is now exactly 0.5 dm (v_star^2 + v_term^2);
the spurious cross term dm v_term v_star . <e> is gone (it was +9.8e-4 of 12.6
in that test). Per cell the speed changes by at most 2.8 % and the direction
by at most 3 degrees (cos 0.9986, 50 random stars at 3 dx). On a node <e> is
round-off and the deposit changes at round-off (3e-15 relative), not bitwise.
Next to an outflow face the window is one-sided and <e> is large: the
correction then also pushes some cells back toward the wall, which is what
zero net momentum requires there. A window of one cell has no zero-momentum
radial pattern and is refused (`ValueError`), not dropped. Hopkins et al. 2018
(MNRAS 477, 1578, FIRE mechanical feedback) balance the vector weights
instead, rescaling the positive and negative half of each axis separately;
that keeps every component's sign but needs both halves non-empty and a
second normalisation for the energy, which equal-mass cells do not need. The
FLASH-Torch modes below are ports and keep FLASH's own weights, so off a node
they still carry net momentum: 5.5e-4 to 1.1e-3 of dm v_term for
`cons_quant = "momentum"`, 3.1e-3 to 9.6e-3 for `"energy"`, at two off-node
stars (round-off on a node; CPU, 24^3, r_ref, 2026-10-14).

**FLASH-Torch's wind, as an option (2026-10-01).** `wind_mass_load_mode`
chooses where the loaded mass comes from; the default `"conservative"` is the
paragraph above and is untouched (bitwise main). The other two port the
routine Torch's production build actually runs: `Couple_AMUSE/wind/
inject_direct.F90` (setup shortcut `amuseWinds`, parameter names `mass_load`,
`cons_quant`, `var_radius`, `ref_radius` as in Torch's `flash.par`), not the
`GridInject_wind.F90` of the `windGinj` unit, which differs in details (its
radius cap, its energy-mode internal energy).

- `"torch"`: inject_direct as written (`feedback.make_torch_wind_injector`).
  F = v_w / v_ref (`wind_cons_quant = "momentum"`) or (v_w / v_ref)^2
  (`"energy"`), v_ref = sqrt(T / 1.38e7) 1000 km/s from `wind_target_temp_K`;
  F dm goes in at v_w / F (= v_ref) radially from the star, the star's own
  velocity not added; there is no floor at F = 1, so a wind slower than v_ref
  goes in lighter and faster, as in FLASH. The cells are weighted by FLASH's
  `overlap` kernel (1 to R/sqrt 3, then 3d/r - 2, then a quadratic tail to
  R; 10^3 points per cell) normalised over the sphere. The radius is
  `wind_ref_radius` (0 = 3.5 sqrt(3) dx, FLASH's `ref_radius = -1`); with
  `wind_var_radius` Weaver's R_1 = 0.74296 (Mdot/rho0)^0.3 v_w^0.1 t_w^0.4
  clipped to [sqrt(3) dx, ref_radius], rho0 the overlap-weighted density over
  ref_radius at the star's first wind call (kept on the star as `wind_bgdy`,
  checkpointed with it), t_w the star's age at the end of the step. Momentum
  mode keeps the internal energy density (`wind_add_heat` = FLASH's
  `add_therm_e`: the energy rises by the unloaded 0.5 dm v_w^2 instead);
  energy mode mixes v|v| per component and keeps the specific internal
  energy. The (F - 1) dm it creates is booked: `_fb["dm_load"]` in the
  worker's feedback carry (checkpointed), subtracted in the `LUNA mass:`
  budget line, so gas + sinks + stars - created stays constant.
- `"sink"`: the same injection, but the (F - 1) dm is taken from the nearest
  active sink (distance through a periodic face unless `particle_boundary`
  is outflow), in star order, never more than the sink holds (a capped step
  loads less: v_inj = v_w dm / (dm + taken), counted in `n_load_capped`). The
  moved mass keeps the sink's velocity: the sink's velocity is unchanged, the
  gas gains the wind's radial momentum as in "torch" plus taken x v_sink,
  and with `wind_add_heat` the gas energy rises by 0.5 dm v_w^2 + 0.5 taken
  v_sink^2. Nothing is created. The carry's sink masses (and the sink store
  Torch reads, if mirrored) are reduced before the step, so the step's
  accretion and Torch's star formation after it see them. Momentum mode only.

Where the port is not literal: several stars are injected in one scatter
(their increments summed where spheres overlap, FLASH goes star by star; only
the energy mode is nonlinear in that); the unloaded FLASH path (solid-angle
weights, which its own par file calls buggy) is not ported; `min_radius` is 0
as in Torch's par; FLASH's `min_wind_mass` (7 Msun, current mass) gate is not
applied (Torch's Python gates on `min_feedback_mass` already).

What sets the step in res4 fb_256, and what the modes do to it (measured
2026-10-01). From 1.2 Myr the CFL cells are ionised gas at the density floor
(n ~ 0.3 cm^-3, 1.0-1.3 x the floor) moving at 1300-1800 km/s, 2-4 cells from
the 38.7 Msun star (Mdot 0.57 Msun/Myr, v_w ~3100 km/s); c_s is 15 km/s,
since `rad_ionization` prescribes the pressure, so the signal speed is |v|.
The star sits in a cavity at the floor out to 2-3 cells, its shell at 6-10.
Momentum into floor gas leaves at v_w: sweeping up the gas there cannot slow
a steady wind, since the swept gas has left. Restarting the run from step
50387 for 13 minutes (apudev, one GPU): the frozen tree keeps bridge steps of
3.9 yr (hydro dt median 5.2 yr, peak |v| near the stars 1880 km/s); "torch"
and "sink" give 7.8 yr (hydro dt median 14.9 yr, |v| 627 km/s ~ v_ref; the
bridge dt is a power-of-two fraction of 2 kyr, so 14.9 yr buys a factor 2),
~120 kyr of model time per wall hour against ~63. "torch" created 6.1e-2
Msun in 25.6 kyr (2.4 Msun/Myr, 4x the stars' own wind mass); "sink" moved
6.0e-2 Msun out of the sinks instead, never capped.

The density floor (`cosmo_rho_floor`) used to create mass: a cell under the
floor was set to it with its momentum zeroed and its energy reset. In the
same restarts it added 8.8e-4 Msun in 12.3 kyr (frozen tree) and 4.8e-4 in
25.6 kyr ("torch"); with it counted, gas + sinks + stars - created moved by
the winds the stars had not yet lost through SeBa and nothing else
(+2.1e-5 frozen tree, +1.5e-3 "torch"/"sink": with 7.8-yr bridge steps
Torch's stellar-evolution step came every ~113 yr, not ~102, each taking the
same 5.8e-5 Msun off the stars, so SeBa's mass loss lags the injected winds;
why on Torch's side is not traced; the injection itself books to round-off in
every interval without an SE step).

**The density floor conserves mass (2026-10-01).** The floored cell still
gets rho_floor, zero momentum and the floor energy, but the mass it gains,
D = rho_floor - rho, now comes from its 26 neighbours
(`cosmo_floors.conservative_density_floor`). Each neighbour can give its
excess S = rho - rho_floor; a floored cell asks every neighbour for the same
fraction q = D / sum(S) of its excess, a neighbour asked for more than it
has gives all of it, and what the neighbourhood cannot cover is taken from
the whole grid in proportion to the excess left (one global sum; it only
fires when a hole is deeper than its surroundings). A donor loses mass at
its own velocity and specific energy. Mass is conserved to round-off, no
cell drops under the floor, and nothing crosses a non-periodic face. The
whole branch sits behind a `lax.cond` on "any cell under the floor", so a
run that never reaches the floor is bitwise what it was. The AMR floors
(`amr/cosmo.py`, `amr_rho_floor`) still create mass; they are not on the
coupled path.

**The budget line counted winds twice.** It subtracted the winds and SN
ejecta injected into the gas, as if the stars kept their mass. They do not:
Torch takes SeBa's loss and the SN ejecta off each star and pushes the new
mass with `set_particle_mass`, so `stars` has already lost them. The line is
now gas + sinks + sinks gone + stars - created (torch mode) - in + out, and
it also prints what Torch took off the stars (`dm_star_loss`) next to the
winds and SNe injected; their difference is the SeBa lag above.

## Traps

- `Config` is frozen: `replace`, never assign.
- `sink_sink_gravity` **must** be off under PeTar, or star–star gravity is
  counted twice. It is also the only O(N²) term in `Sinks`.
- The IC registry keys off the test problem's **name**, not
  `SPATIAL_DIMENSION`. `"Sod-Shock-3D"` resolves to a tube with `Ny = Nz = 1`.
- Testing a worktree needs `PYTHONPATH=<worktree>/src`; the editable install
  points at the main checkout, so without it you test main.
- AMUSE is not in every environment here. A test module that imports it errors
  at collection rather than failing, which reads as "not run" and hides red
  tests.
