# LUNA environment variables

Every numerics option is a `Config` field (`docs/CONFIG.md`), set in the
config file or as `field=value` on the command line. The environment
variables below are what is left: machine and start-up settings, diagnostics,
the AMR package's communication strategy, and the Torch worker's config file.
Set them in the shell (or the SLURM script) before launching; they are read
when the solver or plan is built, so changing one mid-run has no effect.
Boolean ones are on when set to `1` unless noted.

The numerics variables `LUNA_HYDRO_FUSED`, `LUNA_HYDRO_RECON32`,
`LUNA_AMR_REFLUX`, `LUNA_MG_MIXED`, `LUNA_MG_CA`, `LUNA_MG_SMOOTHER` and
`LUNA_KH_NOISE` are no longer read: use `hydro_fused`, `hydro_recon32`,
`amr_reflux`, `gravity_precision`, `gravity_mg_ca`, `gravity_mg_smoother` and
`problem.noise_amp` (Kelvin-Helmholtz).

## Start-up (read in `luna/setup/jax_init.py`)

| Variable | Default | Values | Effect |
|---|---|---|---|
| `LUNA_JAX_INIT_TIMEOUT` | 300 s | seconds | `jax.distributed.initialize` bootstrap deadline. |
| `LUNA_XLA_CACHE_DIR` | `~/.jax_cache` | path | Persistent XLA compilation cache directory. |
| `AMD_SERIALIZE_COPY` | `3` on ROCm | ROCm runtime | Set by LUNA itself unless already set, see below. |

### `AMD_SERIALIZE_COPY` (the ROCm copy race)

The HIP runtime of ROCm 7.2 races in its copy path. With jax 0.10.2
(`jax_env2`) runs hang at a random step, waiting on a host read:
dark-matter particle exchanges (2 processes, and 1 process with 2 GPUs), the
halo-plan build of sharded AMR, the Tier-1 turbulent cloud (5 of 8 runs at
32^3 on 1 process x 2 GPUs). `AMD_SERIALIZE_COPY=3` serializes the copies and
the hangs are gone; kernel launches and RCCL stay asynchronous.

`initialize_jax_distributed` sets `AMD_SERIALIZE_COPY=3`, and prints one line
saying so, when the run is on ROCm and the variable is not in the
environment:

- ROCm: `JAX_PLATFORMS` names `rocm`, or, when `JAX_PLATFORMS` is unset (or
  says `gpu`), a `jax-rocm*` plugin is installed. The check does not start the
  backend: the runtime reads the variable only once, when it starts.
- A value you set, even an empty one, wins: `AMD_SERIALIZE_COPY=` switches the
  workaround off.
- If the backend has already started (a script that touched `jax.devices()`
  first), it is too late: LUNA prints a warning and leaves it to the job script.

Measured on MI300A (apudev, off vs on, same code):

| | jax 0.10.0 | jax 0.10.2 |
|---|---|---|
| hydro roofline 512³, Pallas | 52.53 vs 52.49 ms (0 %) | 51.88 vs 51.85 ms (0 %) |
| self-gravity, Jeans 256³, 1 GPU | 778.0 vs 775.4 Mcells/s (-0.3 %, two pairs) | 778 vs 756 Mcells/s (-2.9 %, three pairs) |
| dark matter l9 (512³ particles), 2 processes | 41.2 vs 40.8 ms/step (0 %) | hangs after step 1 vs 41.4 ms/step |
| start-up failures, 10 two-process launches with it | 0 of 10 | 0 of 10 |

## Diagnostics

| Variable | Default | Values | Effect |
|---|---|---|---|
| `LUNA_FLOP_COUNT` | off | `1` | Print the per-step FLOP estimate at compile time. `luna/setup/hydro.py`. |
| `LUNA_MEM_REPORT` | off | any | Every rank prints its device memory in use and the limit before the first step. `luna/setup/evolve.py`. |
| `LUNA_ROCTX` | off | any | Emit the step parts as roctx ranges even when rocprofv3 is not detected in `LD_PRELOAD`. `luna/setup/step_timing.py`. |

## Block AMR (read in `luna/setup/runner_amr.py`, `luna/amr/`)

| Variable | Default | Values | Effect |
|---|---|---|---|
| `LUNA_AMR_HALO_MODE` | `sparse` | `sparse` / `gather` / `ring` | Halo/seam exchange strategy when sharded. `sparse` = host-built per-device-pair `all_to_all` plans, O(surface) wire (benched fastest AND lowest memory). `gather` = plain `all_gather`. `ring` = chunked ring gather for OOM-scale hierarchies. |
| `LUNA_AMR_HALO_RING` | `auto` | `auto` / `0` / `1` | Force the ring gather on/off; `auto` engages it when a full gather would exceed the threshold below. |
| `LUNA_AMR_HALO_RING_GB` | `16` | float (GB) | Gather-size threshold for the auto ring gate. |
| `LUNA_AMR_NO_SHRINK` | off | `1` | Freeze the hysteretic pow2 capacity ladder (no shrink-triggered recompiles); costs padded memory. `luna/amr/driver.py`. |
| `LUNA_AMR_SHARD_DBG` | off | any | Per-process device-memory report while building the sharded hierarchy. |
| `LUNA_AMR_LOOP_DBG` | off | any | Per-op loop trace + per-level regrid trace (slow; host sync per op). |
| `LUNA_AMR_DBG_COLLECTIVE` | off | any | Log every collective kernel dispatch in `luna/amr/shard.py` (multi-process lockstep debugging). |

## Torch coupling (read in `src/amuse_luna/interface.py`)

| Variable | Default | Values | Effect |
|---|---|---|---|
| `LUNA_CONFIG` | unset | path | The LUNA worker loads this config file (as `python main.py <file>` would) before the values set through the AMUSE interface; docs/TORCH.md. |

## Autodiff experiments: command-line options now

The experiment variables became options of the example tasks in
`scripts/examples/autodiff/`, given on the run script's command line:

| was | now |
|---|---|
| `LUNA_EMERG_MOVIE`, `_MOVIE_EVERY`, `_MOVIE_NFRAME`, `_CKPT`, `_HOSTLOOP`, `_MAXIT_PER_PROC` | `movie=`, `movie_every=`, `movie_nframe=`, `ckpt=`, `hostloop=1`, `maxit_per_proc=` (`blast_emergence_run`, `turb_emergence_run`) |
| `LUNA_M82_FWDONLY=1` | removed with the X-ray sensitivity example |
| `LUNA_KHC_F`, `LUNA_KHC_NTRAIN` | `fine_factor=`, `n_train=` (`kh_corrector_run`) |

## Tests

`LUNA_JAXAMR` (path to a JAX-AMR checkout for the parity test in
`tests/test_amr_geometry.py`) and `LUNA_REFERENCE_SOLUTIONS_UNLOCKED=1` (accept a
luna-reference-solutions checkout that is not the commit in `reference-solutions.lock`;
logged, see `src/luna/reference_data.py`).
