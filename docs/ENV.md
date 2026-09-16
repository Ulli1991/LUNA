# LUNA environment-variable knobs

The **library-level** knobs — variables read inside `amr/`, `setup/`,
`time_integration/` that change numerics, communication strategy, or
performance. These are the ones every user should know about; set them in the
shell (or the SLURM script) before launching.

Per-experiment overrides (`LUNA_M82_*`, `LUNA_EMERG_*`, `LUNA_BB_*`, … read by
individual `scripts/validation/runs/*.py`) are documented in the run scripts
themselves and are not listed here.

## Distribution / mesh

| Variable | Default | Values | Effect |
|---|---|---|---|
| `LUNA_MESH_SHAPE` | auto | e.g. `2,2,4` | Pins the device-mesh decomposition in the run scripts (product must equal the GPU count). Equivalent to `cfg.MESH_SHAPE`. |

## Dense hydro

| Variable | Default | Values | Effect |
|---|---|---|---|
| `LUNA_HYDRO_FUSED` | `0` | `0` / `1` | Fused 3D MUSCL-Hancock step (single traced kernel; 2nd order at shard seams, boundary 0/1 only). Perf knob, validated at 512³ FP64 on MI300A. `time_integration/integrators.py`. |
| `LUNA_DBG_STEP` | off | `1` | Per-step debug prints in the dense runner (rank 0). |

## Block-AMR distribution (read in `setup/runner_amr.py` / `amr/shard.py`)

| Variable | Default | Values | Effect |
|---|---|---|---|
| `LUNA_AMR_HALO_MODE` | `sparse` | `sparse` / `gather` / `ring` | Halo/seam exchange strategy when sharded. `sparse` = host-built per-device-pair `all_to_all` plans, O(surface) wire (benched fastest AND lowest memory). `gather` = plain `all_gather`. `ring` = chunked ring gather for OOM-scale hierarchies. |
| `LUNA_AMR_HALO_RING` | `auto` | `auto` / `0` / `1` | Force the ring gather on/off; `auto` engages it when a full gather would exceed the threshold below. |
| `LUNA_AMR_HALO_RING_GB` | `16` | float (GB) | Gather-size threshold for the auto ring gate. |
| `LUNA_AMR_REFLUX` | `repl` | `repl` / `sharded` / `iface` | Coarse-fine reflux mode: replicated wrapper (default), explicit-psum distributed, or plan-routed interface flux (`iface` is opt-in: +23% at the 8-GPU run, but a 2-process hang is still unresolved). |
| `LUNA_AMR_NO_SHRINK` | off | `1` | Freeze the hysteretic pow2 capacity ladder (no shrink-triggered recompiles); costs padded memory. `amr/driver.py`. |

## Block-AMR debugging

| Variable | Default | Effect |
|---|---|---|
| `LUNA_AMR_SHARD_DBG` | off | Per-process device-memory report while building the sharded hierarchy. |
| `LUNA_AMR_LOOP_DBG` | off | Per-op loop trace + per-level regrid trace (slow; host sync per op). |
| `LUNA_AMR_DBG_COLLECTIVE` | off | Log every collective kernel dispatch in `amr/shard.py` (multi-process lockstep debugging). |

## Conventions

Boolean knobs are read as `== "1"` unless noted. Knobs are read at solver/plan
build time — set them before the run starts; changing them mid-run has no
effect. New library-level knobs belong in this table; new per-experiment knobs
belong in their run script's docstring.
