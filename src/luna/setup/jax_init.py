"""Process start-up: JAX distributed under SLURM (one GPU per task), float64
as the config says, the compilation cache. Called once, before `run(cfg)`."""
import os

import jax


def _local_device_ids_from_slurm():
    """Return [local_gpu_index] for this process, or None to let JAX decide.

    Returns None (JAX default: use all visible local devices) when we're not
    under a multi-task SLURM step, or when the per-task GPU binding already
    narrowed this process to a single visible device — both cases JAX handles
    correctly on its own. Returns [SLURM_LOCALID] only when every task on the
    node can see multiple GPUs, so JAX would otherwise pick the wrong one.
    """
    localid = os.environ.get("SLURM_LOCALID")
    ntasks = int(os.environ.get("SLURM_STEP_NUM_TASKS") or os.environ.get("SLURM_NTASKS") or 1)
    nnodes = int(os.environ.get("SLURM_STEP_NUM_NODES") or os.environ.get("SLURM_NNODES") or 1)
    if localid is None or ntasks <= 1:
        return None

    visible = (
        os.environ.get("ROCR_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("HIP_VISIBLE_DEVICES")
    )
    # One task per node (e.g. one process owning both GPUs of a node): it
    # takes every GPU it can see. Say so explicitly -- given None, JAX's own
    # SLURM detection fills in [SLURM_LOCALID] and the process gets one GPU.
    if ntasks <= nnodes:
        n = (len([x for x in visible.split(",") if x != ""]) if visible
             else int(os.environ.get("SLURM_GPUS_ON_NODE") or 0))
        return list(range(n)) if n > 0 else None
    if visible is not None:
        n_visible = len([x for x in visible.split(",") if x != ""])
        # Per-task binding already isolated one GPU -> it's local index 0.
        if n_visible <= 1:
            return [0]
    # No binding (all local GPUs visible to every task): claim ours by localid.
    return [int(localid)]


def _on_rocm():
    """Whether JAX will run on ROCm, without starting a backend (JAX_PLATFORMS, else a plugin)."""
    plats = [p.strip().lower() for p in os.environ.get("JAX_PLATFORMS", "").split(",") if p.strip()]
    if plats and "gpu" not in plats:
        return "rocm" in plats
    from importlib.metadata import distributions
    return any((d.metadata["Name"] or "").lower().replace("_", "-").startswith("jax-rocm")
               for d in distributions())


def serialize_rocm_copies(print=print):
    """On ROCm, set AMD_SERIALIZE_COPY=3 unless it is set already; True if it was set."""
    # the HIP runtime's copy race hangs runs (docs/ENV.md); it reads the variable once
    if "AMD_SERIALIZE_COPY" in os.environ or not _on_rocm():
        return False
    from jax._src import xla_bridge
    if xla_bridge.backends_are_initialized():
        print("⚠ ROCm without AMD_SERIALIZE_COPY and the backend already started: "
              "set AMD_SERIALIZE_COPY=3 in the job script (docs/ENV.md)", flush=True)
        return False
    os.environ["AMD_SERIALIZE_COPY"] = "3"
    print("✓ AMD_SERIALIZE_COPY=3 (the ROCm copy-race workaround, docs/ENV.md)", flush=True)
    return True


def with_host_platform():
    """Add "cpu" to an explicit JAX_PLATFORMS (cuda -> cuda,cpu; the first stays the default): the AMR face
    report (amr/hydro.report_amr_faces) is a jax.debug.callback, which needs a CPU device on the host."""
    plats = jax.config.jax_platforms
    if plats and "cpu" not in [p.strip() for p in plats.split(",")]:
        jax.config.update("jax_platforms", plats + ",cpu")


def initialize_jax_distributed(enable_x64: bool = False):
    """Start JAX distributed from the SLURM environment (single process
    outside a multi-task step), then set float64 (`enable_x64`), the matmul
    precision and the compilation cache, and print the device topology."""
    with_host_platform()
    # Mute every rank but 0: unmuted, these banners + the device listing were
    # ~17M lines / 777 MB of log per variant at 4096 ranks (job 1264426), all
    # funneled through srun onto the shared FS.
    import builtins
    print = ((lambda *a, **k: None)
             if int(os.environ.get("SLURM_PROCID", "0") or 0) != 0
             else builtins.print)
    serialize_rocm_copies(print)                 # before anything starts the backend
    print("=" * 60)
    print("MULTI-NODE SHARDING MODE (1 GPU per process)")
    print("=" * 60)

    # Which local GPU this process should own. On clusters where SLURM binds one
    # GPU per task (per-task CUDA_/ROCR_VISIBLE_DEVICES, e.g. Raven's A100 nodes)
    # each process already sees a single device at local index 0, so passing
    # nothing is correct. On clusters where every task on a node sees ALL local
    # GPUs (e.g. viper's MI300A nodes: ROCR_VISIBLE_DEVICES=0,1 for both tasks),
    # JAX must be told which one to take via local_device_ids, otherwise every
    # process tries to grab its *global* device id and the ROCm/CUDA backend
    # fails ("hipDeviceGet HIP_ERROR_InvalidDevice" / topology-exchange timeout).
    local_device_ids = _local_device_ids_from_slurm()

    if jax.distributed.is_initialized():           # a second launch in the same processes
        print(f"✓ JAX distributed already initialized: process {jax.process_index()} / {jax.process_count()}")
    else:
        try:
            init_kwargs = dict(
                coordinator_address=os.environ.get("JAX_COORDINATOR_ADDRESS"),
                num_processes=int(os.environ.get("SLURM_NTASKS", 1)),
                process_id=int(os.environ.get("SLURM_PROCID", 0)),
            )
            # Bootstrap deadline (default 300 s): a rank that can't reach the
            # coordinator in time takes the whole job down with DEADLINE_EXCEEDED.
            # Keep it below the Slurm walltime so the failure is a clean error.
            if os.environ.get("LUNA_JAX_INIT_TIMEOUT"):
                init_kwargs["initialization_timeout"] = int(
                    os.environ["LUNA_JAX_INIT_TIMEOUT"])
            if local_device_ids is not None:
                init_kwargs["local_device_ids"] = local_device_ids
            jax.distributed.initialize(**init_kwargs)
            print("✓ JAX distributed initialized")
            if local_device_ids is not None:
                print(f"  local_device_ids={local_device_ids} (SLURM_LOCALID pinning)")
            print(f"  Process: {jax.process_index()} / {jax.process_count()}")
        except (RuntimeError, ValueError) as e:
            if int(os.environ.get("SLURM_NTASKS", 1)) > 1:
                # A multi-task job must never continue as N independent rank-0
                # processes (they would all wipe and write the same output_dir).
                raise RuntimeError(
                    "jax.distributed.initialize failed in a multi-task job "
                    f"(SLURM_NTASKS={os.environ.get('SLURM_NTASKS')}, "
                    f"JAX_COORDINATOR_ADDRESS={os.environ.get('JAX_COORDINATOR_ADDRESS')})"
                ) from e
            print("  Running in single-process mode (use SLURM/MPI for multi-node)")
            if os.environ.get("JAX_COORDINATOR_ADDRESS"):   # a coordinator was asked for: say why it failed
                print(f"  jax.distributed.initialize failed: {e}")

    # Serialized launches deadlock RCCL across nodes: a bare ring ppermute
    # over 2 nodes never completes its first call (docs/AUTODIFF.md).
    nnodes = int(os.environ.get("SLURM_STEP_NUM_NODES") or os.environ.get("SLURM_NNODES") or 1)
    if nnodes > 1 and (os.environ.get("HIP_LAUNCH_BLOCKING") or os.environ.get("AMD_SERIALIZE_KERNEL")):
        print(f"⚠ HIP_LAUNCH_BLOCKING/AMD_SERIALIZE_KERNEL set on {nnodes} nodes: the "
              "first cross-node collective will hang. Unset it.", flush=True)

    # Configure JAX
    jax.config.update("jax_enable_x64", enable_x64)
    jax.config.update("jax_default_matmul_precision", "high")
    # Per-config cache dirs (size/ndev/precision) on scratch are load-bearing
    # on shared machines — see docs/history/JUPITER.md. Fall back to the old default.
    jax.config.update(
        "jax_compilation_cache_dir",
        os.environ.get("LUNA_XLA_CACHE_DIR", os.path.expanduser("~/.jax_cache")),
    )
    if enable_x64:
        print("✓ float64 enabled (jax_enable_x64=True)")

    print(f"Platform: {jax.default_backend()}")
    print(f"Total devices: {len(jax.devices())}")
    print(f"Local devices: {len(jax.local_devices())}")

    # Show device topology (capped: at 4-digit process counts the full listing
    # is thousands of lines per run and slows srun stdout plumbing).
    devs = jax.devices()
    for i, dev in enumerate(devs[:8]):
        print(f"  Device {i}: {dev.device_kind} (process {dev.process_index})")
    if len(devs) > 8:
        print(f"  ... and {len(devs) - 8} more devices")
    print()
