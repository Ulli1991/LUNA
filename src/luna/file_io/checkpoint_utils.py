"""Checkpoint housekeeping: rotation, config-compatibility checks and run provenance."""
import os
import glob
import json
import socket
import subprocess
import sys
import datetime
import functools
from dataclasses import asdict, is_dataclass

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_json(cfg):
    """Return the full config as a JSON string (non-JSON values stringified) for provenance."""
    d = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    return json.dumps(d, default=str, sort_keys=True)


@functools.lru_cache(maxsize=None)
def code_revision():
    """The full git commit of the running code, '-dirty' if tracked files differ from it; without git
    the CODE_REVISION file a deployment wrote next to the code; an installed wheel gives its version."""
    if not os.path.isfile(os.path.join(_REPO_ROOT, "..", "..", "pyproject.toml")):
        rev = _revision_file()           # not a source checkout: never ask git of whatever repo encloses it
        return rev if rev != "unknown" else _installed_version()
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
        if not head:
            return _revision_file()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=_REPO_ROOT,
                               capture_output=True, text=True, timeout=30).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return _revision_file()


def _revision_file():
    fn = os.path.join(os.path.dirname(os.path.dirname(_REPO_ROOT)), "CODE_REVISION")
    try:
        with open(fn) as fh:
            return fh.read().strip() or "unknown"
    except OSError:
        return "unknown"


def _installed_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        return "luna-hydro " + version("luna-hydro")
    except PackageNotFoundError:
        return "unknown"


def _git_revision():
    return code_revision()


def write_provenance(attrs, cfg):
    """Every output header: the full configuration and the exact code that wrote it."""
    attrs["config_json"] = config_json(cfg)
    attrs["code_revision"] = code_revision()


def write_run_json(out_dir, cfg, is_main_process=True, extra=None):
    """Write <out_dir>/run.json: the config plus git revision, argv, host, JAX version, start time."""
    if not is_main_process:
        return None
    try:
        import jax
        jax_ver, nproc = jax.__version__, jax.process_count()
    except Exception:
        jax_ver, nproc = "unknown", 1
    rec = {"config": json.loads(config_json(cfg)),
           "git_revision": _git_revision(),
           "argv": sys.argv,
           "hostname": socket.gethostname(),
           "jax_version": jax_ver,
           "process_count": int(nproc),
           "started": datetime.datetime.now().isoformat(timespec="seconds")}
    if extra:
        rec.update(extra)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "run.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    return path


def _write_meta(f, cfg, t, step, local_shape=None):
    """Write the state/grid/physics/numerics/time_control groups shared by the checkpoint writers."""
    state = f.create_group("state")
    state.attrs["time"] = float(t)
    state.attrs["step"] = int(step)
    write_provenance(f.attrs, cfg)

    grid = f.create_group("grid")
    grid.attrs["Nx"] = cfg.Nx
    grid.attrs["Ny"] = cfg.Ny
    grid.attrs["Nz"] = cfg.Nz
    grid.attrs["Lx"] = cfg.Lx
    grid.attrs["Ly"] = cfg.Ly
    grid.attrs["Lz"] = cfg.Lz
    grid.attrs["SPATIAL_DIMENSION"] = cfg.SPATIAL_DIMENSION
    grid.attrs["BOUNDARY"] = cfg.BOUNDARY
    if local_shape is not None:
        grid.attrs["local_shape"] = local_shape

    physics = f.create_group("physics")
    physics.attrs["gamma"] = cfg.gamma

    numerics = f.create_group("numerics")
    numerics.attrs["ORDER"] = cfg.ORDER
    numerics.attrs["LIMITER"] = cfg.LIMITER
    numerics.attrs["CFL"] = cfg.CFL

    time_control = f.create_group("time_control")
    time_control.attrs["t_start"] = cfg.t_start
    time_control.attrs["t_end"] = cfg.t_end
    time_control.attrs["time_bet_snaps"] = cfg.time_bet_snaps


def _write_units(f, cfg, full=False):
    """Write the units group if cfg.units is enabled; full=True adds the derived units."""
    if not cfg.units.is_enabled:
        return
    units_grp = f.create_group("units")
    units_grp.attrs["UnitLength_in_cm"] = cfg.units.UnitLength_in_cm
    units_grp.attrs["UnitMass_in_g"] = cfg.units.UnitMass_in_g
    units_grp.attrs["UnitVelocity_in_cm_per_s"] = cfg.units.UnitVelocity_in_cm_per_s
    if full:
        units_grp.attrs["UnitTime_in_s"] = cfg.units.UnitTime_in_s
        units_grp.attrs["UnitDensity_in_cgs"] = cfg.units.UnitDensity_in_cgs
        units_grp.attrs["UnitPressure_in_cgs"] = cfg.units.UnitPressure_in_cgs
        units_grp.attrs["UnitEnergy_in_cgs"] = cfg.units.UnitEnergy_in_cgs


def _shard_context(U_sharded, cfg, mesh, per_process=False):
    """Return (process_id, process_count, mesh, local_shards) for the distributed writers.

    per_process=True sizes a reconstructed mesh by process count in multi-process runs.
    """
    import jax
    from luna.utils.distributed import create_device_mesh

    process_id = jax.process_index()
    process_count = jax.process_count()
    if mesh is None:
        n = process_count if (per_process and process_count > 1) else jax.device_count()
        mesh, _ = create_device_mesh(n, cfg.SPATIAL_DIMENSION)
    local_shards = U_sharded.addressable_shards
    if len(local_shards) == 0:
        print(f"Warning: Process {process_id} has no addressable shards")
    return process_id, process_count, mesh, local_shards


def manage_checkpoints(checkpoint_dir, max_checkpoints):
    """Delete all but the newest max_checkpoints steps (every process file of a step; <= 0 keeps all)."""
    import re
    if max_checkpoints <= 0:
        return

    pat = re.compile(r"checkpoint_(\d+)(?:\.\d+)?\.hdf5$")
    by_step = {}
    for fn in glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.hdf5")):
        m = pat.match(os.path.basename(fn))
        if m:
            by_step.setdefault(int(m.group(1)), []).append(fn)
    steps = sorted(by_step)
    for st in steps[:max(len(steps) - max_checkpoints, 0)]:
        for fn in by_step[st]:
            os.remove(fn)
            print(f"  Deleted old checkpoint: {os.path.basename(fn)}")


def validate_checkpoint_compatibility(checkpoint_data, cfg):
    """Raise ValueError if the checkpoint's grid, domain, dimension, BOUNDARY or gamma differ from cfg."""
    grid = checkpoint_data["grid"]
    physics = checkpoint_data["physics"]

    if grid["Nx"] != cfg.Nx or grid["Ny"] != cfg.Ny or grid["Nz"] != cfg.Nz:
        raise ValueError(
            f"Grid mismatch: checkpoint has ({grid['Nx']}, {grid['Ny']}, {grid['Nz']}), "
            f"config has ({cfg.Nx}, {cfg.Ny}, {cfg.Nz})"
        )

    if (
        abs(grid["Lx"] - cfg.Lx) > 1e-10
        or abs(grid["Ly"] - cfg.Ly) > 1e-10
        or abs(grid["Lz"] - cfg.Lz) > 1e-10
    ):
        raise ValueError(
            f"Domain size mismatch: checkpoint has ({grid['Lx']}, {grid['Ly']}, {grid['Lz']}), "
            f"config has ({cfg.Lx}, {cfg.Ly}, {cfg.Lz})"
        )

    if grid["SPATIAL_DIMENSION"] != cfg.SPATIAL_DIMENSION:
        raise ValueError(
            f"Spatial dimension mismatch: checkpoint has {grid['SPATIAL_DIMENSION']}, "
            f"config has {cfg.SPATIAL_DIMENSION}"
        )

    if abs(physics["gamma"] - cfg.gamma) > 1e-10:
        raise ValueError(f"gamma mismatch: checkpoint has {physics['gamma']}, "
                         f"config has {cfg.gamma}")
    if "BOUNDARY" in grid and grid["BOUNDARY"] != cfg.BOUNDARY:
        raise ValueError(f"BOUNDARY mismatch: checkpoint has {grid['BOUNDARY']}, "
                         f"config has {cfg.BOUNDARY}")

    print("✓ Checkpoint compatibility validated")


def checkpoint_dir(cfg):
    """Return cfg.checkpoint_dir, or "<output_dir>/checkpoints" by default."""
    import os
    return cfg.checkpoint_dir or os.path.join(cfg.output_dir, "checkpoints")
