"""The output directory of a run: the opt-in wipe, then create."""
from pathlib import Path


def prepare_output_dir(cfg, repo):
    """Wipe `cfg.output_dir` when `clear_output_dir` asks (validation analysis
    globs every snapdir, so stale output would corrupt it; default off keeps
    production and restarts safe), refusing the cwd, home, repo root and any
    top-level path; then create it. Only process 0 deletes, the rest wait."""
    import jax
    if cfg.clear_output_dir:
        target = Path(cfg.output_dir).resolve()
        forbidden = {Path.cwd().resolve(), Path.home().resolve(), Path("/")}
        if target in forbidden or target == Path(repo).resolve() or len(target.parts) < 3:
            raise ValueError(f"clear_output_dir=True refuses to wipe {target} "
                             "(cwd, home, repo root, filesystem root or a top-level dir)")
        if jax.process_index() == 0:
            import shutil
            shutil.rmtree(cfg.output_dir, ignore_errors=True)
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils
            multihost_utils.sync_global_devices("clear_output_dir")
    try:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass   # multi-process NFS race: the dir exists, which is what we wanted
