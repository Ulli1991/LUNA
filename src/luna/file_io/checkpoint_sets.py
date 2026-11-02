"""Checkpoint sets: when one is due, when one is complete, which one to restart from.

See docs/RUNNING.md (Checkpoints).
"""
import glob
import os
import re

_STEP = re.compile(r"^(?:checkpoint|radiation|coupled)_(\d+)\.")


def marker(d, step):
    return os.path.join(d, f"checkpoint_{int(step):06d}.complete")


def mark_complete(d, step):
    """Write the set's completion marker (rank 0, after the barrier following the last file)."""
    tmp = marker(d, step) + ".partial"
    with open(tmp, "w") as fh:
        fh.write(f"{int(step)}\n")
    os.replace(tmp, marker(d, step))


def steps_on_disk(d):
    """Return every step with any file of a set, sorted."""
    out = set()
    for f in os.listdir(d) if os.path.isdir(d) else ():
        m = _STEP.match(f)
        if m and not f.endswith(".partial"):
            out.add(int(m.group(1)))
    return sorted(out)


def complete_steps(d):
    """Return the steps with a marker and their gas files (a stray marker is not a set)."""
    return [s for s in steps_on_disk(d) if os.path.isfile(marker(d, s))
            and glob.glob(os.path.join(d, f"checkpoint_{s:06d}*.hdf5"))]


def latest_step(d):
    """Return the newest complete set's step (newest on disk if there are no markers at all), or None."""
    done = complete_steps(d)
    if done:
        return done[-1]
    if glob.glob(os.path.join(d, "checkpoint_*.complete")):
        return None
    allsteps = steps_on_disk(d)
    return allsteps[-1] if allsteps else None


def prune(d, keep):
    """Delete every file of all but the newest `keep` complete sets (keep <= 0 keeps all)."""
    if keep <= 0:
        return
    done = complete_steps(d)
    if not done:
        return
    keep_steps = set(done[-keep:])
    newest = done[-1]
    for s in steps_on_disk(d):
        if s in keep_steps or s > newest:
            continue
        for f in glob.glob(os.path.join(d, f"*_{s:06d}.*")):
            os.remove(f)
            print(f"  Deleted old checkpoint file: {os.path.basename(f)}")


def due_everywhere(local_due):
    """Return rank 0's answer on every rank; the write is collective, so all ranks must agree."""
    import jax
    if jax.process_count() == 1:
        return bool(local_due)
    import numpy as np
    from jax.experimental import multihost_utils
    return bool(multihost_utils.broadcast_one_to_all(np.int32(bool(local_due))))


def write_atomically(path, write):
    """Call `write(tmp_path)`, then rename into place, so a killed job never leaves a truncated file."""
    tmp = path + ".partial"
    write(tmp)
    os.replace(tmp, path)
