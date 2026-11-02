"""Checkpoint sets (luna/file_io/checkpoint_sets.py): a restart takes the newest
COMPLETE set, never a newer one a killed job left half-written; pruning keeps
whole complete sets, radiation and coupled files with them; a writer that dies
leaves a .partial, not a truncated file under the real name."""
import os

import pytest

from luna.file_io import checkpoint_sets as cs
from luna.file_io.read_checkpoint import find_latest_checkpoint

pytestmark = pytest.mark.fast


def _touch(d, name, text="x"):
    with open(os.path.join(d, name), "w") as fh:
        fh.write(text)


def _set(d, step, n_proc=2, radiation=True, coupled=True, complete=True):
    for p in range(n_proc):
        _touch(d, f"checkpoint_{step:06d}.{p}.hdf5")
        if radiation:
            _touch(d, f"radiation_{step:06d}.{p}.hdf5")
    if coupled:
        _touch(d, f"coupled_{step:06d}.json", "{}")
        _touch(d, f"coupled_{step:06d}.npz")
    if complete:
        cs.mark_complete(d, step)


def test_the_newest_complete_set_wins_over_a_newer_partial_one(tmp_path):
    d = str(tmp_path)
    _set(d, 100)
    _set(d, 200)
    _set(d, 300, complete=False)              # the job died writing it
    assert cs.latest_step(d) == 200
    assert os.path.basename(find_latest_checkpoint(d)).startswith("checkpoint_000200.")


def test_a_directory_from_before_markers_restarts_from_its_newest_set(tmp_path):
    d = str(tmp_path)
    _set(d, 100, complete=False)
    _set(d, 200, complete=False)
    assert cs.latest_step(d) == 200
    assert os.path.basename(find_latest_checkpoint(d)).startswith("checkpoint_000200.")


def test_markers_but_none_complete_is_no_restart_point(tmp_path):
    """Once a directory uses markers, an unmarked set is never trusted."""
    d = str(tmp_path)
    _set(d, 100)
    os.remove(cs.marker(d, 100))
    _touch(d, "checkpoint_000050.complete")   # an old marker whose set was pruned
    _set(d, 200, complete=False)
    assert cs.latest_step(d) is None
    assert find_latest_checkpoint(d) is None


def test_prune_keeps_whole_complete_sets(tmp_path):
    d = str(tmp_path)
    for s in (100, 200, 300, 400):
        _set(d, s)
    _set(d, 250, complete=False)              # older than the newest complete: leftover
    _set(d, 500, complete=False)              # newer: possibly being written, kept
    cs.prune(d, 2)
    left = sorted(os.listdir(d))
    assert not any(f"_{s:06d}." in f for f in left for s in (100, 200, 250))
    for s in (300, 400):
        assert os.path.isfile(cs.marker(d, s))
        assert f"radiation_{s:06d}.0.hdf5" in left and f"coupled_{s:06d}.json" in left
    assert "checkpoint_000500.0.hdf5" in left


def test_a_writer_that_dies_leaves_no_file_under_the_real_name(tmp_path):
    path = str(tmp_path / "checkpoint_000100.0.hdf5")

    def dies(tmp):
        with open(tmp, "w") as fh:
            fh.write("half")
        raise OSError("killed")
    with pytest.raises(OSError):
        cs.write_atomically(path, dies)
    assert not os.path.exists(path)
    assert os.path.exists(path + ".partial")
    assert cs.steps_on_disk(str(tmp_path)) == []   # a .partial is not a set


def test_one_process_decides_for_itself():
    assert cs.due_everywhere(True) is True
    assert cs.due_everywhere(False) is False
