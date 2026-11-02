"""Sharded AMR through main.run on 2 CPU processes against 1 process of 2 devices: no hang,
the same multi-process programs on every rank, and the same blocks bit for bit."""
import ast
import os
import signal
import subprocess
import sys

import numpy as np
import pytest

from tests.test_sinks_multiprocess import _env, _free_port

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(ROOT, "tests", "_amr_multiprocess_run.py")
TIMEOUT = 400          # a run fits well inside; a hang never ends


def _run(n_proc, n_dev, out):
    """Rank 0's RESULT plus every rank's PROGRAMS; fails with every rank's tail on a hang."""
    port = _free_port()
    args = [sys.executable, DRIVER, f"output_dir={out}", f"dump_after={TIMEOUT - 30}"]
    procs = [subprocess.Popen(args, cwd=ROOT, env=_env(n_dev, n_proc, r, port),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              start_new_session=True) for r in range(n_proc)]
    outs, hung = [], False
    for p in procs:
        try:
            outs.append(p.communicate(timeout=TIMEOUT if not hung else 5)[0])
        except subprocess.TimeoutExpired:
            hung = True
            for q in procs:
                if q.poll() is None:
                    os.killpg(q.pid, signal.SIGKILL)
            outs.append(p.communicate()[0])
    tails = "\n".join(f"--- rank {r} (rc {p.returncode}) ---\n{o[-4000:]}"
                      for r, (p, o) in enumerate(zip(procs, outs)))
    if hung:
        pytest.fail(f"{n_proc}-process sharded AMR run hung (> {TIMEOUT} s):\n{tails}")
    for p in procs:
        assert p.returncode == 0, f"a rank failed:\n{tails}"
    line = next((ln for ln in outs[0].splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, f"no RESULT line:\n{tails}"
    res = ast.literal_eval(line[len("RESULT "):])
    res["programs"] = []
    for r, o in enumerate(outs):
        tag = f"PROGRAMS {r} "
        got = [ln for ln in o.splitlines() if ln.startswith(tag)]
        assert len(got) == 1, f"no {tag}line:\n{tails}"
        res["programs"].append(ast.literal_eval(got[0][len(tag):]))
    return res


def test_sharded_amr_two_processes_equal_one(tmp_path):
    one = _run(1, 2, str(tmp_path / "one"))
    two = _run(2, 1, str(tmp_path / "two"))
    assert (one["processes"], one["devices"]) == (1, 2)
    assert (two["processes"], two["devices"]) == (2, 2)
    assert one["n_steps"] >= 4 and one["levels"] == 2, one["n_steps"]
    assert len(one["coords"][1]) > 0, "nothing refined"
    # the same programs over both processes, in the same order, on each rank
    p0, p1 = two["programs"]
    assert p0 and p0 == p1, ("ranks compiled different multi-process programs:\n"
                             f"rank 0: {p0}\nrank 1: {p1}")
    # 2 processes == 1 process, bit for bit
    assert (two["n_steps"], two["t"], two["snap"]) == (one["n_steps"], one["t"], one["snap"])
    for L in range(one["levels"]):
        np.testing.assert_array_equal(np.array(two["coords"][L]), np.array(one["coords"][L]))
        np.testing.assert_array_equal(np.array(two["data"][L]), np.array(one["data"][L]),
                                      err_msg=f"level {L}")
