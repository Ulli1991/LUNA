"""Standalone sinks across PROCESSES, on CPU: 2 processes of one device each
(gloo) against 1 process of two devices, the same 2x1x1 mesh.

The rest of the suite runs every mesh inside ONE process, where a host-side
call that only rank 0 makes cannot hang. Here the production loop
(`luna.main.run`: evolve, the progress line, diagnostics, snapshots) runs a
32^3 periodic self-gravitating box with a cold, converging Gaussian clump
that forms a sink in the first step, to t_end (a handful of steps):

  * the clump in rank 0's block only, and on the x seam between the blocks
  * a hang is a failure (timeout; each rank's stack is in the message)
  * every rank compiles the same programs over both processes, in the same
    order: on GPU compiling one waits for the other process, so a rank-0-only
    reduction hangs there (main: the progress line's sink report, in
    backend_compile_and_load); on CPU it runs, so this list is the check
  * the same for FLASH-Torch's sink rules (sink_accretion, sink_creation_checks,
    sink_merging)
  * steps, sink count, sink masses, positions, velocities and the gas mass
    equal the 1-process run to round-off; gas + sinks equals the initial gas
    mass to round-off in both
"""
import ast
import json
import os
import signal
import socket
import subprocess
import sys

import numpy as np
import pytest

from tests._sinks_multiprocess_run import L, N, T_END, config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(ROOT, "tests", "_sinks_multiprocess_run.py")
TIMEOUT = 180          # a run takes ~15 s on the CPU backend; a hang never ends
RTOL = 1e-12


def _env(n_dev, n_proc, rank, port):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1",
               XLA_FLAGS=f"--xla_force_host_platform_device_count={n_dev}",
               SLURM_NTASKS=str(n_proc), SLURM_PROCID=str(rank),
               JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{port}")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(ROOT, "src"), ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_ic(d, centre):
    """A cold clump, rho = 1 + 20 g, converging (v = -0.5 (r - c) g), on
    P = 0.1: the global potential minimum sits at `centre`."""
    from luna.ic_generation.ic import resolve
    from luna.ic_generation.ic_file import _fingerprint
    os.makedirs(d, exist_ok=True)
    x = (np.arange(N) + 0.5) * (L / N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = [X - centre[0], Y - centre[1], Z - centre[2]]
    g = np.exp(-(R[0] ** 2 + R[1] ** 2 + R[2] ** 2) / (2.0 * (2.0 * L / N) ** 2))
    rho = 1.0 + 20.0 * g
    v = [-0.5 * r * g for r in R]
    E = 0.1 / (5.0 / 3.0 - 1.0) + 0.5 * rho * sum(c * c for c in v)
    np.save(os.path.join(d, "U.npy"), np.stack([rho, rho * v[0], rho * v[1], rho * v[2], E]))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(_fingerprint(resolve(config(d, "", "1,1,1"))), f, indent=1)
    return float(rho.sum()) * (L / N) ** 3


def _run(n_proc, n_dev, ic, out, rules="luna"):
    """Launch the driver on `n_proc` processes; rank 0's RESULT, or a failure
    with every rank's output tail (a hang: its stack) if any rank fails or hangs."""
    port = _free_port()
    args = [sys.executable, DRIVER, f"ic_dir={ic}", f"output_dir={out}", "mesh=2,1,1", f"rules={rules}",
            f"dump_after={TIMEOUT - 30}"]
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
        pytest.fail(f"{n_proc}-process sink run hung (> {TIMEOUT} s):\n{tails}")
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


# (0.26, 0.41, 0.59): the clump and its accretion window inside rank 0's
# block (x < 0.5); (0.51, ...): on the seam, the window straddles both blocks
# and with FLASH-Torch's sink rules: every check is a collective over both
# processes (psum/pmax of the windows, the candidates gathered), so every rank
# has to compile the same programs here too
@pytest.mark.parametrize("rules", ["luna", "flash"])
@pytest.mark.parametrize("centre", [(0.26, 0.41, 0.59), (0.51, 0.41, 0.59)],
                         ids=["rank0-only", "on-seam"])
def test_sinks_two_processes_equal_one(centre, rules, tmp_path):
    ic = str(tmp_path / "ic")
    m0 = _write_ic(ic, centre)
    one = _run(1, 2, ic, str(tmp_path / "one"), rules)
    two = _run(2, 1, ic, str(tmp_path / "two"), rules)
    assert (one["processes"], one["devices"]) == (1, 2)
    assert (two["processes"], two["devices"]) == (2, 2)
    print({k: one[k] for k in ("n_steps", "t", "n_sinks", "m_sinks", "m_gas")})

    # a sink formed, and the run reached t_end
    assert one["n_sinks"] >= 1 and one["snap"] == "sinks_001.npz", one
    assert one["t"] == pytest.approx(T_END, rel=1e-12)
    if centre[0] < 0.5:                             # and stayed in rank 0's block
        x = np.array(one["pos"])[np.array(one["active"]) > 0.5, 0]
        assert ((x > 4.0 / N) & (x < 0.5 - 4.0 / N)).all(), x
    # mass: gas + sinks is the initial gas, in each run
    for r in (one, two):
        assert (r["m_gas"] + r["m_sinks"]) == pytest.approx(m0, rel=RTOL, abs=0.0)
    # the same programs over both processes, in the same order, on each rank
    p0, p1 = two["programs"]
    assert p0 and p0 == p1, ("ranks compiled different multi-process programs:\n"
                             f"rank 0: {p0}\nrank 1: {p1}")
    # 2 processes == 1 process
    assert two["n_steps"] == one["n_steps"] and two["snap"] == one["snap"]
    assert two["n_sinks"] == one["n_sinks"]
    assert two["active"] == one["active"]
    np.testing.assert_allclose(two["m_gas"], one["m_gas"], rtol=RTOL, atol=0.0)
    for k in ("mass", "pos", "vel"):
        np.testing.assert_allclose(np.array(two[k]), np.array(one[k]), rtol=RTOL,
                                   atol=RTOL * max(1.0, np.abs(one[k]).max()), err_msg=k)
