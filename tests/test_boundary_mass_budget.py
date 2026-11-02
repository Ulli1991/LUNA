"""Mass is accounted for at every boundary condition: on an outflow run with
isolated gravity and sinks (the M3 cloud at 32^3 with 10x its mass, 20 CFL steps), box gas +
sinks - initial = face inflow - outflow (cfg.boundary_mass_budget) to
round-off, for BOUNDARY 1 (zero-gradient, mass flows in) and 4 (diode).

Teeth: every face's net flow is far above the tolerance, so leaving any one
face out breaks the budget. The accumulator is its own program: with it on,
the gas equals the run without it bitwise. Two processes of one device each
(gloo) close the budget and equal one process of two devices.
"""
import ast
import os
import socket
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.slow

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(ROOT, "tests", "_boundary_mass_run.py")
TOL = 1e-12            # of the initial mass
FACES = ("x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")


def _env(n_dev, n_proc, rank, port):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1",
               XLA_FLAGS=f"--xla_force_host_platform_device_count={n_dev}",
               SLURM_NTASKS=str(n_proc), SLURM_PROCID=str(rank),
               JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{port}")
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.join(ROOT, "src"), ROOT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def _run(boundary, mesh, n_proc=1, **kw):
    """Rank 0's RESULT dict from `n_proc` processes over the mesh `mesh`."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    n_dev = int(np.prod(mesh)) // n_proc
    args = [sys.executable, DRIVER, f"boundary={boundary}", "mesh=" + ",".join(map(str, mesh))]
    args += [f"{k}={v}" for k, v in kw.items()]
    procs = [subprocess.Popen(args, cwd=ROOT, env=_env(n_dev, n_proc, r, port),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for r in range(n_proc)]
    outs = [p.communicate(timeout=900)[0] for p in procs]
    tails = "\n".join(f"--- rank {r} ---\n{o[-3000:]}" for r, o in enumerate(outs))
    assert all(p.returncode == 0 for p in procs), tails
    line = next((ln for ln in outs[0].splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, tails
    return ast.literal_eval(line[len("RESULT "):])


def _check_budget(r):
    """(box + sinks) - initial == in - out to TOL; returns the per-face net."""
    b = np.array(r["budget"])
    net = b[0] - b[1]
    gain = r["m_gas"] + r["m_sinks"] - r["m0"]
    resid = gain - net.sum()
    print(f"gain {gain:.6e} (in {b[0].sum():.3e}, out {b[1].sum():.3e}), residual "
          f"{resid / r['m0']:.1e} of M0; sinks {r['n_sinks']}, {r['m_sinks']:.4e}; "
          "net per face " + " ".join(f"{f} {v:+.2e}" for f, v in zip(FACES, net)))
    assert r["steps"] == 20
    assert r["n_sinks"] >= 1 and r["m_sinks"] > 0.0      # the sinks take part
    assert abs(resid) <= TOL * r["m0"]
    # teeth: without any one face the budget does not close
    for f, v in zip(FACES, net):
        assert abs(gain - (net.sum() - v)) > 100.0 * TOL * r["m0"], f
    return net


@pytest.mark.parametrize("boundary", [1, 4])
def test_budget_closes(boundary):
    r = _run(boundary, [2, 2, 2])
    net = _check_budget(r)
    if boundary == 1:
        assert np.all(net > 0.0)      # zero-gradient: the infall is fed through every face


def test_budget_is_off_the_step(tmp_path):
    """Accumulating the budget leaves the gas bitwise as without it."""
    on, off = str(tmp_path / "on.npy"), str(tmp_path / "off.npy")
    _run(4, [2, 1, 1], budget=1, out=on)
    _run(4, [2, 1, 1], budget=0, out=off)
    assert np.array_equal(np.load(on), np.load(off))


def test_budget_two_processes():
    one = _run(4, [2, 1, 1], n_proc=1)
    two = _run(4, [2, 1, 1], n_proc=2)
    assert (two["processes"], two["devices"]) == (2, 2)
    _check_budget(two)
    np.testing.assert_allclose(np.array(two["budget"]), np.array(one["budget"]),
                               rtol=1e-12, atol=TOL * one["m0"])
