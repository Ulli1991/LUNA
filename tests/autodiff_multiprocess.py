"""The IC inversion's derivatives across PROCESSES, on CPU (shared by the
test_autodiff_multiprocess_* files, one per problem).

The other sharded tests split the grid over the devices of ONE process.
Production runs one process per GPU on several nodes: every neighbour of a
shard lives in another process, the halo ppermutes and the multigrid's
psums cross process boundaries, and every host fetch must be of a
replicated array. Here: 8 CPU processes of one device each (gloo
collectives; the production IC path is one device per process), the grid
32^3 on a 4x1x2 mesh (4 along x, so a halo sent the wrong way round shows),
a generic state (rho, v, p perturbed), minmod, FP64, 4 steps -- run by
tools/adj_multinode.py, the tool the multi-node GPU jobs run:

  1. the rollout, its gradient, J^T w (and J v) == on 1 device
  2. <J v, w> == <v, J^T w> (hydro only: gravity and cooling are custom_vjp,
     JAX has no JVP through them), with exactly rounded sums
  3. d(total mass)/dU == (1, 0, 0, 0, 0)

for the hydro alone, the gravity IC inversion's physics (Jeans box, FMG
adjoint) and the cooling one (OU forcing + radiative cooling). All to
1e-11, the identity tests' bound.
"""
import os
import socket
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "adj_multinode.py")
STEPS = 4


def _env(n_dev, n_proc=1, rank=0, port=0):
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update(JAX_PLATFORMS="cpu", PYTHONUNBUFFERED="1",
               XLA_FLAGS=f"--xla_force_host_platform_device_count={n_dev}",
               SLURM_NTASKS=str(n_proc), SLURM_PROCID=str(rank),
               JAX_COORDINATOR_ADDRESS=f"127.0.0.1:{port}")
    return env


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(mode, problem, nx, n_dev, n_proc, extra, timeout=900):
    """Launch `n_proc` processes of the tool; return rank 0's stdout."""
    port = _free_port()
    args = [sys.executable, TOOL, mode, f"problem={problem}", f"nx={nx}",
            f"n_steps={STEPS}", *extra]
    procs = [subprocess.Popen(args, cwd=ROOT, env=_env(n_dev, n_proc, r, port),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True) for r in range(n_proc)]
    outs = [p.communicate(timeout=timeout)[0] for p in procs]
    for p, o in zip(procs, outs):
        assert p.returncode == 0, f"{mode} rank failed (rc {p.returncode}):\n{o[-3000:]}"
    return outs[0]


def check_across_processes(problem, tmp_path, nx=32, mesh="4,1,2"):
    """Run `problem` on 8 processes against 1 device; assert the tool's CHECK passes."""
    ref = tmp_path / "ref.npz"
    fp64 = "gravity_proper_mixed=0"     # 1e-11 layout independence needs FP64 cycles; FP32 rounds per layout
    _run("ref", problem, nx, 1, 1, [f"out={ref}", "mesh=1,1,1", fp64])
    out = _run("check", problem, nx, 1, 8, [f"mesh={mesh}", f"ref={ref}", fp64])
    print(out[out.find("CHECK "):])
    assert "processes=8" in out and "devices=8" in out, out[-2000:]
    assert "CHECK PASS" in out, out[out.find("CHECK "):]
