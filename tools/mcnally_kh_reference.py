"""The A5 field reference: LUNA MC + HLLC FP64 McNally KH at --n and --check cells per side, on a device mesh.

    python tools/mcnally_kh_reference.py --n 4096 --check 2048 --mesh 1x1 --out luna-reference-solutions/luna/mcnally_kh/mcnally2012_luna_mc_ref.npz \
        [--raw DIR]

Stores rho and rho V_y at t = 1.5 of the --n run, block-averaged to 256^2 (float32: the test reads differences
of ~1e-3), M(t) and max(rho V_y^2 / 2)(t) of both runs at t = 0.02 k, the self-convergence L1 of V_y between them
on 256^2 (how good the reference is), and the run's provenance. --raw keeps both runs' end states at 1024^2.
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run(n, mesh):
    """(M, K, rho, my at t = 1.5 averaged to 1024^2 (or n), steps, wall seconds) of one MC run."""
    import tests.test_hydro_hard_mcnally_kh as kh
    from tests._helpers import evolve
    cfg = kh.cfg_for(n, "mc")
    U0 = kh.initial(cfg)
    times = kh.TIMES[1:]
    count = iter(range(len(times)))
    m = min(n, 1024)

    def observe(U):
        keep = next(count) == len(times) - 1
        return (kh.mode_amplitude(U), kh.max_ke(U),
                (kh.coarsen(np.asarray(U[0], np.float64), m), kh.coarsen(np.asarray(U[2], np.float64), m))
                if keep else None)

    t0 = time.time()
    out, steps = evolve(cfg, U0, times, mesh_shape=mesh, observe=observe)
    wall = time.time() - t0
    M = np.array([kh.mode_amplitude(U0)] + [o[0] for o in out])
    K = np.array([kh.max_ke(U0)] + [o[1] for o in out])
    print(f"n={n} mesh={mesh} steps={steps} wall={wall:.0f}s M(1.5)={M[-1]:.8e}", flush=True)
    return M, K, out[-1][2][0], out[-1][2][1], steps, wall


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--check", type=int, default=2048)
    ap.add_argument("--mesh", default="2x2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="")
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    from tests import _helpers
    import tests.test_hydro_hard_mcnally_kh as kh
    _helpers.PRECISION = "fp64"
    mesh = tuple(int(v) for v in a.mesh.split("x"))
    nc = kh.NC
    runs = {n: run(n, mesh) for n in (a.n, a.check)}
    (M, K, rho, my, steps, wall), (Mc, Kc, rhoc, myc, stepsc, wallc) = runs[a.n], runs[a.check]
    if a.raw:
        os.makedirs(a.raw, exist_ok=True)
        for n, (_, _, r, y, *_rest) in runs.items():
            np.savez(os.path.join(a.raw, f"a5_mc_{n}_t1.5_1024.npz"), rho=r, my=y)
    r256, y256 = kh.coarsen(rho, nc), kh.coarsen(my, nc)
    rc256, yc256 = kh.coarsen(rhoc, nc), kh.coarsen(myc, nc)
    self_l1_vy = float(np.abs(y256 / r256 - yc256 / rc256).mean())
    self_l1_rho = float(np.abs(r256 - rc256).mean())
    rev = (subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
           or os.environ.get("LUNA_GIT_REV", ""))          # a `git archive` copy has no .git: the job names it
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out, rho=r256.astype(np.float32), my=y256.astype(np.float32), t=1.5, times=kh.TIMES, M=M, K=K, M_check=Mc,
        K_check=Kc, problem="KH-McNally-2D", n=a.n, n_check=a.check, n_coarse=nc, gamma=kh.G, limiter=1,
        riemann_solver="hllc", precision="fp64", cfl=float(kh.cfg_for(8).CFL), self_l1_vy=self_l1_vy,
        self_l1_rho=self_l1_rho, steps=steps, steps_check=stepsc, wall_s=wall, wall_check_s=wallc, mesh=a.mesh,
        devices=str(jax.devices()), jax=jax.__version__, git_rev=rev, git_dirty=dirty,
        job=os.environ.get("SLURM_JOB_ID", ""), date=datetime.date.today().isoformat(),
        command=" ".join(sys.argv))
    print(f"wrote {a.out}: self-convergence L1 vy {self_l1_vy:.3e}, rho {self_l1_rho:.3e}; "
          f"max |M_n / M_check - 1| {np.abs(M / Mc - 1).max():.3e}", flush=True)


if __name__ == "__main__":
    main()
