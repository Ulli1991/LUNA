"""Tabulate a converged fine-grid reference for a 1D problem without an exact solution.

    python tools/hydro_reference.py Titarev-Toro --lx 10 --t 5 --gamma 1.4 --limiter 1 \
        --n 65536 --check 32768 --coarse 8192 --out luna-reference-solutions/luna/titarev_toro/titarev_toro_mc.npz

Runs the registered IC at --n and --check cells (FP64, HLLC), averages both conservatively onto
--coarse cells, and stores rho, u, p of the --n run with the run's settings, the git revision and
the self-convergence L1(rho) between the two, so a test can see how good its reference is.
"""
import argparse
import datetime
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run(problem, n, lx, t, gamma, limiter, boundary=1):
    """Conserved U (3, n) at t from the registered IC on n cells."""
    from luna.ic_generation.ic import initial_conditions
    from tests._helpers import accuracy_cfg, evolve
    cfg = accuracy_cfg(test_problem=problem, SPATIAL_DIMENSION=1, Nx=n, Lx=lx, BOUNDARY=boundary,
                       gamma=gamma, LIMITER=limiter)
    return evolve(cfg, np.asarray(initial_conditions(cfg)), t)[0]


def coarsen(U, m):
    """Conservative average of conserved U (3, n) onto m cells."""
    return U.reshape(3, m, U.shape[1] // m).mean(axis=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("problem")
    ap.add_argument("--lx", type=float, required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--limiter", type=int, default=1)
    ap.add_argument("--boundary", type=int, default=1)
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--check", type=int, default=32768)
    ap.add_argument("--coarse", type=int, default=8192)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)

    fine, check = (coarsen(run(a.problem, n, a.lx, a.t, a.gamma, a.limiter, a.boundary), a.coarse)
                   for n in (a.n, a.check))
    rho, u = fine[0], fine[1] / fine[0]
    p = (a.gamma - 1.0) * (fine[2] - 0.5 * rho * u * u)
    self_l1 = float(np.abs(fine[0] - check[0]).sum() * a.lx / a.coarse)
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, rho=rho, u=u, p=p, problem=a.problem, lx=a.lx, t=a.t, gamma=a.gamma,
                        limiter=a.limiter, boundary=a.boundary, riemann_solver="hllc", precision="fp64", n_run=a.n,
                        n_check=a.check, n_coarse=a.coarse, self_l1_rho=self_l1, git_rev=rev,
                        date=datetime.date.today().isoformat())
    print(f"wrote {a.out}: {a.problem} N={a.n} on {a.coarse} cells, self-convergence L1(rho) "
          f"vs N={a.check} = {self_l1:.3e}")


if __name__ == "__main__":
    main()
