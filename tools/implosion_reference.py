"""Fine-grid LUNA references for the A10 implosion test (tests/test_hydro_hard_implosion.py).

    python tools/implosion_reference.py --out luna-reference-solutions/luna/lw_implosion/lw_implosion_mc.npz

HLLC + MC, FP64, one device: rho at T_EARLY on --early cells (and --early-check cells), block-averaged onto
--early-store cells; rho at T_LATE on --late cells (and --late-check), averaged onto --late-store cells. Each
field is stored as the lower triangle (i >= j) in float32: the runs are exactly x <-> y symmetric, which this
checks before storing. Also stored: self-convergence L1(rho) of each pair on the stored grid, steps, the git
revision, device and SLURM job id.
"""
import argparse
import datetime
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def block(a, m):
    """Block average of an (n, n) field onto (m, m)."""
    n = a.shape[0]
    return a.reshape(m, n // m, m, n // m).mean(axis=(1, 3))


def tri(a):
    """Lower triangle (i >= j) of a square field, flattened row by row."""
    return a[np.tril_indices(a.shape[0])]


def untri(v):
    """The symmetric square field whose lower triangle is v."""
    m = int(round((np.sqrt(8 * v.size + 1) - 1) / 2))
    out = np.zeros((m, m), v.dtype)
    out[np.tril_indices(m)] = v
    return out + np.tril(out, -1).T


def rho_at(n, t):
    """(rho at t on n x n cells, steps, wall seconds), HLLC + MC; refuses an asymmetric or non-conserving run."""
    import tests.test_hydro_hard_implosion as imp
    from tests._helpers import evolve
    w = time.time()
    U0 = imp.initial(n)
    U, steps = evolve(imp.cfg_for(n, "hllc-mc"), U0, t)
    a = imp.asym(U)
    if a != 0.0:
        raise RuntimeError(f"N={n} t={t}: x <-> y asymmetry {a:.3e}, not stored")
    return U[0], steps, time.time() - w


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--early", type=int, default=6400)
    ap.add_argument("--early-check", type=int, default=3200)
    ap.add_argument("--early-store", type=int, default=800)
    ap.add_argument("--late", type=int, default=2048)
    ap.add_argument("--late-check", type=int, default=1024)
    ap.add_argument("--late-store", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default="", help="directory for the full rho fields (.npy), e.g. on scratch")
    a = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))       # the tree as the run starts: the code it imports
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=here).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           cwd=here).stdout.strip()
    import jax
    jax.config.update("jax_enable_x64", True)
    import tests.test_hydro_hard_implosion as imp

    out = {}
    for tag, t, n, nc, m in (("early", imp.T_EARLY, a.early, a.early_check, a.early_store),
                             ("late", imp.T_LATE, a.late, a.late_check, a.late_store)):
        fine, steps, wall = rho_at(n, t)
        chk, steps_c, wall_c = rho_at(nc, t)
        if a.raw:
            os.makedirs(a.raw, exist_ok=True)
            np.save(os.path.join(a.raw, f"rho_{tag}_N{n}.npy"), fine)
            np.save(os.path.join(a.raw, f"rho_{tag}_N{nc}.npy"), chk)
        f, c = block(fine, m), block(chk, m)
        out |= {f"{tag}_rho": tri(f).astype(np.float32), f"{tag}_t": t, f"{tag}_n": n, f"{tag}_n_check": nc,
                f"{tag}_n_store": m, f"{tag}_steps": steps, f"{tag}_wall_s": wall, f"{tag}_steps_check": steps_c,
                f"{tag}_self_l1_rho": float(np.abs(f - c).mean()),
                f"{tag}_self_l2_rho": float(np.sqrt(((f - c) ** 2).mean()))}
        print(f"{tag}: N={n} t={t} steps={steps} wall={wall:.0f}s; check N={nc} steps={steps_c} "
              f"wall={wall_c:.0f}s; self L1(rho) on {m}^2 = {out[f'{tag}_self_l1_rho']:.4e}", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, problem="LW-Implosion", gamma=imp.G, box=imp.L, boundary=2, riemann_solver="hllc",
                        limiter=1, precision="fp64", cfl=float(imp.cfg_for(8, "hllc-mc").CFL), git_rev=rev,
                        git_dirty=dirty, device=str(jax.devices()[0].device_kind),
                        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""), date=datetime.date.today().isoformat(),
                        **out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
