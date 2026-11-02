"""Exact 1D pieces of the Kurganov-Tadmor 2D Riemann configurations 3 and 6, and the stored LUNA reference.

    python tools/kt_riemann.py run CONFIG N OUTDIR [mesh=2,2] [limiter=1] [solver=hllc] [precision=fp64]
    python tools/kt_riemann.py pack CONFIG REF_NPY CHECK_NPY OUT.npz [seeded=SEEDED_NPY]   # host numpy

No exact 2D solution exists. Exact are the four interface problems (1D Riemann problems of the printed data,
luna.hydro.exact_riemann), hence the strips along the walls that the 2D interaction has not reached, and the
fluxes through the walls while the walls see only those 1D solutions (zero-gradient BOUNDARY=1). `run` evolves
the registered IC (FP64) and saves the end state; `pack` reduces the reference and a coarser check run to the
density block averages the test reads (luna-reference-solutions/luna/kt_riemann/kt_riemann_c<CONFIG>_mc.npz), with provenance.
"""
import json
import os
import subprocess
import sys

import numpy as np

from luna.hydro import exact_riemann as er
from luna.ic_generation.kt_riemann_2d import GAMMA, STATES, T_END

# interface: (left or lower quadrant, right or upper quadrant, normal axis 0 = x, 1 = y), and the wall whose
# whole length carries its 1D solution: (axis of the wall's normal, side 0 = low, 1 = high)
IFACES = {"21": (2, 1, 0), "34": (3, 4, 0), "32": (3, 2, 1), "41": (4, 1, 1)}
WALL = {"21": (1, 1), "34": (1, 0), "32": (0, 0), "41": (0, 1)}
PACK_M = 256                                    # density stored as block averages on PACK_M^2
SELF_MS = (64, 128, 256)                        # grids of the stored self-convergence L1
SEEDED_M = 128                                  # grid of the stored seeded run
SEED_RNG = 20261014                             # numpy default_rng seed of the IC perturbation (as the A11 critic)


def problem(config, name, states=None, gamma=GAMMA):
    """(W_L, W_R, (v_t L, v_t R), axis): the 1D Riemann problem across interface `name`, W = (rho, u_n, p)."""
    q = (states or STATES[config])
    a, b, ax = IFACES[name]
    (ra, ua, va, pa), (rb, ub, vb, pb) = q[a], q[b]
    if ax == 0:
        return (ra, ua, pa), (rb, ub, pb), (va, vb), ax
    return (ra, va, pa), (rb, vb, pb), (ua, ub), ax


def waves(config, name, gamma=GAMMA):
    """exact_riemann.waves of interface `name`: p*, u*, star densities, (head, tail) speeds."""
    WL, WR, _, _ = problem(config, name)
    return er.waves(WL, WR, gamma)


def strip_exact(config, name, n, t=T_END, gamma=GAMMA, sub=256):
    """(rho, u_n, p) exact cell averages on n cells of [0, 1] along interface `name`'s normal, split at 1/2."""
    WL, WR, _, _ = problem(config, name)
    return er.cell_average(WL, WR, gamma, np.linspace(0.0, 1.0, n + 1), 0.5, t, sub=sub)


def strip_rows(n, name, width):
    """Slice along the transverse axis of the cells whose centres lie within `width` of the wall of `name`."""
    c = (np.arange(n) + 0.5) / n
    side = WALL[name][1]
    idx = np.nonzero(c > 1.0 - width if side else c < width)[0]
    return slice(int(idx[0]), int(idx[-1]) + 1)


def strip_profile(rho, name, width):
    """Mean of the density field rho (N, N) over the strip of `name` (cells within `width` of its wall)."""
    ax = IFACES[name][2]
    rows = strip_rows(rho.shape[1 - ax], name, width)
    return rho[:, rows].mean(1) if ax == 0 else rho[rows, :].mean(0)


def row_deviation(rho, config, name, t=T_END):
    """Per transverse row (in order away from the wall), L1 over the normal axis of rho - the exact 1D solution:
    where the strip still holds the interface's 1D solution."""
    ax = IFACES[name][2]
    n = rho.shape[ax]
    ex = strip_exact(config, name, n, t)[0]
    rows = rho.T if ax == 1 else rho                      # rows[:, j]: profile along the normal at row j
    dev = np.abs(rows - ex[:, None]).mean(0)
    return dev[::-1] if WALL[name][1] else dev


def crossing(profile, a, b):
    """Position in [0, 1] where `profile` (cell centres) first crosses (a + b) / 2 going from a's side to b's."""
    n = len(profile)
    x = (np.arange(n) + 0.5) / n
    f = (profile - 0.5 * (a + b)) * np.sign(b - a)
    k = np.nonzero((f[:-1] < 0) & (f[1:] >= 0))[0]
    if len(k) == 0:
        return np.nan
    k = k[0]
    return float(x[k] + (x[k + 1] - x[k]) * (-f[k]) / (f[k + 1] - f[k]))


def flux(rho, u, v, p, axis, gamma=GAMMA):
    """Euler flux (mass, x-momentum, y-momentum, energy) through a face of normal `axis`."""
    E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    un = u if axis == 0 else v
    return np.array([rho * un, rho * u * un + (p if axis == 0 else 0.0 * p),
                     rho * v * un + (p if axis == 1 else 0.0 * p), un * (E + p)])


def _wall_flux(config, name, xi, gamma=GAMMA):
    """Flux through the wall of `name` (normal = the wall's axis) of the 1D solution at speeds xi."""
    WL, WR, (tl, tr), ax = problem(config, name)
    rho, un, p = er.sample(WL, WR, gamma, xi)
    vt = np.where(xi < er.star(WL, WR, gamma)[1], tl, tr)
    u, v = (un, vt) if ax == 0 else (vt, un)
    return flux(rho, u, v, p, WALL[name][0], gamma)


def wall_budget(config, t, gamma=GAMMA, order=16):
    """Exact change of the totals (mass, x-mom, y-mom, energy) on (0, 1)^2 from 0 to t by the four wall fluxes.

    Each wall carries its interface's self-similar 1D solution, so its flux integral is
    t (F_L + F_R) / 2 + t^2 / 2 I, I = int (F(xi) - F_far(xi)) dxi, by Gauss-Legendre between the wave speeds.
    Refused once a wave reaches a corner; valid while the 2D interaction stays off the walls (config 3 to 0.3
    and beyond, config 6 to 0.2586, lit_a11/lit.md)."""
    xg, wg = np.polynomial.legendre.leggauss(order)
    total = np.zeros(4)
    for name in IFACES:
        w = waves(config, name, gamma)
        speeds = [*w["left"], w["u"], *w["right"]]
        if max(abs(s) for s in speeds) * t >= 0.5:
            raise ValueError(f"interface {name}: a wave reaches the corner before t={t}")
        far_l, far_r = (_wall_flux(config, name, np.array([s]), gamma)[:, 0] for s in (-1e3, 1e3))
        br = np.unique(np.r_[min(speeds) - 1.0, speeds, 0.0, max(speeds) + 1.0])
        integral = np.zeros(4)
        for a, b in zip(br[:-1], br[1:]):
            xi = 0.5 * (a + b) + 0.5 * (b - a) * xg
            f = _wall_flux(config, name, xi, gamma) - np.where(xi < 0.0, far_l[:, None], far_r[:, None])
            integral += 0.5 * (b - a) * (f * wg).sum(1)
        ax, side = WALL[name]
        sign = 1.0 if side == 0 else -1.0                   # +normal flux enters at the low wall
        total += sign * (t * 0.5 * (far_l + far_r) + 0.5 * t * t * integral)
    return total


def totals(states, gamma=GAMMA):
    """Totals (mass, x-mom, y-mom, energy) of the four quadrant states on (0, 1)^2."""
    out = np.zeros(4)
    for r, u, v, p in states.values():
        out += 0.25 * np.array([r, r * u, r * v, p / (gamma - 1.0) + 0.5 * r * (u * u + v * v)])
    return out


def block_mean(a, m):
    """Average of an (N, N) field onto m x m blocks (conservative for densities)."""
    n = a.shape[0]
    return a.reshape(m, n // m, m, n // m).mean((1, 3))



def seeded(U, eps):
    """U times (1 + eps r) per cell, r uniform in (-1, 1) from default_rng(SEED_RNG): rho perturbed, velocities kept."""
    if not eps:
        return U
    r = np.random.default_rng(SEED_RNG).uniform(-1.0, 1.0, U.shape[1:])
    return U * (1.0 + eps * r)[None]


# ---------------------------------------------------------------- the stored reference
def _git():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    d = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                       text=True).stdout.strip()
    return h, d


def run(config, n, outdir, mesh=(1, 1), limiter=1, solver="hllc", order=1, precision="fp64", seed=0.0):
    """Evolve KT config `config` on n^2 cells to T_END (one process, `mesh` devices), the IC times (1 + seed r) per
    cell if seed; saves OUTDIR/kt<config>_<solver>_lim<limiter>[_o0][_fp32][_seed<s>]_N<n>.npy and its record (.json)."""
    import time

    import jax
    jax.config.update("jax_enable_x64", precision != "fp32")
    from luna.ic_generation.ic import initial_conditions
    from luna.ic_generation.kt_riemann_2d import Params
    from tests import _helpers
    from tests._helpers import accuracy_cfg, evolve
    _helpers.PRECISION = precision
    cfg = accuracy_cfg(test_problem="KT-Riemann-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0,
                       BOUNDARY=1, gamma=GAMMA, LIMITER=limiter, riemann_solver=solver, ORDER=order,
                       problem=Params(config=config))
    U0 = seeded(np.asarray(initial_conditions(cfg)), seed)
    t0 = time.time()
    U, steps = evolve(cfg, U0, T_END, mesh_shape=mesh)
    wall = time.time() - t0
    tag = ("_o0" if order == 0 else "") + ("" if precision == "fp64" else f"_{precision}") + (f"_seed{seed:g}" if seed else "")
    stem = os.path.join(outdir, f"kt{config}_{solver}_lim{limiter}{tag}_N{n}")
    os.makedirs(outdir, exist_ok=True)
    np.save(stem + ".npy", np.asarray(U))
    h, d = _git()
    rec = dict(config=config, n=n, limiter=limiter, solver=solver, order=order, t=T_END, gamma=GAMMA, boundary=1,
               cfl=cfg.CFL, precision=precision, seed=seed, seed_rng=SEED_RNG, mesh=list(mesh), steps=steps, wall_s=wall, git=h, git_dirty=d,
               run_line="python tools/kt_riemann.py " + " ".join(sys.argv[1:]),
               slurm_job_id=os.environ.get("SLURM_JOB_ID"), node=os.uname().nodename,
               devices=str(jax.devices()[:int(np.prod(mesh))]))
    with open(stem + ".json", "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec), flush=True)
    return U


def pack(config, ref_npy, check_npy, out, seeded_npy=None):
    """The test's reference file: rho of the reference run on PACK_M^2 (float32), the L1 between the reference
    and the check run on SELF_MS grids, both run records; with seeded_npy also that (seeded) run's rho on
    SEEDED_M^2, so the test can see the reference is still on the seed-stable branch."""
    ref, chk = (np.load(p, mmap_mode="r")[0] for p in (ref_npy, check_npy))
    recs = [json.load(open(p[:-4] + ".json")) for p in (ref_npy, check_npy)]
    for r in recs:
        if (r["config"], r["precision"], r.get("order", 1), r["limiter"]) != (config, "fp64", 1, recs[0]["limiter"]):
            raise ValueError(f"not a config-{config} FP64 MUSCL-Hancock pair: {r}")
    extra = {}
    if seeded_npy:
        rec = json.load(open(seeded_npy[:-4] + ".json"))
        if (rec["config"], rec["precision"], rec["limiter"]) != (config, "fp64", recs[0]["limiter"]) or not rec["seed"]:
            raise ValueError(f"not a seeded config-{config} FP64 run of the same limiter: {rec}")
        extra = dict(seeded_rho=block_mean(np.asarray(np.load(seeded_npy, mmap_mode="r")[0]), SEEDED_M).astype(np.float32),
                     seeded_n=rec["n"], seeded_eps=rec["seed"])
        recs.append(rec)
    self_l1 = [float(np.abs(block_mean(ref, m) - block_mean(chk, m)).mean()) for m in SELF_MS]
    np.savez_compressed(out, rho=block_mean(np.asarray(ref), PACK_M).astype(np.float32), m=PACK_M,
                        config=config, t=T_END, gamma=GAMMA, boundary=1, limiter=recs[0]["limiter"],
                        riemann_solver=recs[0]["solver"], precision="fp64", n_ref=recs[0]["n"],
                        n_check=recs[1]["n"], self_ms=np.array(SELF_MS), self_l1_rho=np.array(self_l1),
                        provenance=json.dumps(dict(reference=recs[0], check=recs[1], seeded=recs[2:])), **extra)
    print(f"wrote {out}: config {config}, N={recs[0]['n']} on {PACK_M}^2, self L1(rho) vs N={recs[1]['n']} "
          + ", ".join(f"{m}: {v:.3e}" for m, v in zip(SELF_MS, self_l1)))


def main(argv):
    cmd = argv[0]
    kw = dict(a.split("=", 1) for a in argv if "=" in a)
    pos = [a for a in argv[1:] if "=" not in a]
    if cmd == "run":
        run(int(pos[0]), int(pos[1]), pos[2], mesh=tuple(int(m) for m in kw.get("mesh", "1,1").split(",")),
            limiter=int(kw.get("limiter", 1)), solver=kw.get("solver", "hllc"), order=int(kw.get("order", 1)),
            precision=kw.get("precision", "fp64"), seed=float(kw.get("seed", 0.0)))
    elif cmd == "pack":
        pack(int(pos[0]), pos[1], pos[2], pos[3], seeded_npy=kw.get("seeded"))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
