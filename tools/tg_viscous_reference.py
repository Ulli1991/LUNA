"""Viscous Taylor-Green reference for A15 (test_hydro_hard_taylor_green_reynolds): euler3d.py with nu = 1/Re.

    EULER3D=/path/to/euler3d.py python tools/tg_viscous_reference.py run RE N [T]     # one run -> $LUNA_DATA/hydro/tg_viscous/
    python tools/tg_viscous_reference.py pack OUT RE:N [RE:N ...] [--source TEXT]       # the test's npz

euler3d.py: github.com/Ulli1991/euler_3d, branch viscosity 3940684 (-nu k^2 u in the rhs). u0 = 1, k = 1 (box 2 pi),
so Re = 1/nu. Per-volume kinetic energy K and enstrophy Z = 0.5 <|omega|^2> every DT_OUT; eps = 2 Z / Re."""
import argparse
import datetime
import hashlib
import importlib.util
import os
import subprocess
import time

import numpy as np

DT_OUT = 0.02


def load_solver():
    """The euler3d module from $EULER3D (required; a copy, the user's repository is never touched)."""
    path = os.environ.get("EULER3D")
    if not path:
        raise RuntimeError("EULER3D is not set: euler3d.py of euler_3d branch viscosity (3940684)")
    spec = importlib.util.spec_from_file_location("euler3d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def run(re, n, T):
    """Taylor-Green at Re = 1/nu on n^3 to T; saves t, K, Z, max|omega|, E(k) at T to $LUNA_DATA/hydro/tg_viscous."""
    e3, path = load_solver()
    import jax
    e3.init_distributed()
    P = len(jax.devices())
    g, mesh = e3.setup(n, P=P)
    u_hat = e3.taylor_green(g, mesh)
    advance = e3.make_advance(g, 10, 0.5, "ck", mesh, nu=1.0 / re)
    diag = e3.make_diagnostics(g, mesh)
    vol = (2 * np.pi) ** 3
    rec = {k: [] for k in ("t", "K", "Z", "wmax")}
    t, bkm, wprev, dtprev = 0.0, 0.0, 0.0, 0.0
    wall = time.time()
    for tout in np.round(np.arange(0.0, T + 1e-9, DT_OUT), 10):
        while t < tout - 1e-12:
            u_hat, t, _, bkm, wprev, dtprev = advance(u_hat, t, float(tout), bkm, wprev, dtprev)
            t = float(t)
        d = diag(u_hat)
        for k, v in (("t", t), ("K", float(d["energy"]) / vol), ("Z", float(d["enstrophy"]) / vol),
                     ("wmax", float(d["wmax"]))):
            rec[k].append(v)
        if abs(tout - round(tout)) < 1e-9:
            print(f"Re={re} N={n} t={t:.2f} K={rec['K'][-1]:.10f} eps={2 * rec['Z'][-1] / re:.9f} "
                  f"({time.time() - wall:.0f}s)", flush=True)
    spec = np.asarray(diag(u_hat)["spectrum"])
    out = os.path.join(os.environ["LUNA_DATA"], "hydro", "tg_viscous", f"Re{re}_N{n}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    np.savez(out, Re=re, N=n, devices=P, euler3d_md5=md5, spec_T=spec, **{k: np.array(v) for k, v in rec.items()})
    print(f"wrote {out}", flush=True)


def pack(out, runs, source):
    """The test's npz: Re<r>_t, Re<r>_Z (per volume) for each run 'RE:N', plus Re<r>_N and provenance."""
    data, lines = {}, []
    for rn in runs:
        re, n = (int(x) for x in rn.split(":"))
        src = os.path.join(os.environ["LUNA_DATA"], "hydro", "tg_viscous", f"Re{re}_N{n}.npz")
        with np.load(src) as z:
            data[f"Re{re}_t"], data[f"Re{re}_Z"], data[f"Re{re}_K"] = z["t"], z["Z"], z["K"]
            data[f"Re{re}_N"] = n
            lines.append(f"Re {re} at {n}^3 (euler3d md5 {z['euler3d_md5']})")
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    data["provenance"] = np.array(
        f"packed {datetime.datetime.now().isoformat(timespec='seconds')} by tools/tg_viscous_reference.py (luna {git}) "
        f"from $LUNA_DATA/hydro/tg_viscous: {'; '.join(lines)}. euler_3d branch viscosity 3940684. {source}")
    np.savez_compressed(out, **data)
    print(data["provenance"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("re", type=int)
    r.add_argument("n", type=int)
    r.add_argument("T", type=float, nargs="?", default=20.0)
    p = sub.add_parser("pack")
    p.add_argument("out")
    p.add_argument("runs", nargs="+")
    p.add_argument("--source", default="")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.re, a.n, a.T)
    else:
        pack(a.out, a.runs, a.source)


if __name__ == "__main__":
    main()
