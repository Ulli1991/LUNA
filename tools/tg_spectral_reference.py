"""Inviscid Taylor-Green reference from the pseudo-spectral incompressible Euler solver (euler3d.py).

Usage: EULER3D=/path/to/euler3d.py python tools/tg_spectral_reference.py N [T]; writes $LUNA_DATA/hydro/tg_spectral_N<N>.npz.
euler3d.py: github.com/Ulli1991/euler_3d (b104c2b made the N = 128 reference; tools/tg_reference_pack.py packs it)."""
import importlib.util
import os
import sys
import time

import numpy as np

DT_OUT = 0.1                                     # E, Z, strip width, E(k) every DT_OUT
T_FIELD = np.arange(0.5, 5.01, 0.5)              # velocity coefficients at these times
KC_MAX = 85                                      # stored modes |k_i| <= KC_MAX (all of N = 256)


def load_solver():
    """The euler3d module from $EULER3D (a copy; the user's repo is never touched)."""
    path = os.environ.get("EULER3D")
    if not path:
        raise RuntimeError("EULER3D is not set: the path of euler3d.py (github.com/Ulli1991/euler_3d)")
    spec = importlib.util.spec_from_file_location("euler3d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def truncate(u_hat, n, kc):
    """Normalised coefficients (3, 2kc+1, 2kc+1, kc+1), axes ordered 0..kc, -kc..-1."""
    idx = np.r_[0:kc + 1, n - kc:n]
    c = np.asarray(u_hat)[:, idx][:, :, idx][:, :, :, :kc + 1]
    return (c / float(n) ** 3).astype(np.complex64)


def main(n, T):
    """Run Taylor-Green at n^3 to T and save the diagnostics and coefficients."""
    e3 = load_solver()
    e3.init_distributed()
    g, mesh = e3.setup(n)
    u_hat = e3.taylor_green(g, mesh)
    advance = e3.make_advance(g, 10, 0.5, "ck", mesh)
    diag = e3.make_diagnostics(g, mesh)
    kc = min(n // 3, KC_MAX)
    rec = {k: [] for k in ("t", "E", "Z", "wmax", "delta", "spec")}
    fields, ftimes, trunc = [], [], []
    t, bkm, wprev, dtprev = 0.0, 0.0, 0.0, 0.0
    wall = time.time()
    for tout in np.round(np.arange(0.0, T + 1e-9, DT_OUT), 10):
        while t < tout - 1e-12:
            u_hat, t, _, bkm, wprev, dtprev = advance(u_hat, t, float(tout), bkm, wprev, dtprev)
            t = float(t)
        d = diag(u_hat)
        spec = np.asarray(d["spectrum"])
        delta, _, _ = e3.analyticity_strip(spec, g)
        for k, v in (("t", t), ("E", float(d["energy"])), ("Z", float(d["enstrophy"])),
                     ("wmax", float(d["wmax"])), ("delta", delta), ("spec", spec)):
            rec[k].append(v)
        if np.any(np.isclose(tout, T_FIELD)):
            c = truncate(u_hat, n, kc)
            e_kept = 0.5 * np.sum(np.abs(c.astype(np.complex128)) ** 2
                                  * np.where(np.arange(kc + 1) == 0, 1.0, 2.0)) * (2 * np.pi) ** 3
            fields.append(c)
            ftimes.append(t)
            trunc.append(1.0 - e_kept / rec["E"][-1])
        print(f"N={n} t={t:.2f} E={rec['E'][-1]:.12f} Z={rec['Z'][-1]:.5f} delta={delta:.4e} "
              f"2dx={2 * g.dx:.4e} ({time.time() - wall:.0f}s)", flush=True)
    out = os.path.join(os.environ["LUNA_DATA"], "hydro", f"tg_spectral_N{n}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, N=n, dx=g.dx, kc=kc, t=np.array(rec["t"]), E=np.array(rec["E"]),
             Z=np.array(rec["Z"]), wmax=np.array(rec["wmax"]), delta=np.array(rec["delta"]),
             spec=np.array(rec["spec"]), field_t=np.array(ftimes), field_c=np.array(fields),
             field_trunc=np.array(trunc))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 5.0)
