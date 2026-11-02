"""The batched integrator on the 168 network-5 DVODE trajectories: accuracy table.

    python tools/chem_integrator/validate_net5.py [--ref raw_or_packed.npz] [--rtol R] [--warm]
"""
import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.ode import rodas3 as ig  # noqa: E402

from luna.chemistry import networks  # noqa: E402
from luna.chemistry import rates_net5  # noqa: E402
from luna.chemistry import tables as tb  # noqa: E402
from tools.chem_integrator.pack_reference import packed  # noqa: E402

DATA = "chemistry_net5_trajectories.npz"     # packed test reference, in $LUNA_DATA/chemistry
QTY = ("xH2", "xHp", "xCO", "e", "T")
POOL = {"xH2": 0.5, "xHp": 1.0, "xCO": 1.41e-4}     # compared where ref > 1e-3 of the pool


def load(path=DATA, cases=None):
    """Cases (names, params, output times, y0, reference (case, output, QTY)); packed or raw npz."""
    d = np.load(path if os.path.exists(path) else tb.data_path(path))
    if "ref" not in d:
        d = packed(d)
    sel = slice(None) if cases is None else np.asarray(cases)
    return (d["case_names"][sel], d["case_params"][sel], d["t"], d["y0"][sel],
            d["ref"][sel].astype(np.float64))


def params(cp, tdust=10.0):
    return rates_net5.make_params(cp[:, 0], cp[:, 2], cp[:, 3], cp[:, 4], cp[:, 5], tdust=tdust)


def run(cp, t, y0, opt, warm=False, timed=False):
    """Every case over every output interval, in one compiled scan."""
    n = y0.shape[0]
    p = ig.broadcast_params(params(cp), n)

    @jax.jit
    def go(y0, p, dts):
        def step(c, dt):
            y, p, h = c
            r = ig.integrate(y, p, dt, h0=h if warm else None, opt=opt, net=networks.NET5)
            temp = rates_net5.rhs(r.y, r.params)[1]["T"]
            return (r.y, r.params, r.h), (r.y, temp, r.params.tdust, r.n_steps, r.n_rejected,
                                          r.failed)
        return jax.lax.scan(step, (y0, p, jnp.zeros(n)), dts)[1]

    args = (jnp.asarray(y0), p, jnp.asarray(np.diff(t)))
    out = jax.block_until_ready(go(*args))
    wall = None
    if timed:
        t0 = time.perf_counter()
        out = jax.block_until_ready(go(*args))
        wall = time.perf_counter() - t0
    y, temp, td, ns, nr, fl = (np.moveaxis(np.asarray(a), 0, 1) for a in out)
    got = np.concatenate([y, temp[..., None]], -1)
    return dict(got=got, Td=td, steps=ns, rej=nr, failed=fl, wall=wall)


def rel_errors(got, ref):
    """Relative error per quantity; species only where ref > 1e-3 of the pool (else 0)."""
    errs = {}
    for i, q in enumerate(QTY):
        r, g = ref[..., i], got[..., i]
        mask = r > 1e-3 * POOL[q] if q in POOL else np.ones(r.shape, bool)
        errs[q] = np.where(mask, np.abs(g - r) / np.where(mask, np.abs(r), 1.0), 0.0)
    return errs


def table(names, cp, got, ref):
    """{qty: (max over all outputs, median, worst case, output)} and the equilibrium maxima."""
    errs = rel_errors(got, ref[:, 1:])
    rows = {}
    for q, e in errs.items():
        k = np.unravel_index(np.argmax(e), e.shape)
        nz = e[e > 0]
        rows[q] = (float(e.max()), float(np.median(nz)) if nz.size else 0.0, str(names[k[0]]),
                   int(k[1]) + 1)
    eq = cp[:, 0] >= 1e3
    eqrows = {q: float(e[eq, -1].max()) for q, e in errs.items()}
    return rows, eqrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=DATA, help="packed or raw (run_grid.py) trajectories")
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--atol-scale", type=float, default=1.0)
    ap.add_argument("--warm", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    names, cp, t, y0, ref = load(a.ref)
    atol = None if a.atol_scale == 1.0 else tuple(a.atol_scale * x for x in networks.NET5.atol)
    opt = ig.Options(rtol=a.rtol, atol=atol)
    res = run(cp, t, y0, opt, warm=a.warm, timed=True)
    rows, eqrows = table(names, cp, res["got"], ref)
    print(f"ref={a.ref} rtol={a.rtol} atol_scale={a.atol_scale} warm={a.warm} "
          f"devices={jax.devices()}")
    print(f"wall {res['wall']:.2f} s; failed: {int(res['failed'].sum())}; steps per interval: "
          f"mean {res['steps'].mean():.1f} max {res['steps'].max()}; rejected: mean "
          f"{res['rej'].mean():.2f} max {res['rej'].max()}; accepted {int(res['steps'].sum())}")
    print(f"{'qty':4s} {'max rel err':>12s} {'median':>10s}  worst case @ output | equilibrium")
    for q, (m, med, nm, k) in rows.items():
        print(f"{q:4s} {m:12.3e} {med:10.2e}  {nm} @ {k} | {eqrows[q]:.3e}")
    if a.out:
        np.savez_compressed(a.out, names=names, ref=ref, **{k: v for k, v in res.items()
                                                            if v is not None})


if __name__ == "__main__":
    main()
