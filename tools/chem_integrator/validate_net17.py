"""The batched integrator on the 288 network-17 DVODE trajectories: accuracy table.

    python tools/chem_integrator/validate_net17.py [--ref raw_or_packed.npz] [--fixes17 fixed] [--pack out.npz]
"""
import argparse
import functools
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.ode import rodas3 as ig  # noqa: E402

from luna.chemistry import networks  # noqa: E402
from luna.chemistry import rates_net17  # noqa: E402
from luna.chemistry import tables as tb  # noqa: E402

DATA = "chemistry_net17_trajectories.npz"     # packed test reference, in $LUNA_DATA/chemistry
# the test reference: DVODE at RTOL 1e-8 (ATOL / 1e4), dust converged (run_variants.sbatch)
TIGHT = "n17b/variants/tight_dust/net17_trajectories.npz"     # under the chemistry work directory
QTY = ("xH2", "xHp", "xCO", "xHep", "xHepp", "xDp", "xHD", "e", "T")
KEEP_EVERY = 5                  # packed: every 5th output (the integration still steps every one)
FAST = 0.25                     # |d ln T| over one output interval above this: a thermal runaway


def packed(d):
    """Case names and params, output times, y0 (float64), fixed inputs and QTY at kept outputs (float32)."""
    names = d["case_names"]
    cols = list(d["columns"])
    traj = np.stack([d["traj_" + n] for n in names])
    keep = np.arange(0, traj.shape[1], KEEP_EVERY)
    lnt = np.log(traj[..., cols.index("T")])
    steady = np.abs(np.diff(lnt, axis=1, prepend=lnt[:, :1])) <= FAST
    out = dict(case_names=names, case_params=d["case_params"], t=traj[0, :, 0],
               y0=traj[:, 0, 1:9], keep=keep, steady=steady[:, keep],
               ref=traj[:, keep][..., [cols.index(c) for c in QTY]].astype(np.float32))
    for k in ("fixed_zeta", "fixed_abundD", "fixed_tdust0", "abundc_solar", "abundo_solar"):
        out[k] = d[k]
    return out


def load(path=DATA, cases=None):
    """(names, params, t, y0, ref (case, kept output, QTY), keep, steady, fixed) from a packed or raw npz."""
    d = np.load(path if os.path.exists(path) else tb.data_path(path))
    d = packed(d) if "ref" not in d else dict(d)
    sel = slice(None) if cases is None else np.asarray(cases)
    fixed = {k: float(d[k]) for k in ("fixed_zeta", "fixed_abundD", "fixed_tdust0",
                                       "abundc_solar", "abundo_solar")}
    return (d["case_names"][sel], d["case_params"][sel], d["t"], d["y0"][sel],
            d["ref"][sel].astype(np.float64), d["keep"], d["steady"][sel], fixed)


def params(cp, fixed):
    z = cp[:, 6]
    return rates_net17.make_params(
        cp[:, 0], cp[:, 2], cp[:, 3], cp[:, 4], cp[:, 5], zeta=fixed["fixed_zeta"], z=z,
        abundc=fixed["abundc_solar"] * z,
        abundo=fixed["abundo_solar"] * z, abund_d=fixed["fixed_abundD"],
        tdust=fixed["fixed_tdust0"])


def clamps(y, p):
    """chemcool.cc's input clamps before evolve_abundances ($LUNA_DATA/chemistry/harness/net17/net17_onezone.F)."""
    ion = y[:, 1] > 0.99
    y = y.at[:, 0].set(jnp.where(ion, 0.0, y[:, 0])).at[:, 2].set(jnp.where(ion, 0.0, y[:, 2]))
    y = y.at[:, 6].set(jnp.where(ion, 0.0, y[:, 6]))
    y = y.at[:, 2].set(jnp.minimum(y[:, 2], jnp.maximum(jnp.minimum(p.abundc, p.abundo), 0.0)))
    for i, j, pool in ((3, 4, rates_net17.ABHE + 0 * p.n_h), (5, 6, p.abund_d)):
        s = y[:, i] + y[:, j]
        f = jnp.where((s > pool) & (s > 0.0), pool / jnp.where(s > 0.0, s, 1.0), 1.0)
        y = y.at[:, i].multiply(f).at[:, j].multiply(f)
    return y


def network(fixes17="chemcool", eos=None):
    """NET17 with the RHS of a fixes17 mode and EOS."""
    if fixes17 == "chemcool" and eos is None:
        return networks.NET17
    return networks.NET17._replace(rhs=functools.partial(rates_net17.rhs, fixes17=fixes17, eos=eos))


def run(cp, t, y0, opt, fixed, net=networks.NET17, timed=False):
    """Every case over every output interval, in one compiled scan; photo state at each interval start."""
    n = y0.shape[0]
    p = ig.broadcast_params(params(cp, fixed), n)
    rhs = net.rhs

    @jax.jit
    def go(y0, p, dts):
        kw = {} if rhs is rates_net17.rhs else rhs.keywords

        def photo(y, p):
            tp, xh2, xhd = rates_net17.photo_state(y, p, **kw)
            return p._replace(photo_temp=tp, photo_x_h2=xh2, photo_x_hd=xhd)

        def step(c, dt):
            y, p = c
            p = photo(y, p)
            y = clamps(y, p)
            r = ig.integrate(y, p, dt, net=net, opt=opt)
            temp = rhs(r.y, r.params)[1]["T"]
            return (r.y, r.params), (r.y, temp, r.n_steps, r.n_rejected, r.failed)
        return jax.lax.scan(step, (y0, photo(y0, p)), dts)[1]

    args = (jnp.asarray(y0), p, jnp.asarray(np.diff(t)))
    out = jax.block_until_ready(go(*args))
    wall = None
    if timed:
        t0 = time.perf_counter()
        out = jax.block_until_ready(go(*args))
        wall = time.perf_counter() - t0
    y, temp, ns, nr, fl = (np.moveaxis(np.asarray(a), 0, 1) for a in out)
    got = np.concatenate([y, temp[..., None]], -1)
    got = np.concatenate([np.concatenate([y0, np.full((n, 1), np.nan)], 1)[:, None], got], 1)
    return dict(got=got, steps=ns, rej=nr, failed=fl, wall=wall)


def rel_errors(got, ref, cp, fixed):
    """Relative error per quantity; species only where ref > 1e-3 of their pool (else 0)."""
    z = cp[:, 6][:, None]
    pools = {"xH2": 0.5, "xHp": 1.0, "xCO": fixed["abundc_solar"] * z,
             "xHep": rates_net17.ABHE, "xHepp": rates_net17.ABHE,
             "xDp": fixed["fixed_abundD"], "xHD": fixed["fixed_abundD"]}
    errs = {}
    for i, q in enumerate(QTY):
        r, g = ref[..., i], got[..., i]
        mask = r > 1e-3 * np.broadcast_to(pools[q], r.shape) if q in pools else np.ones(r.shape, bool)
        errs[q] = np.where(mask, np.abs(g - r) / np.where(mask, np.abs(r), 1.0), 0.0)
    return errs


def table(names, cp, got, ref, keep, steady, fixed):
    """{qty: (max, median, worst case, output, max off runaways)} and the n >= 1e3 end-state maxima."""
    errs = rel_errors(got[:, keep][:, 1:], ref[:, 1:], cp, fixed)
    rows = {}
    for q, e in errs.items():
        k = np.unravel_index(np.argmax(e), e.shape)
        nz = e[e > 0]
        rows[q] = (float(e.max()), float(np.median(nz)) if nz.size else 0.0, str(names[k[0]]),
                   int(keep[1:][k[1]]), float(np.where(steady[:, 1:], e, 0.0).max()))
    eq = cp[:, 0] >= 1e3
    eqrows = {q: float(e[eq, -1].max()) for q, e in errs.items()}
    return rows, eqrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="packed or raw (run_grid.py) trajectories")
    ap.add_argument("--fixes17", default="chemcool")
    ap.add_argument("--eos", default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--atol-scale", type=float, default=1.0)
    ap.add_argument("--fortran-dust", action="store_true",
                    help="dust Newton stopped at the first 1 %% step, as the Fortran (not smooth)")
    ap.add_argument("--pack", default=None, help="write the packed reference here and stop")
    ap.add_argument("--save", default=None, help="save the trajectories (case, output, QTY) here")
    a = ap.parse_args()
    if a.pack:
        np.savez_compressed(a.pack, **packed(np.load(a.ref)))
        print(a.pack, os.path.getsize(a.pack))
        return
    names, cp, t, y0, ref, keep, steady, fixed = load(a.ref)
    net = network(a.fixes17, a.eos)
    if a.fortran_dust:
        net = net._replace(converge=())
    atol = None if a.atol_scale == 1.0 else tuple(a.atol_scale * x for x in net.atol)
    res = run(cp, t, y0, ig.Options(rtol=a.rtol, atol=atol), fixed, net=net, timed=True)
    rows, eqrows = table(names, cp, res["got"], ref, keep, steady, fixed)
    print(f"ref={a.ref} fixes17={a.fixes17} eos={a.eos} rtol={a.rtol} atol_scale={a.atol_scale} fortran_dust={a.fortran_dust} devices={jax.devices()}")
    print(f"wall {res['wall']:.2f} s; failed: {int(res['failed'].sum())}; steps per interval: "
          f"mean {res['steps'].mean():.1f} max {res['steps'].max()}; rejected: mean "
          f"{res['rej'].mean():.2f} max {res['rej'].max()}")
    print(f"{'qty':5s} {'max rel err':>12s} {'median':>10s}  worst case @ output | off runaways "
          f"| equilibrium")
    for q, (m, med, nm, k, st) in rows.items():
        print(f"{q:5s} {m:12.3e} {med:10.2e}  {nm} @ {k} | {st:.3e} | {eqrows[q]:.3e}")
    if a.save:
        np.savez_compressed(a.save, names=names, params=cp, t=t, got=res["got"])


if __name__ == "__main__":
    main()
