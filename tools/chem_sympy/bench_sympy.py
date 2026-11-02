"""Existing network-5 RHS (+ jacfwd) against the SymPy-generated f (+ J) on every cell of a saved 512^3 state.
    python tools/chem_sympy/bench_sympy.py --cells DIR --what check,time,prof --out DIR [--only a,b]
"""
import argparse
import contextlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.chemistry import cooling, rates_net5  # noqa: E402
from luna.chemistry import generated as gen  # noqa: E402

CH = 2 ** 20                              # cells per chunk, as the run
REPS = 5
ATOL = np.array([1e-7, 1e-12, 1e-14, 0.0])


def timed(f, *args, reps=REPS):
    """Warm call, then reps timed calls: (out, median s, min s, all s)."""
    out = jax.block_until_ready(f(*args))
    w = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        w.append(time.perf_counter() - t0)
    return out, float(np.median(w)), float(np.min(w)), w


REGIONS = []


def _clocks():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC), time.clock_gettime_ns(time.CLOCK_BOOTTIME)


@contextlib.contextmanager
def region(label):
    """Host clock window of the enclosed synchronised GPU work, for cutting a kernel trace."""
    time.sleep(0.5)
    t0 = _clocks()
    try:
        yield
    finally:
        t1 = _clocks()
        REGIONS.append(dict(label=label, mono=(t0[0], t1[0]), boot=(t0[1], t1[1])))
        time.sleep(0.5)


def chunked(f, *args):
    """lax.map of f over 2^20-cell chunks of flat per-cell arrays."""
    n = jax.tree_util.tree_leaves(args)[0].shape[0]
    out = jax.lax.map(lambda a: f(*a), jax.tree_util.tree_map(
        lambda a: a.reshape((n // CH, CH) + a.shape[1:]), args))
    return jax.tree_util.tree_map(lambda a: a.reshape((n,) + a.shape[2:]), out)


def dust_frozen(n_h, temp, x_hi, chi, g0_dust, dgr, tdust_in, h2_form_ex=0.84, h2_form_kin=0.12,
                rtol=1e-2):
    """cooling.dust_temperature with no Newton: T_dust is the value passed in."""
    ch7 = cooling.interp_rows({7: cooling.chem_rows()[7]}, temp)[7] * dgr
    active = (dgr > 1e-8) & (temp < 1e5)
    temp2 = temp * 1e-2

    def rates(td):
        fa = 1.0 / (1.0 + 1e4 * jnp.exp(-6e2 / td))
        stick = 1.0 / (1.0 + 0.4 * (temp2 + td * 1e-2) ** 0.5 + 0.2 * temp2 + 0.08 * temp2**2.0)
        return fa, stick, ch7 * fa * stick
    td = jnp.where(active, tdust_in, 10.0)
    return cooling._finish(td, td * 0.0 < 1.0, active, temp, x_hi, n_h, rates)


@contextlib.contextmanager
def dust_mode(dust):
    """While tracing: the existing RHS with T_dust frozen at p.tdust, or as it is."""
    orig = cooling.dust_temperature
    if dust == "frozen":
        cooling.dust_temperature = dust_frozen
    try:
        yield
    finally:
        cooling.dust_temperature = orig


def params(meta, n_h, cols, td, rtol):
    """Network-5 Params per cell as the coupled stage builds them (one pixel holding the local column)."""
    z = 0.0 * n_h
    base = rates_net5.Params(
        n_h=n_h, n_h_col=cols[0][:, None], n_h2_col=cols[1][:, None], n_co_col=cols[2][:, None],
        g0=meta["g0"], g0_dust=meta["g0"], zeta=meta["zeta"], dgr=1.0, abundc=meta["abundc"],
        abundo=meta["abundo"], tdust=td, dust_rtol=rtol)
    return rates_net5.Params(*[v if jnp.ndim(v) else z + v for v in base])


def jsum(J):
    """J summed per cell (keeps it live, writes 8 B/cell instead of 128)."""
    return J.sum(-1).sum(-1)


def variants(meta):
    """name -> (dust mode for tracing, per-chunk function (y, n_h, c0, c1, c2, td) -> outputs)."""
    rt = {"implicit": 1e-12, "frozen": 1e-2}

    def ex_rhs(dust):
        def f(y, nh, c0, c1, c2, td):
            ydot, aux = jax.vmap(rates_net5.rhs)(y, params(meta, nh, (c0, c1, c2), td, rt[dust]))
            return ydot, aux["tdust"]
        return f

    def ex_jac(dust):
        def f(y, nh, c0, c1, c2, td):
            p = params(meta, nh, (c0, c1, c2), td, rt[dust])

            def one(yy, pp):
                def fa(z):
                    fz, aux = rates_net5.rhs(z, pp)
                    return fz, (fz, aux)
                J, (fz, aux) = jax.jacfwd(fa, has_aux=True)(yy)
                return fz, aux["tdust"], J
            fz, tdo, J = jax.vmap(one)(y, p)
            return fz, tdo, jsum(J)
        return f

    def g_rhs(dust):
        def f(y, nh, c0, c1, c2, td):
            ydot, aux = gen.rhs(y, params(meta, nh, (c0, c1, c2), td, rt[dust]), dust)
            return ydot, aux["tdust"]
        return f

    def g_jac(dust):
        def f(y, nh, c0, c1, c2, td):
            ydot, J, aux = gen.rhs_and_jac(y, params(meta, nh, (c0, c1, c2), td, rt[dust]), dust)
            return ydot, aux["tdust"], jsum(J)
        return f

    out = {}
    for dust in ("implicit", "frozen"):
        out[f"rhs_{dust}"] = (dust, ex_rhs(dust))
        out[f"rhs_jacfwd_{dust}"] = (dust, ex_jac(dust))
        out[f"gen_f_{dust}"] = (dust, g_rhs(dust))
        out[f"gen_fJ_{dust}"] = (dust, g_jac(dust))
    return out


def compile_variant(dust, fn, args):
    """AOT-compiled chunked fn over all cells (traced inside its dust mode)."""
    f = jax.jit(lambda *a: chunked(fn, *a))
    with dust_mode(dust):
        return f.lower(*args).compile()


def load(cdir):
    meta = json.load(open(os.path.join(cdir, "meta.json")))
    ld = lambda k: jnp.asarray(np.load(os.path.join(cdir, k + ".npy")))  # noqa: E731
    cols = np.load(os.path.join(cdir, "cols.npy"))
    args = (ld("y0"), ld("n_h"), *(jnp.asarray(c) for c in cols), ld("tdust"))
    return meta, args


def do_check(meta, args, res, n_chunks=8):
    """Full f and J of generated vs existing (jacfwd) on n_chunks evenly spaced 2^20-cell chunks."""
    n = args[0].shape[0]
    idx = np.linspace(0, n // CH - 1, n_chunks).astype(int)
    out = {}
    for dust, rt in (("implicit", 1e-12), ("frozen", 1e-2)):
        @jax.jit
        def both(y, nh, c0, c1, c2, td):
            p = params(meta, nh, (c0, c1, c2), td, rt)
            with dust_mode(dust):
                fr, aux = rates_net5.rhs(y, p)
                Jr = jax.vmap(lambda yy, pp: jax.jacfwd(lambda z: rates_net5.rhs(z, pp)[0])(yy))(y, p)
            fg, Jg, ag = gen.rhs_and_jac(y, p, dust)
            return fr, aux["gross"], aux["tdust"], Jr, fg, Jg, ag["tdust"]
        ef, ej, etd, nbad = 0.0, 0.0, 0.0, 0
        for i in idx:
            sl = tuple(a[i * CH:(i + 1) * CH] for a in args)
            fr, gross, tdr, Jr, fg, Jg, tdg = (np.asarray(v) for v in both(*sl))
            d = np.abs(fg - fr)
            ef = max(ef, float(np.max(np.where(d == 0, 0.0, d / np.where(gross > 0, gross, np.inf)))))
            s = np.maximum(np.abs(np.asarray(sl[0])), ATOL / 1e-4)
            Jrs = Jr * s[:, None, :]
            row = np.max(np.abs(Jrs), -1) + gross
            dj = np.abs(Jg - Jr) * s[:, None, :]
            ej = max(ej, float(np.max(np.where(dj == 0, 0.0, dj / row[:, :, None]))))
            etd = max(etd, float(np.max(np.abs(tdg - tdr) / tdr)))
            nbad += int((~np.isfinite(Jg)).any(-1).any(-1).sum() + (~np.isfinite(Jr)).any(-1).any(-1).sum())
        out[dust] = dict(chunks=[int(i) for i in idx], cells=int(n_chunks * CH), f_err_vs_gross=ef,
                         J_err_scaled=ej, tdust_rel=etd, nonfinite_J_cells=nbad)
        print("check", dust, out[dust], flush=True)
    res["check"] = out


def do_time(meta, args, res, only, hlo_dir):
    n = args[0].shape[0]
    out = {}
    for name, (dust, fn) in variants(meta).items():
        if only and name not in only:
            continue
        exe = compile_variant(dust, fn, args)
        if hlo_dir:
            with open(os.path.join(hlo_dir, f"{name}.hlo.txt"), "w") as fh:
                fh.write(exe.as_text())
        _, tm, tmin, w = timed(exe, *args)
        ca = exe.cost_analysis()
        ca = ca[0] if isinstance(ca, list) else ca
        out[name] = dict(ms=tm * 1e3, ms_min=tmin * 1e3, ms_all=[x * 1e3 for x in w],
                         ns_per_cell=tm / n * 1e9,
                         xla_flops_per_cell=float(ca.get("flops", 0)) / CH,
                         xla_transcendentals_per_cell=float(ca.get("transcendentals", 0)) / CH)
        print("time", name, {k: v for k, v in out[name].items() if k != "ms_all"}, flush=True)
        del exe
    res["time"] = out


def do_prof(meta, args, res, only, path):
    """One warm call per variant, then one inside a region (for rocprofv3 traces / counters)."""
    for name, (dust, fn) in variants(meta).items():
        if only and name not in only:
            continue
        exe = compile_variant(dust, fn, args)
        jax.block_until_ready(exe(*args))
        with region(name):
            jax.block_until_ready(exe(*args))
        print("prof", name, "done", flush=True)
        json.dump(dict(res, regions=REGIONS), open(path, "w"), indent=1)     # survives a later crash
        del exe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--what", default="check,time")
    ap.add_argument("--only", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--hlo", action="store_true")
    a = ap.parse_args()
    what, only = a.what.split(","), [s for s in a.only.split(",") if s]
    res = dict(device=str(jax.devices()[0]), jax=jax.__version__, what=what)
    meta, args = load(a.cells)
    res.update(meta=meta, n=int(args[0].shape[0]))
    if "check" in what:
        do_check(meta, args, res)
    if "time" in what:
        do_time(meta, args, res, only, a.out if a.hlo else "")
    path = os.path.join(a.out, f"bench_{'_'.join(what)}{a.tag}.json")
    if "prof" in what:
        do_prof(meta, args, res, only, path)
    res["regions"] = REGIONS
    json.dump(res, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
