"""Fixed-structure network-5 solve on a saved state: k Rodas3 substeps per cell, no rejection, no while loop.

    python tools/chem_coupled/fixed_k.py --run DIR --state ckpt_step1273 --what refs|fixed|grad --out DIR"""
import argparse
import contextlib
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.chemistry import cooling, rates_net5  # noqa: E402
from luna.ode import rodas3 as ig  # noqa: E402
from luna.chemistry import networks  # noqa: E402
from luna.chemistry import tables as tb  # noqa: E402

from luna.sources import chemistry as ch  # noqa: E402
from tools.chem_coupled import speedups as sp  # noqa: E402

CH = 2 ** 20
DUST_ITERS = 4
NAMES = ("x_H2", "x_H+", "x_CO", "e")
NET = networks.NET5._replace(rhs=functools.partial(rates_net5.rhs, lvg=False),
                       converge=(("dust_rtol", 1e-2),))
NET_REF = networks.NET5                      # the run's network: dust converged to 1e-12
REF8 = ig.Options(rtol=1e-8, atol=tuple(1e-4 * a for a in networks.NET5.atol))


def dust_fixed(n_h, temp, x_hi, chi, g0_dust, dgr, tdust_in, h2_form_ex=0.84, h2_form_kin=0.12,
               rtol=1e-2):
    """cooling.dust_temperature with DUST_ITERS masked Newton iterations in place of the while loop."""
    ch7 = cooling.interp_rows({7: cooling.chem_rows()[7]}, temp)[7] * dgr
    cl14 = cooling.interp_rows({14: cooling.cool_rows()[14]}, temp)[14]
    active = (dgr > 1e-8) & (temp < 1e5)
    isrf_heat = cooling.ISRF_DUST_HEATING * n_h * (g0_dust * chi / tb.f32(1.7)) * dgr
    temp2 = temp * 1e-2
    h2_form_dust = 1.0 - h2_form_ex - h2_form_kin

    def rates(td):
        fa = 1.0 / (1.0 + 1e4 * jnp.exp(-6e2 / td))
        stick = 1.0 / (1.0 + 0.4 * (temp2 + td * 1e-2) ** 0.5 + 0.2 * temp2 + 0.08 * temp2**2.0)
        return fa, stick, ch7 * fa * stick

    def step(td):
        fa, stick, rdust = rates(td)
        dust_cool = cooling.DUST_EMISSION * td**6 * n_h * dgr
        gas_dust = cl14 * (temp - td) * n_h**2 * dgr
        dfa = -fa * (1 - fa) * (6e2 / td**2)
        rdust_rat = dfa / fa - 2e-2 * stick / jnp.sqrt(td + temp)
        dust_h2_heat = rdust * x_hi * n_h**2 * h2_form_dust * 4.48 * cooling.EV
        f_nr = isrf_heat - dust_cool + gas_dust + jnp.where(rdust == 0.0, 0.0, dust_h2_heat)
        df_nr = (-(6.0 * dust_cool / td) - cl14 * n_h**2
                 + jnp.where(rdust == 0.0, 0.0, dust_h2_heat * rdust_rat))
        new = td - f_nr / df_nr
        new = jnp.where(new != new, cooling.TMIN, new)
        return jnp.clip(new, cooling.TMIN, 1.5e3)

    td = jnp.where(active, tdust_in, 10.0)
    done = td * 0.0 > 1.0
    for _ in range(DUST_ITERS):
        new = step(td)
        stop = done | (jnp.abs((new - td) / td) < rtol)
        td, done = jnp.where(stop, td, new), stop
    td = cooling._polish(td, rtol, step)
    return cooling._finish(td, done, active, temp, x_hi, n_h, rates)


@contextlib.contextmanager
def fixed_dust():
    """While tracing: the dust Newton as straight-line code."""
    orig = cooling.dust_temperature
    cooling.dust_temperature = dust_fixed
    try:
        yield
    finally:
        cooling.dust_temperature = orig


def fixed_cell(k, y0, p, dt, net=NET):
    """k Rodas3 substeps of dt/k for one cell: (y, T_dust, max embedded error norm, any invalid step)."""
    rtol, atol = net.rtol, jnp.asarray(net.atol)
    p = p._replace(**{f: jnp.asarray(v, jnp.float64) for f, v in net.converge})
    h = dt / k

    def sub(c, _):
        y, p, emax, bad = c
        (f0, aux0), jac = ig._f_and_jac(net, y, p)
        p = ig._with_hidden(net, p, aux0)
        scale = jnp.maximum(rtol * jnp.abs(y) + atol, 1e-300)
        y1, err_vec, b = ig._rodas3(net, y, f0, jac, p, h, scale)
        w = jnp.maximum(rtol * jnp.maximum(jnp.abs(y), jnp.abs(y1)) + atol, 1e-300)
        err = ig._error_norm(err_vec, w)
        ynew, phys, _ = ig._check(net, rtol, atol, y1, p)
        return (ynew, p, jnp.maximum(emax, err), bad | ~phys | b | ig._invalid(aux0)), None

    z = 0.0 * dt
    (y, p, emax, bad), _ = jax.lax.scan(sub, (y0, p, z, z > 1.0), None, length=k)
    _, aux = net.rhs(y, p)
    return y, aux["tdust"], emax, bad


def chunked(f, *args):
    """lax.map of f over 2^20-cell chunks of flat per-cell arrays."""
    n = jax.tree_util.tree_leaves(args)[0].shape[0]
    out = jax.lax.map(lambda a: f(*a), jax.tree_util.tree_map(
        lambda a: a.reshape((n // CH, CH) + a.shape[1:]), args))
    return jax.tree_util.tree_map(lambda a: a.reshape((n,) + a.shape[2:]), out)


def load_cells(run, state):
    """The stage's per-cell inputs on a saved state: y0 (e in erg cm^-3), n_H, columns, T_dust, h, dt [s]."""
    c, U, arr, meta = sp.load(run, state)

    @jax.jit
    def inputs(U, S):
        y0, n_h, cols, _, _ = ch.cell_inputs(U, S, c)
        f = lambda a: a.reshape(-1)  # noqa: E731
        return y0.reshape(-1, 4), f(n_h), tuple(f(x) for x in cols)
    y0, n_h, cols = inputs(jnp.asarray(U), jnp.asarray(arr["S"]))
    del U
    return c, meta["dt"] * c.t_s, y0, n_h, cols, jnp.asarray(arr["tdust"]).reshape(-1), \
        jnp.asarray(arr["h"]).reshape(-1)


def timed(f, *args, reps=5):
    out = jax.block_until_ready(f(*args))
    w = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(*args))
        w.append(time.perf_counter() - t0)
    return out, float(np.median(w)), w


def errors(y, ref, mask=None):
    """Per component: max and p99 relative difference (where |ref| > atol), cells outside 1e-4 |ref| + atol."""
    atol = np.asarray(networks.NET5.atol)
    out = {}
    for j, k in enumerate(NAMES):
        a, b = np.asarray(y[:, j]), np.asarray(ref[:, j])
        if mask is not None:
            a, b = a[mask], b[mask]
        d = np.abs(a - b)
        big = np.abs(b) > atol[j]
        rel = d[big] / np.abs(b[big])
        out[k] = dict(max=float(rel.max()) if rel.size else 0.0,
                      p99=float(np.percentile(rel, 99)) if rel.size else 0.0,
                      n_out_rtol=int(np.sum(d > 1e-4 * np.abs(b) + atol[j])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--what", default="fixed")
    ap.add_argument("--ks", default="1,2,3,4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = os.path.join(a.out, a.state)
    os.makedirs(d, exist_ok=True)
    c, dt, y0, n_h, cols, td0, h0 = load_cells(a.run, a.state)
    n = y0.shape[0]
    res_f = os.path.join(d, f"result_{a.what}.json")
    res = dict(state=a.state, cells=n, dt_yr=dt / 3.15576e7)

    def save():
        json.dump(res, open(res_f, "w"), indent=1)

    if a.what == "refs":
        rec = np.concatenate([np.asarray(n_h)[:, None]] + [np.asarray(v)[:, None] for v in cols]
                             + [np.asarray(y0), np.asarray(td0)[:, None], np.full((n, 1), dt)], 1)
        rec.astype(np.float64).tofile(os.path.join(d, "dvode_in.bin"))
        del rec
        for name, opt, rt in (("ref_ad", ig.Options(), 1e-2), ("ref8", REF8, 1e-12)):
            c.dust_rtol = rt
            f = jax.jit(lambda y0, n_h, cols, td0, h0, opt=opt: ch.integrate_cells(
                c, y0, n_h, cols, td0, h0, dt, opt=opt))
            t0 = time.perf_counter()
            r = jax.block_until_ready(f(y0, n_h, cols, td0, h0))
            att = np.asarray(r[3]) + np.asarray(r[4])
            res[name] = dict(s=time.perf_counter() - t0, mean_attempts=float(att.mean()),
                             max_attempts=int(att.max()), failed=int(np.asarray(r[5]).sum()))
            np.save(os.path.join(d, f"{name}.npy"), np.asarray(r[0]))
            np.save(os.path.join(d, f"{name}_att.npy"), att.astype(np.int32))
            print(name, res[name], flush=True)
            save()
            del r
        ad, r8 = np.load(os.path.join(d, "ref_ad.npy")), np.load(os.path.join(d, "ref8.npy"))
        hard = np.load(os.path.join(d, "ref_ad_att.npy")) > 1
        res["adaptive_vs_ref8"] = dict(all=errors(ad, r8), hard=errors(ad, r8, hard),
                                       n_hard=int(hard.sum()))
        print(json.dumps(res["adaptive_vs_ref8"], indent=1), flush=True)
        save()
        return

    if a.what == "dvode_cmp":
        r8 = np.load(os.path.join(d, "ref8.npy"))
        dv = np.fromfile(os.path.join(d, "dvode_out.bin")).reshape(-1, 5)
        hard = np.load(os.path.join(d, "ref_ad_att.npy")) > 1
        res["dvode_vs_ref8"] = dict(all=errors(dv[:, :4], r8), hard=errors(dv[:, :4], r8, hard))
        print(json.dumps(res, indent=1), flush=True)
        save()
        return

    p_all = lambda n_h, cols, td: ch.params(c, n_h, cols, td)  # noqa: E731
    if a.what == "fixed":
        ad, r8 = np.load(os.path.join(d, "ref_ad.npy")), np.load(os.path.join(d, "ref8.npy"))
        hard = np.load(os.path.join(d, "ref_ad_att.npy")) > 1
        res["n_hard"] = int(hard.sum())
        for k in (int(v) for v in a.ks.split(",")):
            with fixed_dust():
                cell = jax.vmap(lambda y, p, k=k: fixed_cell(k, y, p, dt))
                f = jax.jit(lambda y0, n_h, cols, td0: chunked(
                    lambda y, nh, c0, c1, c2, td: cell(y, p_all(nh, (c0, c1, c2), td)),
                    y0, n_h, *cols, td0))
                out, t_med, w = timed(f, y0, n_h, cols, td0)
            y, td, emax, bad = (np.asarray(v) for v in out)
            np.save(os.path.join(d, f"fixed_k{k}.npy"), y)
            np.save(os.path.join(d, f"fixed_k{k}_err.npy"), emax)
            em = emax
            r = dict(ms=1e3 * t_med, ms_all=[1e3 * x for x in w], ns_per_cell=1e9 * t_med / n,
                     invalid=int(bad.sum()), nonfinite=int((~np.isfinite(y)).any(1).sum()),
                     err_pct={q: float(np.percentile(em, q)) for q in (50, 90, 99, 99.9, 99.99)},
                     err_max=float(em.max()),
                     frac_err_gt={str(t): float(np.mean(em > t)) for t in (0.1, 1, 10, 100)},
                     err_hard_pct={q: float(np.percentile(em[hard], q)) for q in (50, 90, 99)},
                     vs_adaptive=dict(all=errors(y, ad), hard=errors(y, ad, hard)),
                     vs_ref8=dict(all=errors(y, r8), hard=errors(y, r8, hard),
                                  err_gt1=errors(y, r8, em > 1) if (em > 1).any() else None))
            res[f"k{k}"] = r
            print(f"k={k}", json.dumps(r, indent=1), flush=True)
            save()
            del out, y
        return

    if a.what == "hybrid":
        k1, k2, thr = (float(v) for v in a.ks.split(","))
        k1, k2 = int(k1), int(k2)
        with fixed_dust():
            one = jax.vmap(lambda y, p: fixed_cell(k1, y, p, dt))
            two = jax.vmap(lambda y, p: fixed_cell(k2, y, p, dt))

            def first(y0, n_h, cols, td0):
                y, td, e, bad = chunked(lambda y, nh, c0, c1, c2, t: one(y, p_all(nh, (c0, c1, c2), t)),
                                        y0, n_h, *cols, td0)
                return y, td, (e > thr) | bad

            def hybrid(y0, n_h, cols, td0, cap):
                y, td, flag = first(y0, n_h, cols, td0)
                idx = jnp.nonzero(flag, size=cap, fill_value=n)[0]
                g = lambda v: v[jnp.minimum(idx, n - 1)]  # noqa: E731
                y2, td2, _, _ = chunked(lambda yy, nh, c0, c1, c2, t: two(yy, p_all(nh, (c0, c1, c2), t)),
                                        g(y0), g(n_h), *(g(v) for v in cols), g(td0))
                return y.at[idx].set(y2, mode="drop"), td.at[idx].set(td2, mode="drop"), flag
            nflag = int(jnp.sum(jax.jit(first)(y0, n_h, cols, td0)[2]))
            cap = max(CH, -(-nflag // CH) * CH)
            f = jax.jit(hybrid, static_argnums=4)
            out, t_med, w = timed(f, y0, n_h, cols, td0, cap)
            _, t1, _ = timed(jax.jit(first), y0, n_h, cols, td0)
        y = np.asarray(out[0])
        ok = True
        if os.path.exists(os.path.join(d, f"fixed_k{k1}.npy")):
            yk1 = np.load(os.path.join(d, f"fixed_k{k1}.npy"), mmap_mode="r")
            yk2 = np.load(os.path.join(d, f"fixed_k{k2}.npy"), mmap_mode="r")
            fl = np.asarray(out[2])
            ok = bool(np.array_equal(y, np.where(fl[:, None], yk2, yk1)))
        res[f"hybrid_{k1}_{k2}_{thr:g}"] = dict(ms=1e3 * t_med, ms_all=[1e3 * x for x in w],
                                                ms_first_pass=1e3 * t1, flagged=nflag, cap=cap,
                                                equals_offline_composite=ok)
        print(res, flush=True)
        save()
        return

    if a.what == "grad":
        hard = np.load(os.path.join(d, "ref_ad_att.npy")) > 1
        rng = np.random.default_rng(3)
        sel = np.concatenate([rng.choice(np.nonzero(hard)[0], 8, replace=False),
                              rng.choice(n, 8, replace=False)])
        ys, nhs = jnp.asarray(np.asarray(y0)[sel]), jnp.asarray(np.asarray(n_h)[sel])
        cs = tuple(jnp.asarray(np.asarray(v)[sel]) for v in cols)
        tds = jnp.asarray(np.asarray(td0)[sel])
        wts = jnp.asarray(rng.normal(size=(sel.size, 4)))
        for k in (int(v) for v in a.ks.split(",")):
            def F(y, zeta, k=k):
                p = p_all(nhs, cs, tds)._replace(zeta=zeta)
                yk = jax.vmap(lambda yy, pp: fixed_cell(k, yy, pp, dt)[0])(y, p)
                return jnp.sum(wts * jnp.log(jnp.maximum(yk, 1e-300)) * (yk > 0))
            with fixed_dust():
                Fj = jax.jit(F)
                gy, gz = (np.asarray(g) for g in jax.jit(jax.grad(F, argnums=(0, 1)))(
                    ys, jnp.full(sel.size, c.zeta)))
                z0 = np.full(sel.size, c.zeta)
                worst = []
                for i in range(sel.size):
                    ad = np.concatenate([gy[i] * np.asarray(ys[i]), [gz[i] * c.zeta]])
                    fd = np.zeros(5)
                    for j in range(5):
                        eps = 1e-5
                        yp, ym, zp, zm = (np.asarray(ys).copy(), np.asarray(ys).copy(),
                                          z0.copy(), z0.copy())
                        if j < 4:
                            yp[i, j] *= 1 + eps
                            ym[i, j] *= 1 - eps
                        else:
                            zp[i] *= 1 + eps
                            zm[i] *= 1 - eps
                        fd[j] = float(Fj(jnp.asarray(yp), jnp.asarray(zp))
                                      - Fj(jnp.asarray(ym), jnp.asarray(zm))) / (2 * eps)
                    worst.append(float(np.max(np.abs(ad - fd)) / max(np.max(np.abs(ad)), 1e-300)))
            res[f"grad_k{k}"] = dict(max_rel=float(max(worst)), median_rel=float(np.median(worst)),
                                     per_cell=worst)
            print(f"grad k={k}", res[f"grad_k{k}"], flush=True)
            save()


if __name__ == "__main__":
    main()
