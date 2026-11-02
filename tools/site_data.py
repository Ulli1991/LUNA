"""Per-test website bundles for group A against AthenaK: LUNA's picture runs, then site/data/<test>.{npz,json}.

    python tools/site_data.py run TEST [root=$LUNA_SITE_ROOT] [schemes=mc+minmod]   # GPU job
    python tools/site_data.py TEST [root=...] [out=site/data]                                            # bundle

`run` calls the test's own run function (its cfg and IC) at the picture resolution with LUNA's limiters and keeps
the states it evolved: ROOT/luna/TEST/<label>_<scheme>.npz (t, U at each time; 3D in float32) and a .json (commit,
job, GPU, precision, the cfg fields that matter, gravity_energy as resolved). Raw outputs stay on scratch; the
bundle is float32 with fields <= 512^2, profiles and time series, about 1 MB.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

from luna.reference_data import reference_named, reference_path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("LUNA_SITE_ROOT", "site")     # raw outputs and bundles; set it to scratch
LIMITERS = {"minmod": 0, "mc": 1, "vl": 2}
CFG_KEYS = ("test_problem", "SPATIAL_DIMENSION", "Nx", "Ny", "Nz", "Lx", "Ly", "Lz", "BOUNDARY", "gamma", "ORDER",
            "LIMITER", "riemann_solver", "CFL", "enable_x64", "SELF_GRAVITY", "gravity_energy")


# ---------------------------------------------------------------- LUNA picture runs
class Capture:
    """Patch `names` (evolve, evolve_staged) in module `mod` to keep every (cfg, times, states) they return."""

    def __init__(self, mod, names=("evolve", "evolve_staged"), thin=False):
        self.mod, self.calls, self.orig, self.thin = mod, [], {}, thin
        for nm in names:
            if hasattr(mod, nm):
                self.orig[nm] = f = getattr(mod, nm)
                setattr(mod, nm, self._wrap(f, nm))

    def _wrap(self, f, nm):
        def w(cfg, U0, times, *a, **kw):
            r = f(cfg, U0, times, *a, **kw)
            st = r[0] if nm == "evolve" else r
            ts = np.atleast_1d(np.asarray(times, float))
            st = [st] if np.ndim(times) == 0 else list(st)
            if self.thin:                                 # long observe series (RM): its middle and last only
                k = sorted({len(st) // 2, len(st) - 1})
                ts, st = ts[k], [st[i] for i in k]
            self.calls.append((cfg, ts, st))
            return r
        return w

    def restore(self):
        for nm, f in self.orig.items():
            setattr(self.mod, nm, f)


def _limiter(mod, lim):
    """Point mod.accuracy_cfg at LIMITER = lim (the tests' scheme tables know minmod and MC only)."""
    from tests._helpers import accuracy_cfg
    mod.accuracy_cfg = lambda **kw: accuracy_cfg(**{**kw, "LIMITER": lim})
    return accuracy_cfg


def _sel(cap, keep):
    """The captured calls' (cfg, t, U) of the times in `keep` (None: all), observe results passed through."""
    out = []
    for cfg, ts, sts in cap.calls:
        for t, U in zip(ts, sts):
            if keep is None or any(abs(t - k) <= 1e-9 * max(1.0, k) for k in keep):
                out.append((cfg, float(t), U))
    return out


def _meta(cfg):
    import jax

    from luna.setup.gravity_energy import resolve_gravity_energy
    from tests._helpers import PRECISION
    h = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip().splitlines()[:1] if jax.default_backend() == "gpu" else []
    c = {k: getattr(cfg, k) for k in CFG_KEYS if hasattr(cfg, k)}
    if getattr(cfg, "SELF_GRAVITY", False):
        c["gravity_energy_resolved"] = resolve_gravity_energy(cfg)
    return dict(luna_commit=h, luna_dirty=dirty, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                node=os.uname().nodename, gpu=gpu[0] if gpu else "", backend=jax.default_backend(),
                precision=PRECISION, cfg={k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                                          for k, v in c.items()})


def _save(d, label, sch, rows, extra=None, wall=None):
    """ROOT/luna/TEST/<label>_<sch>.npz: t, U<k> per kept time (3D float32), extra arrays; .json the meta."""
    os.makedirs(d, exist_ok=True)
    arr = {"t": np.array([t for _, t, _ in rows])}
    for k, (_, _, U) in enumerate(rows):
        U = np.asarray(U)
        arr[f"U{k}"] = U.astype(np.float32) if U.ndim == 4 else U
    arr |= {k: np.asarray(v) for k, v in (extra or {}).items()}
    p = os.path.join(d, f"{label}_{sch}")
    np.savez(p + ".npz", **arr)
    meta = _meta(rows[0][0]) | dict(label=label, scheme=sch, times=arr["t"].tolist(), wall_s=wall)
    json.dump(meta, open(p + ".json", "w"), indent=1)
    print("wrote", p, arr["t"], {k: v.shape for k, v in arr.items()}, flush=True)


def _kh_runs(sch, n=1024, ns=(256, 512, 1024)):
    """Lecoanet KH: E_y(t) every KH_HST at each N (observe), the field at KH_SNAP and KH_T at the finest."""
    from tests._helpers import evolve_staged
    from tools.athenak_refs import KH_HST, KH_SNAP, KH_T, kh_cfg
    from luna.ic_generation.ic import initial_conditions
    out = []
    for m in ns:
        cfg = kh_cfg(m, LIMITER=LIMITERS[sch])
        U0 = np.asarray(initial_conditions(cfg))
        ts = np.round(np.arange(1, round(KH_T / KH_HST) + 1) * KH_HST, 12)
        field = set(KH_SNAP) | {KH_T} if m == n else set()
        dv = (cfg.Lx / cfg.Nx) * (cfg.Ly / cfg.Ny)

        def obs(U, t, dv=dv, field=field):
            e = dict(Ey=0.5 * float((U[2] ** 2 / U[0]).sum()) * dv, Ex=0.5 * float((U[1] ** 2 / U[0]).sum()) * dv)
            return (e, U) if any(abs(t - f) < 1e-9 for f in field) else (e, None)
        w = time.time()
        r = evolve_staged(cfg, U0, list(ts), observe=obs)
        rows = [(cfg, float(t), U) for t, (_, U) in zip(ts, r) if U is not None] or [(cfg, 0.0, U0)]
        out.append((f"N{m}", rows, dict(t_series=ts, Ey=[e["Ey"] for e, _ in r], Ex=[e["Ex"] for e, _ in r]),
                    time.time() - w))
    return out


def picture_runs(test, sch):
    """[(label, [(cfg, t, U)], extra arrays, wall s)] of LUNA's picture runs of TEST with limiter `sch`."""
    import importlib
    lim = LIMITERS[sch]
    mc = "hllc-mc"                                    # the tests' scheme name; LIMITER is overridden

    def via(modname, fn, label, keep=None, names=("evolve", "evolve_staged"), extra=None, thin=False):
        mod = importlib.import_module(modname)
        orig = _limiter(mod, lim)
        cap = Capture(mod, names, thin)
        w = time.time()
        try:
            r = fn(mod)
        finally:
            cap.restore()
            mod.accuracy_cfg = orig
        return [(label, _sel(cap, keep), extra(r) if extra else None, time.time() - w)]

    if test == "a1":
        return via("tests.test_hydro_accuracy_linear_waves", lambda m: m.run.__wrapped__(1, 1024, "sound_r", mc), "N1024")
    if test == "a3":
        return via("tests.test_hydro_accuracy_lw_smooth", lambda m: m.errors.__wrapped__(1024, mc), "N1024")
    if test == "a4":
        return (via("tests.test_hydro_hard_lw_gresho", lambda m: m.periodic.__wrapped__(1024, mc, False), "static")
                + via("tests.test_hydro_hard_lw_gresho", lambda m: m.periodic.__wrapped__(1024, mc, True), "moving"))
    if test == "a6":
        return via("tests.test_hydro_hard_isothermal_atmosphere", lambda m: m.errors.__wrapped__(2048, mc), "N2048")
    if test == "a7":
        import tools.athenak_refs as ar
        out = []
        for nj, case in ar.A7_CASES:
            mod = importlib.import_module("tests.test_gravity_jeans_wave")
            orig = ar._luna_cfg(mod, lim, None, "auto")
            cap = Capture(mod)
            w = time.time()
            try:
                mod.errors.__wrapped__(128, nj, mc)
            finally:
                cap.restore()
                mod.accuracy_cfg = orig
            out.append((f"{case}_N128", _sel(cap, None), None, time.time() - w))
        return out
    if test == "a8":
        import tools.athenak_refs as ar
        mod = importlib.import_module("tests.test_gravity_evrard_energy")
        orig = ar._luna_cfg(mod, lim, None, "auto")
        cap = Capture(mod)
        w = time.time()
        try:
            b = np.array(mod.run_budget(256, mc, times=(0.75, 3.0)))
        finally:
            cap.restore()
            mod.accuracy_cfg = orig
        return [("N256", _sel(cap, None), dict(budget=b), time.time() - w)]
    if test == "a9":
        return via("tests.test_hydro_hard_woodward_colella", lambda m: m.run(2048, lim), "N2048")
    if test == "a10":
        return via("tests.test_hydro_hard_implosion", lambda m: m.run.__wrapped__(1024, mc, (m.T_EARLY,)), "N1024")
    if test == "a11":
        sk = "minmod" if lim == 0 else "mc"            # uncached: MC and van Leer share the scheme name
        return [r for c in (3, 6) for r in via("tests.test_hydro_hard_kt_riemann",
                                               lambda m, c=c: m._run.__wrapped__(c, 2048, sk, 1, None, 0.5, m.G, 1,
                                                                                 m.T, None, "fp64"), f"c{c}_N2048")]
    if test == "rt":
        return via("tests.test_hydro_hard_rayleigh_taylor", lambda m: m.run.__wrapped__(512, mc, 1.0), "N512")
    if test == "rm":
        return [r for case in ("heavy-light", "light-heavy")
                for r in via("tests.test_hydro_hard_richtmyer_meshkov",
                             lambda m, case=case: m.history.__wrapped__(case, 512, mc), f"{case}_N512",
                             names=("evolve",), extra=lambda r: dict(tau=r[0], psi=r[1]), thin=True)]
    if test == "tg":                                  # _tg_reference.luna_run's cfg with LIMITER, M = 0.1 to t = 20
        import tests._tg_reference as tg
        from tests._helpers import accuracy_cfg, evolve_staged
        cfg = accuracy_cfg(test_problem="Sod-Shock-3D-AMR", SPATIAL_DIMENSION=3, Nx=256, Ny=256, Nz=256, Lx=tg.L,
                           Ly=tg.L, Lz=tg.L, BOUNDARY=0, gamma=tg.GAMMA, LIMITER=lim)
        ts = np.round(np.arange(1, 201) * 0.1, 12)               # K(t) every 0.1 as AthenaK's history, U at 20
        dv = (tg.L / 256) ** 3

        def obs(U, t):
            return 0.5 * float((sum(U[d] ** 2 for d in (1, 2, 3)) / U[0]).sum()) * dv, (U if t > 19.99 else None)
        w = time.time()
        r = evolve_staged(cfg, tg.luna_ic(256, 0.1), list(ts), observe=obs)
        return [("N256", [(cfg, 20.0, r[-1][1])], dict(t_series=ts, K=[k for k, _ in r]), time.time() - w)]
    if test == "sod":
        return via("tests.test_hydro_accuracy_shocktube",
                   lambda m: [m.run_1d(1, n) for n in (128, 256, 512, 800, 1600, 3200)], "c1")
    if test == "so":
        return via("tests.test_hydro_hard_shu_osher", lambda m: [m.errors(n, lim, np.load(m.REF)["rho"])
                                                                for n in m.NS], "ladder")
    if test == "tt":
        return via("tests.test_hydro_hard_titarev_toro", lambda m: [m.errors(n, lim, np.load(m.REF)["rho"])
                                                                   for n in m.NS], "ladder")
    if test == "sedov":
        from dataclasses import replace
        mod = importlib.import_module("tests.test_hydro_accuracy_sedov")
        cfg0 = mod.sedov_cfg
        mod.sedov_cfg = lambda n: replace(cfg0(n), LIMITER=lim)
        out = []
        try:
            for n in (64, 128, 256):
                cap = Capture(mod, ("evolve",))
                w = time.time()
                try:
                    r = mod.run_sedov(n)
                finally:
                    cap.restore()
                out.append((f"N{n}", _sel(cap, None), dict(dR_cells=r[0], L1_rho=r[1], L1_p=r[2]), time.time() - w))
        finally:
            mod.sedov_cfg = cfg0
        return out
    if test == "kh":
        return [(lb, rows, ex, w) for lb, rows, ex, w in _kh_runs(sch)]
    raise SystemExit(f"no picture runs for {test}")


def run(test, root=ROOT, schemes="mc+minmod"):
    import jax
    jax.config.update("jax_enable_x64", True)
    d = os.path.join(root, "luna", test)
    for sch in schemes.split("+"):
        for label, rows, extra, wall in picture_runs(test, sch):
            rows = [r for r in rows if np.size(r[2]) > 0]
            if test == "sod" or test in ("so", "tt"):        # a ladder: one file per N
                for cfg, t, U in rows:
                    _save(d, f"{label}_N{cfg.Nx}", sch, [(cfg, t, U)], wall=wall)
            else:
                _save(d, label, sch, rows, extra, wall)


def main(argv):
    if argv and argv[0] == "run":
        kw = dict(a.split("=", 1) for a in argv[2:])
        return run(argv[1], **kw)
    kw = dict(a.split("=", 1) for a in argv[1:])
    return bundle(argv[0], **kw)


# ---------------------------------------------------------------- bundles
LT = os.path.dirname(ROOT)                                    # the work root above site/
AKD = os.path.join(LT, "athenak")
ROW = {"a1": "A1", "a3": "A3", "a4": "A4", "a5": "A5", "a6": "A6", "a7": "A7", "a8": "A8", "a9": "A9", "a10": "A10",
       "a11": "A11", "rt": "A13", "rm": "A14", "tg": "A15"}
A4_BULK = 1.0 / (0.4 * np.pi)                                # moving Gresho: one box per revolution
CAP = 512                                                     # fields: at most CAP cells per axis


def shrink(a, cap=CAP):
    """Block mean of a 2D field down to <= cap cells per axis (trailing cells that do not fill a block dropped)."""
    a = np.asarray(a, np.float64)
    for ax in range(a.ndim):
        f = -(-a.shape[ax] // cap)
        if f > 1:
            m = a.shape[ax] // f * f
            a = np.take(a, range(m), axis=ax)
            a = a.reshape(a.shape[:ax] + (m // f, f) + a.shape[ax + 1:]).mean(ax + 1)
    return a


def prim(U, g):
    """(rho, velocity components, p) of a LUNA state."""
    rho = U[0]
    v = [U[1 + d] / rho for d in range(U.shape[0] - 2)]
    return rho, v, (g - 1.0) * (U[-1] - 0.5 * rho * sum(x * x for x in v))


def ak_final(rd, sdim):
    """(t, state) of an AthenaK run's final dump."""
    from tools import luna_ic
    t, _, p = luna_ic.final(rd)
    return t, luna_ic.read_dump(p, sdim)


def ak_dump(rd, sdim, t):
    """(t, state) of the history-cadence dump nearest t."""
    from tools import luna_ic
    d = min(luna_ic.dumps(rd), key=lambda x: abs(x[0] - t))
    return d[0], luna_ic.read_dump(d[2], sdim)


def ak_meta(rd):
    """AthenaK run record: job, GPU, binary line and the input's scheme settings."""
    import re
    j = json.load(open(os.path.join(rd, "job.json"))) if os.path.exists(os.path.join(rd, "job.json")) else {}
    a = open(os.path.join(rd, "run.athinput")).read()
    kv = dict(re.findall(r"^(\w+) = (.+)$", a, re.M))
    return dict(dir=rd.replace(LT + "/", "luna_tests/"), slurm_job_id=j.get("slurm_job_id"), gpu=j.get("gpu"),
                run_line=j.get("run_line"), wall_s=j.get("wall_s"), precision="fp64",
                cfg={k: kv.get(k) for k in ("reconstruct", "rsolver", "integrator", "cfl_number", "dfloor", "pfloor",
                                            "tlim", "nx1", "nx2", "nx3", "gamma", "four_pi_G", "self_gravity")})


def luna_pic(test, label, sch, root=ROOT):
    """(times, [states], meta) of a LUNA picture run, or None."""
    p = os.path.join(root, "luna", test, f"{label}_{sch}")
    if not os.path.exists(p + ".npz"):
        return None
    z = np.load(p + ".npz")
    k = sorted((x for x in z.files if x[0] == "U" and x[1:].isdigit()), key=lambda x: int(x[1:]))
    return z["t"], [np.asarray(z[x], np.float64) for x in k], json.load(open(p + ".json")), z


def _src_path(src):
    import re
    m = re.match(r"`([^`]+)` `([^`]+)`", src)
    if not m:
        return None, None
    path, key = m.groups()
    path = os.path.join(LT, path[len("luna_tests/"):]) if path.startswith("luna_tests/") else os.path.join(REPO, path)
    return path, key


def ladder(src):
    """{N: value} over every N of the series a summary source names (its key with N<n> generalised)."""
    import re
    path, key = _src_path(src)
    if path is None:
        return {}
    pat = lambda k: re.compile("^" + re.sub(r"N\d+", r"N(\\d+)", re.escape(k)).replace(r"N\(\\d\+\)", r"N(\d+)") + "$")  # noqa: E731
    if path.endswith(".json"):
        d = json.load(open(path))
        data = d.get("data", d)
        k, met = key.rsplit(".", 1)
        rx = pat(k)
        return {int(m.group(1)): float(v[met]) for kk, v in data.items() if (m := rx.match(kk)) and met in v}
    z = np.load(path)
    rx = pat(key)
    return {int(m.group(1)): float(z[kk]) for kk in z.files if (m := rx.match(kk)) and np.ndim(z[kk]) == 0}


def summary_rows(test, root=ROOT):
    """The test's rows of summary.json (athenak_summary.py), each with its ladders over N per column."""
    import re
    js = json.load(open(os.path.join(LT, "summary", "summary.json")))
    rid = ROW.get(test)
    out = []
    for name, r in js.items():
        if rid and re.match(rf"^{rid}( |$)", name):
            lad = {c: {str(n): v for n, v in sorted(ladder(x["source"]).items())} for c, x in r["values"].items()}
            out.append(dict(name=name, **r, ladders=lad))
    return out


def radial(U, g, centre, box, nb=None, bulk=(0.0, 0.0, 0.0)):
    """Shell averages of rho, v_r, p and entropy p / rho^g against r (3D) or of rho, v_phi, p (2D)."""
    sdim = U.ndim - 1
    n = U.shape[1:]
    xs = [(np.arange(n[d]) + 0.5) * box[d] / n[d] - centre[d] for d in range(sdim)]
    X = np.meshgrid(*xs, indexing="ij")
    r = np.sqrt(sum(x * x for x in X))
    rho, v, p = prim(U, g)
    v = [v[d] - bulk[d] for d in range(sdim)]
    dr = box[0] / n[0]
    edges = np.arange(0.0, min(box) / 2 + dr, dr) if nb is None else np.linspace(0.0, min(box) / 2, nb + 1)
    k = np.digitize(r.ravel(), edges) - 1
    ok = (k >= 0) & (k < edges.size - 1)
    cnt = np.bincount(k[ok], minlength=edges.size - 1)

    def avg(q):
        return np.bincount(k[ok], np.asarray(q).ravel()[ok], minlength=edges.size - 1) / np.maximum(cnt, 1)
    rr = np.maximum(r, 1e-300)
    if sdim == 3:
        vr = sum(v[d] * X[d] for d in range(3)) / rr
        q = dict(rho=avg(rho), v_r=avg(vr), p=avg(p), s=avg(p / rho ** g))
    else:
        vphi = (X[0] * v[1] - X[1] * v[0]) / rr
        q = dict(rho=avg(rho), v_phi=avg(vphi), p=avg(p))
    return dict(r=0.5 * (edges[1:] + edges[:-1]), **q)


def reduce_state(test, U, g, ctx):
    """{name: array} of what the page shows of one state: fields (<= CAP^2), cuts, profiles."""
    sdim = U.ndim - 1
    rho, v, p = prim(U, g)
    if sdim == 1:
        return dict(rho=rho, u=v[0], p=p)
    if sdim == 2:
        out = dict(rho=shrink(rho))
        if test == "a4":
            rad = radial(U, g, (0.5, 0.5), (1.0, 1.0), bulk=(ctx.get("bulk", 0.0), 0.0))
            out |= {f"rad_{k}": x for k, x in rad.items()}
            out["speed"] = shrink(np.sqrt(v[0] ** 2 + v[1] ** 2))
        if test == "a10":
            out |= dict(diag_rho=np.diagonal(rho), wall_rho=rho[:, 0])
        if test == "a11":
            out |= dict(top_rho=rho[:, -1], bottom_rho=rho[:, 0], left_rho=rho[0, :], right_rho=rho[-1, :],
                        top_p=p[:, -1], bottom_p=p[:, 0])
        if test in ("kh", "a5"):
            out |= dict(vy=shrink(v[1]), mean_rho_y=rho.mean(0))
        if test == "rt":
            out |= dict(vy=shrink(v[1]), vy_mid=v[1][:, v[1].shape[1] // 2])
        return out
    k = U.shape[3] // 2
    out = dict(rho_mid=shrink(rho[:, :, k]))
    if test in ("a8", "sedov"):
        L = ctx["box"]
        out |= {f"rad_{n}": x for n, x in radial(U, g, (L / 2,) * 3, (L,) * 3).items()}
    if test == "tg":
        h = 2 * np.pi / U.shape[1]
        w = (np.gradient(v[1], h, axis=0) - np.gradient(v[0], h, axis=1))[:, :, k]
        out = dict(omega_z_mid=shrink(w), rho_mid=out["rho_mid"])
    if test == "a7":
        out["rho_x"] = rho[:, 0, 0]
    return out


def ak_pics(test):
    """{(label, scheme): (run dir, t or None for the final)} of AthenaK's picture states."""
    S = os.path.join(ROOT, "ak", test)
    j = os.path.join
    fl = "_floor1e-8"
    P = {
        "a1": {("N1024", s): (j(AKD, "a1v2", f"{s}_cfl0.4", f"a1_{s}_N1024"), None) for s in ("plm_rk2", "ppmx_rk4")},
        "a3": {("N1024", s): (j(AKD, "a3", f"{s}_cfl0.4", f"a3_{s}_N1024"), None) for s in ("plm_rk2", "ppmx_rk4")},
        "a4": {(lb, s): (j(AKD, "a4", f"{s}_cfl0.4", f"a4_{s}_N1024_{c}"), None)
               for lb, c in (("static", "s"), ("moving", "m")) for s in ("plm_rk2", "ppmx_rk4", "ppm4_rk4")},
        "a6": {("N2048", s): (j(LT, "lit_a06", "ak", s, f"a6_{s}_N2048"), None)
               for s in ("plm_rk2", "ppmx_rk4", "ppm4_rk4")},
        "a7": {(f"{c}_N128", s): (j(LT, "a07_a08_ak", "a7", f"{s}_cfl0.3", f"a7_{s}_{c}_N128"), None)
               for c in ("stable", "unstable") for s in ("plm_rk2", "ppmx_rk4")},
        "a8": {("N256", s): (j(LT, "a07_a08_ak", "a8", f"{s}_cfl0.3_N256", f"a8_{s}_N256"), t)
               for s in ("plm_rk2", "ppmx_rk4") for t in (0.75, None)},
        "a9": {("N2048", s): (j(AKD, "a9", f"{s}_cfl0.4" + ("" if s == "plm_rk2" else fl), f"a9_{s}_N2048_1d"), None)
               for s in ("plm_rk2", "ppmx_rk4", "ppm4_rk4")},
        "a10": {("N1024", s): (j(S, f"{s}_cfl0.4", f"a10_{s}_N1024_e"), None)
                for s in ("plm_rk2", "ppmx_rk4", "ppm4_rk4")},
        "a11": {(f"c{c}_N2048", s): (j(AKD, "a11", f"{s}_cfl0.4{fl}", f"a11_{s}_c{c}_N2048"), None)
                for c in (3, 6) for s in ("plm_rk2", "ppmx_rk4")},
        "rt": {("N512", "ppmx_rk4"): (j(AKD, "rt", "ppmx_rk4_cfl0.4_g", "rt_ppmx_rk4_N512_p"), None),
               ("N512", "plm_rk2"): (j(S, "plm_rk2_cfl0.4", "rt_plm_rk2_N512_p"), None)},
        "tg": {("N256", "ppmx_rk4"): (j(AKD, "tg", "ppmx_rk4_cfl0.4", "tg_ppmx_rk4_N256"), None),
               ("N256", "plm_rk2"): (j(LT, "a15_ext", "ak", "plm_rk2_cfl0.3", "tg_plm_rk2_N256"), None)},
        "kh": {(f"N1024_t{t:g}", s): (j(S, f"{s}_cfl0.4", f"kh_{s}_N1024" + ("" if t == 6 else f"_t{t:g}")), None)
               for s in ("plm_rk2", "ppmx_rk4") for t in (2.0, 4.0, 6.0)},
        "sedov": {(f"N{n}", s): (j(S, f"{s}_cfl0.3", f"sedov_{s}_N{n}_f100"), None)
                  for n in (64, 128, 256) for s in ("plm_rk2", "ppmx_rk4")},
    }
    return P.get(test, {})


SDIM = {"a1": 1, "a6": 1, "a9": 1, "sod": 1, "so": 1, "tt": 1, "a7": 3, "a8": 3, "tg": 3, "sedov": 3}
GAMMA = {"a1": 5 / 3, "a7": 5 / 3, "a8": 5 / 3, "sedov": 5 / 3, "kh": 5 / 3}
LUNA_LABELS = {"a1": ["N1024"], "a3": ["N1024"], "a4": ["static", "moving"], "a6": ["N2048"],
               "a7": ["stable_N128", "unstable_N128"], "a8": ["N256"], "a9": ["N2048"], "a10": ["N1024"],
               "a11": ["c3_N2048", "c6_N2048"], "rt": ["N512"], "rm": ["heavy-light_N512", "light-heavy_N512"],
               "tg": ["N256"], "kh": ["N256", "N512", "N1024"], "sedov": ["N64", "N128", "N256"]}


def _gamma(test, meta=None):
    if meta and meta.get("cfg", {}).get("gamma"):
        return float(meta["cfg"]["gamma"])
    return GAMMA.get(test, 1.4)


def bundle(test, root=ROOT, out=None):
    """site/data/TEST.npz (float32) and .json from the stored AthenaK references, AthenaK's raw picture states and
    LUNA's picture runs; anything missing is listed in the json under `missing`, never silently skipped."""
    out = out or os.path.join(REPO, "site", "data")
    os.makedirs(out, exist_ok=True)
    arr, missing, luna_meta, ak_runs = {}, [], {}, {}
    sdim = SDIM.get(test, 2)
    # LUNA picture states
    labels = LUNA_LABELS.get(test, [])
    if test in ("sod", "so", "tt"):
        labels = sorted({os.path.basename(f)[:-5].rsplit("_", 1)[0]
                         for f in __import__("glob").glob(os.path.join(root, "luna", test, "*.json"))})
    for lb in labels:
        for sch in ("mc", "minmod", "vl"):
            r = luna_pic(test, lb, sch, root)
            if r is None:
                if sch != "vl":
                    missing.append(f"LUNA {lb} {sch}: {root}/luna/{test}/{lb}_{sch}.npz")
                continue
            ts, sts, meta, z = r
            luna_meta[f"{lb}_{sch}"] = meta
            g = _gamma(test, meta)
            ctx = dict(box=meta["cfg"].get("Lx"), bulk=A4_BULK if lb == "moving" else 0.0)
            for t, U in zip(ts, sts):
                for k, x in reduce_state(test, U, g, ctx).items():
                    arr[f"luna_{sch}_{lb}_t{t:g}_{k}"] = x
            for k in z.files:
                if not (k == "t" or (k[0] == "U" and k[1:].isdigit())):
                    arr[f"luna_{sch}_{lb}_{k}"] = z[k]
    # AthenaK picture states
    for (lb, sch), (rd, t) in ak_pics(test).items():
        if not os.path.exists(os.path.join(rd, "run.luna_final.txt")):
            missing.append(f"AthenaK {lb} {sch}: {rd.replace(LT + '/', 'luna_tests/')}")
            continue
        ta, U = ak_final(rd, sdim) if t is None else ak_dump(rd, sdim, t)
        ak_runs[f"{lb}_{sch}"] = ak_meta(rd) | dict(t=ta)
        g = float(ak_runs[f"{lb}_{sch}"]["cfg"]["gamma"] or 1.4)
        for k, x in reduce_state(test, U, g, dict(box=_box(rd), bulk=A4_BULK if lb == "moving" else 0.0)).items():
            arr[f"ak_{sch}_{lb}_t{ta:.4g}_{k}"] = x
    extra = EXTRA.get(test)
    rows = summary_rows(test, root) if test in ROW else []
    if extra:
        e_arr, e_rows, e_missing = extra(root)
        arr |= e_arr
        rows += e_rows
        missing += e_missing
    arr = {k: np.asarray(v, np.float32) for k, v in arr.items() if np.size(v)}
    np.savez_compressed(os.path.join(out, f"{test}.npz"), **arr)
    js = dict(test=test, row_id=ROW.get(test), rows=rows, luna=luna_meta, athenak=ak_runs, missing=missing,
              arrays={k: list(v.shape) for k, v in arr.items()},
              built=time.strftime("%Y-%m-%dT%H:%M:%S"), builder="tools/site_data.py")
    json.dump(js, open(os.path.join(out, f"{test}.json"), "w"), indent=1, default=str)
    size = os.path.getsize(os.path.join(out, f"{test}.npz"))
    print(f"{test}: {len(arr)} arrays, {size / 1e6:.2f} MB, {len(rows)} rows, missing {len(missing)}")
    for m in missing:
        print("  missing", m)


def _box(rd):
    """x1 extent of an AthenaK run (its input's x1max)."""
    import re
    return float(re.search(r"x1max = (\S+)", open(os.path.join(rd, "run.athinput")).read()).group(1))


def _row(name, metric, N, values, ladders, note=""):
    """A metric row in summary.json's shape, for the tests the summary does not have."""
    return dict(name=name, metric=metric, N=N, values={c: dict(value=v[0], source=v[1]) for c, v in values.items()},
                ladders={c: {str(n): x for n, x in sorted(lad.items())} for c, lad in ladders.items()}, note=note)


def _luna_states(root, test, pat):
    """{(scheme, N): (t, U)} of LUNA ladder files ROOT/luna/TEST/<pat>_N<n>_<scheme>.npz."""
    import glob
    import re
    out = {}
    for f in glob.glob(os.path.join(root, "luna", test, f"{pat}_N*_*.npz")):
        m = re.search(rf"{pat}_N(\d+)_(\w+)\.npz$", f)
        z = np.load(f)
        out[m.group(2), int(m.group(1))] = (float(z["t"][0]), np.asarray(z["U0"], np.float64))
    return out


def _ak_states(test, pat, schemes=("plm_rk2", "ppmx_rk4"), cfl="0.4"):
    """{(scheme, N): (t, U)} of AthenaK runs ROOT/ak/TEST/<scheme>_cfl<cfl>/<test>_<scheme>_<pat>N<n>."""
    import glob
    import re
    out = {}
    for s in schemes:
        for rd in glob.glob(os.path.join(ROOT, "ak", test, f"{s}_cfl{cfl}", f"{test}_{s}_{pat}N*")):
            m = re.search(r"N(\d+)$", rd)
            if m and os.path.exists(os.path.join(rd, "run.luna_final.txt")):
                out[s, int(m.group(1))] = ak_final(rd, SDIM.get(test, 2))
    return out


def _tube_rows(test, ref_fn, metric, pat, ak_pat, missing, root=ROOT):
    """L1(rho) per N of every LUNA and AthenaK tube run against ref_fn(n) (cell averages on n cells)."""
    lad, src = {}, {}
    for lab, states in (("luna", _luna_states(root, test, pat)), ("ak", _ak_states(test, ak_pat))):
        if not states:
            missing.append(f"{lab} {test} ladder")
        for (s, n), (_t, U) in states.items():
            col = s if lab == "luna" else s.split("_")[0]
            h = 1.0 / n
            lad.setdefault(col, {})[n] = float(np.abs(U[0] - ref_fn(n)).sum() * h * (10.0 if test != "sod" else 1.0))
            src[col] = f"site/{lab}/{test} (tools/site_data.py)"
    top = max((n for d in lad.values() for n in d), default=None)
    vals = {c: (d[top], src[c]) for c, d in lad.items() if top in d}
    return [_row(f"{test} {top}", metric, str(top), vals, lad)] if top else []


def extra_sod(root):
    import tests.test_hydro_accuracy_shocktube as st
    from luna.hydro import exact_riemann as er
    WL, WR, x0, t = er.TORO[1]
    ex = lambda n: er.cell_average(WL, WR, st.G, np.linspace(0, 1, n + 1), x0, t)  # noqa: E731
    missing = []
    rows = _tube_rows("sod", lambda n: ex(n)[0], "L1(rho) vs the exact cell averages, Sod (Toro 1), t = 0.25",
                      "c1", "c1_", missing, root)
    e = ex(3200)
    return dict(exact_rho=e[0], exact_u=e[1], exact_p=e[2], exact_x=(np.arange(3200) + 0.5) / 3200), rows, missing


def _ref_tube(path):
    z = np.load(path)
    r = np.asarray(z["rho"])
    return r, (lambda n: r.reshape(n, -1).mean(1))


def extra_so(root):
    import tests.test_hydro_hard_shu_osher as so
    r, f = _ref_tube(so.REF)
    missing = []
    rows = _tube_rows("so", f, "L1(rho) vs the LUNA MC 2^16 reference (luna/shu_osher/shu_osher_mc.npz), T = 1.8",
                      "ladder", "", missing, root)
    return dict(ref_rho=r, ref_x=(np.arange(r.size) + 0.5) * so.LX / r.size), rows, missing


def extra_tt(root):
    import tests.test_hydro_hard_titarev_toro as tt
    r, f = _ref_tube(tt.REF)
    missing = []
    rows = _tube_rows("tt", f, "L1(rho) vs the LUNA MC 2^16 reference (luna/titarev_toro/titarev_toro_mc.npz), T = 5",
                      "ladder", "", missing, root)
    return dict(ref_rho=r, ref_x=(np.arange(r.size) + 0.5) * tt.LX / r.size), rows, missing


def sedov_metric(U, t, n):
    """(|R - R_exact| in cells, L1 rho / rho2, L1 p / p2 behind the shock) as the Sedov test's run_sedov."""
    import tests.test_hydro_accuracy_sedov as sd
    from luna.hydro import exact_sedov as se
    from luna.ic_generation.ic import initial_conditions
    cfg = sd.sedov_cfg(n)
    g, L = cfg.gamma, cfg.Lx
    dx = L / n
    U0 = np.asarray(initial_conditions(cfg))
    rho0, p_bg = float(U0[0].min()), float((g - 1.0) * U0[4].min())
    E = float(np.sum(U0[4] - p_bg / (g - 1.0)) * dx ** 3)
    x = (np.arange(n) + 0.5) * dx - 0.5 * L
    r = np.sqrt(x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2)
    R_ex = se.xi0(g) * (E * t * t / rho0) ** 0.2
    dR = (sd.shock_radius(r, U[0], rho0, dx) - R_ex) / dx
    rho_ex, _, p_ex, R = se.solution(r, t, E, rho0, g)
    p = (g - 1.0) * (U[4] - 0.5 * (U[1:4] ** 2).sum(0) / U[0])
    inside = r < R
    Rdot = 0.4 * R / t
    rho2, p2 = rho0 * (g + 1) / (g - 1), rho0 * Rdot ** 2 * 2 / (g + 1)
    rr = np.linspace(0, 0.5 * L, 400)
    prof = se.solution(rr, t, E, rho0, g)
    return (float(abs(dR)), float(np.mean(np.abs(U[0] - rho_ex)[inside])) / rho2,
            float(np.mean(np.abs(p - p_ex - p_bg)[inside])) / p2, rr, prof[0], prof[2] + p_bg)


def extra_sedov(root):
    missing, lad, arr = [], {}, {}
    for (lb, s), (rd, _) in ak_pics("sedov").items():
        if os.path.exists(os.path.join(rd, "run.luna_final.txt")):
            t, U = ak_final(rd, 3)
            dR, er_, ep, rr, rex, pex = sedov_metric(U, t, int(lb[1:]))
            col = s.split("_")[0]
            for k, v in (("dR", dR), ("L1_rho", er_), ("L1_p", ep)):
                lad.setdefault(f"{col}_{k}", {})[int(lb[1:])] = v
            arr |= dict(exact_r=rr, exact_rho=rex, exact_p=pex)
    for sch in ("mc", "minmod"):
        for n in (64, 128, 256):
            r = luna_pic("sedov", f"N{n}", sch, root)
            if r is None:
                continue
            ts, sts, _, _ = r
            dR, er_, ep, *_ = sedov_metric(sts[-1], float(ts[-1]), n)
            for k, v in (("dR", dR), ("L1_rho", er_), ("L1_p", ep)):
                lad.setdefault(f"{sch}_{k}", {})[n] = v
    rows = []
    for k, what in (("L1_rho", "L1 of rho / rho2 behind the shock vs exact (Sedov-Taylor), t_end"),
                    ("dR", "|R - R_exact| in cells at t_end")):
        cols = {c[:-len(k) - 1]: d for c, d in lad.items() if c.endswith(k)}
        top = max((n for d in cols.values() for n in d), default=None)
        if top:
            rows.append(_row(f"Sedov {k} {top}", what, f"{top}^3",
                             {c: (d[top], "tools/site_data.py") for c, d in cols.items() if top in d}, cols))
    return arr, rows, missing


def extra_kh(root):
    from tools import luna_ic
    arr, missing = {}, []
    for s in ("plm_rk2", "ppmx_rk4"):
        for n in (256, 512, 1024):
            h = os.path.join(ROOT, "ak", "kh", f"{s}_cfl0.4", f"kh_{s}_N{n}", "run.hydro.hst")
            if not os.path.exists(h):
                missing.append(f"AthenaK kh {s} N{n} history")
                continue
            d = luna_ic.history(h)
            arr[f"ak_{s}_N{n}_series_t"], arr[f"ak_{s}_N{n}_series_Ey"] = d["time"], d["2-KE"]
    return arr, [], missing


def extra_a5(root):
    arr = {}
    for f, s in (("a5_plm_rk2_2048", "plm_rk2"), ("a5_ppmx_rk4_4096", "ppmx_rk4"), ("a5_ppm4_rk4_2048", "ppm4_rk4")):
        z = np.load(reference_named(f + ".npz"))
        for n in (512, 1024, 2048):
            if f"N{n}_M" in z.files:
                arr[f"ak_{s}_N{n}_series_t"], arr[f"ak_{s}_N{n}_series_M"] = z[f"N{n}_t"], z[f"N{n}_M"]
    z = np.load(os.path.join(LT, "lit_a05", "luna_a5.npz"))
    for s in ("minmod", "mc", "vanleer"):
        for n in (512, 1024, 2048):
            if f"{s}_N{n}_M" in z.files:
                arr[f"luna_{s}_N{n}_series_t"], arr[f"luna_{s}_N{n}_series_M"] = z[f"{s}_N{n}_t"], z[f"{s}_N{n}_M"]
    ref = np.loadtxt(reference_path("pencil/mcnally_kh/mcnally2012_fig4_Mref.txt"))
    arr["pencil_t"], arr["pencil_M"] = ref[:, 0], ref[:, 1]
    from tools.athenak_refs import A5_RAW
    for lab, p in (("luna_mc_N4096", A5_RAW["luna"]), ("ak_ppmx_rk4_N4096", A5_RAW["ak"])):
        z = np.load(p)
        arr[f"{lab}_t1.5_rho"], arr[f"{lab}_t1.5_vy"] = shrink(z["rho"]), shrink(z["my"] / z["rho"])
    return arr, [], []


def extra_a8(root):
    arr = {}
    for s in ("plm_rk2", "ppmx_rk4"):
        z = np.load(reference_path(f"athenak/evrard/a8_{s}_256.npz"))
        for n in (64, 128, 256):
            arr[f"ak_{s}_N{n}_series_t"], arr[f"ak_{s}_N{n}_series_dE"] = z[f"N{n}_t"], z[f"N{n}_dE"]
    z = np.load(os.path.join(LT, "a07_a08_ak", "luna_a8_auto.npz"))
    for s in ("minmod", "mc", "vl"):
        for n in (64, 128, 256):
            if f"hllc-{s}_N{n}_dE" in z.files:
                arr[f"luna_{s}_N{n}_series_t"], arr[f"luna_{s}_N{n}_series_dE"] = (z[f"hllc-{s}_N{n}_t"],
                                                                                  z[f"hllc-{s}_N{n}_dE"])
    return arr, [], []


def extra_tg(root):
    z = np.load(reference_path("athenak/taylor_green/tg_ppmx_rk4_256.npz"))
    arr = {f"ak_ppmx_rk4_N{n}_series_{k}": z[f"N{n}_{k}"] for n in (64, 128, 256) for k in ("t", "K")}
    v = np.load(reference_path("spectral/taylor_green_viscous/tg_viscous_spectral.npz"))
    arr |= {f"spectral_{k}": v[k] for k in v.files if np.ndim(v[k]) and np.size(v[k]) < 100000}
    return arr, [], []


def extra_rm(root):
    z = np.load(reference_path("athenak/richtmyer_meshkov/rm_ppmx_rk4_512.npz"))
    arr = {f"ak_ppmx_rk4_{c}_N512_series_{k}": z[f"{c}_N512_{k}"] for c in ("heavy-light", "light-heavy")
           for k in ("tau", "psi", "exact")}
    return arr, [], []


def extra_exact(test):
    """Exact or reference profile of the 1D smooth tests and A9."""
    def f(root):
        if test == "a1":
            import tests.test_hydro_accuracy_linear_waves as lw
            U0, _ = lw.initial(1, 1024, "sound_r")
            return dict(exact_rho=np.asarray(U0)[0]), [], []
        if test == "a6":
            import tests.test_hydro_hard_isothermal_atmosphere as iso
            return dict(exact_rho=np.asarray(iso.steady(2048))[0]), [], []
        if test == "a9":
            r, _ = _ref_tube(reference_path("luna/woodward_colella/woodward_colella_mc.npz"))
            return dict(ref_rho=r), [], []
        if test == "a4":
            r = np.linspace(0, 0.5, 501)
            return dict(exact_r=r, exact_v_phi=np.where(r < 0.2, 5 * r, np.where(r < 0.4, 2 - 5 * r, 0.0))), [], []
        return {}, [], []
    return f


EXTRA = {"sod": extra_sod, "so": extra_so, "tt": extra_tt, "sedov": extra_sedov, "kh": extra_kh, "a5": extra_a5,
         "a8": extra_a8, "tg": extra_tg, "rm": extra_rm} | {t: extra_exact(t) for t in ("a1", "a6", "a9", "a4")}


if __name__ == "__main__":
    main(sys.argv[1:])
