"""AthenaK references from LUNA's own initial data (pgen luna_ic): set up the runs, then reduce them with the
LUNA tests' metric functions.

    python tools/athenak_refs.py prep TEST ROOT [scheme=ppmx_rk4] [cfl=0.3] [ns=32,64]
    python tools/athenak_refs.py reduce TEST ROOT [scheme=ppmx_rk4] [out=<luna-reference-solutions>/athenak/<problem>/<file>.npz]
    python tools/athenak_refs.py luna TEST OUTSTEM [ns=32,64]     # LUNA's four schemes, same metric (GPU job)

TEST: a1 (linear wave), so, tt (Shu-Osher, Titarev-Toro: prep, reduce), sod, sedov, kh (site pages; prep only), rm (A14), rt (A13), tg (A15), a3 (LW smooth), a4, a5 (McNally KH; chains of 75 runs),
a6 (isothermal atmosphere; the pgen potential of AthenaK branch luna_ic_a6), a7 (Jeans), a8 (Evrard), a9, a10
(implosion; prep also late=..., tag=n), a11 (KT 3, 6; prep ranks=4, reduce configs=3); `luna` also a1, tg, a5, a6, a7, a8, a10, a11; `table` also a5. prep writes ROOT/<run>/{ic/, run.athinput} and
ROOT/runs.txt (one run directory and its rank count per line); the job runs AthenaK in each; reduce writes
<TEST>_<scheme>.npz (+ .md) with the metrics, the run inputs and the provenance found in ROOT.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

from luna.reference_data import reference_root
from tools import luna_ic

ATHENAK = os.environ.get("ATHENAK_DIR", "")      # an AthenaK checkout with the luna_ic pgen (tools/athenak/)
# the binary the runs used: build_gpu_lunaic (886dd2a, no self-gravity) by default; A7/A8 ran
# ATHENAK_DIR=<checkout of branch luna_ic_mg> ATHENAK_EXE=<its build_gpu_lunaic_mg>/src/athena
ATHENAK_EXE = os.environ.get("ATHENAK_EXE", os.path.join(ATHENAK, "build_gpu_lunaic", "src", "athena") if ATHENAK else "")
SCHEMES = {"plm_rk2": ("plm", "rk2"), "ppmx_rk4": ("ppmx", "rk4"), "ppm4_rk4": ("ppm4", "rk4")}
FLOOR = None                                     # dfloor = pfloor for every run of a prep (prep ... floor=1e-8)


def _git(d):
    """(HEAD hash, dirty files) of a git tree; the references live in luna-reference-solutions."""
    h = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    st = subprocess.run(["git", "-C", d, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                        text=True).stdout.splitlines()
    return h, "\n".join(st).strip()


def _run(root, name, U, box, gamma, ranks=1, **kw):
    """One run directory: the IC and the AthenaK input; returns its line for runs.txt."""
    d = os.path.join(root, name)
    luna_ic.write_ic(U, os.path.join(d, "ic"), box, gamma, name)
    n = luna_ic.to_luna5(U).shape[1:]
    # AthenaK (CUDA) runs out of shared memory above 256 cells along x1 in a 2D MeshBlock (512 in 1D)
    kw.setdefault("block", (max(b for b in range(1, 257) if n[0] % b == 0),) + tuple(n[1:]))
    kw.setdefault("floor", FLOOR)
    luna_ic.athinput(os.path.join(d, "run.athinput"), n=n, box=box, gamma=gamma, ic_dir=os.path.join(d, "ic"),
                     **kw)
    return f"{d} {ranks}"


# ---------------------------------------------------------------- A1, linear sound wave (1D) as on viper
def prep_a1(root, scheme, cfl, ns=(32, 64, 128, 256, 512, 1024)):
    """Right-going sound wave, EPS 1e-6, one period, AthenaK's OutputErrors against the IC; MeshBlocks of at most
    512 cells (AthenaK's 1D kernels run out of shared memory at 1024), plus a 2-rank, 8-MeshBlock copy of the
    512 run (must give the 1-block run's errors)."""
    from tests.test_hydro_accuracy_linear_waves import EPS, G, initial
    recon, integ = SCHEMES[scheme]
    lines = []
    for n in ns:
        U0, period = initial(1, n, "sound_r", EPS)
        kw = dict(box=(1.0, 1.0, 1.0), gamma=G, bc=("periodic",) * 3, recon=recon, integrator=integ, cfl=cfl,
                  tlim=period, errors=True)
        lines.append(_run(root, f"a1_{scheme}_N{n}", U0, block=(min(n, 512), 1, 1), **kw))
        if n == 512:
            lines.append(_run(root, f"a1_{scheme}_N{n}_8mb", U0, ranks=2, block=(n // 8, 1, 1), **kw))
    return lines


def reduce_a1(root, scheme):
    """RMS over variables of the L1 error / EPS (AthenaK's metric), from errs.dat and from the final dump."""
    from tests.test_hydro_accuracy_linear_waves import EPS
    res = {}
    for d in sorted(glob.glob(os.path.join(root, f"a1_{scheme}_N*"))):
        name = os.path.basename(d).split("_", 3)[3]
        e = luna_ic.errs(os.path.join(d, "run-errs.dat"))
        U = np.load(luna_ic.final(d)[2])
        U0 = np.load(os.path.join(d, "ic", "U.npy"))
        l1 = np.abs(U - U0).reshape(5, -1).mean(1)
        res[name] = dict(errs_rms=e["RMS-L1"] / EPS, dump_rms=float(np.sqrt((l1 ** 2).sum())) / EPS,
                         ncycle=e["Ncycle"], L1_vars=l1 / EPS)
    return res


# ---------------------------------------------------------------- A14, Richtmyer-Meshkov
def prep_rm(root, scheme, cfl, ns=(32, 64, 128, 256, 512)):
    """Both cases, +psi0 and -psi0, rho dumped every DT_OUT to past the window's end."""
    import tests.test_hydro_hard_richtmyer_meshkov as rm
    recon, integ = SCHEMES[scheme]
    lines = []
    for case in rm.CASES:
        d = rm.setup(case)
        for n in ns:
            for sg, tag in ((1.0, "p"), (-1.0, "m")):
                U0 = rm.initial(case, n, sg)
                tl = (int(d["t_end"] / rm.DT_OUT) + 1) * rm.DT_OUT
                lines.append(_run(root, f"rm_{scheme}_{case}_N{n}_{tag}", U0, box=(d["L"], 0.5 * rm.LAM, 1.0),
                                  gamma=rm.G, bc=("reflect", "reflect", "periodic"), recon=recon, integrator=integ,
                                  cfl=cfl, tlim=tl, hst_dt=rm.DT_OUT, dump=True, dump_vars=1))
    return lines


def reduce_rm(root, scheme):
    """psi/psi0 (odd part) at the dumps' actual times, L1/L2/Linf against the exact curve over the window and
    the fitted-rate error, as test_hydro_hard_richtmyer_meshkov.errors."""
    import tests.test_hydro_hard_richtmyer_meshkov as rm
    from tools.rm_linear import _laplace_psi, psi_history
    res = {}
    for case in rm.CASES:
        d = rm.setup(case)
        kc = rm.K * d["s"]["c_trans_f"]
        for n in sorted({int(p.split("_N")[1].split("_")[0])
                         for p in glob.glob(os.path.join(root, f"rm_{scheme}_{case}_N*_p"))}):
            amp, tim = [], []
            for tag in ("p", "m"):
                rd = os.path.join(root, f"rm_{scheme}_{case}_N{n}_{tag}")
                ds = luna_ic.dumps(rd)[1:]                                     # dump 0 is the IC
                tim.append(np.array([t for t, _, _ in ds]))
                amp.append(np.array([rm.amplitude(case, t, np.load(p)[0, :, :, 0]) for t, _, p in ds]))
            m = min(len(tim[0]), len(tim[1]))
            dt_pm = float(np.abs(tim[0][:m] - tim[1][:m]).max())
            t = 0.5 * (tim[0][:m] + tim[1][:m])
            psi = (amp[0][:m] - amp[1][:m]) / (2 * rm.KPSI0 / rm.K)
            tau = kc * (t - d["t_hit"])
            w = (tau >= rm.TAU0) & (tau <= rm.TAU)
            assert np.isfinite(psi[w]).all(), f"{case} N={n}: interface lost inside the window"
            # exact by the Laplace route: rm_linear's Bessel series diverges for light-heavy beyond tau ~ 12 here
            # (scipy 1.16.3; agrees with Laplace to 8 digits where it converges, e.g. heavy-light throughout)
            ex = _laplace_psi(d["s"].get("raw", d["s"]), tau[w])[0]
            exb = psi_history(dict(d["s"]), tau[w])[0]
            e = psi[w] - ex
            rate_rel = np.polyfit(tau[w], psi[w], 1)[0] / np.polyfit(tau[w], ex, 1)[0] - 1
            res[f"{case}_N{n}"] = dict(L1=float(np.abs(e).mean()), L2=float(np.sqrt((e * e).mean())),
                                       Linf=float(np.abs(e).max()), rate=float(abs(rate_rel)),
                                       rate_signed=float(rate_rel), tau=tau, psi=psi, exact=ex, window=w,
                                       bessel_minus_laplace=float(np.abs(exb - ex).max()), t_pm_max=dt_pm,
                                       t_off_max=float(np.abs(t - np.round(t / rm.DT_OUT) * rm.DT_OUT).max()))
        done = [int(k.split("_N")[1]) for k in res if k.startswith(f"{case}_N")]
        if done:                                      # the test reads <case>_L1: the finest N
            res[case] = {k: v for k, v in res[f"{case}_N{max(done)}"].items() if np.ndim(v) == 0} | dict(N=max(done))
    return res


# ---------------------------------------------------------------- A13, Rayleigh-Taylor
def prep_rt(root, scheme, cfl, ns=(32, 64, 128, 256, 512)):
    """+AMP and -AMP eigenmode seeds, g = 0.1 in -y by <hydro> const_accel, to T."""
    import tests.test_hydro_hard_rayleigh_taylor as rt
    recon, integ = SCHEMES[scheme]
    lines = []
    for nx in ns:
        for s, tag in ((1.0, "p"), (-1.0, "m")):
            lines.append(_run(root, f"rt_{scheme}_N{nx}_{tag}", rt.initial(nx, 4 * nx, s), box=(rt.LX, 1.0, 1.0),
                              gamma=rt.GAM, bc=("reflect", "reflect", "periodic"), recon=recon, integrator=integ,
                              cfl=cfl, tlim=rt.T, accel=(2, -rt.G), block=(min(nx, 64), 64, 1)))   # small blocks: ~5x faster
    return lines


def reduce_rt(root, scheme):
    """test_hydro_hard_rayleigh_taylor.series on AthenaK's final states (its run() replaced)."""
    import tests.test_hydro_hard_rayleigh_taylor as rt

    def run(nx, sch, s, t=rt.T, sign=1.0):
        assert t == rt.T and sign == 1.0
        rd = os.path.join(root, f"rt_{scheme}_N{nx}_{'p' if s > 0 else 'm'}")
        tf, _, p = luna_ic.final(rd)
        assert abs(tf - rt.T) < 1e-12, (rd, tf)
        return luna_ic.read_dump(p, 2)

    rt.run = run
    res = {}
    for nx in sorted({int(p.split("_N")[1].split("_")[0])
                      for p in glob.glob(os.path.join(root, f"rt_{scheme}_N*_p", "run.luna_final.txt"))
                      if os.path.exists(p.replace("_p/", "_m/"))}):
        res[f"N{nx}"] = rt.series(nx, f"athenak-{scheme}")
    return res


# ---------------------------------------------------------------- A15, Taylor-Green effective Reynolds number
def prep_tg(root, scheme, cfl, ns=(32, 64, 128, 256)):
    """M = 0.1 TG to t = 20, kinetic energy from the history every DT; one GPU, 4 MeshBlocks along z from
    128^3."""
    import tests.test_hydro_hard_taylor_green_reynolds as tg
    from tests._tg_reference import luna_ic as tg_ic
    recon, integ = SCHEMES[scheme]
    lines = []
    for n in ns:
        nb = 1 if n <= 64 else 4
        lines.append(_run(root, f"tg_{scheme}_N{n}", tg_ic(n, tg.MACH), box=(tg.L,) * 3, gamma=tg.GAMMA,
                          bc=("periodic",) * 3, recon=recon, integrator=integ, cfl=cfl, tlim=tg.T_END,
                          hst_dt=tg.DT, block=(n, n, n // nb)))
    return lines


def reduce_tg(root, scheme):
    """K(t) per unit mass from the history, eps = -dK/dt on the window; Re_eff and D by the test's measure()
    when $LUNA_DATA/hydro/tg_viscous_spectral.npz is present."""
    import tests.test_hydro_hard_taylor_green_reynolds as tg
    res = {}
    for rd in sorted(glob.glob(os.path.join(root, f"tg_{scheme}_N*"))):
        n = int(rd.rsplit("_N", 1)[1])
        h = luna_ic.history(os.path.join(rd, "run.hydro.hst"))
        t = h["time"]
        K = (h["1-KE"] + h["2-KE"] + h["3-KE"]) / h["mass"]
        r = dict(t=t, K=K)
        if os.path.exists(tg.REF):
            win = tg.references()[1]
            eps = np.interp(win, t, -np.gradient(K, t))
            tg.dissipation = lambda n_, scheme_, order=1, e=eps: e      # noqa: E731
            r["Re_eff"], r["D"] = tg.measure(n, "athenak")
            r["eps_window"] = eps
        res[f"N{n}"] = r
    return res


# ---------------------------------------------------------------- group A phase 1: shared pieces
LUNA_SCHEMES = {"hllc-minmod": ("hllc", 0), "hll-minmod": ("hll", 0), "hllc-mc": ("hllc", 1), "hll-mc": ("hll", 1),
                "hllc-vl": ("hllc", 2)}          # vl: van Leer, AthenaK's only PLM slope (src/luna/hydro/slopes.py)


def _blk(n, cap=64):
    """Largest MeshBlock edge <= cap dividing n (2D: 64^2 blocks run ~5x faster than one block)."""
    return max(b for b in range(1, min(n, cap) + 1) if n % b == 0)


def _state(rd, sdim, t, U0=None):
    """AthenaK's final state in run dir rd as a LUNA state; checks the end time and, given U0, the IC bitwise."""
    tf, _, p = luna_ic.final(rd)
    assert abs(tf - t) <= 1e-12 * max(1.0, t), (rd, tf, t)
    if U0 is not None:
        ic = luna_ic.from_luna5(np.load(os.path.join(rd, "ic", "U.npy")), sdim)
        assert np.array_equal(ic, np.asarray(U0, np.float64)), f"{rd}: IC differs from the test's U0"
    return luna_ic.read_dump(p, sdim)


def _ns(root, pat):
    """Sorted N of the finished runs ROOT/pat ('{}' stands for N)."""
    pre = pat.split("{}")[0]
    return sorted({int(os.path.basename(os.path.dirname(p))[len(pre):].split("_")[0].split("x")[0])
                   for p in glob.glob(os.path.join(root, pat.format("*"), "run.luna_final.txt"))})


# ---------------------------------------------------------------- A3, Liska-Wendroff smooth advection
def prep_a3(root, scheme, cfl, ns=(32, 64, 128, 200, 256, 512, 1024)):
    """rho = 1 + 0.2 sin(pi(x + y - t(u + v))) on [0, 2]^2 to T = 4 from the test's initial(); 200 is LW's grid."""
    import tests.test_hydro_accuracy_lw_smooth as lws
    recon, integ = SCHEMES[scheme]
    return [_run(root, f"a3_{scheme}_N{n}", lws.initial(n), box=(lws.L, lws.L, 1.0), gamma=lws.G,
                 bc=("periodic",) * 3, recon=recon, integrator=integ, cfl=cfl, tlim=lws.T,
                 block=(_blk(n), _blk(n), 1)) for n in ns]


def reduce_a3(root, scheme):
    """test_hydro_accuracy_lw_smooth.errors on AthenaK's final states: LW rel. L1 %, L1, L2, Linf of rho."""
    import tests.test_hydro_accuracy_lw_smooth as lws
    lws.evolve = lambda cfg, U0, t: (_state(os.path.join(root, f"a3_{scheme}_N{cfg.Nx}"), 2, t, U0), 0)
    res = {}
    for n in _ns(root, f"a3_{scheme}_N{{}}"):
        rel, l1, l2, li = lws.errors(n, "hllc-mc")             # the scheme name only builds LUNA's cfg here
        res[f"N{n}"] = dict(relL1_pct=rel, L1=l1, L2=l2, Linf=li)
        if n == 64:                                            # one final field, for a re-reduction
            res[f"N{n}"]["rho"] = _state(os.path.join(root, f"a3_{scheme}_N{n}"), 2, lws.T)[0]
    return res


def luna_a3(ns=(32, 64, 128, 200, 256, 512, 1024), schemes=None):
    """LUNA's schemes (or the "+" list `schemes`), the test's errors()."""
    import tests.test_hydro_accuracy_lw_smooth as lws
    lws.SCHEMES.update(LUNA_SCHEMES)
    return {f"{sc}_N{n}": dict(zip(("relL1_pct", "L1", "L2", "Linf"), lws.errors(n, sc)))
            for sc in (schemes.split("+") if schemes else LUNA_SCHEMES) for n in ns}


# ---------------------------------------------------------------- A4, Gresho vortex (LW p(0) = 5)
A4_LW = ((40, 40, False), (160, 40, True))       # LW's grids, transmissive edges, T = 3


def prep_a4(root, scheme, cfl, ns=(32, 64, 128, 256, 512, 1024)):
    """Static and moving vortex, periodic, one revolution (the test's periodic()); LW's 40^2 static and 160 x 40
    moving runs to T = 3 with outflow edges (its lw_metrics())."""
    import tests.test_hydro_hard_lw_gresho as gr
    recon, integ = SCHEMES[scheme]
    kw = dict(gamma=gr.G, recon=recon, integrator=integ, cfl=cfl)
    lines = []
    for n in ns:
        for mv in (False, True):
            U0 = gr._boost(gr._ic(n, n, 1.0, 0), 1.0 / gr.REV if mv else 0.0)
            lines.append(_run(root, f"a4_{scheme}_N{n}_{'m' if mv else 's'}", U0, box=(1.0, 1.0, 1.0),
                              bc=("periodic",) * 3, tlim=gr.REV, block=(_blk(n), _blk(n), 1), **kw))
    for nx, ny, mv in A4_LW:
        U0 = gr._boost(gr._ic(nx, ny, nx / ny, 1), 1.0 if mv else 0.0)
        lines.append(_run(root, f"a4_{scheme}_N{nx}x{ny}_lw", U0, box=(nx / ny, 1.0, 1.0),
                          bc=("outflow", "outflow", "periodic"), tlim=3.0, block=(_blk(nx), _blk(ny), 1), **kw))
    return lines


def _a4_rows(r, fields, norms):
    return {f"{f}_{nm}": r[f][j] for f in fields for j, nm in enumerate(norms)}


def reduce_a4(root, scheme):
    """periodic() and lw_metrics() of test_hydro_hard_lw_gresho on AthenaK's final states."""
    import tests.test_hydro_hard_lw_gresho as gr

    def evolve(cfg, U0, t):
        if cfg.BOUNDARY == 1:
            name = f"a4_{scheme}_N{cfg.Nx}x{cfg.Ny}_lw"
        else:                                                  # the moving vortex carries net x momentum
            mv = abs(float(np.asarray(U0)[1].sum())) > 1e-6 * float(np.asarray(U0)[0].sum())
            name = f"a4_{scheme}_N{cfg.Nx}_{'m' if mv else 's'}"
        return _state(os.path.join(root, name), 2, t, U0), 0

    gr.evolve = evolve
    res = {}
    for n in _ns(root, f"a4_{scheme}_N{{}}_s"):
        for mv in (False, True):
            res[f"{'moving' if mv else 'static'}_N{n}"] = _a4_rows(gr.periodic(n, "hllc-mc", mv), gr.FIELDS,
                                                                   gr.NORMS)
    for nx, ny, mv in A4_LW:
        if os.path.exists(os.path.join(root, f"a4_{scheme}_N{nx}x{ny}_lw", "run.luna_final.txt")):
            res[f"lw{nx}x{ny}"] = dict(zip(("vort_relL1_pct", "rho_L1_pct", "tke_rel_pct"),
                                           gr.lw_metrics(nx, ny, "hllc-mc", mv)))
    return res


def luna_a4(ns=(32, 64, 128, 256, 512, 1024), schemes=None):
    """LUNA, the four schemes (or the "+" list `schemes`), the test's periodic() and lw_metrics() on LW's grids."""
    import tests.test_hydro_hard_lw_gresho as gr
    gr.SCHEMES.update(LUNA_SCHEMES)
    res = {}
    for sc in (schemes.split("+") if schemes else LUNA_SCHEMES):
        for n in ns:
            for mv in (False, True):
                res[f"{sc}_{'moving' if mv else 'static'}_N{n}"] = _a4_rows(gr.periodic(n, sc, mv), gr.FIELDS,
                                                                            gr.NORMS)
        for nx, ny, mv in A4_LW:
            res[f"{sc}_lw{nx}x{ny}"] = dict(zip(("vort_relL1_pct", "rho_L1_pct", "tke_rel_pct"),
                                                gr.lw_metrics(nx, ny, sc, mv)))
    return res


# ---------------------------------------------------------------- A9, odd-even decoupling (WC blast on a strip)
A9_REF = 16384                                   # 1D reference, 8x LUNA's finest strip (as SO/TT at 16384)


def prep_a9(root, scheme, cfl, ns=(256, 512, 1024, 2048)):
    """The test's y-uniform WC strip (N x 16, square cells, reflecting x and y as its BOUNDARY=2; at the test's
    N also LW's periodic y) and the 1D run from the same IC at every N, plus the 1D reference at A9_REF."""
    import tests.test_hydro_hard_odd_even as oe
    recon, integ = SCHEMES[scheme]
    kw = dict(gamma=1.4, recon=recon, integrator=integ, cfl=cfl, tlim=oe.T)
    lines = []
    for n in tuple(ns) + (A9_REF,):
        _, _, U1, U2 = oe.strip(n, oe.NY, "hllc")
        lines.append(_run(root, f"a9_{scheme}_N{n}_1d", np.asarray(U1), box=(1.0, 1.0, 1.0),
                          bc=("reflect",) * 3, block=(min(n, 512), 1, 1), **kw))
        if n == A9_REF:
            continue
        for ybc in ("reflect", "periodic") if n == oe.N else ("reflect",):
            lines.append(_run(root, f"a9_{scheme}_N{n}_strip{ybc[0]}", U2, box=(1.0, oe.NY / n, 1.0),
                              bc=("reflect", ybc, "periodic"), block=(min(n, 256), oe.NY, 1), **kw))
    return lines


A9_REFS = {"luna": ("luna/woodward_colella/woodward_colella_mc.npz", "rho"),               # LUNA MC 2^16, primary
           "akplm": ("athenak/woodward_colella/a9_plm_rk2_16384.npz", "ref_rho")}     # AthenaK PLM 16384


def _a9_refs():
    """{name: 1D reference rho} of A9_REFS found in luna-reference-solutions."""
    lr = str(reference_root())
    out = {}
    for name, (path, key) in A9_REFS.items():
        p = os.path.join(lr, path)
        if os.path.exists(p) and key in np.load(p).files:
            out[name] = np.asarray(np.load(p)[key], np.float64)
    return out


def _avg(ref, n):
    return np.asarray(ref).reshape(n, -1).mean(1)


def _a9_metrics(U2, U1, refs):
    """The test's three checks (max |v_y|, row spread, max |strip - 1D|) and L1(rho) of strip row 0 and of the
    1D run against each 1D reference averaged onto the run's cells (`L1_rho_strip` against 'luna', the primary;
    `L1_rho_strip_<name>` against the others)."""
    U2, U1 = np.asarray(U2, np.float64), np.asarray(U1, np.float64)
    out = dict(vy_max=float(np.abs(U2[2] / U2[0]).max()), row_spread=float(np.abs(U2 - U2[:, :, :1]).max()),
               strip_minus_1d=float(np.abs(U2[[0, 1, 3], :, 0] - U1).max()))
    for name, ref in refs.items():
        r, sfx = _avg(ref, U1.shape[1]), "" if name == "luna" else f"_{name}"
        out |= {f"L1_rho_strip{sfx}": float(np.abs(U2[0, :, 0] - r).mean()),
                f"L1_rho_1d{sfx}": float(np.abs(U1[0] - r).mean())}
    return out


def reduce_a9(root, scheme):
    """The odd-even checks on AthenaK's strip against AthenaK's own 1D run (its own dt), L1(rho) against the
    references (A9_REFS; for plm_rk2 its own run at A9_REF is 'akplm'); stores this scheme's run at A9_REF
    (`ref_rho`) and its L1 against the LUNA 2^16 reference at 1024 and 8192 cells."""
    import tests.test_hydro_hard_odd_even as oe

    def nonfinite(rd):
        """Fraction of non-finite values in the final dump (AthenaK runs on to tlim through NaN)."""
        return float(1.0 - np.isfinite(np.load(luna_ic.final(rd)[2])).mean())

    rd = os.path.join(root, f"a9_{scheme}_N{A9_REF}_1d")
    ref = None
    res = {}
    if os.path.exists(os.path.join(rd, "run.luna_final.txt")):
        if nonfinite(rd) == 0.0:
            ref = _state(rd, 1, oe.T)[0]
        res["ref"] = dict(N=A9_REF, nonfinite=nonfinite(rd)) | ({"rho": ref} if ref is not None else {})
    refs = _a9_refs()
    if scheme == "plm_rk2" and ref is not None:
        refs["akplm"] = ref
    if ref is not None and "luna" in refs:
        res["ref"] |= {f"L1_vs_luna_at{m}": float(np.abs(_avg(ref, m) - _avg(refs["luna"], m)).mean())
                       for m in (1024, 8192)}
    for n in _ns(root, f"a9_{scheme}_N{{}}_1d"):
        if n == A9_REF:
            continue
        d1 = os.path.join(root, f"a9_{scheme}_N{n}_1d")
        for ybc in ("r", "p"):
            d2 = os.path.join(root, f"a9_{scheme}_N{n}_strip{ybc}")
            if os.path.exists(os.path.join(d2, "run.luna_final.txt")):
                bad = dict(nonfinite_strip=nonfinite(d2), nonfinite_1d=nonfinite(d1),
                           ncycle_strip=luna_ic.final(d2)[1], ncycle_1d=luna_ic.final(d1)[1])
                if bad["nonfinite_strip"] or bad["nonfinite_1d"]:
                    res[f"N{n}_{ybc}"] = bad                     # no metric on a NaN state: reported as such
                else:
                    res[f"N{n}_{ybc}"] = _a9_metrics(_state(d2, 2, oe.T), _state(d1, 1, oe.T), refs) | bad
    return res


def luna_a9(ns=(256, 512, 1024, 2048), schemes=None):
    """LUNA's schemes (or the "+" list `schemes`): the test's equal-dt strip / 1D loop; L1(rho) against A9_REFS."""
    import tests.test_hydro_hard_odd_even as oe
    from tests._helpers import _make_step_fns, accuracy_cfg
    refs = _a9_refs()
    res = {}
    for sc in (schemes.split("+") if schemes else LUNA_SCHEMES):
        solver, lim = LUNA_SCHEMES[sc]
        for n in ns:
            _, _, U1, U2 = oe.strip(n, oe.NY, solver)
            cfg = accuracy_cfg(test_problem="Woodward-Colella", SPATIAL_DIMENSION=2, Nx=n, Ny=oe.NY, Lx=1.0,
                               Ly=oe.NY / n, BOUNDARY=2, gamma=1.4, riemann_solver=solver, LIMITER=lim)
            one = accuracy_cfg(test_problem="Woodward-Colella", SPATIAL_DIMENSION=1, Nx=n, Lx=1.0, BOUNDARY=2,
                               gamma=1.4, riemann_solver=solver, LIMITER=lim)
            (dt1, step1), (dt2, step2) = _make_step_fns(one), _make_step_fns(cfg)
            t, k = 0.0, 0
            while t < oe.T * (1 - 1e-14):
                dt = min(float(dt1(U1)), float(dt2(U2)), oe.T - t)
                U1, U2, t, k = step1(U1, dt), step2(U2, dt), t + dt, k + 1
            res[f"{sc}_N{n}"] = _a9_metrics(U2, U1, refs) | dict(steps=k, rho=np.asarray(U2)[0, :, 0])
    return res


# ---------------------------------------------------------------- A11, Kurganov-Tadmor configurations 3 and 6
A11_TB = {6: (0.25, (256, 512))}                 # config 6's wall budget at t = 0.25 (the test's T_BUDGET, CPU_NS)
A11_AKREF = "athenak/kt_riemann/a11_c{}_ppmx_rk4_8192.npz"   # AthenaK PPMX 8192^2, the second reference


def _a11_ic(config, n):
    """The registered IC (jumps on faces) on n^2, FP64, and its cfg."""
    from luna.ic_generation import kt_riemann_2d as ktic
    from tests._helpers import accuracy_cfg
    cfg = accuracy_cfg(test_problem="KT-Riemann-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0, BOUNDARY=1,
                       gamma=ktic.GAMMA, problem=ktic.Params(config=config))
    return np.asarray(ktic.kt_riemann_2d(cfg), np.float64)


def prep_a11(root, scheme, cfl, ns=(256, 512, 1024, 2048), ranks=1):
    """Both configurations from the registered IC, outflow (zero-gradient) walls as the test's BOUNDARY=1, to
    T = 0.3, 64^2 MeshBlocks, `ranks` GPUs from 4096 on; config 6 also to 0.25 at 256, 512 (wall budget). Pass
    floor=1e-8 (FLT_MIN floors NaN PPMX at shocks)."""
    from luna.ic_generation import kt_riemann_2d as ktic
    recon, integ = SCHEMES[scheme]
    kw = dict(box=(1.0, 1.0, 1.0), gamma=ktic.GAMMA, bc=("outflow", "outflow", "periodic"), recon=recon,
              integrator=integ, cfl=cfl)
    lines = []
    for config in (3, 6):
        for n in ns:
            U0, blk = _a11_ic(config, n), (_blk(n), _blk(n), 1)
            lines.append(_run(root, f"a11_{scheme}_c{config}_N{n}", U0, ranks=ranks if n >= 4096 else 1,
                              tlim=ktic.T_END, block=blk, **kw))
            tb, tb_ns = A11_TB.get(config, (None, ()))
            if n in tb_ns:
                lines.append(_run(root, f"a11_{scheme}_c{config}_N{n}_tb", U0, tlim=tb, block=blk, **kw))
    return lines


def _a11_vs(rho, ref, config, m=128):
    """(L1, L1 inside the test's CORE, L1 outside) of rho - ref, both block-averaged onto m^2."""
    import tests.test_hydro_hard_kt_riemann as kt11
    from tools.kt_riemann import block_mean
    a = np.abs(block_mean(np.asarray(rho, np.float64), m) - block_mean(np.asarray(ref, np.float64), m))
    x = (np.arange(m) + 0.5) / m
    lo, hi = kt11.CORE[config]
    inside = ((x > lo) & (x < hi))[:, None] & ((x > lo) & (x < hi))[None, :]
    return float(a.mean()), float((a * inside).sum() / m / m), float((a * ~inside).sum() / m / m)


def _a11_akref(config):
    """rho on 256^2 of the AthenaK PPMX 8192^2 run (A11_AKREF), or None."""
    p = os.path.join(str(reference_root()), A11_AKREF.format(config))
    return np.asarray(np.load(p)[f"c{config}_ref_rho256"], np.float64) if os.path.exists(p) else None


def _a11_metrics(U, U0, config, U_tb=None):
    """The test's metrics on one end state: L1, L2, L1 core / outside (M = 128; L1 also on 64^2, 256^2) against
    the stored LUNA reference, the same against the AthenaK reference (`_ak`), the strips against exact 1D,
    config 3's asymmetry (max and L1 of rho - rho^T), config 6's pressure strips, the wall budget at T (config 3) or from U_tb (config 6)."""
    import tests.test_hydro_hard_kt_riemann as kt11
    rho = np.asarray(U[0], np.float64)
    l1, l2, core, out = kt11.ref_errors(rho, config)
    r = dict(L1=l1, L2=l2, L1_core=core, L1_out=out, L1_M64=kt11.ref_errors(rho, config, 64)[0],
             L1_M256=kt11.ref_errors(rho, config, 256)[0])
    ak = _a11_akref(config)
    if ak is not None:
        r |= dict(zip(("L1_ak", "L1_core_ak", "L1_out_ak"), _a11_vs(rho, ak, config)))
    for name, (sl1, cells) in kt11.strip_errors(rho, config).items():
        r |= {f"strip{name}_L1": sl1, f"strip{name}_front_cells": cells}
    if config == 3:
        r |= dict(asym=kt11.asymmetry(U), asym_rho_L1=float(np.abs(rho - rho.T).mean()),
                  budget=float(kt11.budget_errors(U0, U, 3, kt11.T).max()))
    if config == 6:                              # p is constant across the slip lines: the test's pressure strips
        r |= {f"pstrip{name}_L1": v for name, v in kt11.pressure_strips(U).items()}
    if U_tb is not None:
        r["budget"] = float(kt11.budget_errors(U0, U_tb, config, kt11.T_BUDGET[config]).max())
    return r


def reduce_a11(root, scheme, configs=(3, 6)):
    """_a11_metrics on AthenaK's final states (IC and end time checked), step counts and wall times; rho on 256^2
    of the finest run (`ref_rho256`) and, for two runs >= 4096, their self-difference on 64^2, 128^2, 256^2."""
    import tests.test_hydro_hard_kt_riemann as kt11
    from tools.kt_riemann import block_mean
    res = {}
    for config in configs:
        ns = _ns(root, f"a11_{scheme}_c{config}_N{{}}")
        big = {}
        for n in ns:
            rd = os.path.join(root, f"a11_{scheme}_c{config}_N{n}")
            U0 = _a11_ic(config, n)
            U = _state(rd, 2, kt11.T, U0)
            tbd = rd + "_tb"
            U_tb = (_state(tbd, 2, kt11.T_BUDGET[config], U0)
                    if os.path.exists(os.path.join(tbd, "run.luna_final.txt")) else None)
            jj = os.path.join(rd, "job.json")
            r = _a11_metrics(U, U0, config, U_tb) | dict(
                ncycle=luna_ic.final(rd)[1], wall_s=json.load(open(jj))["wall_s"] if os.path.exists(jj) else -1)
            if n >= 4096:
                big[n] = U[0]
            res[f"c{config}_N{n}"] = r
            if n == ns[-1]:
                res[f"c{config}_ref"] = dict(N=n, rho256=block_mean(np.asarray(U[0]), 256).astype(np.float32))
            del U
        if len(big) >= 2:
            a, b = sorted(big)[-2:]
            res[f"c{config}_ref"] |= {f"self_L1_M{m}_{a}_{b}": float(np.abs(block_mean(big[a], m)
                                                                          - block_mean(big[b], m)).mean())
                                      for m in (64, 128, 256)}
            res[f"c{config}_ref"] |= dict(zip((f"self_L1_{a}_{b}", f"self_L1_core_{a}_{b}",
                                               f"self_L1_out_{a}_{b}"), _a11_vs(big[a], big[b], config)))
        if ns:
            ref = res[f"c{config}_ref"]["rho256"]
            luna = kt11.reference(config)[0]
            res[f"c{config}_ref"] |= {f"vs_luna_L1_M{m}": float(np.abs(block_mean(ref, m) - block_mean(luna, m))
                                                                  .mean()) for m in (64, 128, 256)}
            res[f"c{config}_ref"] |= dict(zip(("vs_luna_L1", "vs_luna_L1_core", "vs_luna_L1_out"),
                                              _a11_vs(ref, luna, config)))
    return res


def luna_a11(ns=(256, 512, 1024, 2048), schemes="minmod+mc+vl", cfl=None, configs="3+6"):
    """LUNA (HLLC; minmod, MC, van Leer = AthenaK's PLM slope) by the test's cfg, at its CFL or `cfl`:
    _a11_metrics, steps, wall time; config 6 also to 0.25 at 256, 512 for the wall budget."""
    import time

    import tests.test_hydro_hard_kt_riemann as kt11
    from luna.ic_generation import kt_riemann_2d as ktic
    from tests._helpers import accuracy_cfg, evolve
    lim = dict(minmod=0, mc=1, vl=2)
    res = {}
    for config in (int(c) for c in str(configs).split("+")):
        for sc in schemes.split("+"):
            for n in ns:
                cfg = accuracy_cfg(test_problem="KT-Riemann-2D", SPATIAL_DIMENSION=2, Nx=n, Ny=n, Lx=1.0, Ly=1.0,
                                   BOUNDARY=1, gamma=ktic.GAMMA, LIMITER=lim[sc],
                                   problem=ktic.Params(config=config), **({"CFL": float(cfl)} if cfl else {}))
                U0 = np.asarray(ktic.kt_riemann_2d(cfg), np.float64)
                t0 = time.time()
                U, steps = evolve(cfg, U0, kt11.T)
                wall = time.time() - t0
                tb, tb_ns = A11_TB.get(config, (None, ()))
                U_tb = evolve(cfg, U0, tb)[0] if n in tb_ns else None
                res[f"c{config}_hllc-{sc}_N{n}"] = _a11_metrics(U, U0, config, U_tb) | dict(steps=steps, wall_s=wall)
    return res


# ---------------------------------------------------------------- A10, Hui / Liska-Wendroff implosion
A10_REF = (3200, 6400)                           # AthenaK's own early references (PPMX), checked against each other
A10_LATE = (128, 256, 400, 512, 1024)            # t = 2.5: LW's 400^2 and the test's late ladder
A10_REF_KEY = ("athenak/lw_implosion/a10_ppmx_rk4_6400.npz", "ref_rho")


def prep_a10(root, scheme, cfl, ns=(100, 200, 400, 800), late=A10_LATE, tag="", jet=""):
    """The test's exact cell-average IC (its initial()), reflecting x1 and x2, to T_EARLY at ns (A10_REF too if
    listed) and to T_LATE at `late`; 64^2 MeshBlocks where 64 divides N (else the largest divisor below). Run names
    end in _<tag>e / _<tag>l: tag 'n' for runs of the -fmad=false build (run_ak.sh AKBIN=...). jet=1: only the
    test's jet runs instead, 400^2 to each of its JET_TIMES (_<tag>j<100 t>)."""
    import tests.test_hydro_hard_implosion as imp
    recon, integ = SCHEMES[scheme]
    kw = dict(box=(imp.L, imp.L, 1.0), gamma=imp.G, bc=("reflect", "reflect", "periodic"), recon=recon,
              integrator=integ, cfl=cfl)
    if jet:
        return [_run(root, f"a10_{scheme}_N{imp.NS_SYM}_{tag}j{round(100 * t)}", imp.initial(imp.NS_SYM), tlim=t,
                     block=(_blk(imp.NS_SYM),) * 2 + (1,), **kw) for t in imp.JET_TIMES]
    lines = [_run(root, f"a10_{scheme}_N{n}_{tag}e", imp.initial(n), tlim=imp.T_EARLY, block=(_blk(n), _blk(n), 1),
                  **kw) for n in ns]
    return lines + [_run(root, f"a10_{scheme}_N{n}_{tag}l", imp.initial(n), tlim=imp.T_LATE,
                         block=(_blk(n), _blk(n), 1), **kw) for n in late]


def _asym3(U):
    """(max|rho - rho^T|, max|m_x - m_y^T|, max|E - E^T|) of a 2D state."""
    T = np.transpose
    return (float(np.abs(U[0] - T(U[0])).max()), float(np.abs(U[1] - T(U[2])).max()),
            float(np.abs(U[3] - T(U[3])).max()))


def _a10_metrics(U, U0, n, t, refs):
    """Symmetry, |dM|/M, |dE|/E; at T_EARLY L1/L2(rho) against each reference block-averaged onto n (the test's
    early_errors for 'luna'), at T_LATE s_jet and the test's late L1 (LUNA 2048^2 reference)."""
    import tests.test_hydro_hard_implosion as imp
    U, U0 = np.asarray(U, np.float64), np.asarray(U0, np.float64)
    sr, sm, se = _asym3(U)
    out = dict(asym_rho=sr, asym_m=sm, asym_E=se, dM=float(abs(U[0].sum() / U0[0].sum() - 1)),
               dE=float(abs(U[3].sum() / U0[3].sum() - 1)))
    if t == imp.T_EARLY:
        for name, ref in refs.items():
            if ref.shape[0] % n == 0:
                e = U[0] - imp.block(ref, n)
                sfx = "" if name == "luna" else f"_{name}"
                out |= {f"L1{sfx}": float(np.abs(e).mean()), f"L2{sfx}": float(np.sqrt((e * e).mean()))}
    else:
        m = min(n, 256)
        out["s_jet"] = imp.s_jet(U[0])
        if n % m == 0:                                # 400^2 does not block onto the 256^2 late reference
            out["late_L1"] = float(np.abs(imp.block(U[0], m) - imp.block(imp.reference()["late"], m)).mean())
    return out


def _a10_refs():
    """{'luna': the test's 6400^2 reference (800^2), 'akppmx': AthenaK PPMX 6400^2 (800^2) if stored}."""
    import tests.test_hydro_hard_implosion as imp
    from tools.implosion_reference import untri
    out = {"luna": imp.reference()["early"]}
    p = os.path.join(str(reference_root()), A10_REF_KEY[0])
    if os.path.exists(p) and A10_REF_KEY[1] in np.load(p).files:
        out["akppmx"] = untri(np.asarray(np.load(p)[A10_REF_KEY[1]], np.float64))
    return out


def _cost(rd):
    """AthenaK's cost of a run: zone-cycles per second of its loop (its log), the loop time n^2 ncycle / that,
    and the job's wall time for the run (start-up and I/O included)."""
    zc = [ln for ln in open(os.path.join(rd, "run.log")) if ln.startswith("zone-cycles/cpu_second")]
    t, nc, p = luna_ic.final(rd)
    cells = int(np.prod(np.load(p, mmap_mode="r").shape[1:]))
    out = dict(zcps=float(zc[-1].split("=")[1])) if zc else {}
    if zc:
        out["loop_s"] = cells * nc / out["zcps"]
    j = os.path.join(rd, "job.json")
    if os.path.exists(j) and "wall_s" in json.load(open(j)):
        out["wall_s"] = float(json.load(open(j))["wall_s"])
    return out


def reduce_a10(root, scheme):
    """The implosion metrics on AthenaK's final states (IC and end time checked). Early runs at A10_REF give this
    scheme's own reference: block averages onto 800^2, their mutual L1 and L1 against LUNA's reference, stored as
    the symmetrised lower triangle (float32, `ref_rho`) with its asymmetry."""
    import tests.test_hydro_hard_implosion as imp
    from tools.implosion_reference import tri
    refs = _a10_refs()
    res, fine = {}, {}
    rtag = "ne" if any(n in A10_REF for n in _ns(root, f"a10_{scheme}_N{{}}_ne")) else "e"
    for n in _ns(root, f"a10_{scheme}_N{{}}_{rtag}"):
        if n in A10_REF:
            rd = os.path.join(root, f"a10_{scheme}_N{n}_{rtag}")
            U0 = imp.initial(n)
            U = _state(rd, 2, imp.T_EARLY, U0)
            fine[n] = imp.block(U[0], 800)
            res[f"N{n}_{rtag}"] = _a10_metrics(U, U0, n, imp.T_EARLY, {}) | dict(ncycle=luna_ic.final(rd)[1]) | _cost(rd)
            del U, U0
    if fine:
        top = max(fine)
        rf = fine[top]
        res["ref"] = dict(N=top, nofma=float(rtag == "ne"), asym800=float(np.abs(rf - rf.T).max()),
                          L1_vs_luna=float(np.abs(rf - refs["luna"]).mean()),
                          L2_vs_luna=float(np.sqrt(((rf - refs["luna"]) ** 2).mean())),
                          rho=tri(0.5 * (rf + rf.T)).astype(np.float32))
        if len(fine) > 1:
            lo = min(fine)
            res["ref"] |= dict(self_N=lo, L1_self=float(np.abs(rf - fine[lo]).mean()),
                               L2_self=float(np.sqrt(((rf - fine[lo]) ** 2).mean())))
        refs["akppmx" if scheme == "ppmx_rk4" else f"ak{scheme}"] = rf
    for tag, t in (("e", imp.T_EARLY), ("l", imp.T_LATE), ("ne", imp.T_EARLY), ("nl", imp.T_LATE)):
        for n in _ns(root, f"a10_{scheme}_N{{}}_{tag}"):
            if tag == rtag and n in A10_REF:
                continue
            rd = os.path.join(root, f"a10_{scheme}_N{n}_{tag}")
            U0 = imp.initial(n)
            res[f"N{n}_{tag}"] = _a10_metrics(_state(rd, 2, t, U0), U0, n, t, refs) | dict(
                ncycle=luna_ic.final(rd)[1]) | _cost(rd)
    for tag in ("", "n"):                             # the test's jet metric: median of s_jet over JET_TIMES
        rds = [os.path.join(root, f"a10_{scheme}_N{imp.NS_SYM}_{tag}j{round(100 * t)}") for t in imp.JET_TIMES]
        if all(os.path.exists(os.path.join(rd, "run.luna_final.txt")) for rd in rds):
            U0 = imp.initial(imp.NS_SYM)
            st = [_state(rd, 2, t, U0) for rd, t in zip(rds, imp.JET_TIMES)]
            r = {f"s_jet_t{round(100 * t)}": imp.s_jet(U[0]) for U, t in zip(st, imp.JET_TIMES)}
            r |= dict(median=float(np.median(list(r.values()))), asym_max=max(_asym3(U)[0] for U in st))
            res[f"N{imp.NS_SYM}_{tag}jet"] = r
    # the test's own early_errors / late_l1 on AthenaK's states: must give the numbers above
    imp.run = lambda n, sc, times, *a, **k: [                         # noqa: E731
        _state(os.path.join(root, f"a10_{scheme}_N{n}_{'e' if len(times) == 1 else 'l'}"), 2, times[-1])] * len(times)
    for n in _ns(root, f"a10_{scheme}_N{{}}_e"):
        if n in imp.NS_EARLY:
            assert imp.early_errors(n, "hllc-mc") == (res[f"N{n}_e"]["L1"], res[f"N{n}_e"]["L2"]), n
    for n in (n for n in _ns(root, f"a10_{scheme}_N{{}}_l") if n in imp.NS_LATE):
        assert imp.late_l1(n, "hllc-mc") == res[f"N{n}_l"]["late_L1"], n
    return res


def luna_a10(ns=(100, 200, 400, 800), late=A10_LATE, schemes="hllc-minmod+hllc-mc+hllc-vl", cfl=None, timed=""):
    """LUNA's schemes, the metrics of reduce_a10 (at CFL `cfl` if given, e.g. AthenaK's step count). With timed=1
    each early run is timed: `wall_s` of the run (compile included), then `wall2_s` of the same run again and
    `wall1_s` of a one-step run, so loop_s = wall2_s - wall1_s and zcps = n^2 steps / loop_s."""
    import time

    import tests.test_hydro_hard_implosion as imp
    from tests._helpers import evolve
    refs = _a10_refs()
    kw = dict(CFL=float(cfl)) if cfl else {}
    res = {}
    for sc in schemes.split("+"):
        for tag, t, nn in (("e", imp.T_EARLY, ns), ("l", imp.T_LATE, late)):
            for n in nn:
                U0 = imp.initial(n)
                w = time.time()
                U, steps = evolve(imp.cfg_for(n, sc, **kw), U0, t)
                U = np.asarray(U)
                res[f"{sc}_N{n}_{tag}"] = _a10_metrics(U, U0, n, t, refs) | dict(steps=steps, wall_s=time.time() - w)
                if timed and tag == "e":
                    tm = {}
                    for name, t1 in (("wall2_s", t), ("wall1_s", 1e-9)):
                        w = time.time()
                        np.asarray(evolve(imp.cfg_for(n, sc, **kw), U0, t1)[0])
                        tm[name] = time.time() - w
                    tm["loop_s"] = tm["wall2_s"] - tm["wall1_s"]
                    res[f"{sc}_N{n}_{tag}"] |= tm | dict(zcps=n * n * steps / tm["loop_s"])
                if tag == "e" and n == 800:
                    res[f"{sc}_N{n}_{tag}"]["rho"] = np.asarray(U)[0]
    return res


# ---------------------------------------------------------------- A7, A8: self-gravity (AthenaK multigrid)
def _luna_cfg(mod, limiter=None, cfl=None, energy=None):
    """Point a test module's accuracy_cfg at LIMITER / CFL / gravity_energy overrides (van Leer is LIMITER 2);
    returns the original."""
    from tests._helpers import accuracy_cfg
    over = {k: v for k, v in (("LIMITER", limiter), ("CFL", cfl), ("gravity_energy", energy)) if v is not None}
    mod.accuracy_cfg = lambda **kw: accuracy_cfg(**{**kw, **over})
    return accuracy_cfg


def _luna_scheme(sc):
    """'hllc-vl@0.4' -> (the test's scheme name, LIMITER override, CFL override)."""
    name, _, cfl = sc.partition("@")
    return ("hllc-mc", 2, float(cfl) if cfl else None) if name == "hllc-vl" else (name, None,
                                                                                    float(cfl) if cfl else None)


A7_CASES = ((0.5, "stable"), (1.5, "unstable"))


def _a7_t_end(nj):
    import tests.test_gravity_jeans_wave as jw
    w2 = jw.omega2(nj)
    return float(2.0 * np.pi / np.sqrt(w2) if w2 > 0 else 1.0 / np.sqrt(-w2))


def prep_a7(root, scheme, cfl, ns=(32, 64, 128)):
    """Both Jeans cases from the test's initial(), periodic N x N/2 x N/2, two (N/2)^3 MeshBlocks, four_pi_G from
    the test's cfg_for, to one period (stable) or one e-fold (unstable)."""
    import tests.test_gravity_jeans_wave as jw
    recon, integ = SCHEMES[scheme]
    lines = []
    for nj, case in A7_CASES:
        for n in ns:
            G = float(jw.cfg_for(n, nj, "hllc-minmod").GRAVITATIONAL_G)
            lines.append(_run(root, f"a7_{scheme}_{case}_N{n}", jw.initial(n, nj), box=jw.BOX, gamma=jw.GAM,
                              bc=("periodic",) * 3, recon=recon, integrator=integ, cfl=cfl, tlim=_a7_t_end(nj),
                              block=(n // 2,) * 3, gravity=4.0 * np.pi * G))
    return lines


def _a7_series(errs, nj):
    """{metric: value} from errors()' (L1, L2, Linf, ratio): the test's four checked series."""
    exact = 1.0 if nj < 1 else np.e
    return dict(L1=errs[0], L2=errs[1], Linf=errs[2], amp=abs(errs[3] / exact - 1.0), ratio=errs[3])


def reduce_a7(root, scheme):
    """test_gravity_jeans_wave.errors on AthenaK's final states (exact end time, IC checked bitwise)."""
    import tests.test_gravity_jeans_wave as jw
    res = {}
    for nj, case in A7_CASES:
        for n in _ns(root, f"a7_{scheme}_{case}_N{{}}"):
            rd = os.path.join(root, f"a7_{scheme}_{case}_N{n}")
            jw.evolve_staged = lambda cfg, U0, t, rd=rd: _state(rd, 3, t, U0)
            res[f"{case}_N{n}"] = _a7_series(jw.errors.__wrapped__(n, nj, "hllc-minmod"), nj) | dict(
                ncycle=luna_ic.final(rd)[1])
    return res


def _flux(cfg):
    """1.0 if cfg's gravity energy coupling resolves to the face mass fluxes, else 0.0 (kick)."""
    from luna.setup.gravity_energy import resolve_gravity_energy
    return float(resolve_gravity_energy(cfg) == "flux")


def luna_a7(ns=(32, 64, 128), schemes="hllc-minmod+hllc-mc+hllc-vl", energy="auto"):
    """LUNA's schemes ('hllc-vl' van Leer; '@cfl' sets LUNA's CFL), the test's errors(); `energy` overrides the
    test's gravity_energy='kick' (auto: LUNA's default), `flux` records what it resolved to."""
    import tests.test_gravity_jeans_wave as jw
    res = {}
    for sc in schemes.split("+"):
        name, lim, cfl = _luna_scheme(sc)
        orig = _luna_cfg(jw, lim, cfl, energy)
        for nj, case in A7_CASES:
            for n in ns:
                res[f"{sc}_{case}_N{n}"] = _a7_series(jw.errors.__wrapped__(n, nj, name), nj) | dict(
                    flux=_flux(jw.cfg_for(n, nj, name)))
        jw.accuracy_cfg = orig
    return res


A8_CHAIN = (32, 64)                              # N whose 0.25 k states come from runs with tlim = 0.25 k


def prep_a8(root, scheme, cfl, ns=(32, 64)):
    """Periodic Evrard (L = 4R) from the test's initial_conditions(cfg_for), min(N, 64)^3 MeshBlocks (4 ranks from
    256), 4 pi G; one run to t = 3 dumping all variables at the history cadence 0.25 (AthenaK does not land on
    output times: the dumps are a little late), and for N in A8_CHAIN one run per output time with tlim = 0.25 k
    (exact times). Pass floor=1e-8."""
    import tests.test_gravity_evrard_energy as ev
    from luna.ic_generation.ic import initial_conditions
    recon, integ = SCHEMES[scheme]
    lines = []
    for n in ns:
        U0 = np.asarray(initial_conditions(ev.cfg_for(n, "hllc-minmod")), np.float64)
        kw = dict(box=(ev.L,) * 3, gamma=ev.GAM, bc=("periodic",) * 3, recon=recon, integrator=integ, cfl=cfl,
                  block=(min(n, 64),) * 3, gravity=4.0 * np.pi * ev.G)
        lines.append(_run(root, f"a8_{scheme}_N{n}", U0, tlim=ev.TIMES[-1], hst_dt=ev.TIMES[0], dump=True,
                          ranks=4 if n >= 256 else 1, **kw))
        if n in A8_CHAIN:
            lines += [_run(root, f"a8_{scheme}_N{n}_t{k:02d}", U0, tlim=t, **kw)
                      for k, t in enumerate(ev.TIMES[:-1], 1)]
    return lines


def _a8_at(dE):
    """{'dE_t<t>': value} at every output time (t = 0.25 .. 3)."""
    import tests.test_gravity_evrard_energy as ev
    return {f"dE_t{t:g}": float(v) for t, v in zip(ev.TIMES, dE)}


def reduce_a8(root, scheme):
    """|E - E0| / |E0| by the test's energies() at the 12 output times: exact times from the chained runs (and the
    final dump at t = 3), else the dumps at their recorded times (`t`); `dE_dump` is the dump series for every N
    (with the chain, it measures what the late dumps cost); the energies K, Th, W and max rho (the collapse itself),
    mass drift and non-finite cells."""
    import tests.test_gravity_evrard_energy as ev
    from luna.ic_generation.ic import initial_conditions
    res = {}
    for n in _ns(root, f"a8_{scheme}_N{{}}"):
        rd = os.path.join(root, f"a8_{scheme}_N{n}")
        U0 = np.asarray(initial_conditions(ev.cfg_for(n, "hllc-minmod")), np.float64)
        ic = np.load(os.path.join(rd, "ic", "U.npy"))
        assert np.array_equal(ic, U0), f"{rd}: IC differs from the test's U0"
        e0, m0 = sum(ev.energies(U0, n)), U0[0].sum()

        def budget(U):
            """(|dE/E0|, E_kin, E_th, W, max rho)."""
            if not np.isfinite(U).all():
                return (np.nan,) * 5
            k, th, w = ev.energies(U, n)
            return abs(k + th + w - e0) / abs(e0), k, th, w, float(U[0].max())
        ds = [d for d in luna_ic.dumps(rd) if d[0] > 0.0][:len(ev.TIMES) - 1]
        t_dump = np.array([d[0] for d in ds] + [ev.TIMES[-1]])
        Uf = _state(rd, 3, ev.TIMES[-1])
        dump = np.array([budget(np.load(p)) for _, _, p in ds] + [budget(Uf)])
        t, b, exact = t_dump.copy(), dump.copy(), np.zeros(len(ev.TIMES), bool)
        exact[-1] = True
        for k, tk in enumerate(ev.TIMES[:-1]):
            rk = os.path.join(rd + f"_t{k + 1:02d}", "run.luna_final.txt")
            if os.path.exists(rk):
                t[k], b[k], exact[k] = tk, budget(_state(os.path.dirname(rk), 3, tk)), True
        res[f"N{n}"] = dict(t=t, dE=b[:, 0], K=b[:, 1], Th=b[:, 2], W=b[:, 3], rho_max=b[:, 4], exact=exact,
                            t_dump=t_dump, dE_dump=dump[:, 0], **_a8_at(b[:, 0]),
                            max_t_late=float(np.abs(t - np.asarray(ev.TIMES)).max()),
                            mass_drift=float(abs(Uf[0].sum() / m0 - 1.0)),
                            nonfinite=float(1 - np.isfinite(Uf).mean()), ncycle=luna_ic.final(rd)[1])
    return res


def luna_a8(ns=(32, 64), schemes="hllc-minmod+hllc-mc+hllc-vl", energy="auto"):
    """LUNA's schemes by the test's run_budget(): |dE/E0| at the 12 times (`dE`, `dE_t3`); a run that stops (MC 32:
    p < 0 at the bounce) is recorded with `failed` and the error text. `energy` as luna_a7."""
    import tests.test_gravity_evrard_energy as ev
    res = {}
    for sc in schemes.split("+"):
        name, lim, cfl = _luna_scheme(sc)
        orig = _luna_cfg(ev, lim, cfl, energy)
        for n in ns:
            flux = _flux(ev.cfg_for(n, name))
            try:
                b = np.array(ev.run_budget(n, name))
                res[f"{sc}_N{n}"] = dict(t=b[:, 0], K=b[:, 1], Th=b[:, 2], W=b[:, 3], dE=b[:, 4], **_a8_at(b[:, 4]),
                                         failed=0.0, flux=flux)
            except RuntimeError as e:
                print(f"{sc} N={n}: {e}")
                res[f"{sc}_N{n}"] = dict(failed=1.0, flux=flux)
        ev.accuracy_cfg = orig
    return res


# ---------------------------------------------------------------- shock tubes, Sedov, Lecoanet KH (site pages)
def prep_sod(root, scheme, cfl, ns=(128, 256, 512, 800, 1600, 3200), cases=(1,)):
    """Toro's tests (1: Sod) on [0, 1] from the shock-tube test's exact cell-average IC, zero-gradient ends."""
    import tests.test_hydro_accuracy_shocktube as st
    from luna.hydro import exact_riemann as er
    recon, integ = SCHEMES[scheme]
    lines = []
    for c in cases:
        WL, WR, x0, t = er.TORO[c]
        for n in ns:
            U0 = st._cons(*er.cell_average(WL, WR, st.G, np.linspace(0.0, 1.0, n + 1), x0, 1e-12))
            lines.append(_run(root, f"sod_{scheme}_c{c}_N{n}", U0, box=(1.0, 1.0, 1.0), gamma=st.G,
                              bc=("outflow",) * 3, recon=recon, integrator=integ, cfl=cfl, tlim=t,
                              block=(_blk(n, 512), 1, 1)))
    return lines


def _tube_ic(problem, n, lx):
    """The 1D IC of a LUNA tube problem (Shu-Osher, Titarev-Toro) on n cells, as its test builds it."""
    from luna.ic_generation.ic import initial_conditions
    from tests._helpers import accuracy_cfg
    cfg = accuracy_cfg(test_problem=problem, SPATIAL_DIMENSION=1, Nx=n, Lx=lx, BOUNDARY=1, gamma=1.4)
    return np.asarray(initial_conditions(cfg), np.float64)


def prep_so(root, scheme, cfl, ns=(256, 512, 1024, 2048)):
    """Shu-Osher on [0, 10] to T = 1.8 from the test's IC, zero-gradient ends."""
    import tests.test_hydro_hard_shu_osher as so
    recon, integ = SCHEMES[scheme]
    return [_run(root, f"so_{scheme}_N{n}", _tube_ic("Shu-Osher", n, so.LX), box=(so.LX, 1.0, 1.0), gamma=1.4,
                 bc=("outflow",) * 3, recon=recon, integrator=integ, cfl=cfl, tlim=so.T,
                 block=(_blk(n, 512), 1, 1)) for n in ns]


def prep_tt(root, scheme, cfl, ns=(1024, 2048, 4096, 8192)):
    """Titarev-Toro on [0, 10] to T = 5 from the test's IC, zero-gradient ends."""
    import tests.test_hydro_hard_titarev_toro as tt
    recon, integ = SCHEMES[scheme]
    return [_run(root, f"tt_{scheme}_N{n}", _tube_ic("Titarev-Toro", n, tt.LX), box=(tt.LX, 1.0, 1.0), gamma=1.4,
                 bc=("outflow",) * 3, recon=recon, integrator=integ, cfl=cfl, tlim=tt.T,
                 block=(_blk(n, 512), 1, 1)) for n in ns]


# Shu-Osher and Titarev-Toro: AthenaK PPMX + RK4 at TUBE_REF[test] is the reference (8x the tests' finest N), its run
# at half that N the self-convergence check; PLM + RK2 and PPMX at the tests' N are scored against it
TUBE = {"so": ("Shu-Osher", "tests.test_hydro_hard_shu_osher", "shu_osher", "luna/shu_osher/shu_osher_mc.npz"),
        "tt": ("Titarev-Toro", "tests.test_hydro_hard_titarev_toro", "titarev_toro",
               "luna/titarev_toro/titarev_toro_mc.npz")}
TUBE_REF = {"so": 16384, "tt": 65536}


def _tube_l12(rho, ref, lx):
    """(L1, L2) of rho on n cells against ref averaged onto them, h-weighted as the tests' errors()."""
    e, h = rho - _avg(ref, rho.size), lx / rho.size
    return float(h * np.abs(e).sum()), float(np.sqrt(h * (e * e).sum()))


def _tube_akref(test):
    """rho of the stored AthenaK PPMX reference of a tube test, or None."""
    p = os.path.join(str(reference_root()), f"athenak/{TUBE[test][2]}/{test}_ppmx_rk4_{TUBE_REF[test]}.npz")
    return np.asarray(np.load(p)["ref_rho"], np.float64) if os.path.exists(p) else None


def _reduce_tube(test, root, scheme):
    """Every finished run's final rho, ncycle, non-finite fraction and (L1, L2) against the AthenaK PPMX reference
    (this root's run at TUBE_REF for ppmx_rk4, else the stored file) and the LUNA MC 2^16 reference (`_luna`)."""
    import importlib
    mod = importlib.import_module(TUBE[test][1])
    lx, t = mod.LX, mod.T
    with np.load(os.path.join(str(reference_root()), TUBE[test][3])) as z:
        luna_ref = np.asarray(z["rho"], np.float64)

    def rho_of(n):
        rd = os.path.join(root, f"{test}_{scheme}_N{n}")
        U = np.asarray(_state(rd, 1, t), np.float64)
        return U[0], luna_ic.final(rd)[1], float(1.0 - np.isfinite(U).mean())

    ns = _ns(root, f"{test}_{scheme}_N{{}}")
    res = {}
    if scheme == "ppmx_rk4":
        nr = TUBE_REF[test]
        assert nr in ns and nr // 2 in ns, f"{root}: the reference runs {nr} and {nr // 2} are needed"
        ref, ncyc, bad = rho_of(nr)
        half = rho_of(nr // 2)[0]
        assert bad == 0.0, f"{test} reference {nr}: non-finite values"
        res["ref"] = dict(N=nr, rho=ref, ncycle=ncyc, nonfinite=bad, self_N=nr // 2)
        res["ref"] |= dict(zip(("L1_self", "L2_self"), _tube_l12(half, ref, lx)))
        for m in (luna_ref.size, 1024):
            res["ref"] |= dict(zip((f"L1_vs_luna_at{m}", f"L2_vs_luna_at{m}"),
                                   _tube_l12(_avg(ref, m), luna_ref, lx)))
    else:
        ref = _tube_akref(test)
        assert ref is not None, f"reduce {test} ppmx_rk4 first (its reference is read from luna-reference-solutions)"
    for n in ns:
        if scheme == "ppmx_rk4" and n == TUBE_REF[test]:
            continue
        rho, ncyc, bad = rho_of(n)
        r = dict(rho=rho, ncycle=ncyc, nonfinite=bad)
        if bad == 0.0:
            r |= dict(zip(("L1", "L2"), _tube_l12(rho, ref, lx)))
            if n <= luna_ref.size:
                r |= dict(zip(("L1_luna", "L2_luna"), _tube_l12(rho, luna_ref, lx)))
        res[f"N{n}"] = r
    return res


def reduce_so(root, scheme):
    """Shu-Osher: _reduce_tube."""
    return _reduce_tube("so", root, scheme)


def reduce_tt(root, scheme):
    """Titarev-Toro: _reduce_tube."""
    return _reduce_tube("tt", root, scheme)


def prep_sedov(root, scheme, cfl, ns=(64, 128, 256)):
    """Sedov 3D from the Sedov test's IC (sedov3d_run in Arepo units, E = 1e51 erg in 8 cells), zero-gradient
    faces, 64^3 MeshBlocks, to t_end, and to each of its earlier FRACS for the radius history."""
    import tests.test_hydro_accuracy_sedov as sd
    from luna.ic_generation.ic import initial_conditions
    recon, integ = SCHEMES[scheme]
    lines = []
    for n in ns:
        cfg = sd.sedov_cfg(n)
        U0 = np.asarray(initial_conditions(cfg), np.float64)
        kw = dict(box=(cfg.Lx,) * 3, gamma=cfg.gamma, bc=("outflow",) * 3, recon=recon, integrator=integ, cfl=cfl,
                  block=(_blk(n),) * 3)
        lines += [_run(root, f"sedov_{scheme}_N{n}_f{round(100 * f)}", U0, tlim=f * cfg.t_end, **kw)
                  for f in sd.FRACS]
    return lines


KH_L = (1.0, 2.0)                                # Lecoanet et al. 2016: x in [0, 1), z in [0, 2), N x 2N
KH_T, KH_SNAP, KH_HST = 6.0, (2.0, 4.0), 0.05    # end, extra field times (own runs, exact), history cadence


def kh_cfg(n, **kw):
    """LUNA's Kelvin-Helmholtz (Lecoanet's IC without dye, inviscid, gamma 5/3) on Lecoanet's n x 2n box."""
    from tests._helpers import accuracy_cfg
    return accuracy_cfg(test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2, Nx=n, Ny=2 * n, Lx=KH_L[0],
                        Ly=KH_L[1], BOUNDARY=0, gamma=5.0 / 3.0, t_end=KH_T, **kw)


def prep_kh(root, scheme, cfl, ns=(256, 512, 1024)):
    """Lecoanet KH from LUNA's IC, periodic, to KH_T with the history (KE per axis) every KH_HST; at the finest N
    also one run to each KH_SNAP time for its field."""
    from luna.ic_generation.ic import initial_conditions
    recon, integ = SCHEMES[scheme]
    lines = []
    for n in ns:
        U0 = np.asarray(initial_conditions(kh_cfg(n)), np.float64)
        kw = dict(box=KH_L + (1.0,), gamma=5.0 / 3.0, bc=("periodic",) * 3, recon=recon, integrator=integ, cfl=cfl,
                  block=(_blk(n), _blk(2 * n), 1))
        lines.append(_run(root, f"kh_{scheme}_N{n}", U0, tlim=KH_T, hst_dt=KH_HST, **kw))
        if n == max(ns):
            lines += [_run(root, f"kh_{scheme}_N{n}_t{t:g}", U0, tlim=t, **kw) for t in KH_SNAP]
    return lines


# ---------------------------------------------------------------- A5, McNally smooth Kelvin-Helmholtz
A5_SEG, A5_NSEG = 0.02, 75                       # one AthenaK run per output interval, 75 to t = 1.5
A5_REF = "athenak/mcnally_kh/a5_ppmx_rk4_4096.npz"    # AthenaK PPMX + RK4 4096^2, the independent field reference
_A5 = os.environ.get("LUNA_A5_RAW", "")            # the A5 raw-reference directory (lit_a05), if present
A5_RAW = {"luna": os.path.join(_A5, "impl/ref/a5_mc_4096_t1.5_1024.npz"),       # LUNA MC 4096
          "ak": os.path.join(_A5, "ak/ppmx_rk4_cfl0.4_floor1e-8/a5_ppmx_rk4_N4096_t1.5_1024.npz")}
A5_LUNA_RAW = A5_RAW["luna"]                     # both at t = 1.5, rho and rho V_y block-averaged to 1024^2


def prep_a5(root, scheme, cfl, ns=(128, 256, 512, 1024, 2048)):
    """The test's IC (point-sampled eqs. 1-5) on n^2, 64^2 MeshBlocks, as a chain ROOT/a5_<scheme>_N<n>/sKK of 75
    runs to tlim = 0.02, each from the previous run's final dump (ic/U.npy a symlink): AthenaK clamps dt to tlim
    only, not to outputs (`mesh.cpp` NewTimeStep), and M must be read at t = 0.02 k exactly. 4096^2 on 4 ranks."""
    import tests.test_hydro_hard_mcnally_kh as kh
    recon, integ = SCHEMES[scheme]
    lines = []
    for n in ns:
        d = os.path.join(root, f"a5_{scheme}_N{n}")
        U0 = kh.initial(kh.cfg_for(n))
        for k in range(A5_NSEG):
            s = os.path.join(d, f"s{k:02d}")
            if k == 0:
                luna_ic.write_ic(U0, os.path.join(s, "ic"), (1.0, 1.0, 1.0), kh.G, f"a5_N{n}")
            else:
                os.makedirs(os.path.join(s, "ic"), exist_ok=True)
                with open(os.path.join(d, "s00", "ic", "meta.json")) as f, \
                        open(os.path.join(s, "ic", "meta.json"), "w") as g:
                    g.write(f.read())
                link = os.path.join(s, "ic", "U.npy")
                if not os.path.lexists(link):
                    os.symlink(f"../../s{k - 1:02d}/run.luna.final.npy", link)
            luna_ic.athinput(os.path.join(s, "run.athinput"), n=(n, n, 1), box=(1.0, 1.0, 1.0), gamma=kh.G,
                             bc=("periodic",) * 3, block=(_blk(n), _blk(n), 1), recon=recon, integrator=integ,
                             cfl=cfl, tlim=A5_SEG, ic_dir=os.path.join(s, "ic"), floor=FLOOR)
        lines.append(f"{d} {4 if n >= 4096 else 1}")
    return lines


def _a5_chain(d, U0):
    """(M, K, steps, rho, rho V_y at t = 1.5) of a finished chain d, at t = 0.02 k; every run checked to end at
    0.02, the first run's IC checked bitwise against the test's U0."""
    import tests.test_hydro_hard_mcnally_kh as kh
    U = luna_ic.from_luna5(np.load(os.path.join(d, "s00", "ic", "U.npy")), 2)
    assert np.array_equal(U, np.asarray(U0, np.float64)), f"{d}: IC differs from the test's U0"
    M, K, steps = [kh.mode_amplitude(U)], [kh.max_ke(U)], 0
    for k in range(A5_NSEG):
        tf, nc, p = luna_ic.final(os.path.join(d, f"s{k:02d}"))
        assert abs(tf - A5_SEG) <= 1e-15, (d, k, tf)
        V = np.load(p, mmap_mode="r")
        U = np.stack([np.array(V[i, :, :, 0]) for i in range(3)])
        M.append(kh.mode_amplitude(U))
        K.append(kh.max_ke(U))
        steps += nc
    return np.array(M), np.array(K), steps, U[0], U[2]


def _a5_scores(r, ref=None):
    """The test's m_errors (max, time integral, signed end), ke_errors and field_errors of a run dict r (M, K, rho,
    my on min(N, 256)^2); the fields against the LUNA MC 4096 reference and, given ref = (rho, rho V_y) on 256^2,
    also against that one (suffix _ak: the test's field_errors with its reference swapped)."""
    import tests.test_hydro_hard_mcnally_kh as kh
    out = dict(zip(("eM_max", "eM_int", "eM_end"), kh.m_errors(r)))
    out |= dict(zip(("eK_max", "eK_end"), kh.ke_errors(r)))
    for sfx, rf in (("", None), ("_ak", ref)):
        if sfx and rf is None:
            continue
        saved = kh.field_reference
        if rf is not None:
            kh.field_reference = lambda rf=rf: (np.asarray(rf[0], np.float64), np.asarray(rf[1], np.float64), 4096,
                                                float("nan"))
        try:
            fe = kh.field_errors(r)
        finally:
            kh.field_reference = saved
        out |= {f"{f}_{nm}{sfx}": fe[f][j] for f in ("rho", "vy") for j, nm in enumerate(("L1", "L2", "Linf"))}
    return out


def _a5_ak_ref():
    """(rho, rho V_y) on 256^2 of the stored AthenaK PPMX 4096 reference, or None."""
    p = os.path.join(str(reference_root()), A5_REF)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z["N4096_rho"], z["N4096_my"]


def _a5_native(rho, my):
    """L1 of rho and V_y on the run's own grid (N <= 1024) against each 4096 reference of A5_RAW averaged to N, and
    MUSIC's measure (Goffrey+ 2017): relative L1 of V_y, sum |V_y - <V_y>_ref| / sum |<V_y>_ref|."""
    import tests.test_hydro_hard_mcnally_kh as kh
    n, out = rho.shape[0], {}
    if n > 1024:
        return out
    for name, p in A5_RAW.items():
        if not os.path.exists(p):
            continue
        z = np.load(p)
        rr, mr = kh.coarsen(z["rho"], n), kh.coarsen(z["my"], n)
        vy, vr = my / rho, mr / rr
        out |= {f"nat_rho_L1_{name}": float(np.abs(rho - rr).mean()),
                f"nat_vy_L1_{name}": float(np.abs(vy - vr).mean()),
                f"nat_vy_rel_{name}": float(np.abs(vy - vr).sum() / np.abs(vr).sum())}
    return out


def _l12inf(x):
    return dict(L1=float(np.abs(x).mean()), L2=float(np.sqrt((x * x).mean())), Linf=float(np.abs(x).max()))


def reduce_a5(root, scheme):
    """M(t), max(rho V_y^2 / 2)(t) and the t = 1.5 fields of every finished chain, scored with the test's functions
    against fig. 4, fig. 8 and the LUNA MC 4096 reference (and the AthenaK PPMX 4096 reference: this root's 4096
    run, else the stored file); fields block-averaged to min(N, 256)^2 kept (float32) for the 4096 reference
    only; at 4096 the distance to LUNA's 4096 reference (on 256^2 and 1024^2) and to this scheme's 2048 run."""
    import tests.test_hydro_hard_mcnally_kh as kh
    ns = sorted(int(os.path.basename(d).rsplit("_N", 1)[1]) for d in glob.glob(os.path.join(root, f"a5_{scheme}_N*"))
                if os.path.exists(os.path.join(d, f"s{A5_NSEG - 1:02d}", "run.luna_final.txt")))
    runs = {}
    for n in ns[::-1]:                                 # 4096 first: its raw 1024^2 file is the native-grid reference
        M, K, steps, rho, my = _a5_chain(os.path.join(root, f"a5_{scheme}_N{n}"), kh.initial(kh.cfg_for(n)))
        m = min(n, kh.NC)
        runs[n] = dict(t=kh.TIMES.copy(), M=M, K=K, steps=steps, rho=kh.coarsen(rho, m), my=kh.coarsen(my, m))
        if n >= 1024:                                  # finer than the repo keeps: 1024^2 on ptmp
            runs[n]["raw"] = (kh.coarsen(rho, 1024), kh.coarsen(my, 1024))
            np.savez(os.path.join(root, f"a5_{scheme}_N{n}_t1.5_1024.npz"), rho=runs[n]["raw"][0],
                     my=runs[n]["raw"][1])
        runs[n]["nat"] = _a5_native(rho, my)
    ref = (runs[4096]["rho"], runs[4096]["my"]) if 4096 in runs else _a5_ak_ref()
    res = {}
    for n, r in sorted(runs.items()):
        res[f"N{n}"] = dict(t=r["t"], M=r["M"], K=r["K"], steps=r["steps"]) | _a5_scores(r, ref) | r["nat"]
        if n >= 4096:                                  # the field reference; nothing reads other fields
            res[f"N{n}"] |= dict(rho=r["rho"].astype(np.float32), my=r["my"].astype(np.float32))
    if 4096 in runs:
        a = runs[4096]
        rl, ml, _, _ = kh.field_reference()
        res["ref"] = {f"vs_luna_{f}_{nm}": v for f, x in (("rho", a["rho"] - rl), ("vy", a["my"] / a["rho"] - ml / rl))
                      for nm, v in _l12inf(x).items()}
        if os.path.exists(A5_LUNA_RAW):
            z = np.load(A5_LUNA_RAW)
            rr, mr = a["raw"]
            res["ref"] |= {f"vs_luna_1024_{f}_{nm}": v
                           for f, x in (("rho", rr - z["rho"]), ("vy", mr / rr - z["my"] / z["rho"]))
                           for nm, v in _l12inf(x).items()}
        if 2048 in runs:
            b = runs[2048]
            res["ref"] |= dict(self_l1_rho=float(np.abs(a["rho"] - b["rho"]).mean()),
                               self_l1_vy=float(np.abs(a["my"] / a["rho"] - b["my"] / b["rho"]).mean()))
    return res


def luna_a5(ns=(64, 128, 256, 512, 1024, 2048), schemes="minmod+mc+vanleer", cfl=None):
    """LUNA's limiters on the test's run() (HLLC), scored as reduce_a5; cfl overrides the test's 0.8 (equal dt)."""
    import tests.test_hydro_hard_mcnally_kh as kh
    if cfl is not None:
        base = kh.accuracy_cfg
        kh.accuracy_cfg = lambda **k: base(**k, CFL=float(cfl))
    ref = _a5_ak_ref()
    res = {}
    for sc in schemes.split("+"):
        for n in ns:
            kh.NC = 1024                         # run() keeps the fields at min(N, 1024)^2; the test's 256^2 below
            try:
                r = kh.run(n, sc)
            finally:
                kh.NC = 256
            m = min(n, 256)
            r256 = dict(M=r["M"], K=r["K"], rho=kh.coarsen(r["rho"], m), my=kh.coarsen(r["my"], m))
            res[f"{sc}_N{n}"] = dict(t=r["t"], M=r["M"], K=r["K"], steps=r["steps"], rho=r["rho"], my=r["my"]) \
                | _a5_scores(r256, ref) | (_a5_native(r["rho"], r["my"]) if n <= 1024 else {})
    return res


# ---------------------------------------------------------------- A6, isothermal atmosphere in phi = sin(2 pi x)
A6_NS = (64, 128, 256, 512, 1024, 2048)
A6_FLIP = 256                                    # the test's teeth: phi -> -phi at 256


def prep_a6(root, scheme, cfl, ns=A6_NS):
    """The test's steady(n) (rho = p = exp(-sin 2 pi x), exact cell averages), periodic, gamma 1.4, to T = 2 /
    sqrt(gamma); phi through the luna_ic_a6 pgen's user source term (g = -dphi/dx at the cell centres); plus the
    flipped potential at A6_FLIP. Pass floor=1e-8."""
    import tests.test_hydro_hard_isothermal_atmosphere as ta
    recon, integ = SCHEMES[scheme]
    kw = dict(box=(1.0, 1.0, 1.0), gamma=ta.G, bc=("periodic",) * 3, recon=recon, integrator=integ, cfl=cfl,
              tlim=float(ta.T))
    lines = [_run(root, f"a6_{scheme}_N{n}", ta.steady(n), potential=(1.0, 1.0, 1), **kw) for n in ns]
    if A6_FLIP in ns:
        lines.append(_run(root, f"a6_{scheme}_N{A6_FLIP}_flip", ta.steady(A6_FLIP), potential=(-1.0, 1.0, 1), **kw))
    return lines


def _a6_flat(r):
    """{field_norm: value, budget} from the test's errors()."""
    out = {f"{fl}_{nm}": r[fl][j] for fl in ("rho", "mom", "E") for j, nm in enumerate(("L1", "L2", "Linf"))}
    return out | dict(budget=r["budget"])


def reduce_a6(root, scheme):
    """test_hydro_hard_isothermal_atmosphere.errors on AthenaK's final states (exact end time, IC checked bitwise),
    with the run's step count and cost; the flipped potential as N<n>flip."""
    import tests.test_hydro_hard_isothermal_atmosphere as ta
    res = {}
    for n, tag, sign in [(n, "", 1.0) for n in _ns(root, f"a6_{scheme}_N{{}}")] + [(A6_FLIP, "_flip", -1.0)]:
        rd = os.path.join(root, f"a6_{scheme}_N{n}{tag}")
        if not os.path.exists(os.path.join(rd, "run.luna_final.txt")):
            continue
        ta.evolve_staged = lambda cfg, U0, t, rd=rd: _state(rd, 1, t, U0)
        res[f"N{n}{tag.strip('_')}"] = _a6_flat(ta.errors.__wrapped__(n, "hllc-minmod", sign)) | dict(
            ncycle=luna_ic.final(rd)[1]) | _cost(rd)
    return res


def luna_a6(ns=A6_NS, schemes="hllc-minmod+hll-minmod+hllc-mc+hllc-vl"):
    """LUNA's schemes ('hllc-vl' van Leer, '@cfl' a CFL) by the test's errors(), with steps and wall time (the
    first run of a scheme and N includes compilation; wall2_s is the same run again)."""
    import time

    import luna.setup.evolve as ev
    import tests.test_hydro_hard_isothermal_atmosphere as ta
    step0, count = ev.step, [0]

    def counted(*a, **k):
        count[0] += 1
        return step0(*a, **k)

    ev.step = counted
    res = {}
    for sc in schemes.split("+"):
        name, lim, cfl = _luna_scheme(sc)
        orig = _luna_cfg(ta, lim, cfl)
        for n in ns:
            count[0] = 0
            w = time.time()
            r = _a6_flat(ta.errors.__wrapped__(n, name))
            r |= dict(steps=count[0], wall_s=time.time() - w)
            w = time.time()
            ta.errors.__wrapped__(n, name)
            res[f"{sc}_N{n}"] = r | dict(wall2_s=time.time() - w)
        if A6_FLIP in ns:
            res[f"{sc}_N{A6_FLIP}flip"] = _a6_flat(ta.errors.__wrapped__(A6_FLIP, name, -1.0))
        ta.accuracy_cfg = orig
    ev.step = step0
    return res


# ---------------------------------------------------------------- LUNA counterparts of the phase 0 references
def luna_a1(ns=(32, 64, 128, 256, 512, 1024), schemes=None):
    """LUNA's schemes on the A1 right sound wave in AthenaK's metric: RMS over variables of L1(U - U0) / EPS."""
    import tests.test_hydro_accuracy_linear_waves as lw
    lw.SCHEMES.update(LUNA_SCHEMES)
    res = {}
    for sc in (schemes.split("+") if schemes else LUNA_SCHEMES):
        for n in ns:
            U0, U = lw.run(1, n, "sound_r", sc)
            l1 = np.abs(np.asarray(U) - U0).reshape(U0.shape[0], -1).mean(1)
            res[f"{sc}_N{n}"] = dict(rms=float(np.sqrt((l1 ** 2).sum())) / lw.EPS)
    return res


def luna_tg(ns=(64, 128), schemes="minmod+mc+vl"):
    """LUNA's (Re_eff, D) on the A15 Taylor-Green vortex by the test's measure(); vl is van Leer (LIMITER 2)."""
    import tests.test_hydro_hard_taylor_green_reynolds as tg
    tg.SCHEMES.update(vl=2)
    return {f"{sc}_N{n}": dict(zip(tg.NAMES, tg.measure(n, sc))) for sc in schemes.split("+") for n in ns}


def _order(ns, vals):
    """Least-squares order -d ln v / d ln N."""
    return float(-np.polyfit(np.log(ns), np.log(vals), 1)[0])


def _columns(test, luna_json, data):
    """{column: {key: {metric: value}}} for LUNA's four schemes (luna_json, '+'-joined files) and AthenaK's npz
    in data/."""
    L = {k: v for p in luna_json.split("+") for k, v in json.load(open(p))["data"].items()}
    if test == "a11":                                 # LUNA keys 'c3_hllc-mc_N256' -> column hllc-mc, key 'c3_N256'
        cols = {sc: {} for sc in LUNA_SCHEMES}
        for k, v in L.items():
            c, sc, n = k.split("_", 2)
            cols.setdefault(sc, {})[f"{c}_{n}"] = v
    else:
        cols = {sc: {k[len(sc) + 1:]: v for k, v in L.items() if k.startswith(sc + "_")} for sc in LUNA_SCHEMES}
    if test in ("a7", "a8"):                          # only the LUNA schemes that ran
        cols = {sc: c for sc, c in cols.items() if c}
    for sch in SCHEMES:                               # <test>eqdt_*: AthenaK at LUNA's dt (A7)
        for tag, sfx in ((test, ""), (f"{test}eqdt", " at LUNA's dt")):
            pat = f"a11_c*_{sch}_*.npz" if test == "a11" else f"{tag}_{sch}_*.npz"     # A11: one file per config
            if test == "a11" and sfx:
                continue
            for f in sorted(glob.glob(os.path.join(data, "**", pat), recursive=True)):
                z = np.load(f)
                col = cols.setdefault(f"AthenaK {sch}{sfx}", {})
                for k in z.files:
                    if k == "provenance" or z[k].ndim or k.startswith("ref_") or "_ref_" in k:
                        continue
                    key, met = _split(test, k)
                    col.setdefault(key, {})[met] = float(z[k])
    return cols


def _split(test, k):
    """npz key -> (case key, metric) as LUNA's json: 'N64_L1' -> ('N64', 'L1'), 'static_N64_v_L1' ->
    ('static_N64', 'v_L1'), 'lw40x40_rho_L1_pct' -> ('lw40x40', 'rho_L1_pct'); A9: 'N1024_r_vy_max' -> ('N1024',
    'vy_max'), periodic y 'N1024_p_...' -> ('N1024_p', ...)."""
    if test in ("a4", "a7") and not k.startswith("lw"):
        a, b, rest = k.split("_", 2)
        return f"{a}_{b}", rest
    if test == "a9":
        a, b, rest = k.split("_", 2)
        return (a if b == "r" else f"{a}_{b}"), rest
    if test in ("a10", "a11"):                        # 'N400_l_s_jet' -> ('N400_l', 's_jet'); 'c3_N256_L1_out' too
        a, b, rest = k.split("_", 2)
        return f"{a}_{b}", rest
    return tuple(k.split("_", 1))


def table(test, luna_json, data=None):
    """Markdown tables of both codes' metrics per N (LUNA's json from `luna`, AthenaK's npz in data/)."""
    data = data or os.path.join(reference_root(), "athenak")
    cols = _columns(test, luna_json, data)
    if test in ("a6", "a10", "a11"):                     # only the columns that ran; AthenaK's ncycle as steps
        cols = {c: v for c, v in cols.items() if v}
        for v in cols.values():
            for d in v.values():
                if "ncycle" in d:
                    d["steps"] = d["ncycle"]
    names = list(cols)
    out = []

    def tab(title, keyf, met, ns, fits=()):
        out.append(f"\n{title}\n\n| N | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
        for n in ns:
            out.append(f"| {n} | " + " | ".join(
                f"{cols[c][keyf(n)][met]:.3g}" if met in cols[c].get(keyf(n), {}) else "-" for c in names) + " |")
        for lo, hi in fits:
            row = []
            for c in names:
                sub = [n for n in ns if lo <= n <= hi and met in cols[c].get(keyf(n), {})]
                row.append(f"{_order(sub, [cols[c][keyf(n)][met] for n in sub]):.2f}" if len(sub) > 1 else "-")
            out.append(f"| order {lo}-{hi} | " + " | ".join(row) + " |")

    if test == "a3":
        ns = (32, 64, 128, 256, 512, 1024)
        for met in ("L1", "Linf"):
            tab(f"A3 {met}(rho)", lambda n: f"N{n}", met, ns, ((32, 256), (256, 1024)))
        tab("A3 LW rel. L1 % at 200^2", lambda n: f"N{n}", "relL1_pct", (200,))
    elif test == "a4":
        ns = (32, 64, 128, 256, 512, 1024)
        for case in ("static", "moving"):
            for met in ("v_L1", "rho_L1", "p_L1", "v_Linf"):
                tab(f"A4 {case} {met}", lambda n, c=case: f"{c}_N{n}", met, ns, ((32, 256), (256, 1024)))
        for g in ("lw40x40", "lw160x40"):
            for met in ("vort_relL1_pct", "rho_L1_pct", "tke_rel_pct"):
                tab(f"A4 {g} {met}", lambda n, g=g: g, met, (g,))
    elif test == "a9":
        ns = (256, 512, 1024, 2048)
        for met in ("vy_max", "row_spread", "strip_minus_1d", "L1_rho_strip", "L1_rho_strip_akplm"):
            tab(f"A9 {met}", lambda n: f"N{n}", met, ns, ((256, 2048),) if met.startswith("L1") else ())
    elif test == "a10":
        ns, late = (100, 200, 400, 800), (128, 256, 400, 512, 1024)
        for met in ("L1", "L2", "L1_akppmx", "L2_akppmx"):
            tab(f"A10 {met}(rho), t = 0.045", lambda n: f"N{n}_e", met, ns, ((100, 800),))
        for met in ("asym_rho", "asym_m", "asym_E", "dM", "dE", "steps", "loop_s", "zcps", "wall_s"):
            tab(f"A10 {met}, t = 0.045", lambda n: f"N{n}_e", met, ns)
        for met in ("s_jet", "late_L1", "asym_rho", "asym_m", "asym_E", "dM", "dE", "steps", "loop_s"):
            tab(f"A10 {met}, t = 2.5", lambda n: f"N{n}_l", met, late)
        for met in ("L1", "L1_akppmx", "asym_rho"):          # AthenaK -fmad=false runs (tag n)
            tab(f"A10 {met}(rho), t = 0.045, AthenaK -fmad=false", lambda n: f"N{n}_ne", met, ns)
        for met in ("s_jet", "late_L1", "asym_rho"):
            tab(f"A10 {met}, t = 2.5, AthenaK -fmad=false", lambda n: f"N{n}_nl", met, late)
        for g, title in (("jet", "AthenaK default build"), ("njet", "AthenaK -fmad=false")):
            for met in ("s_jet_t240", "s_jet_t245", "s_jet_t250", "s_jet_t255", "s_jet_t260", "median", "asym_max"):
                tab(f"A10 jet {met} at 400^2, {title}", lambda n, g=g: f"N{n}_{g}", met, (400,))
    elif test == "a11":
        ns = (256, 512, 1024, 2048, 4096, 8192)
        for c, mets in ((3, ("L1", "L1_core", "L1_out", "L1_out_ak", "strip21_L1", "strip34_L1", "strip21_front_cells",
                             "asym", "budget", "steps", "wall_s")),
                        (6, ("L1", "L1_core", "L1_out", "strip21_L1", "strip34_L1", "strip34_front_cells",
                             "pstrip21_L1", "pstrip34_L1", "budget", "steps", "wall_s"))):
            for met in mets:
                tab(f"A11 config {c} {met}", lambda n, c=c: f"c{c}_N{n}", met, ns,
                    ((256, 2048),) if met.endswith(("L1", "L1_out", "L1_core", "L1_out_ak")) else ())
    elif test == "a6":
        for met in ("rho_L1", "rho_Linf", "mom_L1", "mom_Linf", "E_L1", "budget"):
            tab(f"A6 {met} of U(T) - U(0)" if met != "budget" else "A6 |d(E + rho phi)| / (E + rho phi) at T",
                lambda n: f"N{n}", met, A6_NS, ((128, 1024), (512, 2048)))
        for met in ("steps", "wall_s", "loop_s", "zcps"):
            tab(f"A6 {met}", lambda n: f"N{n}", met, A6_NS)
        for met in ("rho_L1", "mom_L1", "budget"):
            tab(f"A6 flipped potential (teeth) {met}", lambda n: f"N{n}flip", met, (A6_FLIP,))
    elif test == "a7":
        for case in ("stable", "unstable"):
            for met in ("L1", "Linf", "amp"):
                tab(f"A7 {case} {met}", lambda n, c=case: f"{c}_N{n}", met, (32, 64, 128), ((32, 128),))
    elif test == "a8":
        for t in (0.5, 1, 1.5, 2, 3):
            tab(f"A8 |dE/E0| at t = {t:g}", lambda n: f"N{n}", f"dE_t{t:g}", (32, 64, 128, 256), ((32, 64), (64, 256)))
    print("\n".join(out))


A5_AK = {"AthenaK PLM+RK2": "a5_plm_rk2_2048.npz", "AthenaK PPMX+RK4": "a5_ppmx_rk4_4096.npz",
         "AthenaK PPM4+RK4": "a5_ppm4_rk4_2048.npz"}


def table_a5(luna_json, data=None):
    """Markdown tables of A5 for LUNA's limiters (luna_json from `luna a5`) and AthenaK's npz: M and K against the
    Pencil curves, fields at t = 1.5 against both 4096^2 references, MUSIC's native-grid measure, orders."""
    data = data or os.path.join(reference_root(), "athenak")
    L = json.load(open(luna_json))["data"]
    cols = {f"LUNA {sc}": {int(k.split("_N")[1]): v for k, v in L.items() if k.startswith(sc + "_N")}
            for sc in ("minmod", "mc", "vanleer")}
    for name, f in A5_AK.items():
        p = os.path.join(data, "mcnally_kh", f)
        if os.path.exists(p):
            z = np.load(p)
            col = {}
            for k in z.files:
                if k.startswith("N") and z[k].ndim == 0:
                    n, met = k[1:].split("_", 1)
                    col.setdefault(int(n), {})[met] = float(z[k])
            cols[name] = col
    names = list(cols)
    out = []

    def fmt(v):
        return f"{v:.0f}" if met_is_count[0] else f"{v:.3g}"

    met_is_count = [False]

    def tab(title, met, ns, fit):
        met_is_count[0] = met == "steps"
        out.append(f"\n{title}\n\n| N | " + " | ".join(names) + " |\n|---|" + "---|" * len(names))
        for n in ns:
            out.append(f"| {n} | " + " | ".join(fmt(cols[c][n][met]) if met in cols[c].get(n, {}) else "-"
                                                for c in names) + " |")
        if fit is None:
            return
        row = []
        for c in names:
            sub = [n for n in ns if fit[0] <= n <= fit[1] and met in cols[c].get(n, {})]
            row.append(f"{_order(sub, [cols[c][n][met] for n in sub]):.2f}" if len(sub) > 1 else "-")
        out.append(f"| order {fit[0]}-{fit[1]} | " + " | ".join(row) + " |")

    ns = (64, 128, 256, 512, 1024, 2048, 4096)
    tab("A5 max_t |M/M_ref - 1| (fig. 4)", "eM_max", ns, (128, 2048))
    tab("A5 integral over t of |M/M_ref - 1| / 1.5 (the test's bound)", "eM_int", ns, (128, 2048))
    tab("A5 signed M/M_ref - 1 at t = 1.5 (MC, van Leer, AthenaK PLM near a zero crossing)", "eM_end", ns, None)
    tab("A5 max_t |K/K_ref - 1| (fig. 8)", "eK_max", ns, (128, 2048))
    for met, what in (("rho_L1", "L1(rho)"), ("vy_L1", "L1(V_y)")):
        tab(f"A5 {what} at t = 1.5 on min(N, 256)^2 vs LUNA MC 4096^2", met, ns, (64, 1024))
        tab(f"A5 {what} at t = 1.5 on min(N, 256)^2 vs AthenaK PPMX 4096^2", met + "_ak", ns, (64, 1024))
    tab("A5 L1(V_y) on the run's own grid vs AthenaK PPMX 4096^2", "nat_vy_L1_ak", ns, (256, 1024))
    tab("A5 MUSIC's relative L1(V_y), own grid, vs AthenaK PPMX 4096^2", "nat_vy_rel_ak", ns, (256, 1024))
    tab("A5 MUSIC's relative L1(V_y), own grid, vs LUNA MC 4096^2", "nat_vy_rel_luna", ns, (256, 1024))
    tab("A5 steps to t = 1.5", "steps", ns, None)
    p = os.path.join(data, "mcnally_kh", A5_AK["AthenaK PPMX+RK4"])
    if os.path.exists(p):                         # the gate on the IC / pgen / diagnostic chain
        z = np.load(p)
        mref = np.loadtxt(os.path.join(reference_root(), "pencil", "mcnally_kh", "mcnally2012_fig4_Mref.txt"))[:, 1]   # fig. 4, no JAX
        rich = z["N4096_M"] + (z["N4096_M"] - z["N2048_M"]) / 3
        out.append(f"\nAthenaK PPMX 4096^2 against fig. 4: max_t {float(z['N4096_eM_max']):.3g}; Richardson (order 2) "
                   f"from 2048 and 4096: max_t {float(np.abs(rich / mref - 1).max()):.3g}. Fields at t = 1.5 against "
                   f"LUNA MC 4096^2 on 256^2: L1(rho) {float(z['ref_vs_luna_rho_L1']):.3g}, L1(V_y) "
                   f"{float(z['ref_vs_luna_vy_L1']):.3g}; AthenaK 4096 vs 2048: {float(z['ref_self_l1_rho']):.3g}, "
                   f"{float(z['ref_self_l1_vy']):.3g}.")
    print("\n".join(out))


def luna(test, out, **kw):
    """luna_<test> on this machine's first device; writes OUT.json (scalars, device, job) and OUT.npz (arrays)."""
    import time

    import jax
    jax.config.update("jax_enable_x64", True)
    t0 = time.time()
    res = globals()[f"luna_{test}"](**kw)
    lr = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from tests._helpers import PRECISION
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip() if jax.default_backend() == "gpu" else ""
    meta = dict(luna_commit=_git(lr)[0], luna_dirty=_git(lr)[1], devices=str(jax.devices()), gpu=gpu,
                precision=PRECISION,
                backend=jax.default_backend(), slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                node=os.uname().nodename, wall_s=time.time() - t0, kw={k: str(v) for k, v in kw.items()})
    scal = {k: {kk: float(vv) for kk, vv in v.items() if np.ndim(vv) == 0} for k, v in res.items()}
    with open(out + ".json", "w") as f:
        json.dump(dict(meta=meta, data=scal), f, indent=1)
    np.savez(out + ".npz", **{f"{k}_{kk}": np.asarray(vv) for k, v in res.items() for kk, vv in v.items()})
    print(json.dumps(meta))


# ---------------------------------------------------------------- driver
def provenance(root):
    """Both git hashes, the build, every run's input, log tail and SLURM/hardware record found in ROOT."""
    luna = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lh, ld = _git(luna)
    if not ATHENAK:
        raise SystemExit("set ATHENAK_DIR to the AthenaK checkout the runs used (tools/athenak/README.md)")
    ah, ad = _git(ATHENAK)
    runs = {}
    for rd in sorted(glob.glob(os.path.join(root, "*", "run.athinput"))):
        d = os.path.dirname(rd)
        rec = {"athinput": open(rd).read()}
        for f in ("job.json", "run.log"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                rec[f] = open(p).read()[-1500:] if f == "run.log" else json.load(open(p))
        runs[os.path.basename(d)] = rec
    for rd in sorted(glob.glob(os.path.join(root, "*", "s00", "run.athinput"))):    # A5 chains: first input, chain
        d = os.path.dirname(os.path.dirname(rd))                                    # job record, last run's log
        rec = {"athinput": open(rd).read()}
        last = sorted(glob.glob(os.path.join(d, "s*", "run.log")))
        if os.path.exists(os.path.join(d, "job.json")):
            rec["job.json"] = json.load(open(os.path.join(d, "job.json")))
        if last:
            rec["run.log"] = open(last[-1]).read()[-1500:]
        runs[os.path.basename(d)] = rec
    import datetime
    import hashlib
    exe = ATHENAK_EXE
    sha = hashlib.sha256(open(exe, "rb").read()).hexdigest()
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(exe)).isoformat(timespec="seconds")
    base = subprocess.run(["git", "-C", ATHENAK, "merge-base", "HEAD", "origin/main"], capture_output=True,
                          text=True).stdout.strip()
    # athenak_commit is the source tree's HEAD at reduction, not the binary's source: the binary is pinned by
    # sha256 and mtime (build_gpu_lunaic: 2026-10-11 15:24, from the luna_ic working tree before c0d1427d and
    # c52f88f0, which change src/pgen/luna_ic.cpp I/O only; a fresh c52f88f0 build reproduces its runs bitwise,
    # critic_a10 job 31168524); athenak_upstream is HEAD's merge-base with origin/main (A7/A8: ATHENAK_DIR,
    # ATHENAK_EXE select luna_ic_mg and build_gpu_lunaic_mg)
    return dict(luna_commit=lh, luna_dirty=ld, athenak_commit=ah, athenak_dirty=ad, athenak_upstream=base,
                athenak_binary=exe, athenak_binary_sha256=sha, athenak_binary_mtime=mtime, runs=runs,
                binaries=_binaries(runs))


def _binaries(runs):
    """{AthenaK binary of any run (its job.json run line): {sha256, mtime}}."""
    import datetime
    import hashlib
    out = {}
    for rec in runs.values():
        line = rec.get("job.json", {}).get("run_line", "")
        exe = next((w for w in line.split() if w.endswith("/athena")), None)
        if exe and exe not in out and os.path.exists(exe):
            out[exe] = dict(sha256=hashlib.sha256(open(exe, "rb").read()).hexdigest(),
                            mtime=datetime.datetime.fromtimestamp(os.path.getmtime(exe)).isoformat(timespec="seconds"))
    return out


# ---------------------------------------------------------------- G2, Agertz et al. (2007) blob
def prep_blob(root, scheme, cfl, ns=(64, 128, 256)):
    """The test's IC (cfg_for), periodic {10, 10, 40} R, to 2.5 tau_KH, dumping all variables every 0.25 tau_KH;
    64^3 MeshBlocks, 4 ranks at 256; from 512 (slices) density only every 0.5 tau_KH on 8 ranks. Pass floor=1e-8."""
    import tests.test_hydro_hard_agertz_blob as bl
    from luna.ic_generation.ic import initial_conditions
    recon, integ = SCHEMES[scheme]
    lines = []
    for n in ns:
        cfg, tau = bl.cfg_for(n, "hllc-minmod")
        U0 = np.asarray(initial_conditions(cfg), np.float64)
        lines.append(_run(root, f"blob_{scheme}_N{n}", U0, box=(10.0, 10.0, 40.0), gamma=cfg.gamma,
                          bc=("periodic",) * 3, recon=recon, integrator=integ, cfl=cfl, tlim=2.5 * tau,
                          hst_dt=(0.5 if n >= 512 else 0.25) * tau, dump=True, dump_vars=1 if n >= 512 else 5,
                          block=(min(n, 64),) * 3, ranks=8 if n >= 512 else 4 if n >= 256 else 1))
    return lines


def reduce_blob(root, scheme):
    """Cloud fraction (the test's cloud_mass: T < 0.9 T_ext, rho > 0.64 rho_cl) at every dump's own time (AthenaK
    does not land on output times) and at the end, in units of tau_KH."""
    import tests.test_hydro_hard_agertz_blob as bl
    from luna.ic_generation import agertz_blob as B
    res = {}
    for n in _ns(root, f"blob_{scheme}_N{{}}"):
        rd = os.path.join(root, f"blob_{scheme}_N{n}")
        cfg, tau = bl.cfg_for(n, "hllc-minmod")
        U0 = np.load(os.path.join(rd, "ic", "U.npy"))
        m0 = float(B.cloud_mass(U0, cfg))
        ds = [d for d in luna_ic.dumps(rd) if d[0] > 0.0]
        tf, _, pf = luna_ic.final(rd)
        t = [d[0] for d in ds] + [tf]
        frac = [float(B.cloud_mass(np.load(p), cfg)) / m0 for _, _, p in ds] + \
               [float(B.cloud_mass(np.load(pf), cfg)) / m0]
        res[f"N{n}"] = dict(t_over_tauKH=np.array(t) / tau, cloud=np.array(frac))
    return res


def main(argv):
    cmd, test, root = argv[:3]
    kw = dict(a.split("=", 1) for a in argv[3:])
    scheme = kw.get("scheme", "ppmx_rk4")
    if cmd == "prep":
        extra = {}
        for k in ("ns", "late"):
            if k in kw:
                extra[k] = tuple(int(x) for x in kw[k].replace("+", ",").split(",") if x)
        for k in ("tag", "jet"):
            if k in kw:
                extra[k] = kw[k]
        os.makedirs(root, exist_ok=True)
        if "ranks" in kw:
            extra["ranks"] = int(kw["ranks"])
        if "floor" in kw:
            global FLOOR
            FLOOR = float(kw["floor"])
        lines = globals()[f"prep_{test}"](root, scheme, float(kw.get("cfl", 0.4)), **extra)
        with open(os.path.join(root, "runs.txt"), "a") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
    elif cmd == "reduce":
        extra = {"configs": tuple(int(c) for c in kw["configs"].split("+"))} if "configs" in kw else {}
        res = globals()[f"reduce_{test}"](root, scheme, **extra)
        out = kw.get("out")
        prov = provenance(root)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if np.ndim(vv) == 0} for k, v in res.items()},
                         indent=1, default=float))
        if out:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            flat = {f"{k}_{kk}": np.asarray(vv) for k, v in res.items() for kk, vv in v.items()}
            np.savez_compressed(out, **flat, provenance=json.dumps(prov))
            print("wrote", out)
    elif cmd == "luna":                                    # argv[2] is the output stem here, not ROOT
        for k in ("ns", "late"):
            if k in kw:
                kw[k] = tuple(int(x) for x in kw[k].replace("+", ",").split(",") if x)
        kw.pop("scheme", None)
        luna(test, root, **kw)
    elif cmd == "table":                                   # argv[2] is LUNA's json here
        if test == "a5":
            table_a5(root, **kw)
        else:
            table(test, root, **kw)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
