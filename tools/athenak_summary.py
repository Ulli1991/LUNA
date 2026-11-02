"""Group A against AthenaK: the "where LUNA stands" table and figure, rebuilt from the stored files only.

    python tools/athenak_summary.py root=WORKDIR [out=ROOT/summary] [paper=1]

Every number is read from a file: AthenaK npz in luna-reference-solutions athenak/ (job ids from their provenance), LUNA's json
from `tools/athenak_refs.py luna` under ROOT (job id in its meta), or the 'measured' comment of a test's GPU bounds
(file and line). A missing file stops the script. Writes OUT/summary.md, OUT/summary.json and
OUT/athenak_group_a.{png,pdf}; paper=1 also writes code_paper_figs/athenak_group_a.png.
"""
import json
import os
import re
import sys

import numpy as np

from luna.reference_data import reference_named, reference_root

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_TOP = 1600.0                     # top of the A15 viscous reference range: Re_eff there is a lower bound
A10_NOTE = ("About 0.8x PLM and 1.2x PPMX with a ~5 % reference-bias uncertainty (against the exact planar solution, "
            "critic_a10: MC / PLM 0.72, MC / PPMX 1.20). Cost to t = 0.045 at 800^2, loop time on one A100: LUNA MC "
            "0.55 s (8.7e8 zone-steps/s; 2.0 s with compilation, `luna_tests/lit_a10/athenak/luna_timed.json`, job "
            "31169580), AthenaK PLM 3.1 s, PPMX 8.2 s, PPM4 6.7 s (MeshBlock 50^2, from its zone-cycles/s); LUNA MC "
            "reaches PPMX's 800^2 error at N ~ 970-1020 (estimate), 0.66 s at 1024^2. Equal dt (LUNA CFL 0.614, 978 "
            "vs 980 steps, job 31169579): MC 4.49e-4. AthenaK's no-FMA PPMX 6400^2 reference is 5.9e-5 (L1) from "
            "LUNA's (self-convergence 8.0e-5 and 8.7e-5); against it the ratios are given below. AthenaK dfloor = "
            "pfloor = 1e-8; its default (FMA) build breaks the diagonal mirror symmetry from round-off, the "
            "-fmad=false build is exact, as LUNA.")
COLS = ("minmod", "mc", "vl", "plm", "ppmx", "ppm4")
HEAD = ("LUNA minmod (default)", "LUNA MC", "LUNA van Leer", "AthenaK PLM+RK2", "AthenaK PPMX+RK4",
        "AthenaK PPM4+RK4")


def npz(name, key, root=None):
    """(value, source) of a scalar in an AthenaK npz (luna-reference-solutions by file name, or ROOT-relative); job ids from
    its provenance."""
    p = os.path.join(root, name) if root else reference_named(name)
    z = np.load(p)
    jobs = sorted({str(r["job.json"]["slurm_job_id"]) for r in json.loads(str(z["provenance"]))["runs"].values()
                   if "job.json" in r})
    return float(z[key]), f"`{_short(p)}` `{key}` (AthenaK jobs {', '.join(jobs)})"


def luna_json(path, key, met):
    """(value, source) from a LUNA json: `athenak_refs.py luna` output (meta + data) or the critic's flat json."""
    d = json.load(open(path))
    data, job = (d["data"], d["meta"]["slurm_job_id"]) if "meta" in d else (d, None)
    return float(data[key][met]), f"`{_short(path)}` `{key}.{met}`" + (f" (job {job})" if job else "")


def _short(p):
    """A ptmp path as luna_tests/..., a reference as luna-reference-solutions/..., a repo path relative to the repo."""
    rr = str(reference_root())
    if os.path.abspath(p).startswith(rr + os.sep):
        return "luna-reference-solutions/" + os.path.relpath(p, rr)
    if os.path.abspath(p).startswith(REPO + os.sep):          # the repo first: a worktree may live in luna_tests/
        return os.path.relpath(p, REPO)
    return p[p.index("luna_tests/"):] if "luna_tests/" in p else os.path.relpath(p, REPO)


def comment(test, anchor, field, idx=2):
    """(value, source): entry idx of `field a, b, c;` in the comment after `anchor` within a test file's
    BOUNDS_GPU (the 'measured' values of its GPU ladder 128, 256, 512)."""
    rel = os.path.join("tests", test)
    lines = open(os.path.join(REPO, rel)).read().splitlines()
    g = next(k for k, ln in enumerate(lines) if ln.startswith("BOUNDS_GPU"))
    assert "128" in lines[g + 1] and "512" in lines[g + 1], f"{rel}: GPU ladder is not 128..512"
    i = next(k for k in range(g, len(lines)) if anchor in lines[k])
    j = next(k for k in range(i, len(lines)) if lines[k].lstrip().startswith("#"))
    for k in range(j, len(lines)):
        if not lines[k].lstrip().startswith("#"):
            break
        m = re.search(rf"\b{field} ([-0-9.e+]+), ([-0-9.e+]+), ([-0-9.e+]+);", lines[k])
        if m:
            return float(m.group(idx + 1)), f"`{rel}:{k + 1}` ({field}, measured, no job id)"
    raise KeyError(f"{field} not in the comment after {anchor!r} in {rel}")


def rows(root):
    """The table rows: id, metric, N, {column: (value, source) or None}, higher-is-better, caveat."""
    s, a3, a4, a9 = (os.path.join(root, d) for d in ("summary", "lit_a03", "lit_a04", "lit_a09"))
    a10 = os.path.join(root, "lit_a10", "athenak")
    crit = os.path.join(root, "critic_p1", "out", "luna_extra.json")
    a5l = os.path.join(root, "lit_a05", "luna_a5.json")
    a6l = os.path.join(root, "lit_a06", "luna_a6.json")
    rt, rm = "test_hydro_hard_rayleigh_taylor.py", "test_hydro_hard_richtmyer_meshkov.py"
    out = [
        dict(id="A1", metric="RMS over variables of L1(U - U0) / EPS, sound wave after one period (AthenaK's "
             "`errs`), vs exact", N="1024 (1D)",
             v=dict(minmod=luna_json(f"{s}/luna_a1.json", "hllc-minmod_N1024", "rms"),
                    mc=luna_json(f"{s}/luna_a1.json", "hllc-mc_N1024", "rms"),
                    vl=luna_json(f"{s}/luna_a1.json", "hllc-vl_N1024", "rms"),
                    plm=npz("a1_plm_rk2_1024.npz", "N1024_errs_rms"),
                    ppmx=npz("a1_ppmx_rk4_1024.npz", "N1024_errs_rms")),
             note="Single sign of EPS (AthenaK's metric), so the EPS^2 nonlinear floor (~5.5e-6, PPMX's value from "
                  "N = 128 on) is in every number; the PPMX ratio compares with that floor, not with a scheme error. "
                  "LUNA CFL 0.8, AthenaK 0.4."),
        dict(id="A3", metric="L1(rho) vs exact cell averages, LW smooth advection, T = 4", N="1024^2",
             v=dict(minmod=luna_json(f"{a3}/luna_gpu.json", "hllc-minmod_N1024", "L1"),
                    mc=luna_json(f"{a3}/luna_gpu.json", "hllc-mc_N1024", "L1"),
                    vl=luna_json(f"{a3}/luna_gpu_vl.json", "hllc-vl_N1024", "L1"),
                    plm=npz("a3_plm_rk2_1024.npz", "N1024_L1"), ppmx=npz("a3_ppmx_rk4_1024.npz", "N1024_L1")),
             note="LUNA CFL 0.8 (sum rule), AthenaK 0.4 (min rule). LUNA MC's error grows as dt shrinks "
                  "(MUSCL-Hancock space-time cancellation), AthenaK's does not: see the equal-dt row."),
        dict(id="A3 equal dt", metric="as A3, LUNA at CFL 0.714 = AthenaK's step count", N="1024^2",
             v=dict(mc=luna_json(crit, "a3_hllc-mc-m_N1024", "L1"), vl=luna_json(crit, "a3_hllc-vl-m_N1024", "L1"),
                    plm=npz("a3_plm_rk2_1024.npz", "N1024_L1"), ppmx=npz("a3_ppmx_rk4_1024.npz", "N1024_L1")),
             note="Critic run, job 31107131 (`critic_p1/jobs/luna.luna_extra.31107131.out`: 11896 steps vs AthenaK 11893); minmod not run at equal dt."),
        dict(id="A4 static", metric="L1 of abs(v - v0) vs exact, Gresho p(0) = 5, one revolution", N="1024^2",
             v=dict(minmod=luna_json(f"{a4}/luna_gpu_minmod.json", "hllc-minmod_static_N1024", "v_L1"),
                    mc=luna_json(f"{a4}/luna_gpu_mc.json", "hllc-mc_static_N1024", "v_L1"),
                    vl=luna_json(f"{a4}/luna_gpu_vl.json", "hllc-vl_static_N1024", "v_L1"),
                    plm=npz("a4_plm_rk2_1024.npz", "static_N1024_v_L1"),
                    ppmx=npz("a4_ppmx_rk4_1024.npz", "static_N1024_v_L1"),
                    ppm4=npz("a4_ppm4_rk4_1024.npz", "static_N1024_v_L1")),
             note="dt has no effect (LUNA MC at CFL 0.4: 1.329e-4). PPMX flattens from 512 (its extremum-preserving "
                  "limiter); PPM4 also beats MC in rho and p here."),
        dict(id="A4 moving", metric="as A4 static, bulk flow one box per revolution", N="1024^2",
             v=dict(minmod=luna_json(f"{a4}/luna_gpu_minmod.json", "hllc-minmod_moving_N1024", "v_L1"),
                    mc=luna_json(f"{a4}/luna_gpu_mc.json", "hllc-mc_moving_N1024", "v_L1"),
                    vl=luna_json(f"{a4}/luna_gpu_vl.json", "hllc-vl_moving_N1024", "v_L1"),
                    plm=npz("a4_plm_rk2_1024.npz", "moving_N1024_v_L1"),
                    ppmx=npz("a4_ppmx_rk4_1024.npz", "moving_N1024_v_L1"),
                    ppm4=npz("a4_ppm4_rk4_1024.npz", "moving_N1024_v_L1")),
             note="In rho and p LUNA MC is below PPMX and PPM4 (1.49e-6 vs 1.91e-6 / 1.84e-6 rho L1)."),
        dict(id="A5 M", metric="max over t <= 1.5 of abs(M / M_ref - 1), McNally KH mode amplitude vs the Pencil "
             "4096^2 curve (fig. 4)", N="2048^2",
             v=dict(minmod=luna_json(a5l, "minmod_N2048", "eM_max"), mc=luna_json(a5l, "mc_N2048", "eM_max"),
                    vl=luna_json(a5l, "vanleer_N2048", "eM_max"), plm=npz("a5_plm_rk2_2048.npz", "N2048_eM_max"),
                    ppmx=npz("a5_ppmx_rk4_4096.npz", "N2048_eM_max"),
                    ppm4=npz("a5_ppm4_rk4_2048.npz", "N2048_eM_max")),
             note="AthenaK runs chained per output interval (75 runs of 0.02 from the previous final dump) so M is "
                  "read at t = 0.02 k exactly. LUNA CFL 0.8 takes 0.90x AthenaK's steps; at equal dt (LUNA CFL "
                  "0.72) M moves by <= 5 % (`lit_a05/luna_a5_cfl0.72.json`), at CFL 0.4 MC's 1024 value by +19 %. "
                  "MC / PPMX is 0.24-0.53x at 512-2048 in every metric (A5 critic); against the PLM family the "
                  "ranking depends on metric and CFL. Point-sampled IC (the paper's): cell averages raise every "
                  "code's M error 1.7-2.7x. AthenaK PPMX 4096^2 is 2.1e-5 from "
                  "fig. 4, its order-2 Richardson value from 2048 and 4096 1.2e-6 (LUNA MC 4096^2: 1.1e-5)."),
        dict(id="A5 fields", metric="L1(V_y) at t = 1.5 on 256^2 vs AthenaK PPMX + RK4 4096^2 "
             "(`athenak/mcnally_kh/a5_ppmx_rk4_4096.npz`)", N="1024^2",
             v=dict(minmod=luna_json(a5l, "minmod_N1024", "vy_L1_ak"), mc=luna_json(a5l, "mc_N1024", "vy_L1_ak"),
                    vl=luna_json(a5l, "vanleer_N1024", "vy_L1_ak"), plm=npz("a5_plm_rk2_2048.npz", "N1024_vy_L1_ak"),
                    ppmx=npz("a5_ppmx_rk4_4096.npz", "N1024_vy_L1_ak"),
                    ppm4=npz("a5_ppm4_rk4_2048.npz", "N1024_vy_L1_ak")),
             note="The test's field_errors with the AthenaK reference in place of the LUNA MC 4096^2 one; the two "
                  "4096^2 references differ by L1(V_y) 2.3e-6 on 256^2, AthenaK 4096 vs its 2048 by 4.5e-6. "
                  "L1 orders 64-1024 are ~2 for every scheme against either reference. MC / PPMX is 1.3-1.5x in V_y, "
                  "1.5-2.4x in rho (L1, any reference or grid), 3-4x in L2 and Linf."),
        dict(id="A6", metric="L1(rho(T) - rho(0)), drift from the exact hydrostatic state, isothermal atmosphere "
             "rho = p = exp(-phi), phi = sin 2 pi x, T = 2 / sqrt(gamma)", N="2048 (1D)",
             v={sc: luna_json(a6l, f"hllc-{k}_N2048", "rho_L1") for sc, k in (("minmod", "minmod"), ("mc", "mc"),
                                                                                ("vl", "vl"))}
             | {sc: npz(f"a6_{f}_2048.npz", "N2048_rho_L1") for sc, f in (("plm", "plm_rk2"), ("ppmx", "ppmx_rk4"),
                                                                           ("ppm4", "ppm4_rk4"))},
             note="Neither code is well balanced; both add rho g at the cell centres (LUNA Strang-split around "
                  "MUSCL-Hancock, AthenaK in every RK stage, pgen luna_ic_a6). AthenaK PLM converges at order 2.9 "
                  "(128-1024; 2.6 at 512-2048), every LUNA limiter and AthenaK PPMX/PPM4 at 2.0. AthenaK's drift does "
                  "not depend on dt: at LUNA's dt (CFL 0.8, LUNA's step counts) PLM gives "
                  f"{npz('a6eqdt_plm_rk2_2048.npz', 'N2048_rho_L1')[0]:.3g}, PPMX "
                  f"{npz('a6eqdt_ppmx_rk4_2048.npz', 'N2048_rho_L1')[0]:.3g}. LUNA's does: MC at 1024 8.8e-7 at "
                  "CFL 0.8, 6.8e-7 at 0.4, 3.8e-7 at 0.2 (`lit_a06/luna_a6_cfl.json`, job 31172324), still 3.4x "
                  "AthenaK PLM at 0.2. Cost is launch latency at 2048 cells in 1D (AthenaK PLM 0.48 ms per step)."),
        dict(id="A6 energy", metric="abs(d(E + rho phi)) / (E + rho phi) at T, the static-potential energy budget",
             N="2048 (1D)",
             v={sc: luna_json(a6l, f"hllc-{k}_N2048", "budget") for sc, k in (("minmod", "minmod"), ("mc", "mc"),
                                                                                ("vl", "vl"))}
             | {sc: npz(f"a6_{f}_2048.npz", "N2048_budget") for sc, f in (("plm", "plm_rk2"), ("ppmx", "ppmx_rk4"),
                                                                           ("ppm4", "ppm4_rk4"))},
             note="A truncation drift in both codes (g at cell centres, not the discrete gradient of phi). AthenaK "
                  "PPMX sits at 1e-11..1e-14 from 256 on, not a converging series; LUNA MC, van Leer and AthenaK PLM "
                  "converge at 3.0, PPM4 at 4.0, minmod at 2.0."),
        dict(id="A9", metric="L1(rho) of the N x 16 strip vs reference (LUNA MC 2^16, "
             "`luna/woodward_colella/woodward_colella_mc.npz`), WC blast, T = 0.038", N="2048",
             v=dict(minmod=luna_json(f"{a9}/luna_gpu_v2.json", "hllc-minmod_N2048", "L1_rho_strip"),
                    mc=luna_json(f"{a9}/luna_gpu_v2.json", "hllc-mc_N2048", "L1_rho_strip"),
                    vl=luna_json(f"{a9}/luna_gpu_v2.json", "hllc-vl_N2048", "L1_rho_strip"),
                    plm=npz("a9_plm_rk2_16384.npz", "N2048_r_L1_rho_strip"),
                    ppmx=npz("a9_ppmx_rk4_16384.npz", "N2048_r_L1_rho_strip"),
                    ppm4=npz("a9_ppm4_rk4_16384.npz", "N2048_r_L1_rho_strip")),
             alt=dict(mc=luna_json(f"{a9}/luna_gpu_v2.json", "hllc-mc_N2048", "L1_rho_strip_akplm"),
                      plm=npz("a9_plm_rk2_16384.npz", "N2048_r_L1_rho_strip_akplm"),
                      ppmx=npz("a9_ppmx_rk4_16384.npz", "N2048_r_L1_rho_strip_akplm")),
             alt_name="AthenaK PLM 16384",
             note="Reference choice: the LUNA MC 2^16 run (PPMX 16384 is 0.0019 from it at 1024 cells); against "
                  "AthenaK PLM 16384 instead the ratios are given below. AthenaK PPMX/PPM4 need dfloor = pfloor = "
                  "1e-8 (default FLT_MIN floors: NaN at the collision). Symmetry checks exactly 0 for all."),
        dict(id="A10", metric="L1(rho) vs reference (LUNA HLLC-MC 6400^2, `luna/lw_implosion/lw_implosion_mc.npz`), "
             "LW implosion, t = 0.045", N="800^2",
             v=dict(minmod=luna_json(f"{a10}/luna_hllc-minmod.json", "hllc-minmod_N800_e", "L1"),
                    mc=luna_json(f"{a10}/luna_hllc-mc.json", "hllc-mc_N800_e", "L1"),
                    vl=luna_json(f"{a10}/luna_hllc-vl.json", "hllc-vl_N800_e", "L1"),
                    plm=npz("a10_plm_rk2_1024.npz", "N800_e_L1"), ppmx=npz("a10_ppmx_rk4_6400.npz", "N800_e_L1"),
                    ppm4=npz("a10_ppm4_rk4_1024.npz", "N800_e_L1")),
             alt=dict(mc=luna_json(f"{a10}/luna_hllc-mc.json", "hllc-mc_N800_e", "L1_akppmx"),
                      plm=npz("a10_plm_rk2_1024.npz", "N800_e_L1_akppmx"),
                      ppmx=npz("a10_ppmx_rk4_6400.npz", "N800_e_L1_akppmx")),
             alt_name="AthenaK PPMX 6400^2 (-fmad=false)",
             note=A10_NOTE),
    ]
    a11 = os.path.join(root, "lit_a11", "ak", "out", "luna_gpu.json")
    ak11 = {"plm": "a11_c{}_plm_rk2_2048.npz", "ppmx": "a11_c{}_ppmx_rk4_8192.npz"}

    def kt(c, n, met):
        return {sc: luna_json(a11, f"c{c}_hllc-{sc}_N{n}", met) for sc in ("minmod", "mc", "vl")} | {
            sc: npz(f.format(c), f"c{c}_N{n}_{met}") for sc, f in ak11.items()}

    out += [
        dict(id="A11 c3 strip", metric="KT config 3 (four shocks), T = 0.3: L1(rho) of the strip along the y = 1 wall "
             "(interface 21, backward shock) vs exact 1D cell averages", N="2048^2", v=kt(3, 2048, "strip21_L1"),
             note="Exact 1D, no reference involved. LUNA HLLC vs AthenaK HLLC: AthenaK's error is its HLLC on a slowly "
                  "moving shock (the 1D Riemann problem alone gives it; order ~0.2). AthenaK PLM with HLLE matches "
                  "LUNA MC (1D, 2048: 2.20e-4 vs 2.32e-4; `luna_tests/critic_a11/critic.md` O1)."),
        dict(id="A11 c3", metric="KT config 3: L1(rho) outside the core box on 128^2 blocks vs reference (LUNA MC "
             "8192^2, `luna/kt_riemann/kt_riemann_c3_mc.npz`), core (0.05, 0.65)^2", N="2048^2", v=kt(3, 2048, "L1_out"),
             alt={sc: v for sc, v in kt(3, 2048, "L1_out_ak").items() if sc in ("mc", "plm", "ppmx")},
             alt_name="AthenaK PPMX 8192^2",
             note="AthenaK PPMX/PLM with dfloor = pfloor = 1e-8. All five schemes converge at order ~1 (256-2048); "
                  "the two 8192^2 references differ by 1.5e-4 here (LUNA's own 4096 vs 8192: 1.1e-4)."),
        dict(id="A11 c6 strip", metric="KT config 6 (four slip lines), T = 0.3: L1(rho) of the strip along the y = 0 "
             "wall (interface 34, slip line) vs exact 1D cell averages", N="2048^2", v=kt(6, 2048, "strip34_L1"),
             note="Exact 1D. AthenaK PPMX is best to 2048; from 4096 its slip lines roll up (Kelvin-Helmholtz): "
                  "front offsets 5.8 cells (strip 21, 4096) and 16 cells (strip 34, 8192)."),
        dict(id="A11 c6", metric="KT config 6: L1(rho) outside the core box on 128^2 blocks vs reference (LUNA MC "
             "8192^2, `luna/kt_riemann/kt_riemann_c6_mc.npz`)", N="1024^2", v=kt(6, 1024, "L1_out"),
             note="The reference is the laminar solution, code-independent (AthenaK PLM 4096 is 2.75e-4 from LUNA MC "
                  "4096); against an AthenaK PLM 4096 reference the ratios move <= ~15 % (MC / PLM 0.66, MC / PPMX "
                  "1.75 at 1024). From 2048 AthenaK PPMX's slip lines roll up (L1 5.1e-3 at 2048, 2.7e-2 at 8192), "
                  "so AthenaK PPMX 8192^2 is no reference for config 6."),
    ]
    out += [
        dict(id="A13", metric="L1 of the odd part of v_y vs the exact linear eigenmode, RT, T fixed", N="512 x 2048",
             v=dict(minmod=comment(rt, '"hllc-minmod": dict(L1={128', "L1"),
                    mc=comment(rt, '"hllc-mc": dict(L1={128', "L1"),
                    ppmx=npz("rt_ppmx_rk4_512.npz", "N512_L1")),
             note="AthenaK PLM and LUNA van Leer not run. LUNA values are the test's recorded GPU measurements."),
        dict(id="A13 rate", metric="abs(sigma_meas / sigma - 1), RT growth-rate error", N="512 x 2048",
             v=dict(minmod=comment(rt, '"hllc-minmod": dict(L1={128', "sigma"),
                    mc=comment(rt, '"hllc-mc": dict(L1={128', "sigma"),
                    ppmx=npz("rt_ppmx_rk4_512.npz", "N512_sigma")), note="As A13."),
    ]
    for case in ("heavy-light", "light-heavy"):
        for met, name in (("L1", "L1 of psi / psi0 vs exact (Laplace route), 2 <= tau <= 20"),
                          ("rate", "abs(rate error), fitted post-shock growth rate vs exact")):
            out.append(dict(
                id=f"A14 {case}" + (" rate" if met == "rate" else ""), metric=f"RM {name}", N="512 per wavelength",
                v=dict(minmod=comment(rm, f"('{case}', 'hllc-minmod'): dict(", met),
                       mc=comment(rm, f"('{case}', 'hllc-mc'): dict(", met),
                       ppmx=npz("rm_ppmx_rk4_512.npz", f"{case}_N512_{met}")),
                note="AthenaK PLM and LUNA van Leer not run; both codes converge at ~first order (vortex sheet)."))
    g = os.path.join(root, "a07_a08_ak")
    for case, what in (("stable", "n_J = 0.5, one period"), ("unstable", "n_J = 1.5, one e-fold")):
        out.append(dict(
            id=f"A7 {case}", metric=f"L1 of (rho - exact) / AMP, Jeans wave {what}", N="128 x 64^2",
            v=dict(minmod=luna_json(f"{g}/luna_a7_auto.json", f"hllc-minmod_{case}_N128", "L1"),
                   mc=luna_json(f"{g}/luna_a7_auto.json", f"hllc-mc_{case}_N128", "L1"),
                   vl=luna_json(f"{g}/luna_a7_auto.json", f"hllc-vl_{case}_N128", "L1"),
                   plm=npz("a7_plm_rk2_128.npz", f"{case}_N128_L1"),
                   ppmx=npz("a7_ppmx_rk4_128.npz", f"{case}_N128_L1")),
             note="LUNA gravity_energy='auto' (flux, job 31174675; kick moved the unstable L1 by <= 2 %). "
                  "AthenaK multigrid self-gravity (upstream c5a0d7f, binary build_gpu_lunaic_mg), CFL 0.3 (PLM + RK2 "
                  "at 0.4 grows a 3D instability by 128); at LUNA's dt (CFL 0.8/3, `a7eqdt_*`) AthenaK PLM gives "
                  f"{npz('a7eqdt_plm_rk2_128.npz', f'{case}_N128_L1')[0]:.3g}, PPMX "
                  f"{npz('a7eqdt_ppmx_rk4_128.npz', f'{case}_N128_L1')[0]:.3g}."))
    out.append(dict(
        id="A8", metric="abs(E(t) - E0) / abs(E0) at t = 3, periodic Evrard collapse (L = 4R), W from the 7-point "
        "periodic solve", N="256^3",
        v=dict(minmod=luna_json(f"{g}/luna_a8_auto.json", "hllc-minmod_N256", "dE_t3"),
               mc=luna_json(f"{g}/luna_a8_auto.json", "hllc-mc_N256", "dE_t3"),
               vl=luna_json(f"{g}/luna_a8_auto.json", "hllc-vl_N256", "dE_t3"),
               plm=npz("a8_plm_rk2_256.npz", "N256_dE_t3"), ppmx=npz("a8_ppmx_rk4_256.npz", "N256_dE_t3")),
        note="Both codes couple gravity to the energy through the face mass fluxes (Mullen, Hanawa & Gammie 2020): "
             "LUNA's default gravity_energy='auto' resolves to flux here (job 31174675); the old kick coupling gave "
             "0.24 (MC) .. 0.58 (minmod) at 256^3 (`luna_a8_all.json`, job 31167760). AthenaK floors 1e-8, CFL 0.3. "
             "LUNA flux mode goes NaN at N = 32 (step 1, all limiters) and at 64 near the bounce (minmod, MC; van "
             "Leer 64 runs), where kick mode ran; 128 and 256 are clean."))
    for n in (64, 128):
        out.append(dict(
            id="A15", metric="Re_eff (higher is better): eps(t) matched to viscous spectral TG, M = 0.1; "
            "error ratio = Re_eff(AthenaK) / Re_eff(LUNA)", N=f"{n}^3", higher=True,
            v=dict(minmod=luna_json(f"{s}/luna_tg.json", f"minmod_N{n}", "Re_eff"),
                   mc=luna_json(f"{s}/luna_tg.json", f"mc_N{n}", "Re_eff"),
                   vl=luna_json(f"{s}/luna_tg.json", f"vl_N{n}", "Re_eff"),
                   ppmx=npz("phase0/tg_ppmx_rk4_reeff.npz", f"N{n}_Re_eff", root=root)),
            note="Reference range Re 25..1600: a value at 1600 is a lower bound (AthenaK from 128^3, LUNA MC at "
                 "256^3), so 64^3 is the finest N with both finite. AthenaK Re_eff reduced by job 31083046 from "
                 "the K(t) in `athenak/taylor_green/tg_ppmx_rk4_256.npz`."))
    return out


def ratio(r, a, b):
    """Error ratio a / b of two columns (Re_eff: b / a), with a bound flag; None if either is missing."""
    va, vb = r["v"].get(a), r["v"].get(b)
    if va is None or vb is None:
        return None
    x, y = va[0], vb[0]
    if r.get("higher"):
        lb = y >= RE_TOP * (1 - 1e-4)
        return y / x, (">=" if lb else "")
    return x / y, ""


def fmt(r, c):
    v = r["v"].get(c)
    if v is None:
        return "not measured"
    if r.get("higher") and v[0] >= RE_TOP * (1 - 1e-4):
        return f">= {RE_TOP:.0f}"
    return f"{v[0]:.3g}"


def fr(q):
    return "-" if q is None else f"{q[1]} {q[0]:.3g}".strip()


def table(rs):
    """Markdown: the table, then per row the caveat and the source of every number."""
    t = ["| row | metric | N | " + " | ".join(HEAD) + " | MC / PLM | MC / PPMX |",
         "|---|---|---|" + "---|" * (len(HEAD) + 2)]
    for r in rs:
        t.append(f"| {r['id']} | {r['metric'].replace('|', '/')} | {r['N']} | " + " | ".join(fmt(r, c) for c in COLS)
                 + f" | {fr(ratio(r, 'mc', 'plm'))} | {fr(ratio(r, 'mc', 'ppmx'))} |")
    t.append("\nNotes and sources (AthenaK: CFL 0.4, HLLC, FP64, one A100; LUNA: MUSCL-Hancock, HLLC, CFL 0.8, "
             "FP64, one A100):\n")
    for r in rs:
        src = "; ".join(f"{c}: {r['v'][c][1]}" for c in COLS if r["v"].get(c))
        extra = ""
        if "alt" in r:
            al = r["alt"]
            extra = (f" Against {r['alt_name']}: MC {al['mc'][0]:.3g}, PLM {al['plm'][0]:.3g}, PPMX "
                     f"{al['ppmx'][0]:.3g}; MC / PLM {al['mc'][0] / al['plm'][0]:.3g}, MC / PPMX "
                     f"{al['mc'][0] / al['ppmx'][0]:.3g}.")
        t.append(f"- {r['id']}: {r['note']}{extra} Sources: {src}.")
    return "\n".join(t)


def figure(rs, paths):
    """One panel: LUNA / AthenaK error ratio per row, three LUNA limiters against PLM (filled) and PPMX (open)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(minmod="#2a78d6", mc="#eb6834", vl="#1baf7a")
    names = dict(minmod="LUNA minmod (default)", mc="LUNA MC", vl="LUNA van Leer")
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ys = np.arange(len(rs))[::-1]
    for y, r in zip(ys, rs):
        for k, sc in enumerate(colors):
            for ak, mk, fill in (("plm", "o", True), ("ppmx", "s", False)):
                q = ratio(r, sc, ak)
                if q is None:
                    continue
                yy = y + (k - 1) * 0.18 + (0.06 if ak == "ppmx" else -0.06)
                ax.plot(q[0], yy, mk, ms=7, mew=1.6, color=colors[sc], mfc=colors[sc] if fill else "white")
                if q[1]:
                    ax.annotate("", xy=(q[0] * 1.8, yy), xytext=(q[0], yy),
                                arrowprops=dict(arrowstyle="->", color=colors[sc], lw=1.2))
    ax.axvline(1.0, color="#52514e", lw=1.0)
    ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['id']} ({r['N']})" for r in rs], fontsize=8.5)
    ax.set_xlabel("error ratio LUNA / AthenaK  (A15: Re_eff AthenaK / Re_eff LUNA)")
    ax.grid(axis="x", which="major", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.text(1.0 / 1.15, len(rs) - 0.2, "LUNA smaller error", ha="right", fontsize=8, color="#52514e")
    ax.text(1.15, len(rs) - 0.2, "AthenaK smaller error", ha="left", fontsize=8, color="#52514e")
    ax.set_ylim(-0.7, len(rs) + 0.2)
    h = [plt.Line2D([], [], ls="", marker="o", ms=7, color=c, label=names[s]) for s, c in colors.items()]
    h += [plt.Line2D([], [], ls="", marker="o", ms=7, color="#52514e", label="vs AthenaK PLM+RK2 (filled)"),
          plt.Line2D([], [], ls="", marker="s", ms=7, mew=1.6, color="#52514e", mfc="white",
                     label="vs AthenaK PPMX+RK4 (open)")]
    ax.legend(handles=h, fontsize=8, loc="center right", frameon=False)
    ax.set_title("Group A: LUNA against AthenaK at the finest common N", fontsize=10)
    fig.tight_layout()
    for p in paths:
        fig.savefig(p, dpi=150)


def main(argv):
    kw = dict(a.split("=", 1) for a in argv)
    root = kw["root"]                    # the directory holding the lit_aNN/ work directories
    out = kw.get("out", os.path.join(root, "summary"))
    os.makedirs(out, exist_ok=True)
    rs = rows(root)
    md = table(rs)
    print(md)
    open(os.path.join(out, "summary.md"), "w").write(md + "\n")
    js = {f"{r['id']} {r['N']}": dict(metric=r["metric"], N=r["N"], values={c: dict(value=v[0], source=v[1])
                                                              for c, v in r["v"].items()},
                        mc_over_plm=ratio(r, "mc", "plm"), mc_over_ppmx=ratio(r, "mc", "ppmx")) for r in rs}
    json.dump(js, open(os.path.join(out, "summary.json"), "w"), indent=1)
    paths = [os.path.join(out, "athenak_group_a.png"), os.path.join(out, "athenak_group_a.pdf")]
    if kw.get("paper") == "1":
        paths.append(os.path.join(REPO, "code_paper_figs", "athenak_group_a.png"))
    figure([r for r in rs if r.get("figure", True)], paths)      # A8: energy coupling, table only
    print("wrote", out, *paths)


if __name__ == "__main__":
    main(sys.argv[1:])
