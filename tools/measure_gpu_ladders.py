"""Measure the GPU-tier resolution ladders of the accuracy tests and write them to JSON (bounds come from it).

    python tools/measure_gpu_ladders.py --out gpu_ladders.json            # on the GPU node
    python tools/measure_gpu_ladders.py --out dry.json --dry               # tiny N, CPU check of the plumbing"""
import argparse
import json
import os
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

SOLVERS = ("hllc", "hll")


def norms(e):
    e = np.asarray(e)
    return [float(np.abs(e).mean()), float(np.sqrt((e * e).mean())), float(np.abs(e).max())]


def cases(dry):
    """(key, thunk) for every measurement; dry: the smallest sizes only."""
    import tests.test_hydro_accuracy_linear_waves as lw
    import tests.test_hydro_accuracy_lw_smooth as lws
    import tests.test_hydro_accuracy_shocktube as st
    import tests.test_hydro_accuracy_smooth as sm
    import tests.test_hydro_hard_low_mach as lm
    import tests.test_hydro_hard_lw_gresho as gr
    import tests.test_hydro_hard_vortex as vx
    from luna.hydro import exact_riemann as er

    pick = (lambda big, small: small) if dry else (lambda big, small: big)
    out = []
    for sd, ns in ((1, pick((1024, 2048, 4096), (32,))), (2, pick((512, 1024), (32,)))):
        for sc in lw.SCHEMES:
            for mode in lw.MODES[sd]:
                for n in ns:
                    out.append((f"linear_waves/{sd}/{sc}/{mode}/{n}", lambda sd=sd, n=n, mode=mode, sc=sc: list(lw.errors(sd, n, mode, sc))))
    for sc in lws.SCHEMES:
        for n in pick((512, 1024), (32,)):
            out.append((f"lw_smooth/{sc}/{n}", lambda n=n, sc=sc: list(lws.errors(n, sc))))
    for sc in gr.SCHEMES:
        for mv in (False, True):
            for n in pick((512, 1024), (32,)):
                out.append((f"lw_gresho/{sc}/{mv}/{n}", lambda n=n, sc=sc, mv=mv: {k: list(v) for k, v in gr.periodic(n, sc, mv).items()}))
    for sc in lm.SCHEMES:
        for n in pick((128, 256, 512), (16,)):
            for m in lm.MACHS:
                if n >= 512 and m < 0.01:
                    continue
                out.append((f"low_mach/{sc}/{n}/{m}", lambda n=n, m=m, sc=sc: list(lm.gresho(n, m, sc))))
    for solver in SOLVERS:
        for d in vx.DIRS:
            for n in pick((256, 512), (32,)):
                per = (1, 10, 100) if not dry else (1,)
                out.append((f"vortex/{solver}/{d}/{n}", lambda n=n, d=d, per=per, solver=solver: [list(m) for m in vx.vortex(n, d, per, solver=solver)]))
    for solver in SOLVERS:
        for name, ns in (("sound", pick((512, 1024, 2048), (32,))), ("sound_adv", pick((512, 1024, 2048), (32,))),
                         ("gresho", pick((256, 512), (32,))), ("square", pick((256, 512), (64,))), ("sine", pick((256, 512), (32,)))):
            for n in ns:
                out.append((f"smooth/{name}/{solver}/{n}", lambda name=name, n=n, solver=solver: norms(sm.RUNS[name](n, solver)[2])))
    for solver in SOLVERS:
        for case in er.TORO:
            for n in pick((1024, 2048, 4096), (128,)):
                def toro(case=case, n=n, solver=solver):
                    num, ex, _, _ = st.run_1d(case, n, solver)
                    return [norms(a - b) for a, b in zip(num, ex)]
                out.append((f"toro/{case}/{solver}/{n}", toro))
        for d in st.DIRS:
            n = pick(256, 32) if len(d) == 1 else pick(128, 16)
            out.append((f"sod3d/{solver}/{d}/{n}", lambda d=d, n=n, solver=solver: st.run_tube(d, n, solver=solver)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--only", default="", help="run only keys starting with this prefix")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    res = {"meta": {"git": rev, "backend": jax.default_backend(), "devices": str(jax.devices()), "dry": a.dry}, "data": {}}
    if os.path.exists(a.out):
        res["data"] = json.load(open(a.out)).get("data", {})
    for key, run in cases(a.dry):
        if not key.startswith(a.only) or key in res["data"]:
            continue
        t0 = time.time()
        try:
            res["data"][key] = {"value": run(), "seconds": round(time.time() - t0, 1)}
        except Exception as exc:                     # record and go on: one failure must not lose the night
            res["data"][key] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        print(key, res["data"][key], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
