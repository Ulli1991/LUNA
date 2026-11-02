"""DVODE accuracy gate on a saved state: error |a - ref| / (|ref| + atol_i) against the rtol-1e-8 reference.

    python tools/chem_coupled/gate_fixed.py DIR VARIANT[,VARIANT...]   (numpy only; DIR = fixedk/<state>)"""
import json
import os
import sys

import numpy as np

NAMES = ("x_H2", "x_H+", "x_CO", "e")
ATOL = np.array([1e-7, 1e-12, 1e-14, 0.0])      # chemcool_consts.h ATOL_H2, ATOL_HP, ATOL_CO, ATOL_TMP
FLOOR = 1e-6                                   # per-cell comparison floor: 1 % of rtol 1e-4
STEP = 2 ** 24


def errors(get, ref, mask):
    """Per component: max, p99 of the error over the masked cells, and the error arrays' chunks."""
    out = {q: [] for q in NAMES}
    for s in range(0, ref.shape[0], STEP):
        m = mask[s:s + STEP]
        a, b = get(s, s + STEP)[m], np.asarray(ref[s:s + STEP])[m]
        for j, q in enumerate(NAMES):
            out[q].append((np.abs(a[:, j] - b[:, j]) / (np.abs(b[:, j]) + ATOL[j])).astype(np.float64))
    return {q: np.concatenate(v) for q, v in out.items()}


def summary(err, err_dv=None):
    res = {}
    for q in NAMES:
        e = err[q]
        r = dict(max=float(e.max()), p99=float(np.percentile(e, 99)), cells=int(e.size))
        if err_dv is not None:
            d = err_dv[q]
            r.update(worse_than_dvode=int(np.sum(e > d)),
                     worse_than_dvode_floor=int(np.sum(e > np.maximum(d, FLOOR))))
        res[q] = r
    return res


def main():
    d, variants = sys.argv[1], sys.argv[2].split(",")
    ref = np.load(os.path.join(d, "ref8.npy"), mmap_mode="r")
    dv = np.memmap(os.path.join(d, "dvode_out.bin"), dtype=np.float64, mode="r").reshape(-1, 5)[:, :4]
    hard = np.load(os.path.join(d, "ref_ad_att.npy")) > 1
    sets = {"all": np.ones(ref.shape[0], bool), "hard": hard}
    out = dict(state=os.path.basename(d), cells=int(ref.shape[0]), hard=int(hard.sum()),
               metric="|a - ref8| / (|ref8| + atol_i)", atol=ATOL.tolist(), floor=FLOOR)
    path = os.path.join(d, "gate_fixed.json")
    if os.path.exists(path):
        out = {**json.load(open(path)), **out}
    dv_err = {k: errors(lambda a, b: np.asarray(dv[a:b]), ref, m) for k, m in sets.items()}
    out["dvode"] = {k: summary(v) for k, v in dv_err.items()}
    for v in variants:
        y = np.load(os.path.join(d, f"{v}.npy"), mmap_mode="r")
        err = {k: errors(lambda a, b: np.asarray(y[a:b]), ref, m) for k, m in sets.items()}
        res = {k: summary(err[k], dv_err[k]) for k in sets}
        res["pass"] = all(res[k][q][s] <= out["dvode"][k][q][s]
                          for k in sets for q in NAMES for s in ("max", "p99"))
        out[v] = res
        print(v, json.dumps(res), flush=True)
        json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
