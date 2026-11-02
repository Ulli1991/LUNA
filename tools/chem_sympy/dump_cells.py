"""Save the per-cell network-5 inputs of a saved 512^3 state as .npy (run from a chem-profile checkout).
    PYTHONPATH=<chem-profile>/src:<chem-profile> python tools/chem_sympy/dump_cells.py --out DIR
"""
import argparse
import json
import os

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from luna.sources import chemistry as ch  # noqa: E402
from tools.chem_coupled import profile_chem as pc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--state", default="ckpt_step1273")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c, dt, d = pc.load_cells(a.run, a.state, 0)
    np.save(os.path.join(a.out, "y0.npy"), np.asarray(d["y0"]))
    np.save(os.path.join(a.out, "n_h.npy"), np.asarray(d["n_h"]))
    np.save(os.path.join(a.out, "cols.npy"), np.stack([np.asarray(v) for v in d["cols"]]))
    np.save(os.path.join(a.out, "tdust.npy"), np.asarray(d["td"]))
    meta = dict(run=a.run, state=a.state, cells=int(d["y0"].shape[0]), dt_s=float(dt), g0=c.g0,
                zeta=c.zeta, abundc=ch.ABUNDC, abundo=ch.ABUNDO)
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(meta, flush=True)


if __name__ == "__main__":
    main()
