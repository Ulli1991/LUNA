"""Pack Fortran one-zone trajectories (run_grid.py output) into the compact test format (numpy only).

    python tools/chem_integrator/pack_reference.py <net5_trajectories.npz> [out.npz]
"""
import os
import sys

import numpy as np

DST = os.path.join(os.environ.get("LUNA_DATA", "."), "chemistry", "chemistry_net5_trajectories.npz")
QTY = ("xH2", "xHp", "xCO", "e", "T")


def packed(d):
    """Case names and params, output times, y0 (float64) and x_H2, x_H+, x_CO, e, T (float32)."""
    names = d["case_names"]
    cols = list(d["columns"])
    traj = np.stack([d["traj_" + n] for n in names])
    return dict(case_names=names, case_params=d["case_params"], t=traj[0, :, 0],
                y0=traj[:, 0, 1:5], ref=traj[:, :, [cols.index(c) for c in QTY]].astype(np.float32))


def main():
    out = sys.argv[2] if len(sys.argv) > 2 else DST
    np.savez_compressed(out, **packed(np.load(sys.argv[1])))
    print(out, os.path.getsize(out))


if __name__ == "__main__":
    main()
