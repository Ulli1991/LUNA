"""LUNA <-> AthenaK pgen luna_ic: IC directories in ic_file's layout, AthenaK's float64 dumps and history back.

The pgen (src/pgen/luna_ic.cpp of an AthenaK checkout, branch luna_ic; tools/athenak/) reads U.npy (5, Nx, Ny, Nz) float64, variables
rho, mx, my, mz, E, and writes its dumps in the same layout. 1D (3, Nx) and 2D (4, Nx, Ny) states are padded
with zero transverse momenta. No JAX here: host numpy only.
"""
import json
import os

import numpy as np


def to_luna5(U):
    """(5, Nx, Ny, Nz) float64 from a LUNA conserved state of 1, 2 or 3 dimensions (energy last)."""
    U = np.asarray(U, np.float64)
    sdim = U.ndim - 1
    if U.shape[0] != sdim + 2:
        raise ValueError(f"state {U.shape}: expected {sdim + 2} variables for {sdim}D")
    shape = U.shape[1:] + (1,) * (3 - sdim)
    out = np.zeros((5,) + shape)
    out[0], out[4] = U[0].reshape(shape), U[-1].reshape(shape)
    for d in range(sdim):
        out[1 + d] = U[1 + d].reshape(shape)
    return out


def from_luna5(U5, sdim):
    """The LUNA state of `sdim` dimensions from a (5, Nx, Ny, Nz) array; refuses nonzero dropped momenta."""
    shape = U5.shape[1:1 + sdim]
    drop = [U5[1 + d] for d in range(sdim, 3)]
    if any(np.any(m != 0.0) for m in drop):
        raise ValueError(f"{sdim}D state with nonzero momentum along a collapsed axis")
    return np.stack([U5[0].reshape(shape)] + [U5[1 + d].reshape(shape) for d in range(sdim)]
                    + [U5[4].reshape(shape)])


def write_ic(U, out_dir, box, gamma, name):
    """U.npy and meta.json (ic_file's keys) for pgen luna_ic; box = (Lx, Ly, Lz), 1.0 on collapsed axes."""
    U5 = to_luna5(U)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "U.npy"), U5)
    meta = dict(test_problem=name, Nx=U5.shape[1], Ny=U5.shape[2], Nz=U5.shape[3],
                Lx=float(box[0]), Ly=float(box[1]), Lz=float(box[2]), gamma=float(gamma))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


def read_dump(path, sdim):
    """A luna_ic dump (.npy) as a LUNA state of `sdim` dimensions."""
    return from_luna5(np.load(path), sdim)


def dumps(run_dir, basename="run"):
    """[(time, ncycle, path)] of the run's history-cadence dumps, in order."""
    out = []
    with open(os.path.join(run_dir, f"{basename}.luna_dumps.txt")) as f:
        for line in f:
            i, t, n = line.split()
            out.append((float(t), int(n), os.path.join(run_dir, f"{basename}.luna.{i}.npy")))
    return out


def final(run_dir, basename="run"):
    """(time, ncycle, path) of the run's final dump."""
    with open(os.path.join(run_dir, f"{basename}.luna_final.txt")) as f:
        t, n = f.read().split()
    return float(t), int(n), os.path.join(run_dir, f"{basename}.luna.final.npy")


def history(path):
    """AthenaK .hst as {label: column}, labels as in its header (time, dt, mass, 1-KE, ...)."""
    with open(path) as f:
        head = [ln for ln in f if ln.startswith("#")][-1]
    labels = [tok.split("=", 1)[1] for tok in head[1:].split() if "=" in tok]
    data = np.loadtxt(path, ndmin=2)
    return {lb: data[:, i] for i, lb in enumerate(labels)}


def errs(path):
    """The last line of an AthenaK -errs.dat (OutputErrors) as a dict."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    keys = ["Nx1", "Nx2", "Nx3", "Ncycle", "RMS-L1", "L-infty", "d_L1", "M1_L1", "M2_L1", "M3_L1", "E_L1"]
    return {k: float(v) for k, v in zip(keys, lines[-1])}


def athinput(path, *, n, box, bc, block, recon, integrator, cfl, tlim, gamma, ic_dir, hst_dt=None,
             dump=False, dump_vars=5, errors=False, accel=None, floor=None, gravity=None, potential=None,
             basename="run"):
    """Write an AthenaK input for pgen luna_ic: n, box, block per axis (1 on collapsed axes), bc per axis
    ('periodic' or 'reflect'), accel = (dir 1..3, value) for const_accel (AthenaK reads it from
    <hydro_srcterms>; in <hydro> it is silently ignored), floor = dfloor = pfloor (AthenaK's default FLT_MIN
    lets PPMX/WENOZ face pressures of ~1e-38 into HLLC: NaN at the WC blast collision), gravity = four_pi_G
    for AthenaK's multigrid self-gravity (upstream c5a0d7f, PR #776; 3D, cubic power-of-two MeshBlocks, mean
    subtracted in a periodic box): FMG, then V-cycles to round-off (threshold 0); potential = (phi0, wavelength, axis
    1..3) for phi = phi0 sin(2 pi x / wavelength) as the pgen's user source term (AthenaK branch luna_ic_a6)."""
    ng = 2 if recon == "plm" else 3
    mesh = [f"nghost = {ng}"]
    for a in range(3):
        mesh += [f"nx{a + 1} = {n[a]}", f"x{a + 1}min = 0.0", f"x{a + 1}max = {box[a]!r}",
                 f"ix{a + 1}_bc = {bc[a]}", f"ox{a + 1}_bc = {bc[a]}"]
    hydro = ["eos = ideal", f"reconstruct = {recon}", "rsolver = hllc", f"gamma = {gamma!r}"]
    if floor is not None:
        hydro += [f"dfloor = {floor!r}", f"pfloor = {floor!r}"]
    prob = [f"ic_dir = {ic_dir}", f"user_hist = {'true' if hst_dt else 'false'}",
            f"dump = {'true' if dump else 'false'}", f"dump_vars = {dump_vars}", "final_dump = true",
            f"errors = {'true' if errors else 'false'}"]
    if potential is not None:
        prob += ["user_srcs = true", "potential = sin", f"phi0 = {float(potential[0])!r}",
                 f"wavelength = {float(potential[1])!r}", f"axis = {int(potential[2])}"]
    blocks = {"job": [f"basename = {basename}"], "mesh": mesh,
              "meshblock": [f"nx{a + 1} = {block[a]}" for a in range(3)],
              "time": ["evolution = dynamic", f"integrator = {integrator}", f"cfl_number = {cfl!r}", "nlim = -1",
                       f"tlim = {tlim!r}", "ndiag = 1000000"],
              "hydro": hydro, "problem": prob}
    if accel:
        blocks["hydro_srcterms"] = ["const_accel = true", f"const_accel_dir = {accel[0]}",
                                    f"const_accel_val = {accel[1]!r}"]
    if gravity is not None:
        if any(b & (b - 1) or b != block[0] for b in block):
            raise ValueError(f"MeshBlock {block}: AthenaK multigrid needs cubic power-of-two MeshBlocks")
        blocks["gravity"] = [f"four_pi_G = {float(gravity)!r}", "full_multigrid = true", "threshold = 0.0",
                             "mg_nghost = 1"]
        blocks.setdefault("hydro_srcterms", []).append("self_gravity = true")
    if hst_dt:
        blocks["output1"] = ["file_type = hst", f"dt = {hst_dt!r}", "data_format = %25.17e"]
    with open(path, "w") as f:
        for name, lines in blocks.items():
            f.write(f"<{name}>\n" + "\n".join(lines) + "\n\n")
