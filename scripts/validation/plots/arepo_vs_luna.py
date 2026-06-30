"""AREPO vs LUNA radial profile comparison at the end of each gravity test.

Loads matched final snapshots from
  $AREPO_RUN/<test>/output/snap_XXX.hdf5
and the LUNA equivalent under
  snapshots/<test>_Nx64/snapdir_NNN/*.hdf5
bins by radius from the cloud centre, and overlays ρ(r), |v_r|(r), P(r).

For Evrard we plot at t=0.8 (Springel 2010 Fig 41 reference time).
For BE / BB79 we plot at the simulation end time.
"""
from __future__ import annotations

import os
import re
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()


AREPO_RUN = os.environ.get("AREPO_RUN", "arepo/run/examples")
LUNA_SNAP = "snapshots"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_arepo(path, boxsize_fallback=4.0):
    with h5py.File(path, "r") as h:
        time = float(h["Header"].attrs.get("Time", 0.0))
        L = float(h["Header"].attrs.get("BoxSize", boxsize_fallback))
        p0 = h["PartType0"]
        pos = p0["Coordinates"][:]
        vel = p0["Velocities"][:]
        rho = p0["Density"][:] if "Density" in p0 else (p0["Masses"][:] / p0["Volume"][:])
        u   = p0["InternalEnergy"][:]
    cx = 0.5 * L
    r_vec = pos - cx
    r = np.linalg.norm(r_vec, axis=1)
    # Radial velocity (positive = outward).
    nonzero = r > 0
    rhat = np.zeros_like(r_vec)
    rhat[nonzero] = r_vec[nonzero] / r[nonzero, None]
    v_r = np.sum(vel * rhat, axis=1)
    P = (5.0/3.0 - 1.0) * rho * u
    return dict(r=r, rho=rho, v_r=v_r, P=P, t=time, L=L)


def load_luna(snapdir, boxsize_fallback=4.0):
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    if not files:
        return None
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"]); Nz = int(a["Nz_global"])
        L = float(a["Lx"]); dx = float(a["dx"]); t = float(a["time"])
    rho = np.empty((Nx, Ny, Nz), dtype=np.float64)
    vx  = np.empty_like(rho); vy = np.empty_like(rho); vz = np.empty_like(rho)
    P   = np.empty_like(rho)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"]); zs = int(c.attrs["z_start"])
            shp = c["density"].shape
            rho[xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["density"][:]
            vx [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_x"][:]
            vy [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_y"][:]
            vz [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_z"][:]
            P  [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["pressure"][:]
    coord = (np.arange(Nx) + 0.5) * dx - 0.5 * L
    X, Y, Z = np.meshgrid(coord, coord, coord, indexing="ij")
    r = np.sqrt(X*X + Y*Y + Z*Z).ravel()
    rho_f = rho.ravel()
    vx_f, vy_f, vz_f = vx.ravel(), vy.ravel(), vz.ravel()
    r_safe = np.maximum(r, 1e-10)
    v_r = (vx_f * (X.ravel())/r_safe + vy_f * (Y.ravel())/r_safe + vz_f * (Z.ravel())/r_safe)
    return dict(r=r, rho=rho_f, v_r=v_r, P=P.ravel(), t=t, L=L)


# ---------------------------------------------------------------------------
# Radial binning
# ---------------------------------------------------------------------------

def radial_bins(r, vals, n_bins=40, r_max=None, log=True):
    if r_max is None:
        r_max = r.max()
    r_min = max(r.min(), 1e-3 * r_max)
    if log:
        edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    else:
        edges = np.linspace(r_min, r_max, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    means = np.zeros(n_bins) * np.nan
    for i in range(n_bins):
        mask = (r >= edges[i]) & (r < edges[i+1])
        if mask.sum() > 0:
            means[i] = vals[mask].mean()
    return centers, means


# ---------------------------------------------------------------------------
# Snapshot finder
# ---------------------------------------------------------------------------

def pick_snap_at_time(snapdir_pattern, target_t):
    """Walk snapshots and pick the one whose 'time' attr is closest to target."""
    paths = sorted(glob.glob(snapdir_pattern))
    if not paths:
        return None
    best = paths[-1]
    best_dt = float("inf")
    for p in paths:
        try:
            with h5py.File(p, "r") as h:
                t = float(h["Header"].attrs.get("Time", 0.0))
        except Exception:
            continue
        if abs(t - target_t) < best_dt:
            best_dt = abs(t - target_t); best = p
    return best


def pick_luna_snap_at_time(luna_root, target_t):
    """Walk LUNA snapdirs and pick the one whose 'time' attr is closest."""
    dirs = sorted(glob.glob(os.path.join(luna_root, "snapdir_*")))
    if not dirs:
        return None
    best = dirs[-1]; best_dt = float("inf")
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.hdf5")))
        if not files: continue
        try:
            with h5py.File(files[0], "r") as h:
                t = float(h["cells"].attrs.get("time", 0.0))
        except Exception:
            continue
        if abs(t - target_t) < best_dt:
            best_dt = abs(t - target_t); best = d
    return best


# ---------------------------------------------------------------------------
# One-test plot
# ---------------------------------------------------------------------------

def compare_test(name, arepo_subdir, luna_subdir, target_t, out_path):
    arepo_path = pick_snap_at_time(
        os.path.join(AREPO_RUN, arepo_subdir, "output", "snap_*.hdf5"), target_t)
    luna_path = pick_luna_snap_at_time(
        os.path.join(LUNA_SNAP, luna_subdir), target_t)
    if not arepo_path or not luna_path:
        print(f"  [skip] {name}: missing snapshots — arepo={arepo_path}, luna={luna_path}")
        return False

    # Encode the LUNA resolution in the output filename so runs at different
    # resolutions can't silently overwrite each other. Parse the trailing
    # NxNN token from the luna subdir; fall back to the original name if absent.
    m = re.search(r"Nx(\d+)", luna_subdir)
    if m:
        stem, ext = os.path.splitext(out_path)
        out_path = f"{stem}_Nx{m.group(1)}{ext}"

    A = load_arepo(arepo_path)
    LU = load_luna(luna_path)
    print(f"  {name}: AREPO t={A['t']:.3f}  LUNA t={LU['t']:.3f}  (target {target_t})")

    r_max = 0.5 * A["L"]   # half-box; cloud R typically <0.25 L
    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE[0]*3, FIGSIZE[1]))
    titles = ["ρ", "v_r", "P"]
    keys   = ["rho", "v_r", "P"]
    ylabels = [r"$\rho$", r"$v_r$", r"$P$"]
    for ax, k, ylab in zip(axes, keys, ylabels):
        cA, mA = radial_bins(A["r"], A[k], n_bins=40, r_max=r_max, log=True)
        cL, mL = radial_bins(LU["r"], LU[k], n_bins=40, r_max=r_max, log=True)
        if k == "v_r":
            ax.plot(cA, mA, "-", lw=2.0, color="C0", label="AREPO")
            ax.plot(cL, mL, "-", lw=2.0, color="C1", label="LUNA")
            ax.set_xscale("log")
        else:
            ax.loglog(cA, np.abs(mA), "-", lw=2.0, color="C0", label="AREPO")
            ax.loglog(cL, np.abs(mL), "-", lw=2.0, color="C1", label="LUNA")
        ax.set_xlabel(r"$r$"); ax.set_ylabel(ylab)
        ax.legend()
        decorate(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"    wrote {out_path}")
    return True


def main():
    os.makedirs("figures", exist_ok=True)
    tests = [
        ("Evrard",     "evrard_3d",            "evrard_Nx64",          0.80, "figures/arepo_vs_luna_evrard.png"),
        ("BE sub",     "bonnor_ebert_3d_sub",  "be_xi5.0_Nx64",        3.00, "figures/arepo_vs_luna_be_sub.png"),
        ("BE super",   "bonnor_ebert_3d_super","be_xi10.0_Nx64",       3.00, "figures/arepo_vs_luna_be_super.png"),
        ("BB79",       "bb79_3d",              "bb_Nx64",              2.00, "figures/arepo_vs_luna_bb79.png"),
    ]
    for name, arepo, luna, t, out in tests:
        compare_test(name, arepo, luna, t, out)


if __name__ == "__main__":
    main()
