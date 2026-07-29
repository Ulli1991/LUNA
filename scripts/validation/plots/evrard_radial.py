"""Evrard collapse — radial profiles of ρ, v_r, P, entropy.

Compares LUNA's spherically-averaged radial profiles against the
reference snapshot configuration from Springel 2010 (MNRAS 401:791),
fig 26. The standard comparison time is t=0.8 in the natural units
G=M=R=1, where t_ff ≈ 1.11.

At t≈0.8 the post-shock structure should show:
  - Free-fall envelope: ρ ∝ r^{-3/2}, v_r ≈ -√(2GM/r) (≈ -1 at r=R)
  - Strong accretion shock at r_s ≈ 0.1-0.2
  - Post-shock: nearly hydrostatic, ρ ~ few × pre-shock
  - Central: nearly uniform, hot

Output: figures/evrard_radial.png
"""
import os
import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

from scripts.validation.metrics import radial_profile
from scripts.validation.plots._plotkit import save, load_snap


TARGET_T = 0.8


def _load(snapdir):
    d, a = load_snap(snapdir, fields=("density", "velocity_x", "velocity_y",
                                      "velocity_z", "pressure"))
    return dict(rho=d["density"], vx=d["velocity_x"], vy=d["velocity_y"],
                vz=d["velocity_z"], P=d["pressure"], t=float(a["time"]),
                Lx=float(a["Lx"]), dx=float(a["dx"]), dy=float(a["dy"]),
                dz=float(a["dz"]), Nx=int(a["Nx_global"]))


def _radial_vr(snap):
    Nx = snap["Nx"]; dx = snap["dx"]; dy = snap["dy"]; dz = snap["dz"]
    cx = 0.5 * snap["Lx"]; cy = cx; cz = cx
    x = (np.arange(Nx) + 0.5) * dx - cx
    y = (np.arange(Nx) + 0.5) * dy - cy
    z = (np.arange(Nx) + 0.5) * dz - cz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    r3d = np.sqrt(X * X + Y * Y + Z * Z)
    vr = (X * snap["vx"] + Y * snap["vy"] + Z * snap["vz"]) / np.maximum(r3d, 1e-30)
    return vr


def main():
    # Auto-find the snapshot closest to TARGET_T inside snapshots/evrard_Nx*/
    res_dirs = sorted(glob.glob("snapshots/evrard_Nx*"))
    if not res_dirs:
        raise SystemExit("No Evrard snapshot directories found.")
    runs = []
    for d in res_dirs:
        snaps = sorted(glob.glob(os.path.join(d, "snapdir_*")))
        if not snaps:
            continue
        # Pick snap nearest to TARGET_T by reading the time attr from the first
        # hdf5 file of each snapdir.
        best, best_dt = None, float("inf")
        for s in snaps:
            files = sorted(glob.glob(os.path.join(s, "*.hdf5")))
            if not files:
                continue
            with h5py.File(files[0], "r") as h:
                t = float(h["cells"].attrs["time"])
            if abs(t - TARGET_T) < best_dt:
                best, best_dt = s, abs(t - TARGET_T)
        if best is None:
            continue
        Nx = int(os.path.basename(d).split("Nx")[1])
        runs.append((Nx, best, best_dt))
    runs.sort(key=lambda r: r[0])

    if not runs:
        raise SystemExit("No Evrard snapshots loaded.")

    # Centre-cell, radial-bin geometry.
    GAMMA = 5.0 / 3.0
    R_CLOUD = 1.0
    print(f"Target time: {TARGET_T}")

    fig, axes = plt.subplots(2, 2, figsize=(FIGSIZE[0]*2*0.6, FIGSIZE[1]*2*0.6), constrained_layout=True)
    colors = plt.cm.magma(np.linspace(0.2, 0.85, max(1, len(runs))))

    for (Nx, sd, dt_close), color in zip(runs, colors):
        print(f"  Nx={Nx}: loading {sd} (Δt = {dt_close:.3f} from target)")
        snap = _load(sd)
        cc = ((snap["Nx"] - 1) / 2.0,) * 3
        # Density
        r, rho_r = radial_profile(snap["rho"], snap["dx"], snap["dy"], snap["dz"], center=cc)
        # Radial velocity
        vr_3d = _radial_vr(snap)
        _, vr_r = radial_profile(vr_3d, snap["dx"], snap["dy"], snap["dz"], center=cc)
        # Pressure
        _, P_r = radial_profile(snap["P"], snap["dx"], snap["dy"], snap["dz"], center=cc)
        # Specific entropy s = log(P / rho^gamma); jumps at the shock.
        ent = np.log(snap["P"] / np.maximum(snap["rho"], 1e-30) ** GAMMA)
        _, ent_r = radial_profile(ent, snap["dx"], snap["dy"], snap["dz"], center=cc)

        label = f"Nx={Nx} (t={snap['t']:.3f})"
        axes[0, 0].plot(r, rho_r, color=color, lw=1.8, label=label)
        axes[0, 1].plot(r, vr_r,  color=color, lw=1.8)
        axes[1, 0].plot(r, P_r,   color=color, lw=1.8)
        axes[1, 1].plot(r, ent_r, color=color, lw=1.8)

    # Reference lines.
    r_grid = np.linspace(0.01, 2.0, 200)
    rho_ff = (1.0 / (2.0 * np.pi * R_CLOUD * R_CLOUD)) / r_grid   # IC profile (pre-collapse)
    v_ff = -np.sqrt(2.0 * 1.0 * 1.0 / r_grid)                    # free-fall
    axes[0, 0].plot(r_grid, rho_ff, "-", color="0.5", lw=1.0, label="IC ρ ∝ 1/r")
    axes[0, 1].plot(r_grid, v_ff, "-", color="0.5", lw=1.0, label="v_r = -√(2GM/r)")

    for ax in axes.flat:
        ax.set_xscale("log")
    axes[0, 0].set_yscale("log"); axes[0, 0].set_xlabel("r"); axes[0, 0].set_ylabel("ρ")
    axes[0, 1].set_xlabel("r"); axes[0, 1].set_ylabel("v_r")
    axes[1, 0].set_yscale("log"); axes[1, 0].set_xlabel("r"); axes[1, 0].set_ylabel("P")
    axes[1, 1].set_xlabel("r"); axes[1, 1].set_ylabel("ln(P/ρ^γ)  (entropy)")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="lower right")
    for ax in axes.flat:
        decorate(ax)

    out = "figures/evrard_radial.png"
    save(fig, out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
