"""Overlay radial profiles from the shard_map smoke run against the baseline
auto-sharded 1024^3 Sedov 1024^3 run. If physics is correct, the two
curves should overlay everywhere (modulo small fp32 / collective-order
noise). Any systematic divergence — especially at the shock — points to
a halo bug in the shard_map path.
"""
from pathlib import Path
import glob
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()

from scripts.validation.metrics import radial_profile


BASE_DIR = "snapshots/sedov3d_arepo_Nx1024"           # baseline (auto-sharded)
SM_DIR = "/ptmp/uli/jax-hydro/snapshots/sedov_smoke_shardmap_Nx1024"  # shardmap


def load_snap(snapdir):
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"]); Nz = int(a["Nz_global"])
        Lx = float(a["Lx"]); dx = float(a["dx"]); dy = float(a["dy"]); dz = float(a["dz"])
        t = float(a["time"])
    rho = np.empty((Nx, Ny, Nz), dtype=np.float32)
    vx = np.empty_like(rho); vy = np.empty_like(rho); vz = np.empty_like(rho)
    p = np.empty_like(rho)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"]); zs = int(c.attrs["z_start"])
            shp = c["density"].shape
            rho[xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["density"][:]
            vx [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_x"][:]
            vy [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_y"][:]
            vz [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["velocity_z"][:]
            p  [xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["pressure"][:]
    return dict(rho=rho, vx=vx, vy=vy, vz=vz, p=p, t=t,
                Lx=Lx, dx=dx, dy=dy, dz=dz, Nx=Nx)


def compute_profiles(snap):
    ic = snap["Nx"]/2 - 0.5
    cc = (ic, ic, ic)
    cx = snap["Nx"]/2 * snap["dx"]
    cy = snap["Nx"]/2 * snap["dy"]
    cz = snap["Nx"]/2 * snap["dz"]
    r_bins, rho_r = radial_profile(snap["rho"], snap["dx"], snap["dy"], snap["dz"], center=cc)
    _, p_r = radial_profile(snap["p"], snap["dx"], snap["dy"], snap["dz"], center=cc)
    # radial velocity
    X, Y, Z = np.meshgrid(
        (np.arange(snap["Nx"]) + 0.5) * snap["dx"],
        (np.arange(snap["Nx"]) + 0.5) * snap["dy"],
        (np.arange(snap["Nx"]) + 0.5) * snap["dz"],
        indexing="ij",
    )
    rvec = np.sqrt((X-cx)**2 + (Y-cy)**2 + (Z-cz)**2)
    vr = ((X-cx)*snap["vx"] + (Y-cy)*snap["vy"] + (Z-cz)*snap["vz"]) / np.maximum(rvec, 1e-30)
    _, vr_r = radial_profile(vr, snap["dx"], snap["dy"], snap["dz"], center=cc)
    return r_bins, rho_r, vr_r, p_r


def main():
    print(f"Loading baseline snap_010 from {BASE_DIR} ...")
    base = load_snap(os.path.join(BASE_DIR, "snapdir_010"))
    print(f"Loading shard_map snap_010 from {SM_DIR} ...")
    sm = load_snap(os.path.join(SM_DIR, "snapdir_010"))

    print(f"baseline t = {base['t']:.4e}, shardmap t = {sm['t']:.4e}")

    r_b, rho_b, vr_b, p_b = compute_profiles(base)
    r_s, rho_s, vr_s, p_s = compute_profiles(sm)

    fig, axes = plt.subplots(1, 3, figsize=(FIGSIZE[0]*3, FIGSIZE[1]), constrained_layout=True)
    axes[0].plot(r_b, rho_b, "C0-", lw=2, label="baseline (auto-sharded)")
    axes[0].plot(r_s, rho_s, "C3-", lw=1.5, label="shard_map")
    axes[0].set_xlabel("r [code]"); axes[0].set_ylabel(r"$\rho$")
    axes[0].legend()

    axes[1].plot(r_b, vr_b, "C0-", lw=2)
    axes[1].plot(r_s, vr_s, "C3-", lw=1.5)
    axes[1].set_xlabel("r [code]"); axes[1].set_ylabel(r"$v_r$ [code]")

    axes[2].plot(r_b, p_b, "C0-", lw=2)
    axes[2].plot(r_s, p_s, "C3-", lw=1.5)
    axes[2].set_xlabel("r [code]"); axes[2].set_ylabel(r"$P$")

    for ax in axes:
        decorate(ax)

    Path("figures").mkdir(exist_ok=True)
    out = f"figures/sedov_shardmap_vs_baseline_Nx{base['Nx']}.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")

    # Quantitative diagnostic
    rho_max_b = rho_b.max(); rho_max_s = rho_s.max()
    print(f"shock peak rho: baseline={rho_max_b:.4e}  shardmap={rho_max_s:.4e}  rel diff={(rho_max_s-rho_max_b)/rho_max_b:+.2%}")
    r_shock_b = r_b[np.argmax(rho_b)]; r_shock_s = r_s[np.argmax(rho_s)]
    print(f"shock radius:    baseline={r_shock_b:.4e}  shardmap={r_shock_s:.4e}  rel diff={(r_shock_s-r_shock_b)/r_shock_b:+.2%}")


if __name__ == "__main__":
    main()
