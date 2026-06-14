"""BB79 column-density projection: LUNA vs AREPO at the final time.

Side-by-side log₁₀ Σ(x, y) = ∫ ρ dz across the (x, y) midplane, with the
rotation axis along z. The m=2 perturbation produces a two-bar morphology
once the collapse develops.
"""
from __future__ import annotations

import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()


AREPO_SNAP = "/raven/u/uli/arepo/run/examples/bb79_3d/output"
LUNA_SNAP  = "/raven/u/uli/jax-hydro/snapshots/bb_Nx64"


def pick_snap_closest(pattern, target_t):
    paths = sorted(glob.glob(pattern))
    best, best_dt = paths[-1], float("inf")
    for p in paths:
        try:
            with h5py.File(p, "r") as h:
                t = float(h["Header"].attrs.get("Time", 0.0))
        except Exception:
            continue
        if abs(t - target_t) < best_dt:
            best_dt = abs(t - target_t); best = p
    return best


def luna_pick_snap_closest(root, target_t):
    dirs = sorted(glob.glob(os.path.join(root, "snapdir_*")))
    best, best_dt = dirs[-1], float("inf")
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "*.hdf5")))
        if not files: continue
        with h5py.File(files[0], "r") as h:
            t = float(h["cells"].attrs.get("time", 0.0))
        if abs(t - target_t) < best_dt:
            best_dt = abs(t - target_t); best = d
    return best


def arepo_column_density(snap_path, Ng=128):
    """Bin AREPO particles to a uniform (Ng, Ng) grid integrated along z."""
    with h5py.File(snap_path, "r") as h:
        L = float(h["Header"].attrs["BoxSize"])
        t = float(h["Header"].attrs["Time"])
        pos = h["PartType0"]["Coordinates"][:]
        mass = h["PartType0"]["Masses"][:] if "Masses" in h["PartType0"] else None
        rho  = h["PartType0"]["Density"][:]
        vol  = mass / rho if mass is not None else None
    # Use Mass = ρ·V approach: since AREPO stores cell mass and volume,
    # column density Σ(x,y) = Σ_cells (m / dxdy) where the cell falls in (x,y) bin.
    # If Masses field isn't present, fall back to ρ·V_cell ≈ ρ·(L/N_AREPO)³,
    # which is fine for a Cartesian-lattice IC like ours.
    if mass is None:
        # Approximate cell volume from total mass conservation assumption.
        # For our Cartesian IC, mass ≈ ρ × (L/N)^3 where N = round(Npart^{1/3}).
        N = int(round(pos.shape[0] ** (1.0/3.0)))
        cell_vol = (L / N) ** 3
        mass = rho * cell_vol
    edges = np.linspace(0.0, L, Ng + 1)
    dx = L / Ng
    Sigma, _, _ = np.histogram2d(pos[:, 0], pos[:, 1],
                                  bins=(edges, edges), weights=mass)
    Sigma = Sigma / (dx * dx)
    return Sigma, L, t


def luna_column_density(snapdir):
    """Load LUNA 3D ρ and integrate along z."""
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"]); Nz = int(a["Nz_global"])
        L = float(a["Lx"]); dx = float(a["dx"]); t = float(a["time"])
    rho = np.empty((Nx, Ny, Nz), dtype=np.float64)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"]); zs = int(c.attrs["z_start"])
            shp = c["density"].shape
            rho[xs:xs+shp[0], ys:ys+shp[1], zs:zs+shp[2]] = c["density"][:]
    Sigma = rho.sum(axis=2) * dx
    return Sigma, L, t


def main():
    target_t = 2.0
    arepo_path = pick_snap_closest(os.path.join(AREPO_SNAP, "snap_*.hdf5"), target_t)
    luna_dir   = luna_pick_snap_closest(LUNA_SNAP, target_t)

    Sigma_A, L_A, t_A = arepo_column_density(arepo_path, Ng=64)
    Sigma_L, L_L, t_L = luna_column_density(luna_dir)
    print(f"  AREPO Σ range [{Sigma_A.min():.3e}, {Sigma_A.max():.3e}]  t={t_A:.3f}")
    print(f"  LUNA  Σ range [{Sigma_L.min():.3e}, {Sigma_L.max():.3e}]  t={t_L:.3f}")

    # Common log color scale (handle tiny / negative carefully).
    Sigma_A_pos = np.maximum(Sigma_A, 1e-30)
    Sigma_L_pos = np.maximum(Sigma_L, 1e-30)
    vmin = np.log10(max(min(Sigma_A_pos.min(), Sigma_L_pos.min()), 1e-6))
    vmax = np.log10(max(Sigma_A_pos.max(), Sigma_L_pos.max()))

    fig, axes = plt.subplots(1, 2, figsize=(FIGSIZE[0]*2, FIGSIZE[1]))
    extent_A = (0, L_A, 0, L_A)
    extent_L = (0, L_L, 0, L_L)
    im0 = axes[0].imshow(np.log10(Sigma_A_pos).T, origin="lower",
                          extent=extent_A, vmin=vmin, vmax=vmax, cmap="magma")
    axes[0].grid(False)
    axes[0].set_title(f"AREPO  (t={t_A:.2f})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], shrink=0.85, label=r"$\log_{10} \Sigma$")
    im1 = axes[1].imshow(np.log10(Sigma_L_pos).T, origin="lower",
                          extent=extent_L, vmin=vmin, vmax=vmax, cmap="magma")
    axes[1].grid(False)
    axes[1].set_title(f"LUNA   (t={t_L:.2f})")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1], shrink=0.85, label=r"$\log_{10} \Sigma$")
    fig.suptitle("Boss-Bodenheimer 1979: column density along z (rotation axis)")
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    Nx_luna = Sigma_L.shape[0]   # LUNA column density is (Nx, Ny)
    out = f"figures/arepo_vs_luna_bb79_projection_Nx{Nx_luna}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
