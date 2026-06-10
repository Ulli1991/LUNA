"""Diagnose what's happening in the gravity tests.

For each test problem dump:
  - snapdir_000 (IC):   max|v|, min/max ρ, ρ along x-axis lineout
  - snapdir_001 (1st):  same, plus max|v|/max|v|_initial growth ratio
  - snapdir_005 (mid):  same

Pinpoints whether the IC itself has spurious velocity (= IC bug) or
whether the evolution kicks up motion (= boundary or solver issue).
"""
import os
import glob
import numpy as np
import h5py


def load_snap(snapdir):
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    if not files:
        return None
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"]); Nz = int(a["Nz_global"])
        Lx = float(a["Lx"]); dx = float(a["dx"])
        t = float(a["time"])
    rho = np.empty((Nx, Ny, Nz), dtype=np.float64)
    vx = np.empty_like(rho); vy = np.empty_like(rho); vz = np.empty_like(rho)
    P = np.empty_like(rho)
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
    return dict(rho=rho, vx=vx, vy=vy, vz=vz, P=P, t=t, Lx=Lx, dx=dx, Nx=Nx)


def stat(snap, label=""):
    if snap is None:
        return
    v_mag = np.sqrt(snap["vx"]**2 + snap["vy"]**2 + snap["vz"]**2)
    print(
        f"  {label:>14s}  t={snap['t']:.4f}  ρ ∈ [{snap['rho'].min():.4e}, {snap['rho'].max():.4e}]  "
        f"P ∈ [{snap['P'].min():.4e}, {snap['P'].max():.4e}]  max|v|={v_mag.max():.4e}"
    )
    # Lineout along x at y=z=N/2 — shows the radial profile in 1D
    idx = snap["Nx"] // 2
    rho_line = snap["rho"][:, idx, idx]
    v_line = v_mag[:, idx, idx]
    # find max-gradient cell (interface)
    drho = np.abs(np.diff(rho_line))
    j = int(np.argmax(drho)) if len(drho) > 0 else 0
    print(
        f"       lineout: max(|dρ/dx|) at i={j} (x={(j+0.5)*snap['dx']:.3f}), "
        f"ρ[i]={rho_line[j]:.3e} → ρ[i+1]={rho_line[j+1]:.3e}, "
        f"|v|[i]={v_line[j]:.3e}"
    )


def main():
    tests = {
        "BE sub (ξ=5)":    "snapshots/be_xi5.0_Nx64",
        "BE super (ξ=10)": "snapshots/be_xi10.0_Nx64",
        "Pulse polytrope": "snapshots/pulse_Nx64",
        "Maclaurin e=0.5": "snapshots/maclaurin_e0.50_Nx64",
        "Boss-Bodenheimer":"snapshots/bb_Nx64",
        "Evrard (good)":   "snapshots/evrard_Nx64",
        "Jeans N=64":      "snapshots/jeans_Nx64",
    }
    for name, path in tests.items():
        print(f"\n=== {name} === ({path})")
        for snap_idx in (0, 1, 5, 10):
            sd = os.path.join(path, f"snapdir_{snap_idx:03d}")
            if not os.path.exists(sd):
                print(f"  snapdir_{snap_idx:03d}: missing")
                continue
            snap = load_snap(sd)
            stat(snap, label=f"snap_{snap_idx:03d}")


if __name__ == "__main__":
    main()
