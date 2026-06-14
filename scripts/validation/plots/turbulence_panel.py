"""Driven-turbulence 2x2 montage at a fixed physical time.

Loads the snapshot closest to TARGET_T from each Nx in {128, 256, 512, 1024}
under snapshots/turb_Nx{Nx}/ and renders a density slice through z=Lz/2.
Missing resolutions get a "missing" placeholder.

Output: figures/turbulence_panel.{pdf,png}.
"""
from pathlib import Path
import glob, os
import numpy as np
import h5py
import matplotlib.pyplot as plt

from utils.figstyle import apply_style, decorate, FIGSIZE

apply_style()


LABEL_FS = 11   # in-panel image annotations only (axis labels inherit rcParams)

NX_LIST = [128, 256, 512, 1024]
TARGET_T = 4.0


def _read_density_projection(snapdir):
    """Stitch the density field and integrate along z (column density).

    Returns (col_density, t, Lx) where col_density has shape (Nx, Ny) and
    is the sum of rho along the z axis times dz (so it's a column-integrated
    density in code units of mass/area).
    """
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No shards in {snapdir}")
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"]); Nz = int(a["Nz_global"])
        Lx = float(a["Lx"]); t = float(a["time"])
        dz = float(a["dz"])
    col = np.zeros((Nx, Ny), dtype=np.float64)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"])
            data = c["density"][:]
            col[xs:xs+data.shape[0], ys:ys+data.shape[1]] += data.sum(axis=2)
    col *= dz
    return col.astype(np.float32), t, Lx


def _find_closest(outdir, t_target):
    snaps = sorted(glob.glob(os.path.join(outdir, "snapdir_*")))
    if not snaps:
        return None, None
    best = None; best_t = None; best_dt = float("inf")
    for s in snaps:
        files = sorted(glob.glob(os.path.join(s, "*.hdf5")))
        if not files: continue
        with h5py.File(files[0], "r") as h:
            t = float(h["cells"].attrs["time"])
        dt = abs(t - t_target)
        if dt < best_dt:
            best_dt = dt; best = s; best_t = t
    return best, best_t


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    panels = {128: (0, 0), 256: (0, 1), 512: (1, 0), 1024: (1, 1)}

    # First pass: collect data + global vmin/vmax.
    data = {}
    for Nx in NX_LIST:
        outdir = f"snapshots/turb_Nx{Nx}"
        if not Path(outdir).is_dir():
            data[Nx] = None
            continue
        snapdir, t_snap = _find_closest(outdir, TARGET_T)
        if snapdir is None:
            data[Nx] = None
            continue
        rho, t, Lx = _read_density_projection(snapdir)
        data[Nx] = (rho, t, Lx)

    available = [(Nx, d) for Nx, d in data.items() if d is not None]
    if not available:
        raise SystemExit("No turbulence snapshots found.")
    # Use 1st-99th percentile of available data for a robust shared color range.
    rho_concat = np.concatenate([d[0].ravel() for _, d in available])
    vmin, vmax = np.percentile(rho_concat, [1, 99])

    for Nx, (r, c) in panels.items():
        ax = axes[r, c]
        ax.grid(False)
        if data[Nx] is None:
            ax.text(0.5, 0.5, f"missing\n{Nx}³",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=LABEL_FS, color="0.5")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        rho, t, Lx = data[Nx]
        ax.imshow(rho.T, origin="lower", extent=(0, Lx, 0, Lx),
                  vmin=vmin, vmax=vmax, cmap="magma",
                  interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.03, 0.97, f"{Nx}³\nt={t:.2f}",
                transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=LABEL_FS,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2))

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/turbulence_panel.pdf")
    fig.savefig("figures/turbulence_panel.png", dpi=200)
    print(f"Saved figures/turbulence_panel.{{pdf,png}} "
          f"(vmin={vmin:.3f}, vmax={vmax:.3f})")
    for Nx in NX_LIST:
        if data[Nx] is None:
            print(f"  Nx={Nx}: missing")
        else:
            print(f"  Nx={Nx}: t={data[Nx][1]:.3f}")


if __name__ == "__main__":
    main()
