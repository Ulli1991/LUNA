"""Pretty projection of the z=0 dark-matter cosmic web from a Cosmo-DM snapshot.

Projects all particles to a high-resolution column-density map (log-scaled,
lightly smoothed) -- the large-scale structure of the DM-only run.

  LUNA_DM_SNAP  snapshot path (default the z=0 256^3 run)
  LUNA_DM_NBIN  projection resolution (default 1600)
"""
import os

import h5py
import numpy as np



def main():
    snap = os.environ.get(
        "LUNA_DM_SNAP", "snapshots/cosmo_dm/snap_a1.0004.hdf5")
    nbin = int(os.environ.get("LUNA_DM_NBIN", "1600"))
    with h5py.File(snap, "r") as f:
        box = float(f["Header"].attrs["BoxSize"])
        z = float(f["Header"].attrs["Redshift"])
        pos = np.asarray(f["PartType1"]["Coordinates"], dtype=np.float64)
    print(f"# loaded {pos.shape[0]:,} particles  box={box/1000:.1f} Mpc/h  z={z:.3f}")

    H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=nbin,
                             range=[[0, box], [0, box]])
    try:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=1.0)
    except Exception:
        pass
    img = np.log10(H + 1.0)

    from scripts.validation.plots._plotkit import mpl, save
    plt = mpl()
    from luna.utils.figstyle import FIGSIZE, decorate
    L = box / 1000.0                                      # kpc/h -> Mpc/h
    vmin = float(np.percentile(img, 45))
    vmax = float(np.percentile(img, 99.9))
    side = FIGSIZE[1] + 1.6                               # square panel, canonical scale
    fig, ax = plt.subplots(figsize=(side, side))
    ax.imshow(img.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
              extent=[0, L, 0, L], interpolation="bilinear")
    ax.set_xlabel("x  [Mpc/h]")
    ax.set_ylabel("y  [Mpc/h]")
    decorate(ax)
    ax.grid(False)                                        # no grid over the image
    fig.tight_layout()
    out = "code_paper_figs/cosmo_dm_web.png"
    save(fig, out)
    print(f"# wrote {out}")


if __name__ == "__main__":
    main()
