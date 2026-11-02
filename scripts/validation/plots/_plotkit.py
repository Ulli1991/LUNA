"""Shared scaffold for the figure scripts: headless matplotlib setup, house
style, saving, and stitching sharded LUNA snapshots into global arrays."""
import glob
import os

import numpy as np


def mpl(style=True):
    """Select the headless Agg backend (before pyplot import), optionally
    apply the house figure style, and return pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if style:
        from luna.utils.figstyle import apply_style
        apply_style()
    return plt


def save(fig, path, dpi=None, also_png=False, close=False, **kw):
    """Create the output directory and save fig to path; also_png saves a
    .png sibling at dpi=200. Extra kwargs go to savefig."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, **kw)
    if also_png:
        fig.savefig(os.path.splitext(path)[0] + ".png", dpi=200)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)


def load_snap(snapdir, fields=("density",), dtype=np.float64):
    """Stitch the shard files of one 3D snapshot into global arrays.

    Returns (data, attrs): data maps field name -> (Nx, Ny, Nz) array placed
    by each shard's x/y/z_start; attrs is the first shard's "cells" attribute
    dict (Nx_global, Lx, dx, time, ...).
    """
    import h5py
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No .hdf5 shards in {snapdir}")
    with h5py.File(files[0], "r") as h:
        attrs = dict(h["cells"].attrs)
    shape = (int(attrs["Nx_global"]), int(attrs["Ny_global"]), int(attrs["Nz_global"]))
    data = {name: np.empty(shape, dtype=dtype) for name in fields}
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"]); zs = int(c.attrs["z_start"])
            nx, ny, nz = c["density"].shape
            for name in fields:
                data[name][xs:xs + nx, ys:ys + ny, zs:zs + nz] = c[name][:]
    return data, attrs


def snap_times(root):
    """List snapdir_*/ under root plus each snapshot's time (from shard 0)."""
    import h5py
    dirs = sorted(glob.glob(os.path.join(root, "snapdir_*")))
    times = []
    for d in dirs:
        f = sorted(glob.glob(os.path.join(d, "*.hdf5")))[0]
        with h5py.File(f, "r") as h:
            times.append(float(h["cells"].attrs["time"]))
    return dirs, np.array(times)


def latest_clean(root):
    """Last snapdir before negative density appears in any shard (BB79 runs)."""
    import h5py
    last = None
    for d in sorted(glob.glob(os.path.join(root, "snapdir_*"))):
        for f in sorted(glob.glob(os.path.join(d, "*.hdf5"))):
            with h5py.File(f, "r") as h:
                if h["cells"]["density"][:].min() <= 0:
                    return last
        last = d
    return last
