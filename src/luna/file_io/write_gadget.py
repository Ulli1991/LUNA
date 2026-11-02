"""Gadget/Arepo-style DM particle snapshot writer (+ quick-look figure)."""

import numpy as np


def write_gadget_dm_snapshot(cfg, pos, vel_gadget, a, ic, rank=None, nproc=1,
                             n_total=None, ids=None):
    """Write a Gadget/Arepo-style DM snapshot plus a density-slice PNG.

    With ``rank`` set, each rank writes its own particles to ``snap_a{a}.{rank}.hdf5``.
    ``ids`` become ``PartType1/ParticleIDs``; rows are in no fixed order."""
    import os

    import h5py
    from luna.file_io.checkpoint_utils import write_provenance
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    suffix = "" if rank is None else f".{int(rank)}"
    fn = os.path.join(out_dir, f"snap_a{a:.4f}{suffix}.hdf5")
    n = pos.shape[0]
    n_tot = int(n if n_total is None else n_total)
    with h5py.File(fn, "w") as f:
        H = f.create_group("Header")
        H.attrs["BoxSize"] = float(cfg.Lx)
        H.attrs["Time"] = a
        H.attrs["Redshift"] = 1.0 / a - 1.0
        H.attrs["Omega0"] = cfg.cosmo_Omega_m
        H.attrs["OmegaLambda"] = cfg.cosmo_Omega_L
        H.attrs["HubbleParam"] = ic.h
        H.attrs["NumPart_ThisFile"] = np.array([0, n, 0, 0, 0, 0], dtype=np.int64)
        H.attrs["NumPart_Total"] = np.array([0, n_tot, 0, 0, 0, 0], dtype=np.int64)
        H.attrs["NumFilesPerSnapshot"] = int(nproc)
        H.attrs["MassTable"] = np.array([0, float(ic.mass[0]), 0, 0, 0, 0])
        write_provenance(H.attrs, cfg)
        g = f.create_group("PartType1")
        g.create_dataset("Coordinates", data=pos.astype(np.float32))
        g.create_dataset("Velocities", data=vel_gadget.astype(np.float32))
        if ids is not None:
            g.create_dataset("ParticleIDs", data=np.asarray(ids).astype(np.uint64))
    print(f"[cosmo-dm] wrote {fn}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Nx = cfg.Nx
        H2, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=Nx,
                                  range=[[0, cfg.Lx], [0, cfg.Lx]])
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.log10(H2.T + 0.5), origin="lower", cmap="inferno")
        ax.set_title(f"DM (z={1/a-1:.2f})"); ax.set_xticks([]); ax.set_yticks([])
        png = os.path.join(out_dir, f"cosmo_dm_a{a:.4f}.png")
        fig.tight_layout(); fig.savefig(png, dpi=130)
        print(f"[cosmo-dm] wrote {png}")
    except Exception as e:
        print(f"[cosmo-dm] plot skipped: {e}")
