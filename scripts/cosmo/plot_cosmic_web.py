"""Two-panel cosmic-web figure from a coupled gas+DM snapshot: dark matter
(left) and gas (right), projected column density in log scale.

  python scripts/cosmo/plot_cosmic_web.py <snapshot.hdf5> [out.png]

Snapshot format (written by runner_cosmo_hydro): datasets rho_dm, rho_gas
(super-comoving density grids), attrs a, z, BoxSize [kpc/h]. Reusable as-is for
the 256^3 run now and the 512^3 run when it lands -- just pass the new file.
"""
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Shared cosmic-web overdensity colour scale (also used by plot_dm_validation.py)
CW_VMIN = float(os.environ.get("CW_VMIN", "0.05"))
CW_VMAX = float(os.environ.get("CW_VMAX", "120.0"))


def _proj(rho, axis=2, slab_frac=0.25):
    """Column density over a slab of the box, matching the DM-only comparison
    figures (which project the first `slab_frac` of the box along the same axis).
    slab_frac=1.0 recovers the full-box projection."""
    rho = np.asarray(rho, dtype=np.float64)
    n = rho.shape[axis]
    nslab = max(1, int(round(slab_frac * n)))
    sl = [slice(None)] * rho.ndim
    sl[axis] = slice(0, nslab)
    return rho[tuple(sl)].mean(axis=axis)


def main():
    fn = sys.argv[1] if len(sys.argv) > 1 else (
        "snapshots/cosmo_hydro/coupled_a0.9708.hdf5")
    out = sys.argv[2] if len(sys.argv) > 2 else (
        "code_paper_figs/cosmic_web_gas_dm.png")

    with h5py.File(fn, "r") as f:
        rho_dm = f["rho_dm"][:]
        rho_gas = f["rho_gas"][:]
        a = float(f.attrs.get("a", 1.0))
        z = float(f.attrs.get("z", 1.0 / a - 1.0))
        L = float(f.attrs.get("BoxSize", 25000.0)) / 1000.0   # kpc/h -> Mpc/h
    nx = rho_dm.shape[0]

    dm = _proj(rho_dm)
    gas = _proj(rho_gas)
    # normalise each field to its own mean so the colour scale is dimensionless
    # overdensity (1 = cosmic mean column); log scale spans voids -> filaments.
    dm = dm / dm.mean()
    gas = gas / gas.mean()

    # ONE shared colour scale for both panels AND the DM-only Arepo-vs-LUNA
    # figure (plot_dm_validation.py): projected overdensity Sigma/<Sigma> spans
    # the SAME range in all cosmic-web panels (measured p2~0.03-0.06,
    # p99.95~107-134 across DM/gas/PM/Arepo), so identical overdensities map to
    # identical colours -- the DM-sharper / gas-smoother contrast is then honest.
    norm = LogNorm(vmin=CW_VMIN, vmax=CW_VMAX)

    cmap = sys.argv[3] if len(sys.argv) > 3 else "inferno"  # match DM-only figs
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.6), constrained_layout=True)
    panels = [
        (axes[0], dm, cmap, "Dark matter"),
        (axes[1], gas, cmap, "Gas"),
    ]
    for ax, field, cmap, label in panels:
        _im = ax.imshow(field.T, origin="lower", extent=[0, L, 0, L],
                       cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_xlabel(r"$x\ [h^{-1}\,\mathrm{Mpc}]$")
        ax.set_ylabel(r"$y\ [h^{-1}\,\mathrm{Mpc}]$")
        ax.text(0.035, 0.955, label, transform=ax.transAxes, color="white",
                ha="left", va="top", fontsize=13, fontweight="bold")
    axes[1].text(0.965, 0.955, f"$z={z:.2f}$", transform=axes[1].transAxes,
                 color="white", ha="right", va="top", fontsize=12)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"[cosmic-web] {nx}^3  z={z:.3f}  L={L:.1f} Mpc/h  -> {out}", flush=True)


if __name__ == "__main__":
    main()
