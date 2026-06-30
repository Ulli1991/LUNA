"""LUNA-vs-Arepo DM-only validation figures, in the LUNA paper style.

Two figures, both from the SAME l9 IC (512^3, 134M DM particles) evolved to z=0
by each code, read from their Gadget/Arepo-HDF5 PartType1 snapshots:

  fig32  side-by-side projected DM density (cosmic web), shared colour scale
  fig33  DM power spectrum P(k) + Arepo/LUNA ratio (identical CIC estimator)

Usage:  python -m scripts.cosmo.plot_dm_validation
"""
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
from utils.figstyle import apply_style, decorate, FIGSIZE

LUNA = "snapshots/cosmo_dm_l9_512/snap_a1.0001.hdf5"
AREPO = os.environ.get("LUNA_AREPO_DM_SNAP", "arepo/cosmo_l9_dmo/snap_003.hdf5")
OUT = "code_paper_figs"


def _load_pos(fn):
    f = h5py.File(fn, "r")
    pos = np.asarray(f["PartType1"]["Coordinates"], np.float64)
    box = float(np.asarray(f["Header"].attrs["BoxSize"]).reshape(-1)[0])
    return pos, box


def cosmic_web(Ng=768, slab_frac=0.25):
    """fig32: two density projections (LUNA, Arepo) with a shared colour scale."""
    panels = []
    for tag, fn in [("LUNA (PM)", LUNA), ("Arepo (TreePM)", AREPO)]:
        pos, box = _load_pos(fn)
        sel = pos[:, 2] < slab_frac * box
        H, _, _ = np.histogram2d(pos[sel, 0], pos[sel, 1], bins=Ng,
                                 range=[[0, box], [0, box]])
        panels.append((tag, np.log10(H.T + 0.5), box))
    allv = np.concatenate([p[1].ravel() for p in panels])
    vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99.7)

    fig, ax = plt.subplots(1, 2, figsize=(2 * FIGSIZE[1], FIGSIZE[1]))
    for a, (tag, img, box) in zip(ax, panels):
        L = box / 1000.0
        a.imshow(img, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax,
                 extent=[0, L, 0, L], interpolation="nearest")
        a.text(0.035, 0.965, tag, transform=a.transAxes, ha="left", va="top",
               color="w", fontsize=14,
               bbox=dict(boxstyle="round,pad=0.25", fc="0.1", ec="none", alpha=0.55))
        a.set_xlabel(r"$x\;[h^{-1}\mathrm{Mpc}]$")
        a.tick_params(which="both", direction="in", color="w",
                      top=True, bottom=True, left=True, right=True)
        a.grid(False)
    ax[0].set_ylabel(r"$y\;[h^{-1}\mathrm{Mpc}]$")
    fig.tight_layout()
    p = os.path.join(OUT, "fig32_cosmic_web_arepo_vs_luna.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("wrote", p)


def powerspec():
    """fig33: P(k) overlay + Arepo/LUNA ratio (npz from dm_powerspec parts mode)."""
    L = np.load("figures/pk_luna_dmo.npz")
    A = np.load("figures/pk_arepo_dmo.npz")
    kL, PL = L["k"], L["P"]
    kA, PA = A["k"], A["P"]
    PLi = np.interp(kA, kL, PL)

    fig, ax = plt.subplots(1, 2, figsize=(2 * FIGSIZE[0] * 0.92, FIGSIZE[1]))
    ax[0].loglog(kL, PL, '-', color="C0", lw=2.0, label="LUNA (PM)")
    ax[0].loglog(kA, PA, '--', color="C1", lw=2.0, label="Arepo (TreePM)")
    ax[0].set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax[0].set_ylabel(r"$P(k)\;[h^{-3}\mathrm{Mpc}^3]$")
    ax[0].legend(frameon=False, loc="lower left")
    decorate(ax[0])

    ax[1].semilogx(kA, PA / PLi, '-', color="C3", lw=2.0)
    ax[1].axhline(1.0, color="0.4", ls=":", lw=1.4)
    ax[1].set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax[1].set_ylabel(r"$P_{\mathrm{Arepo}}/P_{\mathrm{LUNA}}$")
    ax[1].set_ylim(0.6, 1.6)
    decorate(ax[1])

    fig.tight_layout()
    p = os.path.join(OUT, "fig33_dm_powerspec_arepo_vs_luna.png")
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    apply_style()
    cosmic_web()
    powerspec()
