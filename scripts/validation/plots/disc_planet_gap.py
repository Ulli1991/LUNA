"""Disc-planet gap figure (de Val-Borro 2006 / GANDALF §6.11).

Reads the LUNA Disc-Planet-2D snapshots under snapshots/disc_planet_Nx{N}/ and
shows (a) the surface-density map at the final time with the planet marked, and
(b) the azimuthally-averaged Σ(R) at t=0 vs the final time — the gap the planet
carves at its orbital radius a=1.

Env: LUNA_DP_SNAP (snapshot dir, default snapshots/disc_planet_Nx512).
"""
import os
import glob
import numpy as np
import h5py
from scripts.validation.plots._plotkit import mpl, save

plt = mpl()
from utils.figstyle import decorate, FIGSIZE


def load_2d(snapdir):
    files = sorted(glob.glob(os.path.join(snapdir, "*.hdf5")))
    if not files:
        return None
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx = int(a["Nx_global"]); Ny = int(a["Ny_global"])
        L = float(a["Lx"]); dx = float(a["dx"]); t = float(a["time"])
    rho = np.empty((Nx, Ny), dtype=np.float64)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs = int(c.attrs["x_start"]); ys = int(c.attrs["y_start"])
            d = c["density"][:]
            d = d.reshape(d.shape[0], d.shape[1])
            rho[xs:xs + d.shape[0], ys:ys + d.shape[1]] = d
    coord = (np.arange(Nx) + 0.5) * dx - 0.5 * L
    return dict(rho=rho, coord=coord, L=L, t=t, dx=dx)


def azimuthal_sigma(snap, n_bins=120, r_max=2.6):
    X, Y = np.meshgrid(snap["coord"], snap["coord"], indexing="ij")
    R = np.sqrt(X * X + Y * Y).ravel()
    rho = snap["rho"].ravel()
    edges = np.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    prof = np.full(n_bins, np.nan)
    for i in range(n_bins):
        m = (R >= edges[i]) & (R < edges[i + 1])
        if m.sum() > 0:
            prof[i] = rho[m].mean()
    return centers, prof


def main():
    snapdir = os.environ.get("LUNA_DP_SNAP", "snapshots/disc_planet_Nx512")
    snaps = sorted(glob.glob(os.path.join(snapdir, "snapdir_*")))
    if not snaps:
        print(f"[skip] no snapshots under {snapdir}")
        return
    first = load_2d(snaps[0])
    last = load_2d(snaps[-1])
    if first is None or last is None:
        print(f"[skip] could not load snapshots in {snapdir}")
        return
    P_orb = 2.0 * np.pi
    n_orb_last = last["t"] / P_orb

    m = ""
    import re
    mm = re.search(r"Nx(\d+)", snapdir)
    if mm:
        m = f"_Nx{mm.group(1)}"

    fig, ax = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2.1, FIGSIZE[1]))

    # (a) surface-density map at the final time, planet marked at (a, 0).
    ext = [-0.5 * last["L"], 0.5 * last["L"], -0.5 * last["L"], 0.5 * last["L"]]
    im = ax[0].imshow(np.log10(last["rho"].T + 1e-6), origin="lower",
                      extent=ext, cmap="magma", aspect="equal")
    ax[0].plot(1.0, 0.0, marker="o", ms=6, mfc="none", mec="cyan", mew=1.5)
    ax[0].set_xlim(-2.6, 2.6); ax[0].set_ylim(-2.6, 2.6)
    ax[0].set_xlabel(r"$x/a$"); ax[0].set_ylabel(r"$y/a$")
    ax[0].grid(False)
    cb = fig.colorbar(im, ax=ax[0], shrink=0.85)
    cb.set_label(r"$\log_{10}\,\Sigma$")

    # (b) azimuthally averaged Σ(R): initial vs final — the gap at R=a.
    c0, p0 = azimuthal_sigma(first)
    c1, p1 = azimuthal_sigma(last)
    ax[1].plot(c0, p0, "-", color="C0", lw=2.0, label=r"$t=0$")
    ax[1].plot(c1, p1, "-", color="C3", lw=2.0,
               label=fr"$t={n_orb_last:.0f}\,P_{{\rm orb}}$")
    ax[1].axvline(1.0, color="0.5", ls="--", lw=1.0, label=r"planet $R=a$")
    ax[1].set_xlabel(r"$R/a$"); ax[1].set_ylabel(r"$\langle\Sigma\rangle_\phi$")
    ax[1].set_xlim(0.0, 2.6)
    ax[1].legend(loc="best")
    decorate(ax[1])

    fig.tight_layout()
    out = f"figures/disc_planet_gap{m}.png"
    save(fig, out, close=True)
    print(f"wrote {out}  (final t={last['t']:.2f} = {n_orb_last:.1f} orbits)")


if __name__ == "__main__":
    main()
