"""Dark-matter power spectrum P(k), the SAME estimator for LUNA and Arepo so the
two are directly comparable: CIC density on a common grid -> FFT -> shell-binned
|delta_k|^2, with the CIC window deconvolved.

  * LUNA   : reads a coupled snapshot's `rho_dm` 512^3 grid directly.
  * Arepo  : reads PartType1 coordinates from a snapshot, CIC-deposits to the
             same Ngrid, then the identical estimator.

Usage:
  python -m scripts.cosmo.dm_powerspec luna  <coupled_a1.0001.hdf5>
  python -m scripts.cosmo.dm_powerspec arepo <snap_003.hdf5> [Ngrid]
  python -m scripts.cosmo.dm_powerspec plot  luna.npz arepo.npz   -> overlay fig
"""
import sys

import numpy as np


def _cic(pos, box, Ng):
    """CIC-deposit particles (N,3) in [0,box) onto an Ng^3 grid (counts)."""
    f = (pos / box) * Ng
    i = np.floor(f).astype(np.int64)
    d = f - i
    rho = np.zeros((Ng, Ng, Ng), np.float64)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((d[:, 0] if dx else 1 - d[:, 0]) *
                     (d[:, 1] if dy else 1 - d[:, 1]) *
                     (d[:, 2] if dz else 1 - d[:, 2]))
                ii = (i[:, 0] + dx) % Ng
                jj = (i[:, 1] + dy) % Ng
                kk = (i[:, 2] + dz) % Ng
                np.add.at(rho, (ii, jj, kk), w)
    return rho


def pk_from_grid(rho, box, deconv_cic=True):
    """Shell-binned P(k) from a real density grid (any normalization)."""
    Ng = rho.shape[0]
    delta = rho / rho.mean() - 1.0
    dk = np.fft.rfftn(delta) / Ng ** 3
    kf = 2 * np.pi / box
    kx = np.fft.fftfreq(Ng, d=1.0 / Ng)
    kz = np.fft.rfftfreq(Ng, d=1.0 / Ng)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing="ij")
    kmag = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)            # in units of kf (integer modes)
    P3 = (np.abs(dk) ** 2) * box ** 3
    if deconv_cic:                                          # divide out CIC window^2
        def w1(k):
            x = np.pi * k / Ng
            return np.where(x == 0, 1.0, (np.sin(x) / x))
        W2 = (w1(KX) * w1(KY) * w1(KZ)) ** 2
        P3 = P3 / W2 ** 2
    kbins = np.arange(0.5, Ng // 2, 1.0)
    which = np.digitize(kmag, kbins)
    nk = np.bincount(which.ravel(), minlength=kbins.size + 1)
    pk = np.bincount(which.ravel(), weights=P3.ravel(), minlength=kbins.size + 1)
    sel = nk > 0
    kcen = kf * np.array([kmag[which == b].mean() if (which == b).any() else np.nan
                          for b in range(1, kbins.size)])
    pkb = (pk[1:kbins.size] / np.maximum(nk[1:kbins.size], 1))
    good = np.isfinite(kcen)
    return kcen[good], pkb[good]


def main():
    mode = sys.argv[1]
    import h5py
    if mode == "luna":
        f = h5py.File(sys.argv[2], "r")
        rho = np.asarray(f["rho_dm"], np.float64)
        box = float(np.asarray(f.attrs["BoxSize"]).reshape(-1)[0])
        k, P = pk_from_grid(rho, box, deconv_cic=False)    # rho_dm already CIC-deposited by LUNA
        np.savez("figures/pk_luna.npz", k=k, P=P, box=box)
        print(f"LUNA P(k): {len(k)} bins, k=[{k.min():.3e},{k.max():.3e}] h/Mpc-ish")
    elif mode in ("arepo", "parts"):
        # Generic Gadget/Arepo PartType1 snapshot (works for both Arepo and the
        # LUNA DM-only snapshot, which writes the identical PartType1 layout).
        #   parts <snap> <out.npz> [Ngrid]      arepo <snap> [Ngrid] -> pk_arepo.npz
        f = h5py.File(sys.argv[2], "r")
        pos = np.asarray(f["PartType1"]["Coordinates"], np.float64)
        box = float(np.asarray(f["Header"].attrs["BoxSize"]).reshape(-1)[0])
        if mode == "parts":
            out = sys.argv[3]
            Ng = int(sys.argv[4]) if len(sys.argv) > 4 else 512
        else:
            out = "figures/pk_arepo.npz"
            Ng = int(sys.argv[3]) if len(sys.argv) > 3 else 512
        rho = _cic(pos, box, Ng)
        k, P = pk_from_grid(rho, box, deconv_cic=True)   # same estimator both sides
        np.savez(out, k=k, P=P, box=box)
        print(f"P(k) [{out}]: N={len(pos)} -> {Ng}^3 CIC, {len(k)} bins, "
              f"k=[{k.min():.3e},{k.max():.3e}]")
    elif mode == "plot":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        L = np.load(sys.argv[2]); A = np.load(sys.argv[3])
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
        ax[0].loglog(L["k"], L["P"], 'o-', ms=3, label='LUNA (PM)')
        ax[0].loglog(A["k"], A["P"], 's-', ms=3, label='Arepo (TreePM)')
        ax[0].set_xlabel('k'); ax[0].set_ylabel('P(k)'); ax[0].legend(frameon=False)
        ax[0].set_title('DM power spectrum, $z=0$ (identical IC)')
        Pi = np.interp(A["k"], L["k"], L["P"])
        ax[1].semilogx(A["k"], A["P"] / Pi, 'o-', ms=3); ax[1].axhline(1, color='k', ls='--')
        ax[1].set_xlabel('k'); ax[1].set_ylabel('Arepo / LUNA'); ax[1].set_ylim(0.7, 1.3)
        ax[1].set_title('ratio')
        fig.tight_layout(); fig.savefig("figures/dm_powerspec_luna_vs_arepo.png", dpi=130)
        print("wrote figures/dm_powerspec_luna_vs_arepo.png")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
