"""Figures of the 3D solver-in-the-loop example from tools/sil3d.py eval
pickles (host only, no JAX):

    python tools/sil3d_figs.py stats  eval.pkl out.png [t_from=0.5] [labels=a,b,..]
    python tools/sil3d_figs.py time   eval.pkl out.png [k_train=128]
    python tools/sil3d_figs.py slices eval.pkl out.png [j=-1]
    python tools/sil3d_figs.py hist   hist.npy[,hist2.npy] out.png

stats: density and velocity spectra, ln-rho PDF, S_2(l), averaged over the
rollout from t_from (in t_turn) on, each solver next to the reference.
time: Mach number, sigma_s and the pointwise error against the reference
vs time; the training horizon marked. slices: mid-plane ln rho.
"""
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

COLORS = {"reference": "k", "coarse": "#d95f02", "sil": "#1b9e77", "offline": "#7570b3"}


def _label(name):
    if name in ("reference", "coarse"):
        return name
    return "offline" if "offline" in name else "sil"


def _kv(argv):
    return dict(a.split("=", 1) for a in argv if "=" in a)


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _mean(rows, key, n_from):
    xs = [r[key] for r in rows if r.get("ok", True) and key in r and r["n"] >= n_from]
    return np.mean(xs, axis=0) if xs else None


def fig_stats(path, out, t_from=0.5, labels=None):
    import matplotlib.pyplot as plt
    from luna.utils.figstyle import apply_style, decorate
    from tools.sil3d import S_BINS
    d = _load(path)
    meta, res = d["meta"], d["res"]
    n_from = d["n0"] + int(round(float(t_from) * meta["t_turn"] / meta["dtc"]))
    apply_style()
    fig, ax = plt.subplots(1, 4, figsize=(4 * 4.6, 4.2))
    names = ["reference"] + [k for k in res if k != "reference"]
    for i, name in enumerate(names):
        rows = res[name]
        lab = labels[i] if labels and i < len(labels) else _label(name)
        col = COLORS.get(_label(name), None)
        pr, pv, pdf, sf = (_mean(rows, k, n_from) for k in ("Prho", "Pv", "pdf", "sf"))
        if pr is None:
            continue
        k = np.arange(1, len(pr) + 1)
        ls = "-" if name == "reference" else "--" if name == "coarse" else "-"
        ax[0].loglog(k, pr, ls, color=col, label=lab)
        ax[1].loglog(k, pv * k ** 2, ls, color=col, label=lab)
        sc = 0.5 * (S_BINS[1:] + S_BINS[:-1])
        ax[2].semilogy(sc, pdf, ls, color=col, label=lab)
        lags = [r for r in rows if "lags" in r][0]["lags"]
        ax[3].loglog(lags / meta["nc"], sf[1], ls, color=col, label=lab)
    ax[0].set(xlabel=r"$k\,L/2\pi$", ylabel=r"$P_\rho(k)$")
    ax[1].set(xlabel=r"$k\,L/2\pi$", ylabel=r"$k^2 P_v(k)$")
    ax[2].set(xlabel=r"$s=\ln(\rho/\rho_0)$", ylabel="PDF", ylim=(1e-5, 1))
    ax[3].set(xlabel=r"$\ell/L$", ylabel=r"$S_2(\ell)$")
    for a in ax:
        decorate(a)
    ax[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def fig_time(path, out, k_train=None):
    import matplotlib.pyplot as plt
    from luna.utils.figstyle import apply_style, decorate
    d = _load(path)
    meta, res = d["meta"], d["res"]
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(3 * 4.6, 4.2))
    tt = lambda n: (np.asarray(n) - d["n0"]) * meta["dtc"] / meta["t_turn"]   # noqa: E731
    for name, rows in res.items():
        col = COLORS.get(_label(name))
        ok = [r for r in rows if "mach" in r]
        n = [r["n"] for r in ok]
        ax[0].plot(tt(n), [r["mach"] for r in ok], color=col, label=_label(name))
        ax[1].plot(tt(n), [r["sigma_s"] for r in ok], color=col)
        e = [r for r in ok if "err" in r]
        if e:
            ax[2].semilogy(tt([r["n"] for r in e]), [r["err"][0] for r in e], color=col)
    if k_train:
        for a in ax:
            a.axvline(float(k_train) * meta["dtc"] / meta["t_turn"], color="0.5", ls=":")
    ax[0].set(xlabel=r"$t/t_{\rm turn}$", ylabel=r"$\mathcal{M}_{\rm rms}$")
    ax[1].set(xlabel=r"$t/t_{\rm turn}$", ylabel=r"$\sigma_s$")
    ax[2].set(xlabel=r"$t/t_{\rm turn}$", ylabel=r"$\langle(s-s_{\rm ref})^2\rangle$")
    for a in ax:
        decorate(a)
    ax[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def fig_slices(path, out, j=-1):
    import matplotlib.pyplot as plt
    d = _load(path)
    res = d["res"]
    ref = {r["n"]: r for r in res["reference"]}
    names = ["reference"] + [k for k in res if k != "reference"]
    common = sorted(set.intersection(*[{r["n"] for r in res[k] if "slice" in r} for k in names]))
    n = common[int(j)]
    fig, ax = plt.subplots(1, len(names), figsize=(4 * len(names), 4.2))
    for a, name in zip(ax, names):
        r = ref[n] if name == "reference" else next(r for r in res[name] if r["n"] == n)
        a.imshow(r["slice"].T, origin="lower", cmap="magma", vmin=-4, vmax=4)
        a.set_axis_off()
        a.text(0.03, 0.95, _label(name), color="w", transform=a.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out, "n =", n)


def fig_hist(paths, out):
    import matplotlib.pyplot as plt
    from luna.utils.figstyle import apply_style, decorate
    apply_style()
    fig, ax = plt.subplots()
    for p in paths.split(","):
        h = np.load(p)
        ax.semilogy(h[:, 0], h[:, 3], ".", ms=2, label=os.path.basename(os.path.dirname(p)))
    ax.set(xlabel="iteration", ylabel="loss")
    decorate(ax)
    ax.legend(frameon=False)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    mode, a, b = sys.argv[1:4]
    kv = _kv(sys.argv[4:])
    if mode == "stats":
        fig_stats(a, b, kv.get("t_from", 0.5),
                  kv["labels"].split(",") if "labels" in kv else None)
    elif mode == "time":
        fig_time(a, b, kv.get("k_train"))
    elif mode == "slices":
        fig_slices(a, b, kv.get("j", -1))
    else:
        fig_hist(a, b)
