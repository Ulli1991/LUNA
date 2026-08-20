"""Figure assembly for the autodiff runner tasks (setup/runner_autodiff).

Pure rendering: each function takes host-side arrays / optimization history
and writes a PNG under figures/. Nothing here feeds back into a simulation.
"""
import numpy as np


def _save_emergence_montage(panel0, panel0_lab, panel0_cmap, frames, fracs,
                            tgt, Nx, label, driver):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, FIGSIZE
    apply_style()
    n = len(frames) + 1
    fig, ax = plt.subplots(1, n, figsize=(FIGSIZE[0] * n * 0.62, FIGSIZE[1] * 0.95))
    vmx = float(np.percentile(np.concatenate([f.ravel() for f in frames]), 99.5))
    ax[0].imshow(panel0.T, origin="lower", cmap=panel0_cmap)
    ax[0].text(0.04, 0.96, panel0_lab, transform=ax[0].transAxes,
               va="top", ha="left", color="white", fontsize=10)
    final_lab = "stirred -> target" if driver == "turbulence" else "merged -> target"
    for a, fr, fc in zip(ax[1:], frames, fracs):
        a.imshow(fr.T, origin="lower", cmap="viridis", vmin=float(tgt.min()), vmax=vmx)
        lab = final_lab if fc == 1.0 else f"t = {fc:.2f} t_end"
        a.text(0.04, 0.96, lab, transform=a.transAxes, va="top", ha="left",
               color="white", fontsize=10)
    for a in ax:
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs("code_paper_figs", exist_ok=True)
    safe = label.replace(" ", "").replace("/", "_").split(".")[0]
    out = f"code_paper_figs/autodiff_luna_emergence_{driver}_{safe}_Nx{Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _save_eta_recovery_figure(its, Ls, etaM, etaE, eta_true, fd_rel, res, cfg):
    """Closed-loop CC85 loading recovery through the full coupled physics. Two
    panels: (a) the η_M, η_E trajectories converging onto the injected truths;
    (b) the matching-loss curve. Caption (README) states this is a circular
    validation of gradient correctness through cooling + self-gravity + external
    potential — we recover what we inject."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()

    its = np.asarray(its)
    fig, ax = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))

    # (a) loading trajectories vs the injected truths.
    ax[0].axhline(eta_true[0], color="C0", ls="--", lw=1.2,
                  label=fr"$\eta_M$ injected = {eta_true[0]:.2f}")
    ax[0].axhline(eta_true[1], color="C1", ls="--", lw=1.2,
                  label=fr"$\eta_E$ injected = {eta_true[1]:.2f}")
    ax[0].plot(its, etaM, "-", color="C0", lw=2.0, label=r"$\eta_M$ recovered")
    ax[0].plot(its, etaE, "-", color="C1", lw=2.0, label=r"$\eta_E$ recovered")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel(r"loading factor $\eta$")
    ax[0].legend(loc="best")
    ax[0].annotate(fr"max rel. err = {res:.1e}" + "\n"
                   fr"AD vs FD: {max(fd_rel):.0e}",
                   xy=(0.97, 0.5), xycoords="axes fraction", ha="right",
                   va="center", fontsize=10)
    decorate(ax[0])

    # (b) matching-loss convergence.
    ax[1].semilogy(its, np.asarray(Ls), "-", color="C3", lw=2.0,
                   label=r"$\|U(\eta)-U_{\rm obs}\|^2$")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("state-matching loss")
    ax[1].legend(loc="best")
    decorate(ax[1])

    fig.tight_layout()
    os.makedirs("code_paper_figs", exist_ok=True)
    out = f"code_paper_figs/autodiff_eta_recovery_Nx{cfg.Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _save_inversion_figure(delta_true, delta_est, rk, res, cfg):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    dt_np = np.asarray(delta_true); de_np = np.asarray(delta_est)
    zc = dt_np.shape[2] // 2
    fig, ax = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 2.2, FIGSIZE[1]),
                           gridspec_kw=dict(wspace=0.35))
    vmax = float(np.max(np.abs(dt_np)))
    for a, fld, lab in ((ax[0], dt_np, r"$\delta_{\rm true}$"),
                        (ax[1], de_np, fr"$\delta_{{\rm recovered}}$ (res {res:.2f})")):
        im = a.imshow(fld[:, :, zc].T, origin="lower", vmin=-vmax, vmax=vmax,
                      cmap="RdBu_r", aspect="auto", interpolation="nearest")
        a.grid(False); a.set_xticks([]); a.set_yticks([])
        a.text(0.04, 0.96, lab, transform=a.transAxes, va="top", ha="left",
               fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.8))
        # thin colorbar inset on the right edge — does not shrink the panel
        cax = inset_axes(a, width="4.5%", height="100%", loc="lower left",
                         bbox_to_anchor=(1.02, 0.0, 1, 1), bbox_transform=a.transAxes,
                         borderpad=0)
        fig.colorbar(im, cax=cax)
    ks = np.arange(1, len(rk))
    rvals = rk[1:]
    # recovery limit k*: last k before r(k) drops through 0.5 (half-recovery).
    below = np.where(rvals < 0.5)[0]
    kstar = int(ks[below[0] - 1]) if len(below) and below[0] > 0 else int(ks[-1])
    ax[2].axhspan(-0.2, 1.05, xmin=0, xmax=1, color="none")            # keep frame
    ax[2].axvspan(0.5, kstar + 0.5, color="C0", alpha=0.10)            # recovered band
    ax[2].axhline(1.0, color="0.6", lw=0.8, ls=":")                    # perfect recovery
    ax[2].axhline(0.5, color="0.6", lw=0.8, ls="--")                  # half-recovery
    ax[2].axhline(0, color="k", lw=0.6)
    ax[2].plot(ks, rvals, "-o", lw=2.0, ms=3.5, color="C0",
               label=r"$r(k)$")
    ax[2].axvline(kstar, color="C3", lw=1.2)
    ax[2].annotate(fr"$k_\ast\!\approx\!{kstar}$", xy=(kstar, 0.5),
                   xytext=(4, 6), textcoords="offset points",
                   color="C3", fontsize=9, va="bottom")
    ax[2].set_ylim(-0.2, 1.05); ax[2].set_xlim(0.5, ks[-1] + 0.5)
    ax[2].set_xlabel(r"wavenumber $k$"); ax[2].set_ylabel(r"recovery $r(k)$")
    ax[2].legend(loc="lower left", frameon=False)
    decorate(ax[2])
    for a in ax:                       # equal-sized square panels
        a.set_box_aspect(1)
    os.makedirs("code_paper_figs", exist_ok=True)
    # tag by test problem AND dominant physics so the turbulence / self-gravity /
    # radiative-cooling / sedov inversions don't overwrite each other (the
    # gravity and cooling cases both reuse the uniform Jeans-box IC).
    base = str(getattr(cfg, "test_problem", "ic")).replace("-", "").replace("_", "")
    if getattr(cfg, "cooling_enabled", False):
        tag, label = base + "_cool", "radiative-cooling box"
    elif getattr(cfg, "SELF_GRAVITY", False):
        tag, label = base + "_selfgrav", f"{getattr(cfg,'test_problem','')} + self-gravity"
    else:
        tag, label = base, str(getattr(cfg, "test_problem", ""))
    out = f"code_paper_figs/autodiff_ic_inversion_{tag}_Nx{cfg.Nx}.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _save_corrector_figure(c_plain, c_corr, snap_ref, snap_plain, snap_corr,
                           mtimes, dt_c, n_train, cfg):
    """diffhydro Fig 16/17 analog for the solver-in-the-loop KH corrector.
    Montage (Fig 16): density and total-energy fields with rows
    {coarse / coarse+CNN / reference} across the montage times. MSE curve
    (Fig 17): density MSE vs time for coarse vs corrected, with the training
    horizon marked (the corrector should track the reference well past it)."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()
    os.makedirs("code_paper_figs", exist_ok=True)
    rows = [("coarse", snap_plain), ("coarse + CNN", snap_corr),
            ("reference", snap_ref)]

    # ---- Fig 16: density + total-energy montages ----
    for fld, cmap, name in ((0, "magma", "density"), (-1, "inferno", "energy")):
        vmin = min(float(s[fld].min()) for s in snap_ref.values())
        vmax = max(float(s[fld].max()) for s in snap_ref.values())
        nC = len(mtimes)
        # Match the kh_grid montage style: square 2.2" panels, in-panel
        # white time labels on a translucent black box (no titles), row
        # labels via set_ylabel, shared color range.
        fig, ax = plt.subplots(3, nC, figsize=(2.2 * nC, 2.2 * 3),
                               squeeze=False)
        for r, (rlab, snaps) in enumerate(rows):
            for c, i in enumerate(mtimes):
                a = ax[r, c]
                a.grid(False)
                if i in snaps:
                    a.imshow(snaps[i][fld].T, origin="lower", cmap=cmap,
                             vmin=vmin, vmax=vmax, interpolation="bilinear",
                             aspect="equal")
                else:
                    a.text(0.5, 0.5, "diverged", transform=a.transAxes,
                           ha="center", va="center", fontsize=11, color="0.6")
                a.set_xticks([]); a.set_yticks([])
                if r == 0:
                    a.text(0.05, 0.92, f"t = {i * dt_c:.1f}",
                           transform=a.transAxes, fontsize=11, color="white",
                           fontweight="bold",
                           bbox=dict(facecolor="black", alpha=0.4,
                                     edgecolor="none", pad=1.5))
                if c == 0:
                    a.set_ylabel(rlab)
        fig.tight_layout()
        out = f"code_paper_figs/autodiff_corrector_{name}_Nc{cfg.Nx}.png"
        fig.savefig(out); plt.close(fig)
        print(f"  wrote {out}", flush=True)

    # ---- Fig 17: density MSE vs time ----
    fig, a = plt.subplots(figsize=FIGSIZE)
    if c_plain:
        t, e = zip(*c_plain); a.semilogy(t, e, "o-", color="C3", label="coarse")
    if c_corr:
        t, e = zip(*c_corr); a.semilogy(t, e, "s-", color="C0",
                                        label="coarse + CNN")
    a.axvline(n_train * dt_c, color="0.5", ls="--", lw=1)
    a.text(n_train * dt_c, a.get_ylim()[1], " trained to here", fontsize=8,
           va="top", ha="left", color="0.4")
    a.set_xlabel("time"); a.set_ylabel("density MSE vs reference")
    a.legend(loc="lower right"); decorate(a)
    fig.tight_layout()
    out = f"code_paper_figs/autodiff_corrector_mse_Nc{cfg.Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)
