"""AMR physics benchmark #1: Sedov-Taylor blast vs the ANALYTIC solution.

Compares three solutions of the identical dimensionless Sedov IC
(rho0=1, E0=0.5 top-hat of radius 6*dx_base, gamma=5/3):

  * AMR      : base Nx^3 + L refinement levels (dynamic pressure regrid),
               dense composite snapshot at the finest resolution
  * uniform  : the SAME setup with amr_levels=0 (pure base grid) -- the
               only difference from the AMR run is refinement
  * analytic : Kamm (2000) closed form (scripts/validation/sedov_analytic)

Metrics (printed as a table + JSON next to the figure):
  * shock radius error |R_num - R_ana| / R_ana  (peak of the radial profile)
  * L1 density-profile error vs analytic over r in (r_core, 1.15 R_ana),
    excluding the finite-size injection core (r_core = 8*dx_base)
  * mass / total-energy conservation drift between first and last snapshot
  * VERDICT: AMR must beat the uniform base run on the L1 profile error
    (the "refinement adds accuracy" claim) and conserve to near machine eps.

Env: AMR_DIR, UNI_DIR (snapshot dirs), BENCH_E0 (0.5), BENCH_TAG.
"""
import glob
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.validation.sedov_analytic import sedov_taylor_3d

AMR_DIR = os.environ.get("AMR_DIR", "snapshots/amr_bench_sedov")
UNI_DIR = os.environ.get("UNI_DIR", "snapshots/amr_bench_sedov_uniform")
E0 = float(os.environ.get("BENCH_E0", "0.5"))
GAMMA = 5.0 / 3.0
TAG = os.environ.get("BENCH_TAG", "")


def load_snapshot(snapdir, index):
    """Stitch one snapshot's shards -> dict(rho, p, vx.., t, dx, Nx, Lx)."""
    snaps = sorted(glob.glob(os.path.join(snapdir, "snapdir_*")))
    if not snaps:                                  # flat layout fallback
        snaps = [snapdir]
    d = snaps[index]
    files = sorted(glob.glob(os.path.join(d, "*.hdf5")))
    with h5py.File(files[0], "r") as h:
        a = h["cells"].attrs
        Nx, Ny, Nz = (int(a[k]) for k in ("Nx_global", "Ny_global", "Nz_global"))
        Lx, dx = float(a["Lx"]), float(a["dx"])
        t = float(a["time"])
    rho = np.empty((Nx, Ny, Nz), np.float64)
    p = np.empty_like(rho)
    v2 = np.zeros_like(rho)
    for f in files:
        with h5py.File(f, "r") as h:
            c = h["cells"]
            xs, ys, zs = (int(c.attrs[k]) for k in ("x_start", "y_start", "z_start"))
            s = c["density"].shape
            sl = np.s_[xs:xs + s[0], ys:ys + s[1], zs:zs + s[2]]
            rho[sl] = c["density"][:]
            p[sl] = c["pressure"][:]
            for comp in ("velocity_x", "velocity_y", "velocity_z"):
                v2[sl] += np.asarray(c[comp][:], np.float64) ** 2
    return dict(rho=rho, p=p, v2=v2, t=t, dx=dx, Nx=Nx, Lx=Lx)


def radial_profile(field, dx, nbins=None):
    N = field.shape[0]
    ctr = (N / 2 - 0.5)
    ii = np.arange(N) - ctr
    R = np.sqrt(ii[:, None, None] ** 2 + ii[None, :, None] ** 2
                + ii[None, None, :] ** 2) * dx
    nbins = nbins or (N // 2)
    edges = np.linspace(0.0, N / 2 * dx, nbins + 1)
    which = np.digitize(R.ravel(), edges) - 1
    ok = (which >= 0) & (which < nbins)
    prof = np.bincount(which[ok], weights=field.ravel()[ok], minlength=nbins)
    cnt = np.bincount(which[ok], minlength=nbins)
    r = 0.5 * (edges[:-1] + edges[1:])
    good = cnt > 0
    return r[good], prof[good] / cnt[good]


def conservation(snapdir):
    first = load_snapshot(snapdir, 0)
    last = load_snapshot(snapdir, -1)
    dV_f, dV_l = first["dx"] ** 3, last["dx"] ** 3

    def totals(s, dV):
        M = float(np.sum(s["rho"])) * dV
        E = float(np.sum(s["p"] / (GAMMA - 1.0) + 0.5 * s["rho"] * s["v2"])) * dV
        return M, E

    M0, E0_ = totals(first, dV_f)
    M1, E1 = totals(last, dV_l)
    return abs(M1 - M0) / M0, abs(E1 - E0_) / E0_, last


def profile_metrics(snap, base_dx):
    r, rho_r = radial_profile(snap["rho"], snap["dx"])
    ana = sedov_taylor_3d(r, snap["t"], E0=E0, rho0=1.0, gamma=GAMMA)
    Rs = ana["R_shock"]
    R_num = r[int(np.argmax(rho_r))]
    r_core = 8.0 * base_dx                       # finite injection region
    m = (r > r_core) & (r < 1.15 * Rs)
    L1 = float(np.mean(np.abs(rho_r[m] - ana["rho"][m]))
               / np.mean(ana["rho"][m]))
    return dict(r=r, rho=rho_r, rho_ana=ana["rho"], Rs_ana=float(Rs),
                Rs_num=float(R_num), Rs_err=float(abs(R_num - Rs) / Rs),
                L1=L1, t=snap["t"])


def main():
    dM_a, dE_a, amr_last = conservation(AMR_DIR)
    dM_u, dE_u, uni_last = conservation(UNI_DIR)
    base_dx = uni_last["dx"]                      # uniform run = base grid
    A = profile_metrics(amr_last, base_dx)
    U = profile_metrics(uni_last, base_dx)

    print("\n=== AMR Sedov benchmark (vs Kamm closed form) ===")
    print(f"  t = {A['t']:.4f}   R_shock analytic = {A['Rs_ana']:.4f}")
    hdr = f"  {'':10s} {'R_s':>8s} {'R_s err':>9s} {'L1(rho)':>9s} " \
          f"{'dM/M':>10s} {'dE/E':>10s}"
    print(hdr)
    print(f"  {'AMR':10s} {A['Rs_num']:8.4f} {A['Rs_err']:9.2%} {A['L1']:9.4f} "
          f"{dM_a:10.2e} {dE_a:10.2e}")
    print(f"  {'uniform':10s} {U['Rs_num']:8.4f} {U['Rs_err']:9.2%} {U['L1']:9.4f} "
          f"{dM_u:10.2e} {dE_u:10.2e}")
    ok = (A["L1"] < U["L1"]) and (A["Rs_err"] <= U["Rs_err"] + 0.02) \
        and dM_a < 1e-10
    print(f"  gain: L1(uniform)/L1(AMR) = {U['L1'] / max(A['L1'], 1e-30):.2f}x")
    print("BENCH RESULT:", "PASS" if ok else "FAIL")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax.plot(A["r"], A["rho_ana"], "k-", lw=1.5, label="analytic (Kamm)")
    ax.plot(U["r"], U["rho"], "o-", ms=2.5, lw=0.8, color="#d62728",
            label=f"uniform {uni_last['Nx']}$^3$")
    ax.plot(A["r"], A["rho"], "o-", ms=2.5, lw=0.8, color="#1f77b4",
            label=f"AMR (eff. {amr_last['Nx']}$^3$)")
    ax.axvline(A["Rs_ana"], color="gray", ls=":", lw=1)
    ax.set_xlim(0, 1.3 * A["Rs_ana"]); ax.set_xlabel("r"); ax.set_ylabel(r"$\rho$")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"Sedov 3D at t={A['t']:.3f}")
    err_a = np.abs(A["rho"] - A["rho_ana"])
    err_u = np.abs(U["rho"] - U["rho_ana"])
    ax2.semilogy(U["r"], err_u, color="#d62728", lw=1, label="uniform")
    ax2.semilogy(A["r"], err_a, color="#1f77b4", lw=1, label="AMR")
    ax2.axvline(A["Rs_ana"], color="gray", ls=":", lw=1)
    ax2.set_xlim(0, 1.3 * A["Rs_ana"]); ax2.set_xlabel("r")
    ax2.set_ylabel(r"$|\rho - \rho_{\rm ana}|$")
    ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = f"code_paper_figs/amr_benchmark_sedov{TAG}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    with open(out.replace(".png", ".json"), "w") as f:
        json.dump({k: v for k, v in dict(
            t=A["t"], Rs_ana=A["Rs_ana"],
            amr=dict(Rs=A["Rs_num"], Rs_err=A["Rs_err"], L1=A["L1"],
                     dM=dM_a, dE=dE_a),
            uniform=dict(Rs=U["Rs_num"], Rs_err=U["Rs_err"], L1=U["L1"],
                         dM=dM_u, dE=dE_u)).items()}, f, indent=2)


if __name__ == "__main__":
    main()
