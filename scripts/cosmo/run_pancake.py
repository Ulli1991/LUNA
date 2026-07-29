"""Run the Zel'dovich pancake THROUGH main.run (production path: registered IC +
super-comoving runner + multigrid FMG gravity) and compare to the exact
Zel'dovich solution.

  JAX_PLATFORM_NAME=cpu python scripts/cosmo/run_pancake.py
"""
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np

sys.path.insert(0, os.getcwd())
from dataclasses import replace

import main
from cosmology import Cosmology
from template_config import Config


def build_cfg():
    a_i, a_c, cs_frac = 0.3, 1.5, 1.0e-2
    k = 2 * np.pi / 1.0
    cs0 = cs_frac * Cosmology(Omega_m=1.0, Omega_L=0.0).dadtau(a_i, 1.0) / (a_c * k)
    p_floor = cs0 ** 2 / (5.0 / 3.0)        # temperature floor at the initial gas temp
    return replace(
        Config(),
        test_problem="Zeldovich-Pancake",
        SPATIAL_DIMENSION=3,
        Nx=64, Ny=64, Nz=64,            # cube -> clean cubic-cell multigrid
        Lx=1.0, Ly=1.0, Lz=1.0,
        gamma=5.0 / 3.0,
        BOUNDARY=0,                     # periodic
        ORDER=1, LIMITER=1, CFL=0.4,
        riemann_solver="hllc",
        SELF_GRAVITY=False,             # gravity handled by the cosmo runner
        CosmologicalIntegration=True,
        cosmo_Omega_m=1.0, cosmo_Omega_L=0.0, cosmo_H0=1.0,
        a_init=0.3, a_end=1.0, zel_a_caustic=1.5, dloga_max=0.01,
        zel_cs_frac=1.0e-2, cosmo_p_floor=p_floor,
        production_mode=True,
        PLOT_AFTER_SIM=False,
        clear_output_dir=True,
        output_dir="snapshots/zeldovich_pancake",
    )


def analytic(a, cfg):
    cosmo = Cosmology(Omega_m=cfg.cosmo_Omega_m, Omega_L=cfg.cosmo_Omega_L)
    H0, a_c = cfg.cosmo_H0, cfg.zel_a_caustic
    k, dx = 2 * np.pi / cfg.Lx, cfg.Lx / cfg.Nx
    x = (np.arange(cfg.Nx) + 0.5) * dx
    A = a / a_c
    q = x.copy()
    for _ in range(80):
        q -= (q - A * np.sin(k * q) / k - x) / (1 - A * np.cos(k * q))
    rho = 1.0 / (1.0 - A * np.cos(k * q))
    v = -(cosmo.dadtau(a, H0) / a_c) * np.sin(k * q) / k
    return x, rho, v


def run_to(a_end):
    """Run the pancake through main to a_end; return profiles + L1 errors."""
    cfg = replace(build_cfg(), a_end=float(a_end))
    res = main.run(cfg)
    U, a, step = res.extra["U"], res.t, res.n_steps
    U = np.asarray(U)
    rho_num = U[0, :, 0, 0]                 # uniform in y,z -> any line in x
    v_num = U[1, :, 0, 0] / U[0, :, 0, 0]
    x, rho_ana, v_ana = analytic(a, cfg)
    l1_rho = np.mean(np.abs(rho_num - rho_ana)) / np.mean(rho_ana)
    l1_v = np.mean(np.abs(v_num - v_ana)) / np.max(np.abs(v_ana))
    return dict(x=x, rho_num=rho_num, v_num=v_num, rho_ana=rho_ana, v_ana=v_ana,
                l1_rho=l1_rho, l1_v=l1_v, a=a, step=step)


def main_run():
    # L1-vs-redshift sweep: integrate to a ladder of scale factors and record the
    # error against the exact solution at each, tracking accuracy through collapse.
    a_ends = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    runs = []
    for ae in a_ends:
        r = run_to(ae)
        runs.append(r)
        print(f"# a={r['a']:.3f} z={1/r['a']-1:5.2f}  L1_rho={r['l1_rho']:.3e}  "
              f"L1_v={r['l1_v']:.3e}  ({r['step']} steps)", flush=True)
    fin = runs[-1]                          # z=0 run for the profile panels
    a_arr = np.array([r["a"] for r in runs])
    z_arr = 1.0 / a_arr - 1.0
    l1r = np.array([r["l1_rho"] for r in runs])
    l1v = np.array([r["l1_v"] for r in runs])
    ok = fin["l1_rho"] < 1e-2 and fin["l1_v"] < 1e-2
    print(f"# PANCAKE-THROUGH-MAIN {'PASS' if ok else 'FAIL'}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from utils.figstyle import apply_style
            apply_style()
        except Exception:
            pass
        from utils.figstyle import decorate
        x = fin["x"]
        os.makedirs("figures", exist_ok=True)
        # save the plot data so the figure can be restyled without re-running
        np.savez("code_paper_figs/zeldovich_pancake_data.npz",
                 x=x, rho_ana=fin["rho_ana"], rho_num=fin["rho_num"],
                 v_ana=fin["v_ana"], v_num=fin["v_num"],
                 z=z_arr, l1r=l1r, l1v=l1v)
        # compact layout: density | velocity on top, L1-vs-z spanning the bottom
        fig = plt.figure(figsize=(10.0, 7.2))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.30, wspace=0.24)
        axd = fig.add_subplot(gs[0, 0])
        axv = fig.add_subplot(gs[0, 1])
        axl = fig.add_subplot(gs[1, :])
        # density at z=0 (legend lives here only)
        axd.plot(x, fin["rho_ana"], "k-", lw=2.4, label="Zel'dovich (exact)")
        axd.plot(x, fin["rho_num"], "--", color="C3", lw=1.9, label="LUNA (main + FMG)")
        axd.set_yscale("log"); axd.set_ylabel(r"$\tilde\rho$  (comoving density)")
        axd.set_xlabel("$x$"); axd.legend(frameon=False, loc="upper center")
        decorate(axd)
        # peculiar velocity at z=0 (shares the density legend)
        axv.plot(x, fin["v_ana"], "k-", lw=2.4)
        axv.plot(x, fin["v_num"], "--", color="C3", lw=1.9)
        axv.set_ylabel(r"$\tilde v$  (peculiar velocity)"); axv.set_xlabel("$x$")
        decorate(axv)
        # L1 error vs analytic through the collapse (z high -> 0), clean lines
        axl.plot(z_arr, l1r, "-", color="C0", lw=2.4, label=r"$L_1(\tilde\rho)$")
        axl.plot(z_arr, l1v, "--", color="C3", lw=2.4, label=r"$L_1(\tilde v)$")
        axl.set_yscale("log"); axl.invert_xaxis()
        axl.set_xlabel("redshift $z$"); axl.set_ylabel(r"$L_1$ vs analytic")
        axl.legend(frameon=False, loc="upper left"); decorate(axl)
        fig.savefig("figures/zeldovich_pancake_main.png", dpi=200, bbox_inches="tight")
        print("# wrote figures/zeldovich_pancake_main.png")
    except Exception as e:
        print(f"# plot skipped: {e}")


if __name__ == "__main__":
    main_run()
