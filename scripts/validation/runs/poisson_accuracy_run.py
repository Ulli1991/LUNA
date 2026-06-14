"""Isolated multigrid Poisson accuracy test — figure G1 AND Jeans-deficit diagnostic.

Feeds a known single-mode density  ρ = δ·cos(kx)  (zero mean) into the SAME FMG
multigrid Poisson solver the self-gravity source uses, and compares the recovered
potential to the analytic

    ∇²φ = 4πG ρ   →   φ = -(4πG/k²) δ cos(kx).

We report the recovered-mode amplitude ratio φ_meas/φ_exact and the residual as a
function of the FMG cycle budget (n_extra_vcycles). The FUNDAMENTAL mode k=2π/L is
the *smoothest* mode — FMG should resolve it on the coarse levels in ~1 pass, so a
ratio < 1 that does NOT improve with cycles indicates a solver bug rather than
under-convergence. If the ratio is < 1 at the production settings, the gravity
term in dynamical tests (Jeans growth) is weakened by the same factor:
    σ² = 4πGρ₀·(ratio) − c_s²k².

Env: LUNA_PA_NX (default 64), LUNA_PA_G (default 7.5, matches the Jeans test).
"""
import os
import sys

sys.path.insert(0, "/raven/u/uli/jax-hydro")

import numpy as np

from setup.distributed_setup import initialize_jax_distributed


def main():
    initialize_jax_distributed(enable_x64=True)
    import jax
    import jax.numpy as jnp
    from gravity.poisson_multigrid import solve_poisson_fmg

    N = int(os.environ.get("LUNA_PA_NX", "64"))
    G = float(os.environ.get("LUNA_PA_G", "7.5"))
    L = 1.0
    dx = L / N
    fourpiG = 4.0 * np.pi * G
    delta = 1.0e-3
    k = 2.0 * np.pi / L                       # fundamental — the Jeans mode

    x = (np.arange(N) + 0.5) * dx
    rho1d = delta * np.cos(k * x)             # zero-mean single mode
    rho = jnp.asarray(np.broadcast_to(rho1d[:, None, None], (N, N, N)).copy())

    phi_exact_amp = -fourpiG * delta / (k * k)
    phi_exact = phi_exact_amp * np.cos(k * x)

    print(f"Poisson accuracy: N={N}, k={k:.4f}, G={G}, "
          f"analytic phi_amp={phi_exact_amp:.6e}", flush=True)

    sweep = [0, 1, 2, 5, 10, 20, 40, 80]
    ratios, l2s, ress = [], [], []
    for nex in sweep:
        phi, res = solve_poisson_fmg(
            rho, dx, fourpiG, 0,                 # boundary=0 periodic
            n_vcycles_per_level=2, n_extra_vcycles=nex,
            return_residual=True,
        )
        phi = np.asarray(phi)
        phi1d = phi.mean(axis=(1, 2))            # uniform in y,z
        phi1d = phi1d - phi1d.mean()             # drop the gauge constant
        amp = 2.0 * np.mean(phi1d * np.cos(k * x))    # cos-mode projection
        ratio = amp / phi_exact_amp
        l2 = np.sqrt(np.mean((phi1d - phi_exact) ** 2)) / abs(phi_exact_amp)
        ratios.append(ratio); l2s.append(l2); ress.append(float(res))
        print(f"  n_extra={nex:3d}: phi_amp={amp:.6e}  ratio={ratio:.4f}  "
              f"L2rel={l2:.3e}  res_inf={float(res):.3e}", flush=True)

    # keep the finest-cycle profile for the figure
    phi_best = phi1d

    # ---- figure G1 ----
    if jax.process_index() != 0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()

    fig, ax = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))
    # (a) convergence: |1-ratio| and L2 error vs cycle budget
    sweep_arr = np.array(sweep)
    ax[0].semilogy(sweep_arr, np.abs(1.0 - np.array(ratios)), "o-",
                   label=r"$|1-\phi_{\rm meas}/\phi_{\rm exact}|$")
    ax[0].semilogy(sweep_arr, np.array(l2s), "s--", label=r"$L_2$ rel. error")
    ax[0].set_xlabel("extra V-cycles"); ax[0].set_ylabel("error")
    ax[0].set_title(f"FMG Poisson convergence, $N$={N}, $k=2\\pi/L$")
    ax[0].legend(loc="best")
    decorate(ax[0])
    # (b) recovered vs analytic potential profile
    ax[1].plot(x, phi_exact, "-", color="k", label="analytic")
    ax[1].plot(x, phi_best, "o", ms=4, color="C0",
               label=f"FMG (ratio {ratios[-1]:.3f})")
    ax[1].set_xlabel("x"); ax[1].set_ylabel(r"$\phi$")
    ax[1].set_title("potential profile")
    ax[1].legend(loc="best")
    decorate(ax[1])
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    figpath = f"figures/poisson_accuracy_Nx{N}.png"
    fig.savefig(figpath)
    print(f"RESULT poisson: ratio(prod n_extra=5)={ratios[sweep.index(5)]:.4f} "
          f"ratio(n_extra=80)={ratios[-1]:.4f}", flush=True)
    print(f"  figure -> {figpath}", flush=True)


if __name__ == "__main__":
    main()
