"""Isolated multigrid Poisson accuracy: figure G1 and the Jeans-deficit diagnostic.

    python -m scripts.examples.runs.poisson_accuracy_run nx=64 g=7.5

Feeds a known single-mode density rho = delta cos(kx) (zero mean) into the
same FMG multigrid Poisson solver the self-gravity source uses and compares
the recovered potential to phi = -(4 pi G / k^2) delta cos(kx), as a function
of the FMG cycle budget. The fundamental mode is the smoothest one, so a ratio
below 1 that does not improve with cycles is a solver bug, not
under-convergence; in a Jeans test it weakens the growth by the same factor.

Parameters: nx, g (matches the Jeans test).
"""
import os


import numpy as np                                    # noqa: E402

from luna.setup.cli import args                            # noqa: E402
from luna.setup.jax_init import initialize_jax_distributed   # noqa: E402


def main(nx=64, g=7.5):
    initialize_jax_distributed(enable_x64=True)
    import jax
    import jax.numpy as jnp
    from luna.gravity.poisson_multigrid import solve_poisson_fmg

    L = 1.0
    dx = L / nx
    fourpiG = 4.0 * np.pi * g
    delta = 1.0e-3
    k = 2.0 * np.pi / L                       # fundamental: the Jeans mode
    x = (np.arange(nx) + 0.5) * dx
    rho1d = delta * np.cos(k * x)
    rho = jnp.asarray(np.broadcast_to(rho1d[:, None, None], (nx, nx, nx)).copy())
    phi_exact_amp = -fourpiG * delta / (k * k)
    phi_exact = phi_exact_amp * np.cos(k * x)
    print(f"Poisson accuracy: N={nx}, k={k:.4f}, G={g}, analytic phi_amp={phi_exact_amp:.6e}",
          flush=True)

    sweep = [0, 1, 2, 5, 10, 20, 40, 80]
    ratios, l2s = [], []
    for nex in sweep:
        phi, res = solve_poisson_fmg(rho, dx, fourpiG, 0, n_vcycles_per_level=2,
                                     n_extra_vcycles=nex, return_residual=True)
        phi1d = np.asarray(phi).mean(axis=(1, 2))
        phi1d = phi1d - phi1d.mean()             # drop the gauge constant
        amp = 2.0 * np.mean(phi1d * np.cos(k * x))
        ratio = amp / phi_exact_amp
        l2 = np.sqrt(np.mean((phi1d - phi_exact) ** 2)) / abs(phi_exact_amp)
        ratios.append(ratio)
        l2s.append(l2)
        print(f"  n_extra={nex:3d}: phi_amp={amp:.6e}  ratio={ratio:.4f}  "
              f"L2rel={l2:.3e}  res_inf={float(res):.3e}", flush=True)
    phi_best = phi1d

    if jax.process_index() != 0:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from luna.utils.figstyle import FIGSIZE, apply_style, decorate
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))
    sweep_arr = np.array(sweep)
    ax[0].semilogy(sweep_arr, np.abs(1.0 - np.array(ratios)), "o-",
                   label=r"$|1-\phi_{\rm meas}/\phi_{\rm exact}|$")
    ax[0].semilogy(sweep_arr, np.array(l2s), "s-", label=r"$L_2$ rel. error")
    ax[0].set_xlabel("extra V-cycles")
    ax[0].set_ylabel("error")
    ax[0].legend(loc="best")
    decorate(ax[0])
    ax[1].plot(x, phi_exact, "-", color="k", label="analytic")
    ax[1].plot(x, phi_best, "o", ms=4, color="C0", label=f"FMG (ratio {ratios[-1]:.3f})")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel(r"$\phi$")
    ax[1].legend(loc="best")
    decorate(ax[1])
    fig.tight_layout()
    os.makedirs("code_paper_figs", exist_ok=True)
    figpath = f"code_paper_figs/poisson_accuracy_Nx{nx}.png"
    fig.savefig(figpath)
    print(f"RESULT poisson: ratio(prod n_extra=5)={ratios[sweep.index(5)]:.4f} "
          f"ratio(n_extra=80)={ratios[-1]:.4f}", flush=True)
    print(f"  figure -> {figpath}", flush=True)


if __name__ == "__main__":
    main(**args(nx=64, g=7.5))
