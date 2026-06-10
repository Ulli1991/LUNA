"""Inverse Sedov / IC recovery via reverse-mode AD.

Setup:
  ρ₀(x) = ρ_bg · (1 + δρ(x))
  δρ_true(x) = a smooth, low-wavenumber Gaussian random field at 32³.

  "Observation" U_obs = forward-simulate(δρ_true, t_end).
  Recover δρ from U_obs by minimizing  L(δρ) = ‖simulate(δρ) − U_obs‖²
  via reverse-mode AD + Adam. 32³ = 32 768 parameters → exactly the regime
  where reverse-mode beats forward (forward would need 32k JVPs).

Memory budget check:
  state 32³ × 5 × 8 B ≈ 1.3 MB, 200 steps × ~3× backward overhead ≈ 1 GB.
  Fits on a single GPU; no checkpointing needed.

The script writes a single figure recovery_progress.png showing
  - loss curve
  - true vs recovered δρ slice
  - residual ‖δρ_est − δρ_true‖² per Adam iteration.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")

from scripts.autodiff.sedov_ad import _step  # reuses the differentiable hydro step


# ---------------------------------------------------------------------------
# Smooth low-k Gaussian random field (the "truth")
# ---------------------------------------------------------------------------

def low_k_field(key, Nx, k_max=4, amp=0.05):
    """Real, zero-mean field built from Fourier modes with |k_i| ≤ k_max."""
    k = jnp.fft.fftfreq(Nx) * Nx
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing="ij")
    mask = (jnp.abs(kx) <= k_max) & (jnp.abs(ky) <= k_max) & (jnp.abs(kz) <= k_max)
    mask = mask & ~((kx == 0) & (ky == 0) & (kz == 0))   # zero DC → zero-mean field
    key_re, key_im = jax.random.split(key)
    coeffs_re = jax.random.normal(key_re, (Nx, Nx, Nx))
    coeffs_im = jax.random.normal(key_im, (Nx, Nx, Nx))
    coeffs = (coeffs_re + 1j * coeffs_im) * mask
    field = jnp.fft.ifftn(coeffs).real
    field = field / jnp.std(field) * amp        # normalize to target amplitude
    return field


# ---------------------------------------------------------------------------
# Perturbed Sedov IC: ρ_bg · (1 + δρ)
# ---------------------------------------------------------------------------

def build_ic_perturbed(delta_rho, E_inj, rho_bg, Nx, L, gamma, p_floor_frac=1e-4):
    dx = L / Nx
    rho = rho_bg * (1.0 + delta_rho)
    vx = jnp.zeros_like(rho); vy = jnp.zeros_like(rho); vz = jnp.zeros_like(rho)
    p_bg = p_floor_frac * rho_bg
    ix0 = Nx // 2 - 1
    ii = jnp.arange(Nx)[:, None, None]
    jj = jnp.arange(Nx)[None, :, None]
    kk = jnp.arange(Nx)[None, None, :]
    in_cube = ((ii >= ix0) & (ii <= ix0 + 1)
               & (jj >= ix0) & (jj <= ix0 + 1)
               & (kk >= ix0) & (kk <= ix0 + 1))
    V_inj = 8.0 * dx ** 3
    p_inj = (gamma - 1.0) * (E_inj / V_inj)
    p = jnp.where(in_cube, p_inj + p_bg, p_bg)
    E = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E], axis=0)


# ---------------------------------------------------------------------------
# Differentiable forward simulation parametrised by δρ
# ---------------------------------------------------------------------------

import functools


@functools.partial(jax.jit, static_argnames=("Nx", "n_steps", "order", "limiter",
                                              "riemann_solver", "gamma_static"))
def simulate(delta_rho, E_inj, rho_bg, *, t_end, Nx=32, L=1.0, n_steps=200,
             order=2, limiter=1, riemann_solver=1, gamma_static=5.0/3.0):
    gamma = gamma_static
    inv_dx = jnp.asarray(Nx / L, dtype=jnp.float64)
    U0 = build_ic_perturbed(delta_rho, E_inj, rho_bg, Nx, L, gamma)
    dt = t_end / n_steps

    def body(U, _):
        U = _step(U, dt, gamma=gamma, limiter=limiter,
                  riemann_solver=riemann_solver, order=order, inv_dx=inv_dx)
        return U, None

    U_final, _ = jax.lax.scan(body, U0, None, length=n_steps)
    return U_final


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def loss_fn(delta_rho, U_obs, E_inj, rho_bg, **sim_kwargs):
    U = simulate(delta_rho, E_inj, rho_bg, **sim_kwargs)
    return jnp.mean((U - U_obs) ** 2)


# ---------------------------------------------------------------------------
# Tiny Adam (no optax dependency)
# ---------------------------------------------------------------------------

def adam_init(params):
    return dict(m=jnp.zeros_like(params), v=jnp.zeros_like(params), t=0)


def adam_step(grads, state, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = b1 * state["m"] + (1 - b1) * grads
    v = b2 * state["v"] + (1 - b2) * (grads ** 2)
    m_hat = m / (1 - b1 ** t)
    v_hat = v / (1 - b2 ** t)
    update = lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return update, dict(m=m, v=v, t=t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    E0 = 0.5
    rho0 = 1.0
    cfg = dict(Nx=32, L=1.0, n_steps=200, order=2, limiter=1,
               riemann_solver=1, gamma_static=5.0/3.0)
    t_end = 0.06        # shorter than the analytic validation: keeps recovery well-posed.

    # Truth field & observation.
    print(f"Inverse Sedov: 32³ δρ field recovery via reverse-mode AD")
    print(f"  Nx={cfg['Nx']}, n_steps={cfg['n_steps']}, t_end={t_end}, "
          f"params = {cfg['Nx']**3}")

    key = jax.random.PRNGKey(42)
    delta_rho_true = low_k_field(key, cfg["Nx"], k_max=4, amp=0.05)
    print(f"  truth field: ‖δρ‖_∞ = {float(jnp.max(jnp.abs(delta_rho_true))):.3e}")

    t0 = time.time()
    U_obs = simulate(delta_rho_true, E0, rho0, t_end=t_end, **cfg)
    jax.block_until_ready(U_obs)
    print(f"  observation generated in {time.time()-t0:.1f}s")

    # Inverse problem: start from δρ = 0, gradient descent.
    delta_rho = jnp.zeros_like(delta_rho_true)
    opt = adam_init(delta_rho)
    loss_grad = jax.jit(jax.value_and_grad(loss_fn),
                        static_argnames=("Nx", "n_steps", "order", "limiter",
                                         "riemann_solver", "gamma_static"))

    losses = []
    residuals = []
    n_iter = 100
    print(f"\n  iter:    loss              ‖δρ_est − δρ_true‖")
    t0 = time.time()
    for it in range(n_iter):
        L_val, grads = loss_grad(delta_rho, U_obs, E0, rho0,
                                  t_end=t_end, **cfg)
        update, opt = adam_step(grads, opt, lr=2e-2)
        delta_rho = delta_rho - update
        res = float(jnp.linalg.norm(delta_rho - delta_rho_true))
        L_val = float(L_val)
        losses.append(L_val)
        residuals.append(res)
        if it < 5 or it % 5 == 0 or it == n_iter - 1:
            print(f"  {it:4d}    {L_val:.4e}     {res:.4e}")
    print(f"\n  total optimization time: {time.time()-t0:.1f}s")

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].semilogy(losses, "o-", lw=1, ms=3, label="loss")
    axes[0].semilogy(residuals, "s-", lw=1, ms=3, label="‖δρ_est − δρ_true‖")
    axes[0].set_xlabel("Adam iter")
    axes[0].set_ylabel("value")
    axes[0].set_title("Loss & IC residual")
    axes[0].legend(); axes[0].grid(True, which="both", alpha=0.3)

    z_slice = cfg["Nx"] // 2
    vmin = float(jnp.min(delta_rho_true)); vmax = float(jnp.max(delta_rho_true))
    im0 = axes[1].imshow(np.asarray(delta_rho_true[:, :, z_slice]),
                          vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[1].set_title("δρ_true (z-mid slice)")
    plt.colorbar(im0, ax=axes[1], shrink=0.85)

    im1 = axes[2].imshow(np.asarray(delta_rho[:, :, z_slice]),
                          vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[2].set_title("δρ_recovered (z-mid slice)")
    plt.colorbar(im1, ax=axes[2], shrink=0.85)

    os.makedirs("figures", exist_ok=True)
    out = "figures/sedov_inverse_recovery.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")

    # Final report.
    final_resid_rel = residuals[-1] / float(jnp.linalg.norm(delta_rho_true))
    print(f"\n  Final ‖δρ_est − δρ_true‖ / ‖δρ_true‖ = {final_resid_rel:.3e}")
    if final_resid_rel < 0.1:
        print("  ✓ Recovery successful (< 10% relative residual)")
    elif final_resid_rel < 0.3:
        print("  ~ Partial recovery (10–30% residual) — bumping n_iter or lr would help")
    else:
        print("  ✗ Recovery FAILED — gradient is noisy or problem is ill-posed at this horizon")


if __name__ == "__main__":
    main()
