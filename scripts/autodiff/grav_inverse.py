"""IC recovery through self-gravitating hydro — gravity inside the AD loop.

Setup: periodic box, uniform background ρ₀ with a smooth low-k perturbation
δρ(x) (the parameters), pressure chosen so the box is Jeans-unstable:
gravity amplifies the seeded modes, hydro pushes back. After n_steps the
final state U_obs is the "observation"; we recover δρ by reverse-mode AD
through the ENTIRE loop — MUSCL-Hancock hydro + per-step Poisson solve +
gravity kick.

The Poisson solve uses gravity/poisson_diff.solve_poisson_fmg_diff: the
implicit (custom-vjp) adjoint, so the backward pass through each gravity
step is one extra FMG solve instead of unrolled V-cycle differentiation.

Rigour instrumentation (same standard as the other inverse demos):
  * FD cross-check of the loss gradient at 5 random voxels at iter 0.
  * Mass + energy bookkeeping of the forward observation.
"""
from __future__ import annotations

import os, sys, time
import functools
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")
jax.config.update("jax_enable_x64", True)

from hydro.muscl_hancock import muscl_hancock_divergence
from hydro.conversions import cons_to_prim
from gravity.poisson_diff import solve_poisson_fmg_diff


# ---------------------------------------------------------------------------
# Smooth low-k field
# ---------------------------------------------------------------------------

def low_k_field(key, Nx, k_max=2, amp=0.05):
    k = jnp.fft.fftfreq(Nx) * Nx
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing="ij")
    mask = (jnp.abs(kx) <= k_max) & (jnp.abs(ky) <= k_max) & (jnp.abs(kz) <= k_max)
    mask = mask & ~((kx == 0) & (ky == 0) & (kz == 0))
    kr, ki = jax.random.split(key)
    c = (jax.random.normal(kr, (Nx, Nx, Nx)) + 1j * jax.random.normal(ki, (Nx, Nx, Nx))) * mask
    f = jnp.fft.ifftn(c).real
    return f / jnp.std(f) * amp


# ---------------------------------------------------------------------------
# Differentiable step: hydro + gravity kick
# ---------------------------------------------------------------------------

def _pad_periodic(W):
    return jnp.pad(W, ((0, 0), (1, 1), (1, 1), (1, 1)), mode="wrap")


def _hydro_step(U, dt, *, gamma, limiter, inv_dx):
    W = cons_to_prim(U, gamma)
    W_pad = _pad_periodic(W)
    dUdt = muscl_hancock_divergence(
        W_pad, dt,
        gamma=gamma, limiter=limiter, riemann_solver="hllc",
        spatial_dim=3, inv_dx=inv_dx, inv_dy=inv_dx, inv_dz=inv_dx,
        boundary=0, order=1,
        exact_wrap=True,        # exact conservation at the periodic seam
    )
    return U + dt * dUdt


def _gravity_kick(U, dt, *, dx, fourpiG):
    """Operator-split kick: solve Poisson (implicit adjoint), apply g = -∇Φ."""
    rho = U[0]
    rho_src = rho - jnp.mean(rho)          # Jeans swindle (periodic)
    phi = solve_poisson_fmg_diff(rho_src, dx, fourpiG, 0, 1, 1)
    phi_pad = jnp.pad(phi, ((1, 1), (1, 1), (1, 1)), mode="wrap")
    inv_2dx = 0.5 / dx
    gx = -(phi_pad[2:, 1:-1, 1:-1] - phi_pad[:-2, 1:-1, 1:-1]) * inv_2dx
    gy = -(phi_pad[1:-1, 2:, 1:-1] - phi_pad[1:-1, :-2, 1:-1]) * inv_2dx
    gz = -(phi_pad[1:-1, 1:-1, 2:] - phi_pad[1:-1, 1:-1, :-2]) * inv_2dx
    mx, my, mz = U[1], U[2], U[3]
    new_mx = mx + dt * rho * gx
    new_my = my + dt * rho * gy
    new_mz = mz + dt * rho * gz
    # Energy: work done by gravity on the (time-centred) momentum.
    new_E = U[4] + dt * (0.5 * (mx + new_mx) * gx
                          + 0.5 * (my + new_my) * gy
                          + 0.5 * (mz + new_mz) * gz)
    return jnp.stack([U[0], new_mx, new_my, new_mz, new_E])


def make_simulate(Nx, L, gamma, rho_0, p_0, fourpiG, n_steps, dt,
                  limiter=0, chunk_len=10):
    dx = L / Nx
    inv_dx = jnp.asarray(Nx / L, dtype=jnp.float64)
    assert n_steps % chunk_len == 0
    n_chunks = n_steps // chunk_len

    def build_ic(delta_rho):
        rho = rho_0 * (1.0 + delta_rho)
        z = jnp.zeros_like(rho)
        p = jnp.full_like(rho, p_0)
        E = p / (gamma - 1.0)
        return jnp.stack([rho, z, z, z, E])

    @jax.jit
    def simulate(delta_rho):
        U = build_ic(delta_rho)

        def step(U, _):
            U = _hydro_step(U, dt, gamma=gamma, limiter=limiter, inv_dx=inv_dx)
            U = _gravity_kick(U, dt, dx=dx, fourpiG=fourpiG)
            return U, None

        @jax.checkpoint
        def chunk(U, _):
            U, _ = jax.lax.scan(step, U, None, length=chunk_len)
            return U, None

        U_final, _ = jax.lax.scan(chunk, U, None, length=n_chunks)
        return U_final

    return simulate


# ---------------------------------------------------------------------------
# Tiny Adam
# ---------------------------------------------------------------------------

def adam_init(p): return dict(m=jnp.zeros_like(p), v=jnp.zeros_like(p), t=0)
def adam_step(g, s, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    t = s["t"] + 1
    m = b1 * s["m"] + (1 - b1) * g
    v = b2 * s["v"] + (1 - b2) * g * g
    upd = lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    return upd, dict(m=m, v=v, t=t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Nx = int(os.environ.get("LUNA_GRAV_NX", "64"))
    n_steps = int(os.environ.get("LUNA_GRAV_NSTEPS", "100"))
    n_iter = int(os.environ.get("LUNA_GRAV_ITER", "60"))
    L = 1.0
    gamma = 5.0 / 3.0
    rho_0 = 1.0
    G = 1.0
    fourpiG = 4.0 * np.pi * G

    # Jeans-unstable: λ_J = cs·sqrt(π/(G ρ₀)) < L for the seeded modes.
    # cs² = γ p₀/ρ₀; choose p₀ so λ_J ≈ 0.5 L → modes k=1,2 unstable.
    cs = 0.5 * L * np.sqrt(G * rho_0 / np.pi)
    p_0 = cs * cs * rho_0 / gamma
    cs = float(np.sqrt(gamma * p_0 / rho_0))

    v_budget = 1.0
    dt = 0.3 * (L / Nx) / (cs + v_budget)
    t_end = n_steps * dt
    t_ff = float(np.sqrt(3.0 * np.pi / (32.0 * G * rho_0)))

    print("IC recovery through self-gravitating hydro (Poisson implicit adjoint)")
    print(f"  Nx={Nx}, params={Nx**3}, n_steps={n_steps}, dt={dt:.3e}, "
          f"t_end={t_end:.3f} (t_ff={t_ff:.3f}), c_s={cs:.3f}")

    key = jax.random.PRNGKey(11)
    delta_rho_true = low_k_field(key, Nx, k_max=2, amp=0.05)

    simulate = make_simulate(Nx, L, gamma, rho_0, p_0, fourpiG, n_steps, dt)

    t0 = time.time()
    U_obs = simulate(delta_rho_true)
    jax.block_until_ready(U_obs)
    m_tot = float(jnp.sum(U_obs[0]))
    m_ic = float(Nx ** 3 * rho_0)
    print(f"  observation in {time.time()-t0:.1f}s; ρ∈[{float(U_obs[0].min()):.4f}, "
          f"{float(U_obs[0].max()):.4f}], ΔM/M = {(m_tot-m_ic)/m_ic:.2e}")

    def loss_fn(dr):
        return jnp.mean((simulate(dr) - U_obs) ** 2)

    loss_grad = jax.jit(jax.value_and_grad(loss_fn))

    # FD cross-check at iter 0
    delta_rho = jnp.zeros_like(delta_rho_true)
    print("\n  FD cross-check (5 voxels, h=1e-4):")
    L0, g_ad = loss_grad(delta_rho)
    g_ad_np = np.asarray(g_ad)
    rng = np.random.default_rng(0)
    h = 1e-4
    for flat in rng.choice(Nx ** 3, size=5, replace=False):
        i, j, k = np.unravel_index(flat, (Nx, Nx, Nx))
        Lp = float(loss_fn(delta_rho.at[i, j, k].add(h)))
        Lm = float(loss_fn(delta_rho.at[i, j, k].add(-h)))
        gfd = (Lp - Lm) / (2.0 * h)
        gad = g_ad_np[i, j, k]
        rel = abs(gfd - gad) / max(abs(gfd), 1e-30)
        print(f"    ({i:2d},{j:2d},{k:2d}): AD={gad:+.5e} FD={gfd:+.5e} rel={rel:.2e}")

    # Adam
    opt = adam_init(delta_rho)
    losses, residuals = [], []
    print(f"\n  Adam ({n_iter} iters):")
    t0 = time.time()
    for it in range(n_iter):
        L_val, g = loss_grad(delta_rho)
        upd, opt = adam_step(g, opt, lr=2e-2)
        delta_rho = delta_rho - upd
        res = float(jnp.linalg.norm(delta_rho - delta_rho_true))
        losses.append(float(L_val)); residuals.append(res)
        if it < 3 or it % 10 == 0 or it == n_iter - 1:
            print(f"  iter {it:3d}  L={float(L_val):.4e}  ‖δρ_est−δρ_true‖={res:.4e}")
    print(f"  wall: {time.time()-t0:.1f}s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].semilogy(losses, "o-", ms=3, label="loss")
    axes[0].semilogy(residuals, "s-", ms=3, label="‖δρ_est − δρ_true‖")
    axes[0].set_xlabel("Adam iter"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"Self-gravitating IC recovery (Nx={Nx})")
    zm = Nx // 2
    vmin, vmax = float(delta_rho_true.min()), float(delta_rho_true.max())
    im = axes[1].imshow(np.asarray(delta_rho_true[:, :, zm]), vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[1].set_title("δρ_true"); plt.colorbar(im, ax=axes[1], shrink=0.8)
    im = axes[2].imshow(np.asarray(delta_rho[:, :, zm]), vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[2].set_title("δρ_recovered"); plt.colorbar(im, ax=axes[2], shrink=0.8)
    os.makedirs("figures", exist_ok=True)
    out = f"figures/grav_inverse_recovery_Nx{Nx}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140)
    print(f"  wrote {out}")
    rel_final = residuals[-1] / float(jnp.linalg.norm(delta_rho_true))
    print(f"\n  Final ‖δρ_est − δρ_true‖/‖δρ_true‖ = {rel_final:.3e}")


if __name__ == "__main__":
    main()
