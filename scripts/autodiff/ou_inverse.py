"""Inverse IC recovery in OU-driven turbulence — the diffhydro-scale demo.

Builds the same IC-reconstruction problem Horowitz+ 2025 ran at 256³ with
OU forcing, but with rigour they don't: we cross-check the AD gradient at
a handful of anchor parameters against central-difference FD.

Setup:
  - 5 conservative fields on Nx³ (default 128³).  Periodic.
  - IC: ρ = ρ_0 (1 + δρ(x)),  v = 0,  P uniform.
  - δρ(x) is a low-k Gaussian random field — the parameters we recover.
  - Dynamics: MUSCL-Hancock + a FIXED-realisation OU forcing on momentum.
    (Same random key sequence on forward and inverse — Horowitz §7.4.)
  - Loss = MSE(U_sim − U_obs) at t = t_end.
  - Optimizer: tiny Adam, no optax dependency.
  - FD cross-check at 5 random voxels of δρ at iteration 0.

If 128³ recovers δρ to <15% relative residual in ~80 Adam steps, we
scale to 256³ next (gradient checkpointing turns on at that point).
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
from hydro.turbulence import OUDriver, OUDriverConfig


# ---------------------------------------------------------------------------
# Smooth low-k δρ field (the "truth")
# ---------------------------------------------------------------------------

def low_k_field(key, Nx, k_max=4, amp=0.05):
    k = jnp.fft.fftfreq(Nx) * Nx
    kx, ky, kz = jnp.meshgrid(k, k, k, indexing="ij")
    mask = (jnp.abs(kx) <= k_max) & (jnp.abs(ky) <= k_max) & (jnp.abs(kz) <= k_max)
    mask = mask & ~((kx == 0) & (ky == 0) & (kz == 0))
    k_re, k_im = jax.random.split(key)
    cre = jax.random.normal(k_re, (Nx, Nx, Nx))
    cim = jax.random.normal(k_im, (Nx, Nx, Nx))
    coeffs = (cre + 1j * cim) * mask
    field = jnp.fft.ifftn(coeffs).real
    return field / jnp.std(field) * amp


# ---------------------------------------------------------------------------
# IC & step
# ---------------------------------------------------------------------------

def build_ic(delta_rho, Nx, rho_0, p_0, gamma):
    rho = rho_0 * (1.0 + delta_rho)
    vx = jnp.zeros_like(rho); vy = jnp.zeros_like(rho); vz = jnp.zeros_like(rho)
    p = jnp.full_like(rho, p_0)
    E = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy + vz * vz)
    return jnp.stack([rho, rho * vx, rho * vy, rho * vz, E])


def _pad_periodic(W, ghost=1):
    return jnp.pad(W, ((0, 0), (ghost, ghost), (ghost, ghost), (ghost, ghost)),
                   mode="wrap")


def _hydro_step(U, dt, *, gamma, limiter, order, inv_dx):
    W = cons_to_prim(U, gamma)
    W_pad = _pad_periodic(W, ghost=1)
    dUdt = muscl_hancock_divergence(
        W_pad, dt,
        gamma=gamma, limiter=limiter, riemann_solver=1,
        spatial_dim=3, inv_dx=inv_dx, inv_dy=inv_dx, inv_dz=inv_dx,
        boundary=0, order=order,
    )
    return U + dt * dUdt


def _apply_ou(U, force, dt):
    """Add momentum from acceleration F; rebalance energy to conserve KE+TE."""
    Fx, Fy, Fz = force
    rho = U[0]
    mx, my, mz, Etot = U[1], U[2], U[3], U[4]
    ke_old = 0.5 * (mx * mx + my * my + mz * mz) / rho
    mx2 = mx + rho * Fx * dt
    my2 = my + rho * Fy * dt
    mz2 = mz + rho * Fz * dt
    ke_new = 0.5 * (mx2 * mx2 + my2 * my2 + mz2 * mz2) / rho
    Etot2 = Etot + (ke_new - ke_old)
    return jnp.stack([rho, mx2, my2, mz2, Etot2])


# ---------------------------------------------------------------------------
# Full simulation: parametrised by δρ + fixed OU key sequence
# ---------------------------------------------------------------------------

def _make_simulate(Nx, L, gamma, rho_0, p_0, n_steps, dt, order, limiter,
                    ou_driver, X, Y, Z, ou_keys):
    inv_dx = jnp.asarray(Nx / L, dtype=jnp.float64)

    @functools.partial(jax.jit)
    def simulate(delta_rho, ou_a0):
        U = build_ic(delta_rho, Nx, rho_0, p_0, gamma)

        def body(carry, key):
            U, a = carry
            # Hydro half-step
            U = _hydro_step(U, dt, gamma=gamma, limiter=limiter,
                            order=order, inv_dx=inv_dx)
            # OU evolve
            a_new = ou_driver.evolve(a, dt, key)
            # Force on momentum
            Fx, Fy, Fz = ou_driver.force_field(a_new, X, Y, Z)
            U = _apply_ou(U, (Fx, Fy, Fz), dt)
            return (U, a_new), None

        (U_final, _), _ = jax.lax.scan(body, (U, ou_a0), ou_keys)
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
    update = lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    return update, dict(m=m, v=v, t=t)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Nx = int(os.environ.get("LUNA_OU_NX", "128"))
    n_steps = int(os.environ.get("LUNA_OU_NSTEPS", "100"))
    n_iter = int(os.environ.get("LUNA_OU_ITER", "80"))
    L = 1.0
    gamma = 5.0 / 3.0
    rho_0 = 1.0
    p_0 = 1.0
    cs = float(np.sqrt(gamma * p_0 / rho_0))
    dt = 0.4 * (L / Nx) / cs    # CFL-based
    t_end = n_steps * dt

    print(f"Inverse IC in OU-driven turbulence")
    print(f"  Nx={Nx}, params={Nx**3}, n_steps={n_steps}, dt={dt:.3e}, "
          f"t_end={t_end:.3f}, c_s≈{cs:.3f}")

    # OU driver: low-k forcing.
    ou_cfg = OUDriverConfig(
        Lx=L, Ly=L, Lz=L,
        k_min=1.0, k_max=4.0,    # integer mode numbers (the config converts to 2π/Lx)
        rms_acceleration=2.0, autocorrelation_time=0.5,
        solenoidal_weight=1.0,
    )
    ou = OUDriver(cfg=ou_cfg)
    print(f"  OU driver: {ou.n_modes} modes, t_ac={ou_cfg.autocorrelation_time}")

    key = jax.random.PRNGKey(7)
    k_true, k_ou0, k_walk = jax.random.split(key, 3)
    delta_rho_true = low_k_field(k_true, Nx, k_max=4, amp=0.05)
    ou_a0 = ou.init_state(k_ou0)
    ou_keys = jax.random.split(k_walk, n_steps)

    # Coordinate grids for force_field.
    coord = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * (L / Nx)
    X, Y, Z = jnp.meshgrid(coord, coord, coord, indexing="ij")

    simulate = _make_simulate(Nx, L, gamma, rho_0, p_0, n_steps, dt,
                               order=2, limiter=0, ou_driver=ou,
                               X=X, Y=Y, Z=Z, ou_keys=ou_keys)

    print(f"  generating observation U_obs (forward sim with truth)...")
    t0 = time.time()
    U_obs = simulate(delta_rho_true, ou_a0)
    jax.block_until_ready(U_obs)
    print(f"  observation ready in {time.time()-t0:.1f}s, "
          f"ρ ∈ [{float(U_obs[0].min()):.3f}, {float(U_obs[0].max()):.3f}]")

    # Loss
    def loss_fn(dr):
        U = simulate(dr, ou_a0)
        return jnp.mean((U - U_obs) ** 2)

    loss_grad = jax.jit(jax.value_and_grad(loss_fn))

    # --- FD cross-check at iteration 0 ---
    delta_rho = jnp.zeros_like(delta_rho_true)
    print("\n  FD cross-check on 5 random voxels of δρ at iter 0...")
    L0, g_ad = loss_grad(delta_rho)
    g_ad = np.asarray(g_ad)
    print(f"    L(0) = {float(L0):.4e}")
    np.random.seed(0)
    h = 1e-4
    flat = g_ad.flatten()
    samples = np.random.choice(flat.size, size=5, replace=False)
    for idx in samples:
        i, j, k = np.unravel_index(idx, g_ad.shape)
        dr_plus  = delta_rho.at[i, j, k].add(h)
        dr_minus = delta_rho.at[i, j, k].add(-h)
        Lp = float(loss_fn(dr_plus))
        Lm = float(loss_fn(dr_minus))
        gfd = (Lp - Lm) / (2.0 * h)
        gad = float(flat[idx])
        rel = abs(gfd - gad) / max(abs(gfd), 1e-30)
        print(f"    voxel ({i:3d},{j:3d},{k:3d}): "
              f"AD={gad:+.4e}  FD={gfd:+.4e}  rel={rel:.2e}")

    # --- Adam optimization ---
    opt = adam_init(delta_rho)
    losses = []; residuals = []
    print(f"\n  Adam optimization for {n_iter} iters")
    t0 = time.time()
    for it in range(n_iter):
        L_val, grads = loss_grad(delta_rho)
        upd, opt = adam_step(grads, opt, lr=2e-2)
        delta_rho = delta_rho - upd
        res = float(jnp.linalg.norm(delta_rho - delta_rho_true))
        L_val = float(L_val)
        losses.append(L_val); residuals.append(res)
        if it < 3 or it % 10 == 0 or it == n_iter - 1:
            print(f"  iter {it:3d}  L={L_val:.4e}  ‖δρ_est−δρ_true‖={res:.4e}")
    print(f"  Adam: {time.time()-t0:.1f}s wall for {n_iter} iter")

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].semilogy(losses, "o-", lw=1, ms=3, label="loss")
    axes[0].semilogy(residuals, "s-", lw=1, ms=3, label="‖δρ_est − δρ_true‖")
    axes[0].set_xlabel("Adam iter"); axes[0].set_ylabel("value")
    axes[0].set_title(f"Loss & IC residual (Nx={Nx}, OU)"); axes[0].grid(alpha=0.3); axes[0].legend()

    z_mid = Nx // 2
    vmin = float(jnp.min(delta_rho_true)); vmax = float(jnp.max(delta_rho_true))
    im0 = axes[1].imshow(np.asarray(delta_rho_true[:, :, z_mid]),
                          vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[1].set_title("δρ_true (z-mid)"); plt.colorbar(im0, ax=axes[1], shrink=0.85)
    im1 = axes[2].imshow(np.asarray(delta_rho[:, :, z_mid]),
                          vmin=vmin, vmax=vmax, cmap="RdBu_r")
    axes[2].set_title("δρ_recovered (z-mid)"); plt.colorbar(im1, ax=axes[2], shrink=0.85)

    os.makedirs("figures", exist_ok=True)
    out = f"figures/ou_inverse_recovery_Nx{Nx}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")
    rel_final = residuals[-1] / float(jnp.linalg.norm(delta_rho_true))
    print(f"\n  Final ‖δρ_est − δρ_true‖ / ‖δρ_true‖ = {rel_final:.3e}")


if __name__ == "__main__":
    main()
