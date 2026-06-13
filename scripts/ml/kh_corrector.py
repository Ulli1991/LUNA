"""Solver-in-the-loop CNN corrector on 2D Kelvin-Helmholtz (#41).

A 64^2 coarse MUSCL-Hancock run accumulates discretization error vs a 256^2
reference. A small CNN applied after every coarse step is trained to cancel
that error, with gradients flowing through K unrolled solver steps
(Um et al. 2020, "Solver-in-the-Loop"). All gradient machinery (limiter
fixes, FD validation) from the autodiff arc is what makes this possible.

Stages (one job):
  1. reference: 256^2 fine run (dt/4), average-pooled 4x to 64^2 at every
     coarse time -> ref trajectory [n_steps+1, 4, 64, 64]
  2. train: random windows, unroll K=8 (coarse step + CNN), normalized MSE
     vs ref, Adam. Final conv zero-init -> training starts from pure solver.
  3. eval: full 64^2 rollouts with/without correction, error-vs-time + maps.

Env: LUNA_KHC_NC (coarse N, default 64), LUNA_KHC_ITERS (default 600).
Output: figures/kh_corrector_Nc{N}.png, checkpoints/kh_corrector_Nc{N}.npz
"""
from __future__ import annotations

import os, sys, time
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")
jax.config.update("jax_enable_x64", True)

from hydro.muscl_hancock import muscl_hancock_divergence
from hydro.conversions import cons_to_prim

GAMMA = 5.0 / 3.0
LX = LY = 2.0


# ---------------------------------------------------------------------------
# 2D KH setup (double tanh shear layers at y=0.5, 1.5 — same as validation)
# ---------------------------------------------------------------------------

def kh_ic(N):
    dx = LX / N
    c = (jnp.arange(N, dtype=jnp.float64) + 0.5) * dx
    X, Y = jnp.meshgrid(c, c, indexing="ij")
    a, sigma = 0.05, 0.2
    band = 0.5 * (jnp.tanh((Y - 0.5) / a) - jnp.tanh((Y - 1.5) / a))
    rho = 1.0 + band                      # 2 inside the band, 1 outside
    vx = 0.5 - band                       # +-0.5 shear
    vy = 0.02 * jnp.sin(4 * jnp.pi * X / LX) * (
        jnp.exp(-((Y - 0.5) ** 2) / sigma**2) +
        jnp.exp(-((Y - 1.5) ** 2) / sigma**2))
    P = jnp.full_like(rho, 2.5)
    E = P / (GAMMA - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return jnp.stack([rho, rho * vx, rho * vy, E])


def make_step(N, dt, limiter=1, order=1):
    inv_d = jnp.asarray(N / LX, dtype=jnp.float64)

    def step(U):
        W = cons_to_prim(U, GAMMA)
        W_pad = jnp.pad(W, ((0, 0), (1, 1), (1, 1)), mode="wrap")
        dUdt = muscl_hancock_divergence(
            W_pad, dt, gamma=GAMMA, limiter=limiter, riemann_solver=1,
            spatial_dim=2, inv_dx=inv_d, inv_dy=inv_d, inv_dz=None,
            boundary=0, order=order)
        return U + dt * dUdt

    return step


def pool4(U):
    """Average-pool 4x4 blocks: (4, 4N, 4N) -> (4, N, N). Conservative."""
    C, H, W = U.shape
    return U.reshape(C, H // 4, 4, W // 4, 4).mean(axis=(2, 4))


# ---------------------------------------------------------------------------
# Tiny CNN, pure JAX. Periodic conv, zero-init last layer (no-op at start).
# ---------------------------------------------------------------------------

# Last layer emits 8 = 4 conserved fields × 2 face-flux directions. The
# correction is the discrete divergence of those fluxes (see cnn_apply),
# which is conservative per channel by construction.
CHANNELS = [4, 48, 48, 48, 8]


def cnn_init(key):
    params = []
    for i, (cin, cout) in enumerate(zip(CHANNELS[:-1], CHANNELS[1:])):
        key, k = jax.random.split(key)
        last = i == len(CHANNELS) - 2
        scale = 0.0 if last else jnp.sqrt(2.0 / (cin * 9))
        W = scale * jax.random.normal(k, (cout, cin, 3, 3), dtype=jnp.float64)
        b = jnp.zeros((cout,), dtype=jnp.float64)
        params.append((W, b))
    return params


def cnn_apply(params, x, norm):
    """x: (4, N, N) conserved state; returns a CONSERVATIVE (4, N, N)
    correction = discrete divergence of learned face fluxes.

    The net (whole-domain) integral of a discrete divergence on a periodic
    grid is exactly zero per channel, so the corrector cannot inject or
    destroy total mass / momentum / energy — it only redistributes them.
    The previous additive form (U + cnn(U)) had no such constraint and was
    driving ρ below its physical floor (see figures/kh_corrector_Nc64.png).
    """
    h = (x / norm[:, None, None])[None]            # (1, 4, N, N)
    for i, (W, b) in enumerate(params):
        h = jnp.pad(h, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="wrap")
        h = jax.lax.conv_general_dilated(
            h, W, window_strides=(1, 1), padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"))
        h = h + b[None, :, None, None]
        if i < len(params) - 1:
            h = jax.nn.gelu(h)
    f = h[0]                                       # (8, N, N) face fluxes
    fx, fy = f[:4], f[4:]                          # per-channel x- and y-flux
    # div: face flux at i+1/2 is fx[...,i,:]; correction_i = F_{i+1/2}-F_{i-1/2}.
    # Sum over i telescopes to 0 on the periodic grid -> conservative.
    div = (fx - jnp.roll(fx, 1, axis=1)) + (fy - jnp.roll(fy, 1, axis=2))
    # 0.02 keeps per-step redistribution at percent-level of field scale;
    # combined with the conservation guarantee this no longer needs the
    # solver to absorb spurious mass.
    return 0.02 * div * norm[:, None, None]


# ---------------------------------------------------------------------------
# Adam (same micro-impl as ou_inverse, but over a pytree)
# ---------------------------------------------------------------------------

def adam_init(p):
    z = jax.tree.map(jnp.zeros_like, p)
    return dict(m=z, v=jax.tree.map(jnp.zeros_like, p), t=0)


def adam_step(g, p, s, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = s["t"] + 1
    m = jax.tree.map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, s["m"], g)
    v = jax.tree.map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, s["v"], g)
    mh = jax.tree.map(lambda m_: m_ / (1 - b1**t), m)
    vh = jax.tree.map(lambda v_: v_ / (1 - b2**t), v)
    p = jax.tree.map(lambda p_, mh_, vh_: p_ - lr * mh_ / (jnp.sqrt(vh_) + eps),
                     p, mh, vh)
    return p, dict(m=m, v=v, t=t)


# ---------------------------------------------------------------------------


def main():
    Nc = int(os.environ.get("LUNA_KHC_NC", "64"))
    n_iter = int(os.environ.get("LUNA_KHC_ITERS", "600"))
    Nf = 4 * Nc
    dt_c = 0.004
    n_steps = 1000                                   # t_end = 4.0
    K = 8                                            # unroll window
    BATCH = 2                                         # small: long-horizon BPTT

    print(f"KH corrector: coarse {Nc}^2, fine {Nf}^2, {n_steps} steps, "
          f"K={K}, {n_iter} iters")

    # --- Stage 1: reference trajectory ---
    t0 = time.time()
    step_f = jax.jit(make_step(Nf, dt_c / 4))
    Uf = kh_ic(Nf)

    @jax.jit
    def four_fine(U):
        for _ in range(4):
            U = step_f(U)
        return U

    ref = [pool4(Uf)]
    for _ in range(n_steps):
        Uf = four_fine(Uf)
        ref.append(pool4(Uf))
    ref = jnp.stack(ref)                              # (n_steps+1, 4, Nc, Nc)
    assert bool(jnp.all(jnp.isfinite(ref))), "reference run not finite"
    print(f"  reference done in {time.time()-t0:.0f}s, "
          f"ref shape {ref.shape}", flush=True)

    norm = jnp.std(ref, axis=(0, 2, 3)) + 1e-12       # per-field scale
    # Coarse solver order: 0 = first-order Godunov (very diffusive, large
    # discretization error → headroom for the corrector to recover fine
    # structure); 1 = PLM (already near-reference at 64², little to correct).
    c_order = int(os.environ.get("LUNA_KHC_ORDER", "1"))
    step_c = make_step(Nc, dt_c, order=c_order)
    print(f"  coarse order={c_order}", flush=True)

    def correction(params, U):
        return cnn_apply(params, U, norm)

    def corrected_step(params, U):
        return step_c(U) + correction(params, U)

    def grid_roughness(c):
        """Mean squared discrete Laplacian of a (4, N, N) field, normalized.
        The 5-point Laplacian's eigenvalue ∝ k², so this term is dominated
        by the 2Δx checkerboard mode — exactly the grid-scale instability
        that blew up the rollout at step 836. Penalizing it starves the
        odd-even mode while leaving smooth corrections almost free."""
        lap = (jnp.roll(c, 1, 1) + jnp.roll(c, -1, 1) +
               jnp.roll(c, 1, 2) + jnp.roll(c, -1, 2) - 4.0 * c)
        return jnp.mean((lap / norm[:, None, None]) ** 2)

    # --- Stage 2: training ---
    # Long-horizon a-posteriori (solver-in-the-loop) training, following
    # Horowitz et al. 2025 (diffhydro §8.1): unroll the corrected solver for
    # HUNDREDS of steps so stability is *learned* under the model's own
    # compounding, rather than imposed via a roughness penalty. Their
    # corrector trains to t≈1.0 (~1000 steps) and stays stable past t=3.
    # We previously unrolled only K≤48 and the model never saw the regime
    # where it blew up (step ~625-840).
    #   * K-curriculum 32→256 (was 8→48).
    #   * LAMBDA_SMOOTH default 0: drop the band-aid; long rollouts make the
    #     checkerboard penalty unnecessary (and at order=0 it actively
    #     fought the sharpening the corrector needs).
    #   * body remat'd so BPTT memory stays bounded over the long unroll.
    LAMBDA_SMOOTH = float(os.environ.get("LUNA_KHC_SMOOTH", "0.0"))

    def window_loss(params, start, K):
        U0 = jax.lax.dynamic_index_in_dim(ref, start, axis=0, keepdims=False)
        @jax.checkpoint
        def body(U, j):
            c = correction(params, U)
            U = step_c(U) + c
            tgt = jax.lax.dynamic_index_in_dim(ref, start + 1 + j, axis=0,
                                               keepdims=False)
            err = jnp.mean(((U - tgt) / norm[:, None, None]) ** 2)
            rough = grid_roughness(c)
            return U, jnp.stack([err, rough])
        _, terms = jax.lax.scan(body, U0, jnp.arange(K))
        err_m, rough_m = jnp.mean(terms, axis=0)
        return err_m + LAMBDA_SMOOTH * rough_m

    def batch_loss(params, starts, K):
        return jnp.mean(jax.vmap(lambda s: window_loss(params, s, K))(starts))

    loss_grad = jax.jit(jax.value_and_grad(batch_loss), static_argnums=2)

    key = jax.random.PRNGKey(0)
    key, k_init = jax.random.split(key)
    params = cnn_init(k_init)
    opt = adam_init(params)
    t0 = time.time()
    for it in range(n_iter):
        K = (32, 64, 128, 256)[min(3, it * 4 // n_iter)]  # long-horizon curriculum
        key, k_s = jax.random.split(key)
        starts = jax.random.randint(k_s, (BATCH,), 0, n_steps - K - 1)
        lr = 5e-4 * (0.1 ** (it / n_iter))            # decay to 5e-5
        L, g = loss_grad(params, starts, K)
        if not bool(jnp.isfinite(L)) or float(L) > 1.0:
            print(f"  iter {it:4d}  loss={float(L):.2e} off-manifold, "
                  f"skipping update", flush=True)
            continue
        gnorm = jnp.sqrt(sum(jnp.sum(W**2) + jnp.sum(b**2) for W, b in g))
        clip = jnp.minimum(1.0, 1.0 / (gnorm + 1e-12))
        g = jax.tree.map(lambda x: x * clip, g)
        params, opt = adam_step(g, params, opt, lr)
        if it < 3 or it % 50 == 0 or it == n_iter - 1:
            print(f"  iter {it:4d}  loss={float(L):.4e}  lr={lr:.1e}",
                  flush=True)
    print(f"  training: {time.time()-t0:.0f}s", flush=True)

    # --- Stage 3: eval rollouts ---
    step_c_jit = jax.jit(step_c)
    corr_jit = jax.jit(lambda U: corrected_step(params, U))

    def rollout(stepper, label):
        U = ref[0]
        errs = []
        first_nan = None
        U_last = U
        for i in range(1, n_steps + 1):
            U = stepper(U)
            if first_nan is None and not bool(jnp.all(jnp.isfinite(U))):
                first_nan = i
                print(f"  [{label}] first non-finite at step {i} "
                      f"(t={i*dt_c:.3f})", flush=True)
            if first_nan is None:
                U_last = U
            if i % 25 == 0 and first_nan is None:
                e = jnp.linalg.norm(U[0] - ref[i][0]) / jnp.linalg.norm(ref[i][0])
                errs.append((i * dt_c, float(e)))
        return U_last, errs

    U_plain, err_plain = rollout(step_c_jit, "coarse")
    U_corr, err_corr = rollout(corr_jit, "corrected")
    c_final = err_corr[-1][1] if err_corr else float("nan")
    print(f"  final rel rho error: coarse-only={err_plain[-1][1]:.3e}  "
          f"corrected={c_final:.3e}  (corr survived "
          f"{len(err_corr)*25 if err_corr else 0} steps)", flush=True)

    os.makedirs("checkpoints", exist_ok=True)
    np.savez(f"checkpoints/kh_corrector_Nc{Nc}.npz",
             norm=np.asarray(norm),
             **{f"W{i}": np.asarray(W) for i, (W, _) in enumerate(params)},
             **{f"b{i}": np.asarray(b) for i, (_, b) in enumerate(params)})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4))
    tp, ep = zip(*err_plain); tc, ec = zip(*err_corr)
    axes[0].semilogy(tp, ep, label="coarse only")
    axes[0].semilogy(tc, ec, label="coarse + CNN")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("rel L2 error (ρ)")
    axes[0].set_title(f"rollout error ({Nc}² vs {Nf}² ref)")
    axes[0].grid(alpha=0.3); axes[0].legend()
    for ax, (U, ttl) in zip(axes[1:],
                            [(ref[-1], f"reference (pooled {Nf}²)"),
                             (U_plain, "coarse only"),
                             (U_corr, "coarse + CNN")]):
        im = ax.imshow(np.asarray(U[0]).T, origin="lower", cmap="RdBu_r")
        ax.set_title(ttl + " — ρ(t=4)")
        plt.colorbar(im, ax=ax, shrink=0.85)
    os.makedirs("figures", exist_ok=True)
    out = f"figures/kh_corrector_Nc{Nc}.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
