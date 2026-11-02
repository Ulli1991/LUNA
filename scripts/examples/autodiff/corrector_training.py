"""A solver-in-the-loop CNN flux corrector (Um et al. 2020), trained by
reverse-mode AD through the unrolled production coarse step so the coarse
rollout tracks a finer run of the same setup (diffhydro Fig. 16/17 analog).

    python -m scripts.examples.autodiff.kh_corrector_run nx=32 n_steps=200 n_iter=100
    python -m scripts.examples.autodiff.kh_corrector_run nx=32 fine_factor=8 n_train=100
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from luna.sources.ml_corrector import adam_init, adam_step
from scripts.examples.autodiff.figs import save_corrector_figure


def task(init, cfg, physics, is_main, *, fine_factor=4, n_train=0, n_iter=100):
    """Train the corrector: the coarse rollout uses the production step and IC,
    the reference is the production setup `fine_factor` times finer, pooled
    back. n_train: training horizon in steps (0 = a third of n_steps); n_iter:
    optimizer iterations."""
    from dataclasses import replace
    from luna.utils.grid_params import GridParams
    from luna.setup.hydro import create_solver_distributed
    from luna.setup.ics import gas_grid
    from luna.setup.mesh import create_distributed_mesh
    from luna.sources.ml_corrector import cnn_init, cnn_correction, pool_factor

    U0 = init.U
    step_fn, compute_dt_fn = physics.hydro.step, physics.hydro.compute_dt
    f = int(fine_factor)                         # coarse->fine gap (8x for the corrector)
    Nfx = f * int(cfg.Nx)
    Nfy = f * int(cfg.Ny)
    n_steps = int(cfg.autodiff_n_steps)                       # eval horizon (~t=3)
    n_train = int(n_train) or max(1, n_steps // 3)            # train horizon (~t=1)
    n_iter = int(n_iter)
    dt_c = (float(cfg.autodiff_fixed_dt) if cfg.autodiff_fixed_dt > 0
            else float(compute_dt_fn(U0)))

    # --- fine reference via the PRODUCTION setup at 4x res ---
    fine_cfg = replace(cfg, Nx=Nfx, Ny=Nfy, autodiff_mode=False)
    fmesh, _ = create_distributed_mesh(fine_cfg)
    Uf, _, _ = gas_grid(fine_cfg, fmesh)
    fgrid = GridParams.from_config(fine_cfg)
    _, fstep, _ = create_solver_distributed(fine_cfg, fgrid, fmesh)
    fstep_j = jax.jit(fstep)

    @jax.jit
    def four_fine(U):
        for _ in range(f):
            U = fstep_j(U, dt_c / f)
        return U

    ref = [pool_factor(Uf, f)]
    for _ in range(n_steps):
        Uf = four_fine(Uf)
        ref.append(pool_factor(Uf, f))
    ref = jnp.stack(ref)
    norm = jnp.std(ref, axis=(0, 2, 3)) + 1e-12
    if is_main:
        print(f"  corrector: coarse {cfg.Nx}x{cfg.Ny}, fine {Nfx}x{Nfy}, {n_steps} steps, "
              f"dt={dt_c:.4e}; fine reference built", flush=True)

    def corrected_step(w, U):
        return step_fn(U, dt_c) + cnn_correction(w, U, norm)

    # Full-horizon solver-in-the-loop loss (Um et al. 2020): unroll the
    # corrected coarse rollout from the TRUE initial state and match the
    # downsampled fine reference at every step. Training on the model's OWN
    # drifted trajectory (not clean reference windows) removes the exposure
    # bias that made the corrector regress at deployment. sqrt-checkpointing
    # keeps reverse-mode memory bounded for long rollouts.
    per = max(1, int(round(n_train ** 0.5)))
    nchunk = (n_train + per - 1) // per

    def full_loss(w):
        @jax.checkpoint
        def chunk(carry, _):
            U, i = carry
            def body(c, _):
                U, i = c
                U = corrected_step(w, U)
                tgt = jax.lax.dynamic_index_in_dim(
                    ref, jnp.minimum(i + 1, n_train), axis=0, keepdims=False)
                e = jnp.mean(((U - tgt) / norm[:, None, None]) ** 2)
                return (U, i + 1), e
            (U, i), errs = jax.lax.scan(body, (U, i), None, length=per)
            return (U, i), jnp.sum(errs)
        _, esum = jax.lax.scan(chunk, (ref[0], 0), None, length=nchunk)
        return jnp.sum(esum) / (nchunk * per)

    loss_grad = jax.jit(jax.value_and_grad(full_loss))

    key = jax.random.PRNGKey(0)
    key, ki = jax.random.split(key)
    w = cnn_init(ki)
    opt = adam_init(w)
    t0 = time.time()
    for it in range(n_iter):
        warm = min(1.0, (it + 1) / 20.0)          # 20-iter warmup, avoids early blow-up
        lr = warm * 4.0e-4 * (0.5 ** (it / max(1, n_iter // 4)))
        L, g = loss_grad(w)
        if not bool(jnp.isfinite(L)):
            continue
        gn = jnp.sqrt(sum(jnp.sum(W ** 2) + jnp.sum(b ** 2) for W, b in g))
        g = jax.tree.map(lambda x: x * jnp.minimum(1.0, 1.0 / (gn + 1e-12)), g)
        w, opt = adam_step(g, w, opt, lr)
        if is_main and (it % max(1, n_iter // 10) == 0 or it == n_iter - 1):
            print(f"    it {it:4d}  loss={float(L):.4e}", flush=True)
    if is_main:
        print(f"  training: {time.time()-t0:.0f}s", flush=True)

    # eval: coarse vs corrected rollout to the FULL horizon (3x the training
    # horizon, ~t=3), capturing field snapshots at 6 montage times + density
    # MSE vs time (diffhydro Fig 16 montage + Fig 17 MSE curve).
    step_c_j = jax.jit(lambda U: step_fn(U, dt_c))
    corr_j = jax.jit(lambda U: corrected_step(w, U))
    mtimes = sorted(set(max(1, n_steps * k // 6) for k in range(1, 7)))
    def rollout(stepper):
        U = ref[0]; curve = []; snaps = {}
        for i in range(1, n_steps + 1):
            U = stepper(U)
            if not bool(jnp.all(jnp.isfinite(U))):
                break
            if i % 25 == 0:
                curve.append((i * dt_c, float(jnp.mean((U[0] - ref[i][0]) ** 2))))
            if i in mtimes:
                snaps[i] = np.asarray(U)
        return curve, snaps
    c_plain, snap_plain = rollout(step_c_j)
    c_corr, snap_corr = rollout(corr_j)
    snap_ref = {i: np.asarray(ref[i]) for i in mtimes}
    bp = c_plain[-1][1] if c_plain else float("nan")
    bc = c_corr[-1][1] if c_corr else float("nan")
    if is_main:
        print(f"  density MSE @ t_end: coarse-only={bp:.3e}  +CNN={bc:.3e}  "
              f"(corr survived {len(c_corr)*25}/{n_steps} steps; "
              f"trained to step {n_train})", flush=True)
        save_corrector_figure(c_plain, c_corr, snap_ref, snap_plain, snap_corr,
                               mtimes, dt_c, n_train, cfg)
    return {"coarse_mse": c_plain, "corr_mse": c_corr, "weights": w}
