"""Recover the M82 wind's CC85 loadings (eta_M, eta_E) by reverse-mode AD
through the full coupled model (wind + cooling + self-gravity + external
potential). Circular by design: inject a known eta, evolve, recover it. It
certifies the gradients through the whole physics stack; it does not
measure the real M82.

    python -m scripts.examples.autodiff.m82_autodiff_run nx=64 task=eta_recovery n_iter=60
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from luna.sources.ml_corrector import adam_init, adam_step
from scripts.examples.autodiff.common import central_fd, rollout
from scripts.examples.autodiff.figs import save_eta_recovery_figure


def task(init, cfg, physics, is_main, *, n_iter=100):
    """Recover the CC85 loadings (eta_M, eta_E) with `n_iter` Adam steps."""
    U0 = init.U
    simulate, _, _, eta_true = rollout(init, cfg, physics, is_main, "eta_recovery")
    t0 = time.time()
    U_obs = simulate(eta_true, U0)               # the "observed" evolved state
    log_true = jnp.log(eta_true)

    # Match the full evolved state; recover in log-η (positivity by construction).
    def loss(log_eta):
        return jnp.mean((simulate(jnp.exp(log_eta), U0) - U_obs) ** 2)

    # Start from a deliberately wrong guess (η_M too high, η_E too low).
    log0 = log_true + jnp.log(jnp.array([1.6, 0.6], eta_true.dtype))

    # FD-validate the reverse-mode gradient at the start point.
    loss_j = jax.jit(loss)
    vg = jax.jit(jax.value_and_grad(loss))
    g0 = vg(log0)[1]
    fd0 = jnp.array([central_fd(loss_j, log0, i) for i in range(2)])
    rel = [abs(float(g0[i]) - float(fd0[i])) / (abs(float(fd0[i])) + 1e-30)
           for i in range(2)]
    if is_main:
        print(f"  eta_recovery: forward+FD ({time.time()-t0:.0f}s); "
              f"∂loss/∂lnη AD={np.asarray(g0)}  FD={np.asarray(fd0)}  "
              f"rel={rel}", flush=True)

    # Adam on log-η (shared pytree Adam: b1=0.9, b2=0.999, eps=1e-8, bias-corr).
    log_eta = log0
    opt = adam_init(log_eta)
    n_iter = int(n_iter)
    hist_it, hist_L, hist_etaM, hist_etaE = [], [], [], []
    for it in range(n_iter):
        L, gg = vg(log_eta)
        log_eta, opt = adam_step(gg, log_eta, opt, 3.0e-2)
        eta_now = np.asarray(jnp.exp(log_eta))
        hist_it.append(it); hist_L.append(float(L))
        hist_etaM.append(float(eta_now[0])); hist_etaE.append(float(eta_now[1]))
        if is_main and (it % max(1, n_iter // 10) == 0 or it == n_iter - 1):
            print(f"    it {it:4d}  L={float(L):.3e}  "
                  f"η_M={eta_now[0]:.4f} (true {float(eta_true[0]):.4f})  "
                  f"η_E={eta_now[1]:.4f} (true {float(eta_true[1]):.4f})",
                  flush=True)
    eta_rec = np.asarray(jnp.exp(log_eta))
    eta_t = np.asarray(eta_true)
    res = float(np.max(np.abs(eta_rec - eta_t) / eta_t))
    if is_main:
        print(f"  recovered η=({eta_rec[0]:.4f}, {eta_rec[1]:.4f})  "
              f"true=({eta_t[0]:.4f}, {eta_t[1]:.4f})  "
              f"max rel err={res:.3e}", flush=True)
        save_eta_recovery_figure(hist_it, hist_L, hist_etaM, hist_etaE,
                                 eta_t, rel, res, cfg)
    return {"eta_recovered": eta_rec.tolist(), "eta_true": eta_t.tolist(),
            "eta_rel_err": res, "grad_fd_rel": rel}
