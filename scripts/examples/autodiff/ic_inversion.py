"""Recover an IC perturbation from the evolved state by reverse-mode AD through
the production solver: the gradient with respect to the full Nx*Ny*Nz field,
Adam on top, and the scale-resolved recovery r(k) as the figure.

    python -m scripts.examples.autodiff.ic_inversion_run problem=jeans nx=32
    python -m scripts.examples.autodiff.turb_autodiff_run nx=32 n_steps=200
    python main.py my_cfg.py autodiff_mode=1 autodiff_task=ic_inversion n_iter=40
"""
import time

import numpy as np
import jax
import jax.numpy as jnp

from luna.ic_generation.spectral import spectral_ic_field
from luna.sources.ml_corrector import adam_init, adam_step
from scripts.examples.autodiff.common import central_fd, rollout
from scripts.examples.autodiff.figs import save_inversion_figure


def task(init, cfg, physics, is_main, *, field="density", n_iter=100, lr=1.0e-2,
         kmax=0, slope=0.0):
    """Recover the IC `field` ("density" or "energy") with `n_iter` Adam steps
    at rate `lr`. The true perturbation is broadband with power-law `slope`
    (|F| ~ k^slope, 0 white), or band-limited to k <= `kmax` when > 0."""
    simulate, _, _, eta_true = rollout(init, cfg, physics, is_main, "ic_inversion")
    return _invert(simulate, init.U, eta_true, cfg, is_main, field=field,
                   n_iter=n_iter, lr=lr, kmax=kmax, slope=slope)


def _recovery_rk(delta_est, delta_true):
    """Scale-resolved recovery correlation r(k) per k-shell."""
    Nx = delta_true.shape[0]
    fe = np.fft.rfftn(np.asarray(delta_est))
    ft = np.fft.rfftn(np.asarray(delta_true))
    kf = np.fft.fftfreq(Nx) * Nx
    kx, ky = np.meshgrid(kf, kf, indexing="ij")
    kz = np.arange(fe.shape[2])
    kmag = np.sqrt(kx[..., None] ** 2 + ky[..., None] ** 2 + kz ** 2)
    kbin = np.rint(kmag).astype(int)
    nk = Nx // 2
    r = np.zeros(nk)
    for k in range(1, nk):
        m = kbin == k
        pt = np.sum(np.abs(ft[m]) ** 2); pe = np.sum(np.abs(fe[m]) ** 2)
        if pt > 0 and pe > 0:
            r[k] = np.sum((fe[m] * np.conj(ft[m])).real) / np.sqrt(pt * pe)
    return r


def _invert(simulate, U0, eta_true, cfg, is_main, *, field, n_iter, lr, kmax, slope):
    """Recover an IC density perturbation by reverse-mode AD through the
    production solver (high-dimensional inverse — the reverse-mode home turf).
    Differentiates w.r.t. the FULL log-density field (Nx*Ny*Nz inputs).
    Emits the recovery + r(k) figure (the reverse-mode proof)."""
    shape = U0.shape[1:]
    # Broadband true IC by default (kmax=0): signal at every scale, so
    # r(k) measures the solver's real recovery reach. Cooling (dissipative) will
    # collapse at its physical limit; self-gravity (lossless FMG adjoint) recovers
    # to high k with the SAME optimizer -> the contrast is pure physics.
    k_max = int(kmax or 0)
    slope = float(slope or 0.0)
    delta_true = spectral_ic_field(jax.random.PRNGKey(1), shape,
                                    k_max=k_max, amp=0.05, slope=slope)
    delta_true_np = np.asarray(delta_true)   # host copy BEFORE any sharding:
    # the r(k)/figure diagnostics need a plain addressable array (np.asarray on a
    # multi-process-sharded delta_true crashes on the non-addressable fetch).

    # The perturbation field must carry the SAME spatial sharding as U0 (its
    # spatial spec with the leading n_vars axis dropped). Otherwise delta is
    # replicated while the field is sharded, and reverse-mode mis-handles the
    # gradient — GSPMD all-reduces the replicated-input grad, summing per-shard
    # contributions that should stay disjoint per spatial slab -> a wrong gradient
    # (measured: r(k)~0, loss stuck). Sharding delta keeps its grad sharded.
    _dshard = None
    try:
        from jax.sharding import NamedSharding, PartitionSpec as P
        _uspec = U0.sharding.spec
        if any(s is not None for s in _uspec):        # U0 is actually sharded
            _dshard = NamedSharding(U0.sharding.mesh, P(*_uspec[1:]))
            delta_true = jax.device_put(delta_true, _dshard)
    except (AttributeError, TypeError):
        _dshard = None

    # perturb density (gravity/turbulence) or energy (Sedov); log-perturbation
    # field*exp(delta) is positive by construction.
    chan = 4 if field == "energy" else 0

    def _perturb(U0_, delta):
        return U0_.at[chan].set(U0_[chan] * jnp.exp(delta))

    U_obs = simulate(eta_true, _perturb(U0, delta_true))

    # loss takes the big arrays (U0, U_obs) as ARGUMENTS, not closure captures:
    # a captured concrete SHARDED array gets embedded as an MLIR constant when the
    # loss is jitted, and the constant handler crashes on the non-addressable
    # multi-process fetch. As arguments they stay first-class sharded values.
    def loss_arg(delta, U0_, U_obs_):
        return jnp.mean((simulate(eta_true, _perturb(U0_, delta)) - U_obs_) ** 2)

    def loss(delta):                       # closure form for the single-proc FD gate
        return loss_arg(delta, U0, U_obs)

    delta0 = jnp.zeros(shape, U0.dtype)
    if _dshard is not None:
        delta0 = jax.device_put(delta0, _dshard)      # match U0's spatial sharding
    # AD-vs-FD gradient gate: single-process only. The per-voxel extraction
    # (float(g[idx])) reads a shard that is non-addressable on other processes,
    # and the eager reductions are collectives — both hang / crash under
    # multi-process sharding. It is a small-Nx correctness diagnostic (already
    # verified), not needed for the production 256^3 sharded figure.
    rel = float("nan")
    if jax.process_count() == 1:
        loss_j = jax.jit(loss)
        g = jax.jit(jax.grad(loss))(delta0)
        idx = tuple(int(s // 2) for s in shape)
        fd = central_fd(loss_j, delta0, idx)
        rel = abs(float(g[idx]) - fd) / (abs(fd) + 1e-30)
        if is_main:
            print(f"  ic_inversion: ∂loss/∂(IC field) shape={g.shape}, "
                  f"finite={bool(jnp.all(jnp.isfinite(g)))}", flush=True)
            print(f"    FD check voxel {idx}: AD={float(g[idx]):.4e}  "
                  f"FD={fd:.4e}  rel={rel:.2e}", flush=True)
    elif is_main:
        print("  ic_inversion: FD gate skipped (multi-process); gradient "
              "verified single-process", flush=True)

    delta = delta0
    opt = adam_init(delta)
    n_iter = int(n_iter)
    lr = float(lr or 3.0e-2)
    # jit the norm so its reduction all-reduces to a REPLICATED scalar; eager
    # jnp.linalg.norm leaves the result sharded (non-addressable -> float() crash
    # under multi-process).
    _jnorm = jax.jit(lambda x: jnp.linalg.norm(x))
    norm_true = float(_jnorm(delta_true))
    # jit value_and_grad through the remat'd rollout: without jit the reverse
    # pass materializes an un-optimized tape every iteration (OOMs a 150-step
    # rollout mid-optimization). One fused, donation-optimized Adam step instead.
    # U0/U_obs passed as args (see loss_arg) so nothing sharded is a constant.
    vgrad = jax.jit(jax.value_and_grad(loss_arg, argnums=0))
    # shared pytree Adam (b1=0.9, b2=0.999, eps=1e-8, bias-corrected), jitted
    # so the update stays one fused device step like the inline version it
    # replaced.
    _adam_jit = jax.jit(lambda gg, d, s: adam_step(gg, d, s, lr))

    _t_loop0 = time.time()
    _t_iter1 = None
    for it in range(n_iter):
        L, gg = vgrad(delta, U0, U_obs)
        delta, opt = _adam_jit(gg, delta, opt)
        if it == 0:
            jax.block_until_ready(delta)          # force compile+first step
            _t_iter1 = time.time()
            if is_main:
                print(f"    [timing] compile+it0 = {_t_iter1 - _t_loop0:.1f}s",
                      flush=True)
        if it % max(1, n_iter // 10) == 0 or it == n_iter - 1:
            # norm is a collective (all ranks must call it) -> compute on EVERY
            # process, print only on rank 0.
            res = float(_jnorm(delta - delta_true)) / norm_true
            L_ = float(L)
            if is_main:
                per = ((time.time() - _t_iter1) / max(it, 1)) if _t_iter1 and it else 0.0
                print(f"    it {it:4d}  L={L_:.3e}  "
                      f"||δ−δ_true||/||δ_true||={res:.3e}  ({per:.2f}s/it)", flush=True)
    res = float(_jnorm(delta - delta_true)) / norm_true
    if is_main:
        stats = jax.local_devices()[0].memory_stats() or {}
        if "peak_bytes_in_use" in stats:
            print(f"    [memory] peak on rank 0's device: "
                  f"{stats['peak_bytes_in_use'] / 2**30:.1f} GiB", flush=True)
    # Gather the (possibly sharded) recovered field to a full replicated array
    # before the host-side r(k) FFT + figure. Resharding to a replicated spec via
    # a jit identity is a collective (all ranks participate, all-gather if it was
    # sharded); the replicated result is fully addressable so np.asarray gives the
    # whole cube on rank 0.
    if jax.process_count() > 1:
        from jax.sharding import NamedSharding, PartitionSpec as P
        repl = NamedSharding(U0.sharding.mesh, P())
        delta_host = np.asarray(jax.jit(lambda x: x, out_shardings=repl)(delta))
    else:
        delta_host = np.asarray(delta)
    rk = _recovery_rk(delta_host, delta_true_np)
    if is_main:
        print(f"  recovered IC perturbation: residual {res:.3e}", flush=True)
        print("  r(k):", "  ".join(f"k{k}={rk[k]:+.2f}" for k in range(1, min(len(rk), 10))),
              flush=True)
        save_inversion_figure(delta_true_np, delta_host, rk, res, cfg)
    return {"ic_residual": res, "grad_fd_rel": rel,
            "rk": rk, "delta": delta_host}
