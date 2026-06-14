"""Autodiffable distributed runner — the differentiable counterpart to
run_simulation_distributed, selected by cfg.autodiff_mode in main.run.

It reuses the SAME setup (mesh / IC / step_fn from create_solver_distributed)
and the SAME source `apply` machinery (cc85, external gravity, self-gravity,
cooling, OU forcing), but runs a pure remat'd fixed-length `lax.scan` with no
IO / checkpoint / NaN-guard, so the whole evolution is differentiable. The
per-step source order matches run_simulation_distributed._attempt_step.

simulate(theta, U0) is differentiable in BOTH:
  * theta — the trainable in-loop source params (cc85 loadings here; CNN
    corrector weights in the solver-in-the-loop task), and
  * U0    — the initial conditions (for IC-inversion tasks).

Timestepping is fixed (cfg.autodiff_fixed_dt>0) or adaptive CFL (compute_dt_fn
wrapped in jax.lax.stop_gradient, coasting at cfg.t_end). The fixed-length scan
means BOTH forward and reverse AD apply, single- and multi-GPU (AD transposes
through shard_map collectives like the HPC step_fn it calls).

The task is chosen by cfg.autodiff_task:
  * "xray_sensitivity" -> d ln L_X / d (cc85 loadings)         [forward/reverse]
  * "ic_inversion"     -> recover an IC density perturbation   [reverse]
"""
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from sources.cc85 import setup_cc85_source
from sources.external_gravity import setup_external_gravity
from sources.self_gravity import setup_self_gravity
from sources.cooling import setup_cooling
from sources.ou_forcing import setup_ou_forcing
from hydro.cooling import apply_cooling, PROTON_MASS_CGS, BOLTZMANN_CGS

MYR_S = 1.0e6 * 3.15576e7


def xray_luminosity(U, cfg, mu=0.6):
    """Soft-band X-ray luminosity L_X = sum n^2 sqrt(T) band(T) dV, physical
    T [K] from the cfg unit system. Absolute scale is a bremsstrahlung proxy;
    the logarithmic sensitivity (elasticity) is scale-free."""
    gamma = cfg.gamma
    u = cfg.units
    rho = U[0]
    v2 = (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / (rho ** 2)
    P = (gamma - 1.0) * (U[4] - 0.5 * rho * v2)
    T = (mu * PROTON_MASS_CGS * jnp.clip(P, 1e-30) * u.UnitPressure_in_cgs
         / (BOLTZMANN_CGS * rho * u.UnitDensity_in_cgs))
    band = jax.nn.sigmoid((T - 1.0e6) / 3.0e5)
    emiss = rho ** 2 * jnp.sqrt(jnp.clip(T, 1.0)) * band
    dV = (cfg.Lx / cfg.Nx) * (cfg.Ly / cfg.Ny) * (cfg.Lz / cfg.Nz)
    return jnp.sum(emiss) * dV


def _low_k_field(key, shape, k_max=4, amp=0.05):
    """Random low-wavenumber real field, std normalized to `amp`."""
    f = jax.random.normal(key, shape)
    F = jnp.fft.rfftn(f)
    kx = jnp.fft.fftfreq(shape[0]) * shape[0]
    ky = jnp.fft.fftfreq(shape[1]) * shape[1]
    kz = jnp.arange(F.shape[2])
    KX, KY, KZ = jnp.meshgrid(kx, ky, kz, indexing="ij")
    kk = jnp.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    F = jnp.where(kk <= k_max, F, 0.0)
    g = jnp.fft.irfftn(F, s=shape)
    return g / (jnp.std(g) + 1e-12) * amp


# ---------------------------------------------------------------------------
# Differentiable evolution: simulate(theta, U0)
# ---------------------------------------------------------------------------

def build_simulate(U0_ref, cfg, step_fn, compute_dt_fn, sharding,
                   is_main_process=True):
    """Return (simulate, step_once, carry0, dt_info, eta_true)."""
    cc85 = setup_cc85_source(cfg, sharding, is_main_process)
    extg = setup_external_gravity(cfg, sharding, is_main_process)
    selfg = setup_self_gravity(cfg, sharding, is_main_process)
    cool = setup_cooling(cfg, is_main_process)
    forcing = setup_ou_forcing(cfg, sharding, is_main_process)

    fixed_dt = float(cfg.autodiff_fixed_dt)
    adaptive = fixed_dt <= 0.0
    dt_mode = getattr(cfg, "autodiff_dt_mode", "stopgrad")   # "stopgrad"|"diff"
    n_steps = int(cfg.autodiff_n_steps)
    t_end = float(cfg.t_end) if adaptive else 0.0
    eta_true = (jnp.array([cfg.cc85_eta_M, cfg.cc85_eta_E], dtype=U0_ref.dtype)
                if cc85 is not None else jnp.zeros(2, dtype=U0_ref.dtype))

    iso_cs2 = float(getattr(cfg, "isothermal_cs2", 0.0))
    gm1 = float(cfg.gamma) - 1.0

    def apply_sources(U, theta, dt, a, key):
        # EXACT order of run_simulation_distributed._attempt_step:
        # step -> forcing -> isothermal -> ext_grav -> self_grav -> cc85 -> cooling
        U = step_fn(U, dt)
        if forcing is not None:
            U, a, key = forcing["apply"](U, a, key, dt)
        if iso_cs2 > 0.0:
            # soft isothermal thermostat: reset E so p = cs2*rho (float64-safe)
            ke = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
            U = U.at[4].set(iso_cs2 * U[0] / gm1 + ke)
        if extg is not None:
            U = extg["apply"](U, dt)
        if selfg is not None:
            U = selfg["apply"](U, dt)
        if cc85 is not None:
            U = cc85["apply_eta"](U, dt, theta)
        if cool is not None:
            U = apply_cooling(U, dt, cfg, cool)
        return U, a, key

    def step_once(carry, theta):
        U, t, a, key = carry
        if adaptive:
            dt = compute_dt_fn(U)
            if dt_mode == "stopgrad":
                dt = jax.lax.stop_gradient(dt)   # drop dt-path (AD != FD)
            # else "diff": differentiate through dt (AD == FD, dt kinks)
            dt = jnp.minimum(dt, jnp.maximum(t_end - t, 0.0))
            active = t < t_end
        else:
            dt = fixed_dt
            active = jnp.asarray(True)
        U_new, a_new, key_new = apply_sources(U, theta, dt, a, key)
        U_new = jnp.where(active, U_new, U)
        return (U_new, t + jnp.where(active, dt, 0.0), a_new, key_new)

    def carry0(U0):
        a0 = forcing["a"] if forcing is not None else jnp.zeros((), U0.dtype)
        key0 = forcing["key"] if forcing is not None else jnp.zeros(2, jnp.uint32)
        return (U0, jnp.asarray(0.0, U0.dtype), a0, key0)

    def simulate(theta, U0):
        @jax.checkpoint
        def body(carry, _):
            return step_once(carry, theta), None
        (U_final, _, _, _), _ = jax.lax.scan(body, carry0(U0), None,
                                             length=n_steps)
        return U_final

    dt_info = f"adaptive({dt_mode})" if adaptive else f"fixed {fixed_dt:.3e}"
    return simulate, step_once, carry0, dt_info, eta_true


# ---------------------------------------------------------------------------
# Entry: dispatched by main.run when cfg.autodiff_mode
# ---------------------------------------------------------------------------

def run_simulation_autodiff(U0, cfg, grid, step_fn, compute_dt_fn, sharding,
                            is_main_process=True):
    task = getattr(cfg, "autodiff_task", "xray_sensitivity")

    # Solver-in-the-loop: a coarse/fine problem that builds its own KH
    # reference, so it doesn't use the single-resolution build_simulate.
    if task == "corrector_training":
        return _corrector_training(U0, cfg, step_fn, compute_dt_fn,
                                   is_main_process)

    simulate, step_once, carry0, dt_info, eta_true = build_simulate(
        U0, cfg, step_fn, compute_dt_fn, sharding, is_main_process)
    n_steps = int(cfg.autodiff_n_steps)
    if is_main_process:
        print(f"  autodiff: task={task}, n_steps(max)={n_steps}, dt={dt_info}, "
              f"eta_true={eta_true}", flush=True)

    # Forward-only diagnostic (any task): stability + L_X(t) maturity gate.
    if os.environ.get("LUNA_M82_FWDONLY", "0") == "1":
        def body(carry, _):
            carry = step_once(carry, eta_true)
            return carry, (xray_luminosity(carry[0], cfg),
                           jnp.all(jnp.isfinite(carry[0])), carry[1])
        t0 = time.time()
        (Uf, tf, _, _), (lxs, oks, ts) = jax.lax.scan(
            body, carry0(U0), None, length=n_steps)
        lxs = np.asarray(lxs); oks = np.asarray(oks); ts = np.asarray(ts)
        all_finite = bool(oks.all())
        t_myr = float(tf) * cfg.units.UnitTime_in_s / MYR_S
        if is_main_process:
            print(f"  forward-only: reached t={t_myr:.2f} Myr in {n_steps} "
                  f"steps, all-finite={all_finite}, {time.time()-t0:.0f}s",
                  flush=True)
            every = max(1, n_steps // 20)
            for k in range(0, n_steps, every):
                print(f"    step {k:5d}  t={float(ts[k])*cfg.units.UnitTime_in_s/MYR_S:6.2f} Myr"
                      f"  L_X={float(lxs[k]):.4e}  finite={bool(oks[k])}", flush=True)
            print(f"    final t={t_myr:.2f} Myr  L_X={float(lxs[-1]):.4e}", flush=True)
        return {"L_X_t": lxs, "t": ts, "all_finite": all_finite, "U_final": Uf}

    if task == "ic_inversion":
        return _ic_inversion(simulate, U0, eta_true, cfg, is_main_process)
    if task == "eta_recovery":
        return _eta_recovery(simulate, U0, eta_true, cfg, is_main_process)
    return _xray_sensitivity(simulate, U0, eta_true, cfg, is_main_process)


def _xray_sensitivity(simulate, U0, eta_true, cfg, is_main):
    """d ln L_X / d ln eta through the production solver (reverse AD + FD)."""
    t0 = time.time()
    U_final = simulate(eta_true, U0)
    LX0 = float(xray_luminosity(U_final, cfg))
    if is_main:
        print(f"  forward: L_X={LX0:.4e}, "
              f"finite={bool(jnp.all(jnp.isfinite(U_final)))}, "
              f"{time.time()-t0:.0f}s", flush=True)

    def lnLX(eta):
        return jnp.log(xray_luminosity(simulate(eta, U0), cfg))

    grad_mode = getattr(cfg, "autodiff_grad_mode", "reverse")
    t0 = time.time()
    if grad_mode == "forward":
        # jvp: NO trajectory tape -> memory-flat in n_steps; the route to high
        # resolution / many steps for a few-parameter sensitivity.
        _, dM = jax.jvp(lnLX, (eta_true,), (jnp.array([1.0, 0.0], eta_true.dtype),))
        _, dE = jax.jvp(lnLX, (eta_true,), (jnp.array([0.0, 1.0], eta_true.dtype),))
        eM = float(eta_true[0] * dM); eE = float(eta_true[1] * dE)
    else:
        g = jax.grad(lnLX)(eta_true)
        eM = float(eta_true[0] * g[0]); eE = float(eta_true[1] * g[1])
    h = 1.0e-4
    def fd(i):
        return (float(lnLX(eta_true.at[i].add(h)))
                - float(lnLX(eta_true.at[i].add(-h)))) / (2.0 * h)
    fdM = eta_true[0] * fd(0); fdE = eta_true[1] * fd(1)
    relM = abs(eM - float(fdM)) / (abs(float(fdM)) + 1e-30)
    relE = abs(eE - float(fdE)) / (abs(float(fdE)) + 1e-30)
    if is_main:
        print(f"\n  === d ln L_X / d ln eta ({grad_mode} AD vs central FD) ===",
              flush=True)
        print(f"  ∂lnL_X/∂lnη_M = {eM:+.6f}  [FD {float(fdM):+.6f}, rel {relM:.2e}]",
              flush=True)
        print(f"  ∂lnL_X/∂lnη_E = {eE:+.6f}  [FD {float(fdE):+.6f}, rel {relE:.2e}]",
              flush=True)
        print(f"  (grad+FD: {time.time()-t0:.0f}s)", flush=True)
    return {"L_X": LX0, "dlnLX_dlneta": (eM, eE),
            "fd": (float(fdM), float(fdE)), "fd_rel": (relM, relE),
            "U_final": U_final}


def _eta_recovery(simulate, U0, eta_true, cfg, is_main):
    """Closed-loop recovery of the CC85 loadings (η_M, η_E) by reverse-mode AD
    through the FULL coupled forward model — cc85 wind + radiative cooling +
    self-gravity + external potential, all active in apply_sources.

    This is a CIRCULAR / self-consistency validation: we inject a KNOWN η,
    evolve, and recover it. It certifies that the gradients are correct through
    the entire physics stack simultaneously — it is NOT a measurement of the real
    M82. "We measure what we put in." (The non-circular feedback-history
    inference is the follow-up, not this code paper.)
    """
    t0 = time.time()
    U_obs = simulate(eta_true, U0)               # the "observed" evolved state
    log_true = jnp.log(eta_true)

    # Match the full evolved state; recover in log-η (positivity by construction).
    def loss(log_eta):
        return jnp.mean((simulate(jnp.exp(log_eta), U0) - U_obs) ** 2)

    # Start from a deliberately wrong guess (η_M too high, η_E too low).
    log0 = log_true + jnp.log(jnp.array([1.6, 0.6], eta_true.dtype))

    # FD-validate the reverse-mode gradient at the start point.
    g0 = jax.grad(loss)(log0)
    h = 1.0e-4
    fd0 = jnp.array([(loss(log0.at[i].add(h)) - loss(log0.at[i].add(-h)))
                     / (2.0 * h) for i in range(2)])
    rel = [abs(float(g0[i]) - float(fd0[i])) / (abs(float(fd0[i])) + 1e-30)
           for i in range(2)]
    if is_main:
        print(f"  eta_recovery: forward+FD ({time.time()-t0:.0f}s); "
              f"∂loss/∂lnη AD={np.asarray(g0)}  FD={np.asarray(fd0)}  "
              f"rel={rel}", flush=True)

    # Adam on log-η.
    log_eta = log0
    m = jnp.zeros_like(log_eta); v = jnp.zeros_like(log_eta)
    n_iter = int(getattr(cfg, "autodiff_n_iter", 60))
    hist_it, hist_L, hist_etaM, hist_etaE = [], [], [], []
    for it in range(n_iter):
        L, gg = jax.value_and_grad(loss)(log_eta)
        m = 0.9 * m + 0.1 * gg
        v = 0.999 * v + 0.001 * gg * gg
        log_eta = log_eta - 3.0e-2 * m / (1 - 0.9 ** (it + 1)) / (
            jnp.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
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
        _save_eta_recovery_figure(hist_it, hist_L, hist_etaM, hist_etaE,
                                  eta_t, rel, res, cfg)
    return {"eta_recovered": eta_rec.tolist(), "eta_true": eta_t.tolist(),
            "eta_rel_err": res, "grad_fd_rel": rel}


def _save_eta_recovery_figure(its, Ls, etaM, etaE, eta_true, fd_rel, res, cfg):
    """Closed-loop CC85 loading recovery through the full coupled physics. Two
    panels: (a) the η_M, η_E trajectories converging onto the injected truths;
    (b) the matching-loss curve. Caption (README) states this is a circular
    validation of gradient correctness through cooling + self-gravity + external
    potential — we recover what we inject."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()

    its = np.asarray(its)
    fig, ax = plt.subplots(1, 2, figsize=(FIGSIZE[0] * 2, FIGSIZE[1]))

    # (a) loading trajectories vs the injected truths.
    ax[0].axhline(eta_true[0], color="C0", ls="--", lw=1.2,
                  label=fr"$\eta_M$ injected = {eta_true[0]:.2f}")
    ax[0].axhline(eta_true[1], color="C1", ls="--", lw=1.2,
                  label=fr"$\eta_E$ injected = {eta_true[1]:.2f}")
    ax[0].plot(its, etaM, "-", color="C0", lw=2.0, label=r"$\eta_M$ recovered")
    ax[0].plot(its, etaE, "-", color="C1", lw=2.0, label=r"$\eta_E$ recovered")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel(r"loading factor $\eta$")
    ax[0].legend(loc="best")
    ax[0].annotate(fr"max rel. err = {res:.1e}" + "\n"
                   fr"AD vs FD: {max(fd_rel):.0e}",
                   xy=(0.97, 0.5), xycoords="axes fraction", ha="right",
                   va="center", fontsize=10)
    decorate(ax[0])

    # (b) matching-loss convergence.
    ax[1].semilogy(its, np.asarray(Ls), "-", color="C3", lw=2.0,
                   label=r"$\|U(\eta)-U_{\rm obs}\|^2$")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("state-matching loss")
    ax[1].legend(loc="best")
    decorate(ax[1])

    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = f"figures/autodiff_eta_recovery_Nx{cfg.Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


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


def _ic_inversion(simulate, U0, eta_true, cfg, is_main):
    """Recover an IC density perturbation by reverse-mode AD through the
    production solver (high-dimensional inverse — the reverse-mode home turf).
    Differentiates w.r.t. the FULL log-density field (Nx*Ny*Nz inputs).
    Emits the recovery + r(k) figure (the reverse-mode proof)."""
    shape = U0.shape[1:]
    delta_true = _low_k_field(jax.random.PRNGKey(1), shape, k_max=4, amp=0.05)

    # perturb density (gravity/turbulence) or energy (Sedov); log-perturbation
    # field*exp(delta) is positive by construction.
    chan = 4 if getattr(cfg, "autodiff_ic_field", "density") == "energy" else 0

    def perturb(delta):
        return U0.at[chan].set(U0[chan] * jnp.exp(delta))

    U_obs = simulate(eta_true, perturb(delta_true))

    def loss(delta):
        return jnp.mean((simulate(eta_true, perturb(delta)) - U_obs) ** 2)

    delta0 = jnp.zeros(shape, U0.dtype)
    g = jax.grad(loss)(delta0)
    idx = tuple(int(s // 2) for s in shape)
    h = 1.0e-4
    fd = (float(loss(delta0.at[idx].add(h)))
          - float(loss(delta0.at[idx].add(-h)))) / (2.0 * h)
    rel = abs(float(g[idx]) - fd) / (abs(fd) + 1e-30)
    if is_main:
        print(f"  ic_inversion: ∂loss/∂(IC field) shape={g.shape}, "
              f"finite={bool(jnp.all(jnp.isfinite(g)))}", flush=True)
        print(f"    FD check voxel {idx}: AD={float(g[idx]):.4e}  "
              f"FD={fd:.4e}  rel={rel:.2e}", flush=True)

    delta = delta0
    m = jnp.zeros_like(delta); v = jnp.zeros_like(delta)
    n_iter = int(getattr(cfg, "autodiff_n_iter", 100))
    norm_true = float(jnp.linalg.norm(delta_true))
    for it in range(n_iter):
        L, gg = jax.value_and_grad(loss)(delta)
        m = 0.9 * m + 0.1 * gg
        v = 0.999 * v + 0.001 * gg * gg
        delta = delta - 1.0e-2 * m / (1 - 0.9 ** (it + 1)) / (
            jnp.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
        if is_main and (it % max(1, n_iter // 10) == 0 or it == n_iter - 1):
            res = float(jnp.linalg.norm(delta - delta_true)) / norm_true
            print(f"    it {it:4d}  L={float(L):.3e}  "
                  f"||δ−δ_true||/||δ_true||={res:.3e}", flush=True)
    res = float(jnp.linalg.norm(delta - delta_true)) / norm_true
    rk = _recovery_rk(delta, delta_true)
    if is_main:
        print(f"  recovered IC perturbation: residual {res:.3e}", flush=True)
        print("  r(k):", "  ".join(f"k{k}={rk[k]:+.2f}" for k in range(1, min(len(rk), 10))),
              flush=True)
        _save_inversion_figure(delta_true, delta, rk, res, cfg)
    return {"ic_residual": res, "grad_fd_rel": rel,
            "rk": rk, "delta": np.asarray(delta)}


def _save_inversion_figure(delta_true, delta_est, rk, res, cfg):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()
    dt_np = np.asarray(delta_true); de_np = np.asarray(delta_est)
    zc = dt_np.shape[2] // 2
    fig, ax = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 2.0, FIGSIZE[1]))
    vmax = float(np.max(np.abs(dt_np)))
    a0 = ax[0].imshow(dt_np[:, :, zc], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax[0].set_title(r"$\delta_{\rm true}$ (z-mid)"); plt.colorbar(a0, ax=ax[0], shrink=0.8)
    a1 = ax[1].imshow(de_np[:, :, zc], vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax[1].set_title(fr"$\delta_{{\rm recovered}}$ (res {res:.2f})"); plt.colorbar(a1, ax=ax[1], shrink=0.8)
    for a in (ax[0], ax[1]):          # image panels: no grid over pixels
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    ks = np.arange(1, len(rk))
    ax[2].plot(ks, rk[1:], "o-", ms=5, label=r"$r(k)$")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_ylim(-0.2, 1.05)
    ax[2].set_xlabel("k"); ax[2].set_ylabel(r"$r(k)$")
    ax[2].set_title("scale-resolved recovery")
    decorate(ax[2])
    os.makedirs("figures", exist_ok=True)
    # tag by test problem AND dominant physics so the turbulence / self-gravity /
    # radiative-cooling / sedov inversions don't overwrite each other (the
    # gravity and cooling cases both reuse the uniform Jeans-box IC).
    base = str(getattr(cfg, "test_problem", "ic")).replace("-", "").replace("_", "")
    if getattr(cfg, "cooling_enabled", False):
        tag, label = base + "_cool", "radiative-cooling box"
    elif getattr(cfg, "SELF_GRAVITY", False):
        tag, label = base + "_selfgrav", f"{getattr(cfg,'test_problem','')} + self-gravity"
    else:
        tag, label = base, str(getattr(cfg, "test_problem", ""))
    fig.suptitle(f"{label}: reverse-mode IC inversion", y=1.02, fontsize=12)
    out = f"figures/autodiff_ic_inversion_{tag}_Nx{cfg.Nx}.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _corrector_training(U0, cfg, step_fn, compute_dt_fn, is_main):
    """Solver-in-the-loop CNN flux corrector (#41) through the PRODUCTION
    solver. Coarse rollout uses the production step_fn + IC (U0); the fine
    reference is the production setup at 4x resolution, downsampled. The CNN
    WEIGHTS are trained by reverse-mode AD through the unrolled coarse rollout
    (long-horizon curriculum, Um et al. 2020)."""
    from dataclasses import replace
    from utils.grid_params import GridParams
    from setup.distributed_setup import (create_distributed_mesh,
                                         initialize_simulation_distributed,
                                         create_solver_distributed)
    from sources.ml_corrector import (cnn_init, cnn_correction, pool_factor,
                                       adam_init, adam_step)

    Nc = int(cfg.Nx)
    f = 4
    Nf = f * Nc
    n_steps = int(cfg.autodiff_n_steps)
    n_iter = int(cfg.autodiff_n_iter)
    dt_c = (float(cfg.autodiff_fixed_dt) if cfg.autodiff_fixed_dt > 0
            else float(compute_dt_fn(U0)))

    # --- fine reference via the PRODUCTION setup at 4x res ---
    fine_cfg = replace(cfg, Nx=Nf, Ny=Nf, autodiff_mode=False)
    fmesh, _ = create_distributed_mesh(fine_cfg)
    Uf, _, _ = initialize_simulation_distributed(fine_cfg, fmesh)
    fgrid = GridParams.from_config(fine_cfg)
    _, fstep, _, _ = create_solver_distributed(fine_cfg, fgrid, fmesh)
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
        print(f"  corrector: coarse {Nc}^2, fine {Nf}^2, {n_steps} steps, "
              f"dt={dt_c:.4e}; fine reference built", flush=True)

    def corrected_step(w, U):
        return step_fn(U, dt_c) + cnn_correction(w, U, norm)

    def window_loss(w, start, K):
        U = jax.lax.dynamic_index_in_dim(ref, start, axis=0, keepdims=False)
        def body(U, j):
            U = corrected_step(w, U)
            tgt = jax.lax.dynamic_index_in_dim(ref, start + 1 + j, axis=0,
                                               keepdims=False)
            return U, jnp.mean(((U - tgt) / norm[:, None, None]) ** 2)
        _, errs = jax.lax.scan(body, U, jnp.arange(K))
        return jnp.mean(errs)

    def batch_loss(w, starts, K):
        return jnp.mean(jax.vmap(lambda s: window_loss(w, s, K))(starts))

    loss_grad = jax.jit(jax.value_and_grad(batch_loss), static_argnums=2)

    key = jax.random.PRNGKey(0)
    key, ki = jax.random.split(key)
    w = cnn_init(ki)
    opt = adam_init(w)
    BATCH = 2
    t0 = time.time()
    for it in range(n_iter):
        K = (32, 64, 128, max(16, min(256, n_steps - 1)))[min(3, it * 4 // n_iter)]
        key, ks = jax.random.split(key)
        starts = jax.random.randint(ks, (BATCH,), 0, n_steps - K - 1)
        lr = 5.0e-4 * (0.1 ** (it / n_iter))
        L, g = loss_grad(w, starts, K)
        if not bool(jnp.isfinite(L)) or float(L) > 1.0:
            continue
        gn = jnp.sqrt(sum(jnp.sum(W ** 2) + jnp.sum(b ** 2) for W, b in g))
        g = jax.tree.map(lambda x: x * jnp.minimum(1.0, 1.0 / (gn + 1e-12)), g)
        w, opt = adam_step(g, w, opt, lr)
        if is_main and (it % max(1, n_iter // 10) == 0 or it == n_iter - 1):
            print(f"    it {it:4d}  K={K}  loss={float(L):.4e}", flush=True)
    if is_main:
        print(f"  training: {time.time()-t0:.0f}s", flush=True)

    # eval: full coarse rollout with/without corrector vs reference
    step_c_j = jax.jit(lambda U: step_fn(U, dt_c))
    corr_j = jax.jit(lambda U: corrected_step(w, U))
    def rollout(stepper):
        U = ref[0]; errs = []
        for i in range(1, n_steps + 1):
            U = stepper(U)
            if not bool(jnp.all(jnp.isfinite(U))):
                break
            if i % 25 == 0:
                errs.append(float(jnp.linalg.norm(U[0] - ref[i][0])
                                  / jnp.linalg.norm(ref[i][0])))
        return errs, U
    e_plain, U_plain = rollout(step_c_j)
    e_corr, U_corr = rollout(corr_j)
    bp = e_plain[-1] if e_plain else float("nan")
    bc = e_corr[-1] if e_corr else float("nan")
    if is_main:
        print(f"  rollout rel-err(rho): coarse-only={bp:.3e}  +CNN={bc:.3e}  "
              f"(corr survived {len(e_corr)*25}/{n_steps} steps)", flush=True)
        _save_corrector_figure(e_plain, e_corr,
                               np.asarray(ref[-1][0]), np.asarray(U_plain[0]),
                               np.asarray(U_corr[0]), cfg)
    return {"coarse_err": e_plain, "corr_err": e_corr, "weights": w}


def _save_corrector_figure(e_plain, e_corr, rho_ref, rho_coarse, rho_corr, cfg):
    """Solver-in-the-loop KH corrector (reproduces the diffhydro/Horowitz KH
    example with the LUNA KH IC): density fields at the final time for the
    downsampled fine reference, the plain coarse solver, and the coarse solver +
    trained CNN corrector — plus the rollout error vs the fine reference. The
    corrected coarse run should recover the fine KH roll-up the coarse run smears
    out."""
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, decorate, FIGSIZE
    apply_style()

    fig, ax = plt.subplots(1, 4, figsize=(FIGSIZE[0] * 2.4, FIGSIZE[1] * 0.95))
    vmin = float(min(rho_ref.min(), rho_coarse.min(), rho_corr.min()))
    vmax = float(max(rho_ref.max(), rho_coarse.max(), rho_corr.max()))
    panels = [(rho_ref, "fine reference\n(downsampled)"),
              (rho_coarse, f"coarse solver\n(err {e_plain[-1]:.2e})" if e_plain else "coarse"),
              (rho_corr, f"coarse + CNN\n(err {e_corr[-1]:.2e})" if e_corr else "coarse+CNN")]
    for a, (rho, ttl) in zip(ax[:3], panels):
        im = a.imshow(rho.T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        a.set_title(ttl, fontsize=11); a.grid(False)
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[2], shrink=0.8, label=r"$\rho$")

    # (d) rollout error vs the fine reference
    if e_plain:
        ax[3].semilogy((np.arange(len(e_plain)) + 1) * 25, e_plain, "o-",
                       color="C3", label=fr"coarse ({e_plain[-1]:.2e})")
    if e_corr:
        ax[3].semilogy((np.arange(len(e_corr)) + 1) * 25, e_corr, "s-",
                       color="C0", label=fr"+ CNN ({e_corr[-1]:.2e})")
    ax[3].set_xlabel("rollout step")
    ax[3].set_ylabel("density rel. error")
    ax[3].set_title("rollout error vs fine ref")
    ax[3].legend(loc="upper left")
    decorate(ax[3])

    fig.suptitle(fr"Solver-in-the-loop KH corrector ($N_c$={cfg.Nx}, fine $4\times$)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = f"figures/autodiff_corrector_Nc{cfg.Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)
