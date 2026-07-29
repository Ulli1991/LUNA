"""Differentiable runner (cfg.autodiff_mode): same setup/step_fn/source stack as
run_simulation_distributed, but a pure remat'd fixed-length lax.scan (no IO /
NaN-guard), so simulate(theta, U0) differentiates in BOTH the in-loop source
params theta and the IC U0, forward and reverse, single- and multi-GPU (AD
transposes through the shard_map collectives). dt fixed or stop_gradient CFL.
Task selected by cfg.autodiff_task (see run_simulation_autodiff)."""
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
from sources.ml_corrector import adam_init, adam_step
from hydro.cooling import apply_cooling, PROTON_MASS_CGS, BOLTZMANN_CGS
from hydro.observables import xray_luminosity
from ic_generation.spectral import spectral_ic_field
from sources.eos_projection import barotropic_project, isothermal_project
from utils.autodiff_figs import (_save_emergence_montage,
                                 _save_eta_recovery_figure,
                                 _save_inversion_figure,
                                 _save_corrector_figure)

MYR_S = 1.0e6 * 3.15576e7


def _central_fd(f, x, i, h=1.0e-4):
    """Central finite difference of scalar f(x) along element/index i."""
    return (float(f(x.at[i].add(h))) - float(f(x.at[i].add(-h)))) / (2.0 * h)


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

    # Sinks and particles carry loop state (sink registry, particle positions)
    # that the differentiable scan carry does not thread. They used to be
    # SILENTLY skipped here while the standard path applied them -- refuse
    # loudly instead so a config that enables them cannot diverge quietly.
    if getattr(cfg, "sinks_enabled", False) or getattr(cfg, "particles_enabled", False):
        raise NotImplementedError(
            "autodiff_mode does not support sinks_enabled/particles_enabled "
            "(their per-step state is not threaded through the scan carry)")

    def apply_sources(U, theta, dt, a, key):
        # Same order as run_simulation_distributed._attempt_step:
        # step -> forcing -> isothermal -> barotropic -> ext_grav -> self_grav
        # -> cc85 -> cooling. One deliberate difference: cc85 uses the
        # theta-parameterized apply_eta (eta_M/eta_E are the differentiated
        # parameters); sinks/particles rejected above.
        U = step_fn(U, dt)
        if forcing is not None:
            U, a, key = forcing["apply"](U, a, key, dt)
        if iso_cs2 > 0.0:
            U = isothermal_project(U, iso_cs2, cfg.gamma)
        if getattr(cfg, "barotropic_eos", False):
            U = barotropic_project(U, cfg)
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

def _render_text_target(Nx, text, lo, hi):
    """Render `text` to an Nx x Nx target field (hi on the glyphs, lo elsewhere)
    -- the late-time morphology the forward run is optimized to form."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jax.image import resize as jresize
    fig = plt.figure(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black"); ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color="white",
            fontsize=260.0 / max(1, len(text)), fontweight="bold",
            transform=ax.transAxes)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.float64) / 255.0
    plt.close(fig)
    r = np.asarray(jresize(jnp.asarray(buf), (Nx, Nx), method="linear"))
    return lo + (hi - lo) * r[::-1].T


def _render_image_target(Nx, path, lo, hi):
    """Load a custom [0,1] target field (bright = where the morphology should
    pile up, plain background = 0), resize to Nx, and map to [lo, hi]. A .npy is
    taken as already in the field convention (built to match _render_text_target);
    an image file is read as grayscale. Used for the 'emerge from explosions'
    demo with an arbitrary picture instead of a rendered word."""
    from jax.image import resize as jresize
    if str(path).endswith(".npy"):
        img = np.load(path).astype(np.float64)
    else:
        from PIL import Image
        img = np.asarray(Image.open(path).convert("L"), np.float64) / 255.0
    img = np.asarray(jresize(jnp.asarray(img), (Nx, Nx), method="linear"))
    img = np.clip(img, 0.0, 1.0)
    return lo + (hi - lo) * img


def _luna_emergence(step_fn, cfg, is_main):
    """Reverse-mode IC optimization so the LATE-TIME field forms a target image
    (diffhydro Fig. 12); the free parameter is the FULL per-cell IC field.
    driver='blast': free internal-energy field e(x,0) of a uniform medium at
    rest (scattered Sedov sites collide into the target); driver='turbulence':
    free band-limited velocity field of a uniform isothermal medium."""
    import time
    driver = getattr(cfg, "autodiff_emergence_driver", "blast")
    Nx, gamma, L, rho0 = cfg.Nx, float(cfg.gamma), cfg.Lx, 1.0
    dx = L / Nx
    n_steps = int(getattr(cfg, "autodiff_n_steps", 300))
    n_iter = int(getattr(cfg, "autodiff_n_iter", 200))
    lr = float(getattr(cfg, "autodiff_emergence_lr", 0.2))
    text = getattr(cfg, "autodiff_emergence_target", "LUNA")
    contrast = float(getattr(cfg, "autodiff_emergence_contrast", 4.0))
    cfl = float(getattr(cfg, "autodiff_emergence_cfl", 0.25))

    if driver == "blast":
        # free internal-energy field e(x,0), uniform medium at rest; sigmoid-
        # bounded to [e_floor, e_max] (positivity + CFL stability).
        match = getattr(cfg, "autodiff_emergence_match", "e")
        e_floor = 1.0
        e_max = e_floor * float(getattr(cfg, "autodiff_emergence_emax", 50.0))
        dt = cfl * dx / float(jnp.sqrt(gamma * (gamma - 1.0) * e_max / rho0))

        def e_of(theta):
            return e_floor + (e_max - e_floor) * jax.nn.sigmoid(theta)

        def build_from_theta(theta):
            z = jnp.zeros((Nx, Nx))
            return jnp.stack([jnp.full((Nx, Nx), rho0), z, z, e_of(theta)], axis=0)

        def body(U, _):
            return step_fn(U, dt), None

        def internal_e(U):
            return U[-1] - 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0]

        observe = (lambda U: U[0]) if match == "rho" else internal_e
        lo = rho0 if match == "rho" else e_floor
        rng = np.random.RandomState(0)
        theta0 = -3.0 + 0.3 * rng.standard_normal((Nx, Nx))
        # optional blast-wave-noise start: seed scattered hot Sedov sites so the
        # IC literally *is* explosions (frame 1 reads as blast noise, then the
        # shells collide into the target). Honest: these random sites do not
        # resemble the target -- only their collisions do.
        n_seeds = int(getattr(cfg, "autodiff_emergence_init_seeds", 0))
        if n_seeds > 0:
            ys = rng.randint(0, Nx, n_seeds)
            xs = rng.randint(0, Nx, n_seeds)
            theta0[ys, xs] = 3.0                       # e_of(3) -> ~e_max hot site
        theta0 = jnp.asarray(theta0)
        nparams = Nx * Nx

        def first_panel(theta):
            return (np.log10(np.maximum(np.asarray(e_of(theta)), 1e-30)),
                    "optimized e(x,0)", "inferno")

    elif driver == "turbulence":
        # free band-limited velocity v(x,0) (coarse control grid upsampled,
        # tanh-bounded) of uniform isothermal gas; thermostat each step.
        Nc = int(getattr(cfg, "autodiff_emergence_nc", 32))
        cs2 = 1.0
        v_max = float(getattr(cfg, "autodiff_emergence_mach", 3.0)) * float(np.sqrt(cs2))
        dt = cfl * dx / float(np.sqrt(gamma * cs2 + v_max ** 2))

        def v_of(theta):
            vx = jax.image.resize(theta[0], (Nx, Nx), method="linear")
            vy = jax.image.resize(theta[1], (Nx, Nx), method="linear")
            return v_max * jnp.tanh(vx), v_max * jnp.tanh(vy)

        def build_from_theta(theta):
            vx, vy = v_of(theta)
            rho = jnp.full((Nx, Nx), rho0)
            E = cs2 * rho0 / (gamma - 1.0) + 0.5 * rho0 * (vx ** 2 + vy ** 2)
            return jnp.stack([rho, rho0 * vx, rho0 * vy, E], axis=0)

        def body(U, _):
            U = step_fn(U, dt)                                  # + isothermal thermostat
            ke = 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0]
            return U.at[-1].set(cs2 * U[0] / (gamma - 1.0) + ke), None

        observe = lambda U: U[0]
        lo = rho0
        rng = np.random.RandomState(0)
        theta0 = jnp.asarray(0.3 * rng.standard_normal((2, Nc, Nc)))
        nparams = 2 * Nc * Nc

        def first_panel(theta):
            vx, vy = v_of(theta)
            return np.asarray(jnp.sqrt(vx ** 2 + vy ** 2)), "optimized v(x,0)", "cividis"
    else:
        raise NotImplementedError(f"luna_emergence driver '{driver}' unknown")

    img_path = getattr(cfg, "autodiff_emergence_image", "")
    if img_path:
        tgt_np = _render_image_target(Nx, img_path, lo, lo * contrast)
        if is_main:
            print(f"  emergence target: image '{img_path}' (not text)", flush=True)
    else:
        tgt_np = _render_text_target(Nx, text, lo, lo * contrast)
    tgt = jnp.asarray(tgt_np)

    def simulate(theta):
        # flat scan with a PER-STEP jax.checkpoint, exactly like the (ROCm-
        # proven) IC-inversion rollout. The sqrt-chunked variant (inner scan
        # nested in a checkpointed chunk) segfaults ROCm XLA beyond ~100 steps.
        U = build_from_theta(theta)
        cbody = jax.checkpoint(body)
        Uf, _ = jax.lax.scan(cbody, U, None, length=n_steps)
        return Uf

    def loss(theta):
        return jnp.mean((observe(simulate(theta)) - tgt) ** 2)

    theta = theta0
    if is_main:
        print(f"  luna_emergence[{driver}]: Nx={Nx} ({nparams} free params), "
              f"n_steps={n_steps}, n_iter={n_iter}, target='{text}'", flush=True)
    loss_grad = jax.jit(jax.value_and_grad(loss))
    # movie hook: LUNA_EMERG_MOVIE=<dir> dumps (a) the optimized final state
    # every LUNA_EMERG_MOVIE_EVERY iterations (the "learning" sequence) and
    # (b) a dense replay of the optimized IC (the physical emergence).
    movie_dir = os.environ.get("LUNA_EMERG_MOVIE", "")
    movie_every = int(os.environ.get("LUNA_EMERG_MOVIE_EVERY", "2"))
    # movie-frame forward sim: PLAIN scan, no jax.checkpoint. Compiling the
    # remat'd simulate as a second forward-only executable next to loss_grad
    # segfaults ROCm XLA (target.npy+opt_00000 then SIGSEGV); remat buys
    # nothing without a reverse pass anyway.
    sim_jit = jax.jit(lambda th: jax.lax.scan(
        body, build_from_theta(th), None, length=n_steps)[0])
    if movie_dir and is_main:
        os.makedirs(movie_dir, exist_ok=True)
        np.save(f"{movie_dir}/target.npy", tgt_np)
    # Device-side Adam loop: ONE lax.scan invocation runs all n_iter
    # iterations. The ROCm runtime deadlocks after ~2 synchronized reverse-
    # mode shard_map executions when the optimizer loop lives on the host
    # (every host-loop variant -- sbatch, srun, cache on/off, whole node,
    # HIP_LAUNCH_BLOCKING -- hangs at iteration 2); a single scan invocation
    # never re-enters the executable, sidestepping the bug entirely. Movie
    # "learning" frames are reconstructed afterwards from the per-iteration
    # thetas the scan emits (forward-only replays don't trigger it).
    t0 = time.time()
    vg = jax.value_and_grad(loss)

    def adam_body(carry, it):
        th, m_, v_ = carry
        L_, g = vg(th)
        m_ = 0.9 * m_ + 0.1 * g
        v_ = 0.999 * v_ + 0.001 * g * g
        bc1 = 1.0 - jnp.power(0.9, it + 1.0)
        bc2 = 1.0 - jnp.power(0.999, it + 1.0)
        th = th - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + 1e-8)
        return (th, m_, v_), (L_, th)

    ckpt = os.environ.get("LUNA_EMERG_CKPT", "")
    hostloop = os.environ.get("LUNA_EMERG_HOSTLOOP", "0") == "1"
    if not hostloop:
        @jax.jit
        def optimize(th0):
            return jax.lax.scan(
                adam_body, (th0, jnp.zeros_like(th0), jnp.zeros_like(th0)),
                jnp.arange(n_iter, dtype=th0.dtype))

        (theta, _, _), (Ls, thetas) = optimize(theta)
        jax.block_until_ready(theta)
        if is_main:
            for it in list(range(0, n_iter, max(1, n_iter // 12))) + [n_iter - 1]:
                print(f"    it {it:4d}  L={float(Ls[it]):.3e}", flush=True)
        if movie_dir and is_main:
            for it in range(0, n_iter, movie_every):
                np.save(f"{movie_dir}/opt_{it:05d}.npy",
                        np.asarray(observe(sim_jit(thetas[it])),
                                   dtype=np.float32))
    else:
        # HOST-LOOP FALLBACK for the ROCm 3rd-invocation deadlock: run at
        # most LUNA_EMERG_MAXIT_PER_PROC iterations (default 2 -- the proven
        # per-process budget), checkpoint ONCE at the end (per-iteration
        # ckpt syncs move the deadlock one call earlier), and let a driver
        # loop restart the process until n_iter is reached.
        m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
        it0 = 0
        if ckpt and os.path.exists(ckpt):
            z = np.load(ckpt)
            theta = jnp.asarray(z["theta"]); m = jnp.asarray(z["m"])
            v = jnp.asarray(z["v"]); it0 = int(z["it"])
            if is_main:
                print(f"  resumed from {ckpt} at it {it0}", flush=True)
        it_end = min(n_iter, it0 + int(os.environ.get(
            "LUNA_EMERG_MAXIT_PER_PROC", "2")))
        for it in range(it0, it_end):
            if movie_dir and is_main and it % movie_every == 0:
                np.save(f"{movie_dir}/opt_{it:05d}.npy",
                        np.asarray(observe(sim_jit(theta)), dtype=np.float32))
            L_, g = loss_grad(theta)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            bc1 = 1 - 0.9 ** (it + 1); bc2 = 1 - 0.999 ** (it + 1)
            theta = theta - lr * (m / bc1) / (jnp.sqrt(v / bc2) + 1e-8)
            if is_main:
                print(f"    it {it:4d}  L={float(L_):.3e}", flush=True)
        if ckpt and is_main:
            np.savez(ckpt + ".tmp.npz", theta=np.asarray(theta),
                     m=np.asarray(m), v=np.asarray(v), it=it_end)
            os.replace(ckpt + ".tmp.npz", ckpt)
        if it_end < n_iter:            # driver loop restarts us; skip movie
            if is_main:
                print(f"  chunk done at it {it_end}/{n_iter}", flush=True)
            return {"loss": float(L_) if is_main else None, "driver": driver}
    if movie_dir and is_main:
        np.save(f"{movie_dir}/theta_final.npy", np.asarray(theta))
        stride = max(1, n_steps // int(os.environ.get("LUNA_EMERG_MOVIE_NFRAME",
                                                      "240")))
        U = build_from_theta(theta)
        np.save(f"{movie_dir}/replay_00000.npy",
                np.asarray(observe(U), dtype=np.float32))
        replay_chunk = jax.jit(lambda U: jax.lax.scan(body, U, None,
                                                      length=stride)[0])
        for ns in range(stride, n_steps + 1, stride):
            U = replay_chunk(U)
            np.save(f"{movie_dir}/replay_{ns:05d}.npy",
                    np.asarray(observe(U), dtype=np.float32))
        print(f"  movie frames -> {movie_dir} (stride {stride})", flush=True)
    if not is_main:
        return {"loss": None, "driver": driver}
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    fracs = [0.0, 0.12, 0.30, 0.55, 1.0]          # time montage of the optimized IC
    panel0, panel0_lab, panel0_cmap = first_panel(theta)
    frames = []
    for f in fracs:
        ns = int(round(f * n_steps))
        U = build_from_theta(theta)
        if ns > 0:
            U, _ = jax.lax.scan(body, U, None, length=ns)
        frames.append(np.asarray(observe(U)))
    _save_emergence_montage(panel0, panel0_lab, panel0_cmap, frames, fracs,
                            tgt_np, Nx, text, driver)
    return {"loss": float(loss(theta)), "Nx": Nx, "driver": driver}


def run_simulation_autodiff(U0, cfg, step_fn, compute_dt_fn, sharding,
                            is_main_process=True):
    """Differentiable-run dispatcher: selects the task by ``cfg.autodiff_task``
    (``xray_sensitivity`` | ``eta_recovery`` | ``ic_inversion`` |
    ``luna_emergence`` | ``corrector_training``) and returns that task's result
    dict; `main.run` wraps it in a RunResult (dict lands in ``.extra``). All
    tasks differentiate through the production step_fn from
    create_solver_distributed."""
    task = getattr(cfg, "autodiff_task", "xray_sensitivity")

    # IC-field optimization so the late-time field forms a target image (fig19);
    # builds its own forward from step_fn, so it doesn't use build_simulate.
    if task == "luna_emergence":
        return _luna_emergence(step_fn, cfg, is_main_process)

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
    fdM = eta_true[0] * _central_fd(lnLX, eta_true, 0)
    fdE = eta_true[1] * _central_fd(lnLX, eta_true, 1)
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
    through the FULL coupled forward model (wind + cooling + self-gravity +
    external potential). CIRCULAR self-consistency validation: inject a KNOWN
    η, evolve, recover it — certifies gradients through the whole physics
    stack; NOT a measurement of the real M82."""
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
    fd0 = jnp.array([_central_fd(loss, log0, i) for i in range(2)])
    rel = [abs(float(g0[i]) - float(fd0[i])) / (abs(float(fd0[i])) + 1e-30)
           for i in range(2)]
    if is_main:
        print(f"  eta_recovery: forward+FD ({time.time()-t0:.0f}s); "
              f"∂loss/∂lnη AD={np.asarray(g0)}  FD={np.asarray(fd0)}  "
              f"rel={rel}", flush=True)

    # Adam on log-η (shared pytree Adam: b1=0.9, b2=0.999, eps=1e-8, bias-corr).
    log_eta = log0
    opt = adam_init(log_eta)
    n_iter = int(getattr(cfg, "autodiff_n_iter", 60))
    hist_it, hist_L, hist_etaM, hist_etaE = [], [], [], []
    for it in range(n_iter):
        L, gg = jax.value_and_grad(loss)(log_eta)
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
        _save_eta_recovery_figure(hist_it, hist_L, hist_etaM, hist_etaE,
                                  eta_t, rel, res, cfg)
    return {"eta_recovered": eta_rec.tolist(), "eta_true": eta_t.tolist(),
            "eta_rel_err": res, "grad_fd_rel": rel}


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
    # Broadband true IC by default (autodiff_ic_kmax=0): signal at every scale, so
    # r(k) measures the solver's real recovery reach. Cooling (dissipative) will
    # collapse at its physical limit; self-gravity (lossless FMG adjoint) recovers
    # to high k with the SAME optimizer -> the contrast is pure physics.
    k_max = int(getattr(cfg, "autodiff_ic_kmax", 0) or 0)
    slope = float(getattr(cfg, "autodiff_ic_slope", 0.0) or 0.0)
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
    chan = 4 if getattr(cfg, "autodiff_ic_field", "density") == "energy" else 0

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
        g = jax.grad(loss)(delta0)
        idx = tuple(int(s // 2) for s in shape)
        fd = _central_fd(loss, delta0, idx)
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
    n_iter = int(getattr(cfg, "autodiff_n_iter", 100))
    lr = float(getattr(cfg, "autodiff_ic_lr", 3.0e-2) or 3.0e-2)
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
        _save_inversion_figure(delta_true_np, delta_host, rk, res, cfg)
    return {"ic_residual": res, "grad_fd_rel": rel,
            "rk": rk, "delta": delta_host}


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
    from sources.ml_corrector import cnn_init, cnn_correction, pool_factor

    import os
    f = int(os.environ.get("LUNA_KHC_F", "4"))   # coarse->fine gap (8x for the corrector)
    Nfx = f * int(cfg.Nx)
    Nfy = f * int(cfg.Ny)
    n_steps = int(cfg.autodiff_n_steps)                       # eval horizon (~t=3)
    n_train = int(os.environ.get("LUNA_KHC_NTRAIN",           # train horizon (~t=1)
                                 str(max(1, n_steps // 3))))
    n_iter = int(cfg.autodiff_n_iter)
    dt_c = (float(cfg.autodiff_fixed_dt) if cfg.autodiff_fixed_dt > 0
            else float(compute_dt_fn(U0)))

    # --- fine reference via the PRODUCTION setup at 4x res ---
    fine_cfg = replace(cfg, Nx=Nfx, Ny=Nfy, autodiff_mode=False)
    fmesh, _ = create_distributed_mesh(fine_cfg)
    Uf, _, _ = initialize_simulation_distributed(fine_cfg, fmesh)
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
        _save_corrector_figure(c_plain, c_corr, snap_ref, snap_plain, snap_corr,
                               mtimes, dt_c, n_train, cfg)
    return {"coarse_mse": c_plain, "corr_mse": c_corr, "weights": w}
