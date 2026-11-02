"""Emergence: optimise a 2-D initial field by reverse-mode AD so that, after
n_steps of hydrodynamics, the gas spells a target word or picture (diffhydro
Fig. 12). driver='blast': the free parameter is the internal-energy field of a
uniform medium at rest (scattered Sedov sites collide into the target);
driver='turbulence': a band-limited velocity field of an isothermal medium.
The hydro step is the production one; the IC and loop are built here.

    python -m scripts.examples.autodiff.blast_emergence_run nx=256 n_steps=300 n_iter=200
    python -m scripts.examples.autodiff.turb_emergence_run nx=256 nc=32 mach=3 movie=frames/
"""
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from scripts.examples.autodiff.figs import save_emergence_montage


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


def task(init, cfg, physics, is_main, *, driver="blast", target="", image="", seeds=0,
         emax=50.0, contrast=4.0, match="e", cfl=0.25, lr=0.2, nc=32, mach=3.0,
         n_iter=100, movie="", movie_every=2, movie_nframe=240, ckpt="", hostloop=False,
         maxit_per_proc=2):
    """Optimise the IC field so the late-time field forms the target.

    driver: "blast" or "turbulence". target: the word (default: the image's
    file name, else "LUNA"); image: a [0,1] target field (.npy/png) instead of
    the word; contrast: target hi/lo; n_iter, lr: Adam; cfl: the fixed-dt CFL.
    blast: seeds hot Sedov sites in e(x,0), emax the e ceiling (x floor), match
    "e" or "rho". turbulence: nc the control-grid side, mach the stirring Mach.
    movie: a directory for the frames (the optimised final state every
    `movie_every` iterations, then a `movie_nframe`-frame replay of the
    optimised IC). hostloop: the host-side Adam loop instead of the one-scan
    device loop, `maxit_per_proc` iterations per process, resumable from `ckpt`
    (the fallback for the ROCm deadlock below)."""
    step_fn = physics.hydro.step
    Nx, gamma, L, rho0 = cfg.Nx, float(cfg.gamma), cfg.Lx, 1.0
    dx = L / Nx
    n_steps = int(cfg.autodiff_n_steps)
    n_iter, lr, contrast, cfl = int(n_iter), float(lr), float(contrast), float(cfl)
    text = target or (os.path.splitext(os.path.basename(image))[0] if image else "LUNA")

    if driver == "blast":
        # free internal-energy field e(x,0), uniform medium at rest; sigmoid-
        # bounded to [e_floor, e_max] (positivity + CFL stability).
        e_floor = 1.0
        e_max = e_floor * float(emax)
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
        n_seeds = int(seeds)
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
        Nc = int(nc)
        cs2 = 1.0
        v_max = float(mach) * float(np.sqrt(cs2))
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

    img_path = image
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
    # movie: (a) the optimized final state every movie_every iterations (the
    # "learning" sequence) and (b) a dense replay of the optimized IC (the
    # physical emergence).
    movie_dir = movie
    movie_every = int(movie_every)
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
    vg = jax.jit(jax.value_and_grad(loss))

    def adam_body(carry, it):
        th, m_, v_ = carry
        L_, g = vg(th)
        m_ = 0.9 * m_ + 0.1 * g
        v_ = 0.999 * v_ + 0.001 * g * g
        bc1 = 1.0 - jnp.power(0.9, it + 1.0)
        bc2 = 1.0 - jnp.power(0.999, it + 1.0)
        th = th - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + 1e-8)
        return (th, m_, v_), (L_, th)

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
        # most maxit_per_proc iterations (default 2 -- the proven
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
        it_end = min(n_iter, it0 + int(maxit_per_proc))
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
        stride = max(1, n_steps // int(movie_nframe))
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
    save_emergence_montage(panel0, panel0_lab, panel0_cmap, frames, fracs,
                            tgt_np, Nx, text, driver)
    return {"loss": float(loss(theta)), "Nx": Nx, "driver": driver}
