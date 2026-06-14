"""Blasts-2D inverse initial conditions — the diffhydro/Horowitz Fig-12 analog.

Optimize the INITIAL INTERNAL-ENERGY field e(x,0) of a 2D uniform medium so that,
after hydro evolution, an ensemble of Sedov-Taylor blast waves collides and
merges into a prescribed late-time target morphology. Reverse-mode AD through
the production 2D MUSCL-Hancock solver; the control variables are the Nx*Ny
energy-field values.

Self-contained: builds the uniform medium + blast field manually and steps it
with the production step_fn (via create_solver_distributed), so it does not rely
on the 3D-shaped orchestrator helpers.

Env: LUNA_BLAST_NX (default 256), LUNA_BLAST_NSTEPS (120), LUNA_BLAST_NITER (120),
     LUNA_BLAST_NBLAST (8).
"""
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")

from dataclasses import replace
from template_config import Config
from setup.distributed_setup import (create_distributed_mesh,
                                     initialize_simulation_distributed,
                                     create_solver_distributed)
from utils.grid_params import GridParams


def _blast_field(key, Nx, L, n_blast, e_amb, e_peak, w):
    """Uniform e_amb background + n_blast Gaussian energy spots (the true IC)."""
    dx = L / Nx
    xc = (jnp.arange(Nx) + 0.5) * dx
    X, Y = jnp.meshgrid(xc, xc, indexing="ij")
    kx, ky = jax.random.split(key)
    cx = jax.random.uniform(kx, (n_blast,), minval=0.2 * L, maxval=0.8 * L)
    cy = jax.random.uniform(ky, (n_blast,), minval=0.2 * L, maxval=0.8 * L)
    e = jnp.full((Nx, Ny := Nx), e_amb)
    for i in range(n_blast):
        e = e + e_peak * jnp.exp(-((X - cx[i]) ** 2 + (Y - cy[i]) ** 2) / (2 * w * w))
    return e


def _text_density(Nx, text, rho_lo, rho_hi):
    """Render `text` to an Nx×Nx density target (rho_hi on the glyphs, rho_lo
    elsewhere) — the late-time morphology the forward run is optimized to form."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from jax.image import resize as jresize
    # render on a FIXED 400px canvas (font sized to the text width), then resize
    # to Nx — so the glyphs never overflow regardless of Nx.
    fig = plt.figure(figsize=(4, 4), dpi=100)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_facecolor("black"); ax.axis("off")
    fam = os.environ.get("LUNA_BLAST_FONT", "sans-serif")     # "cursive" → calligraphic
    fstyle = os.environ.get("LUNA_BLAST_FONTSTYLE", "normal")  # "italic" for flow
    ax.text(0.5, 0.5, text, ha="center", va="center", color="white",
            fontsize=260.0 / max(1, len(text)), fontweight="bold",
            family=fam, style=fstyle, transform=ax.transAxes)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(np.float64) / 255.0
    plt.close(fig)
    r = np.asarray(jresize(jnp.asarray(buf), (Nx, Nx), method="linear"))
    blur = float(os.environ.get("LUNA_BLAST_BLUR", "0"))      # soften strokes (large-scale)
    if blur > 0:
        from scipy.ndimage import gaussian_filter
        r = gaussian_filter(r, sigma=blur * Nx / 512.0)
        r = (r - r.min()) / (r.max() - r.min() + 1e-12)
    return rho_lo + (rho_hi - rho_lo) * r[::-1].T


def _image_density(Nx, path, rho_lo, rho_hi):
    """Load an image file → Nx×Nx density target (grayscale; bright = dense).
    Aspect ratio preserved (fit longest side to Nx, pad the rest)."""
    import matplotlib.image as mpimg
    from jax.image import resize as jresize
    img = mpimg.imread(path).astype(np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    if img.max() > 1.5:
        img = img / 255.0
    H, W = img.shape
    s = Nx / max(H, W)
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    small = np.asarray(jresize(jnp.asarray(img), (nh, nw), method="linear"))
    canvas = np.full((Nx, Nx), float(small.min()))
    y0, x0 = (Nx - nh) // 2, (Nx - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    r = (canvas - canvas.min()) / (canvas.max() - canvas.min() + 1e-12)
    return rho_lo + (rho_hi - rho_lo) * r[::-1].T


def main():
    Nx = int(os.environ.get("LUNA_BLAST_NX", "256"))
    n_steps = int(os.environ.get("LUNA_BLAST_NSTEPS", "120"))
    n_iter = int(os.environ.get("LUNA_BLAST_NITER", "120"))
    n_blast = int(os.environ.get("LUNA_BLAST_NBLAST", "8"))
    L = 1.0
    gamma = 5.0 / 3.0
    rho0 = 1.0

    mesh_env = os.environ.get("LUNA_BLAST_MESH", "")        # e.g. "2x2" for 4 GPUs
    mesh_shape = tuple(int(v) for v in mesh_env.split("x")) if mesh_env else None
    cfg = replace(Config(), test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2,
                  Nx=Nx, Ny=Nx, Lx=L, Ly=L, gamma=gamma, BOUNDARY=0,
                  ORDER=2, LIMITER=1, riemann_solver="hllc", MESH_SHAPE=mesh_shape,
                  PLOT_AFTER_SIM=False, production_mode=True)
    from setup.distributed_setup import initialize_jax_distributed
    initialize_jax_distributed(enable_x64=True)
    mesh, _ = create_distributed_mesh(cfg)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, _, sharding = create_solver_distributed(cfg, grid, mesh)

    dx = L / Nx
    e_amb, e_peak, w = 1.0, 12.0, 2.5 * dx
    gm1 = gamma - 1.0
    # The optimized energy is BOUNDED to [e_amb, e_max] via a sigmoid, and dt is
    # CFL-safe for e_max — so the optimizer can never push the field past what the
    # fixed-dt scan can integrate (otherwise the loss NaNs once the energy grows).
    e_max = e_amb * float(os.environ.get("LUNA_BLAST_EMAX", "14.0"))
    cs_max = float(jnp.sqrt(gamma * gm1 * e_max / rho0))
    dt = 0.12 * dx / cs_max
    # BAND-LIMITED control: optimize the spark field at a fixed COARSE resolution
    # Nc and upsample to Nx. This keeps the energy field smooth — so the optimizer
    # can't carve sharp features that drive locally-supersonic flows past the fixed
    # dt (the high-res NaN) — and makes convergence resolution-independent.
    from jax.image import resize as _jresize
    Nc = min(Nx, int(os.environ.get("LUNA_BLAST_NCTRL", "128")))

    def e_of(theta):                          # coarse control → smooth energy field
        up = _jresize(theta, (Nx, Nx), method="linear")
        return e_amb + (e_max - e_amb) * jax.nn.sigmoid(up)

    def build_U(efield):
        rho = jnp.full((Nx, Nx), rho0)
        mom = jnp.zeros((Nx, Nx))
        U = jnp.stack([rho, mom, mom, efield], axis=0)      # E = efield (v=0)
        return jax.lax.with_sharding_constraint(U, sharding)  # shard across the mesh

    # sqrt-checkpointing: outer scan over CHUNKS (each remat'd), inner over steps.
    # The reverse-mode tape then holds ~2·√(n_steps) carries instead of n_steps,
    # so the 4096² rollout fits in GPU memory (60 GB → a few GB).
    per = max(1, int(round(n_steps ** 0.5)))
    nchunk = (n_steps + per - 1) // per          # ceil; total = nchunk·per ≳ n_steps

    def simulate(efield):
        U = build_U(efield)
        def body(U, _):                       # single GLOBAL fixed dt (below)
            return step_fn(U, dt), None
        @jax.checkpoint
        def chunk(U, _):
            U, _ = jax.lax.scan(body, U, None, length=per)
            return U, None
        Uf, _ = jax.lax.scan(chunk, U, None, length=nchunk)
        return Uf

    # ---- target: (a) blast reconstruction, or (b) a TEXT/IMAGE late-time
    #      morphology — optimize the IC so the explosion set forms "LUNA" (or a
    #      supplied image). LUNA_BLAST_TARGET="LUNA" or LUNA_BLAST_IMAGE=<path>. ----
    text = os.environ.get("LUNA_BLAST_TARGET", "")
    img = os.environ.get("LUNA_BLAST_IMAGE", "")
    mode = "image" if img else ("text" if text else "blast")
    rho_hi = rho0 * float(os.environ.get("LUNA_BLAST_CONTRAST", "2.5"))

    if mode == "blast":
        e_true = _blast_field(jax.random.PRNGKey(0), Nx, L, n_blast, e_amb, e_peak, w)
        U_target = simulate(e_true); e_true_np = np.asarray(e_true)
        def loss(theta):
            return jnp.mean((simulate(e_of(theta)) - U_target) ** 2)
        label = f"{n_blast}blasts"
    else:
        rho_tgt_np = (_image_density(Nx, img, rho0, rho_hi) if mode == "image"
                      else _text_density(Nx, text, rho0, rho_hi))
        rho_tgt = jnp.asarray(rho_tgt_np)
        e_true = None; e_true_np = None
        label = os.path.basename(img) if mode == "image" else text
        def loss(theta):                     # absolute density match (stable);
            return jnp.mean((simulate(e_of(theta))[0] - rho_tgt) ** 2)  # contrast kept achievable

    if jax.process_index() == 0:
        print(f"Blasts-2D: Nx={Nx}, target={mode}({label}), n_steps={n_steps}", flush=True)

    # ---- optimize the (bounded, band-limited Nc×Nc) spark field ----
    from jax.sharding import NamedSharding, PartitionSpec as P
    theta = jax.device_put(jnp.full((Nc, Nc), -2.0), NamedSharding(mesh, P()))  # replicated
    loss_grad = jax.jit(jax.value_and_grad(loss))      # jit for the sharded rollout
    _, g = loss_grad(theta)
    if jax.process_index() == 0:
        print(f"  grad finite={bool(jnp.all(jnp.isfinite(g)))}", flush=True)
    m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
    t0 = time.time()
    for it in range(n_iter):
        L_, gg = loss_grad(theta)
        m = 0.9 * m + 0.1 * gg
        v = 0.999 * v + 0.001 * gg * gg
        theta = theta - 4e-2 * m / (1 - 0.9 ** (it + 1)) / (
            jnp.sqrt(v / (1 - 0.999 ** (it + 1))) + 1e-8)
        if jax.process_index() == 0 and (it % max(1, n_iter // 12) == 0 or it == n_iter - 1):
            extra = ("" if e_true is None else
                     f"  res={float(jnp.linalg.norm(e_of(theta)-e_true))/(np.linalg.norm(e_true_np)+1e-30):.3e}")
            print(f"    it {it:4d}  L={float(L_):.3e}{extra}", flush=True)
    if jax.process_index() != 0:
        return
    e_rec = np.asarray(e_of(theta))
    rho_final = np.asarray(simulate(e_of(theta))[0])
    if mode == "blast":
        res = float(np.linalg.norm(e_rec - e_true_np) / (np.linalg.norm(e_true_np) + 1e-30))
        print(f"  recovered blast IC: residual {res:.3e}  ({time.time()-t0:.0f}s)", flush=True)
        _save_figure(e_true_np, e_rec, np.asarray(simulate(e_true)[0]), rho_final, Nx, res)
    else:
        print(f"  '{label}' emerged from the explosion set  ({time.time()-t0:.0f}s)", flush=True)
        _save_target_figure(e_rec, rho_final, rho_tgt_np, Nx, label)


def _save_figure(e_true, e_rec, rho_target, rho_rec, Nx, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, FIGSIZE
    apply_style()
    fig, ax = plt.subplots(1, 4, figsize=(FIGSIZE[0] * 2.4, FIGSIZE[1] * 0.95))
    panels = [(e_true, "magma", r"$e(x,0)$ true"),
              (e_rec, "magma", fr"$e(x,0)$ recovered (res {res:.2f})"),
              (rho_target, "viridis", r"$\rho$ target (late)"),
              (rho_rec, "viridis", r"$\rho$ recovered (late)")]
    for a, (img, cmap, lab) in zip(ax, panels):
        a.imshow(img.T, origin="lower", cmap=cmap)
        a.grid(False); a.set_xticks([]); a.set_yticks([])
        a.text(0.04, 0.96, lab, transform=a.transAxes, va="top", ha="left",
               color="white", fontsize=10)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = f"figures/autodiff_blast_reconstruction_Nx{Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


def _save_target_figure(e_rec, rho_final, rho_tgt, Nx, label):
    """The 'LUNA out of the explosion set' figure: optimized IC energy field, the
    forward-run output (which forms the target), and the target morphology."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, FIGSIZE
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(FIGSIZE[0] * 2.1, FIGSIZE[1] * 0.95))
    panels = [(e_rec, "magma", r"optimized $e(x,0)$ (the spark field)"),
              (rho_final, "viridis", "forward-run output"),
              (rho_tgt, "viridis", f"target ({label})")]
    for a, (im, cmap, lab) in zip(ax, panels):
        a.imshow(im.T, origin="lower", cmap=cmap)
        a.grid(False); a.set_xticks([]); a.set_yticks([])
        a.text(0.04, 0.96, lab, transform=a.transAxes, va="top", ha="left",
               color="white", fontsize=10)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    safe = label.replace(" ", "").replace("/", "_").split(".")[0]
    out = f"figures/autodiff_luna_emergence_{safe}_Nx{Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
