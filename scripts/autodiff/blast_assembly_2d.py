"""Overlapping blast waves that COLLIDE and MERGE to form a target — the real
diffhydro Fig-12 mechanism (not a smooth mold).

The initial condition is a LATTICE of discrete high-energy blast SEEDS with
optimizable amplitudes. Each seed launches a Sedov blast; the blasts expand,
collide, and merge into a network of shocks whose late-time density forms the
target (the word LUNA, or a supplied photograph). Reverse-mode AD through the
production 2D solver optimizes the seed amplitudes so the *merged* morphology
matches the target. The figure is a TIME MONTAGE — seeds → expanding blasts →
colliding → merged target — so the mechanism is visible.

Env: LUNA_BLAST_TARGET="LUNA" | LUNA_BLAST_IMAGE=<path>, LUNA_BLAST_NX (512),
     LUNA_BLAST_NSEED (lattice per side, 32), LUNA_BLAST_NSTEPS (260),
     LUNA_BLAST_NITER (200), LUNA_BLAST_MESH ("2x2" for 4 GPUs).
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
from setup.distributed_setup import (initialize_jax_distributed,
                                     create_distributed_mesh,
                                     create_solver_distributed)
from utils.grid_params import GridParams
from scripts.autodiff.blast_reconstruction_2d import _text_density, _image_density


def _gauss_kernel(sigma, half):
    x = np.arange(-half, half + 1)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g); k /= k.sum()
    return jnp.asarray(k)


def main():
    Nx = int(os.environ.get("LUNA_BLAST_NX", "512"))
    Ns = int(os.environ.get("LUNA_BLAST_NSEED", "32"))      # seeds per side
    n_steps = int(os.environ.get("LUNA_BLAST_NSTEPS", "260"))
    n_iter = int(os.environ.get("LUNA_BLAST_NITER", "200"))
    L, gamma, rho0 = 1.0, 5.0 / 3.0, 1.0
    while Nx % Ns:                                           # seeds must tile Nx
        Ns -= 1
    s = Nx // Ns

    mesh_env = os.environ.get("LUNA_BLAST_MESH", "")
    mesh_shape = tuple(int(v) for v in mesh_env.split("x")) if mesh_env else None
    cfg = replace(Config(), test_problem="Kelvin-Helmholtz", SPATIAL_DIMENSION=2,
                  Nx=Nx, Ny=Nx, Lx=L, Ly=L, gamma=gamma, BOUNDARY=0,
                  ORDER=2, LIMITER=1, riemann_solver="hllc", MESH_SHAPE=mesh_shape,
                  PLOT_AFTER_SIM=False, production_mode=True)
    initialize_jax_distributed(enable_x64=True)
    mesh, _ = create_distributed_mesh(cfg)
    grid = GridParams.from_config(cfg)
    compute_dt_fn, step_fn, _, sharding = create_solver_distributed(cfg, grid, mesh)
    from jax.sharding import NamedSharding, PartitionSpec as P
    rep = NamedSharding(mesh, P())

    dx = L / Nx
    e_amb = 1.0
    e_max = e_amb * float(os.environ.get("LUNA_BLAST_EMAX", "60.0"))   # strong seeds
    cs_max = float(jnp.sqrt(gamma * (gamma - 1.0) * e_max / rho0))
    dt = 0.05 * dx / cs_max                              # conservative GLOBAL fixed dt
    seed_sigma = float(os.environ.get("LUNA_BLAST_SEEDW", "1.3"))
    kernel = _gauss_kernel(seed_sigma, half=max(2, int(3 * seed_sigma)))
    from jax.scipy.signal import convolve2d

    # base lattice positions (grid coords); blasts may DISPLACE from these.
    _gx, _gy = np.meshgrid((np.arange(Ns) + 0.5) * s, (np.arange(Ns) + 0.5) * s,
                           indexing="ij")
    base_pos = jnp.asarray(np.stack([_gx.ravel(), _gy.ravel()], axis=1))   # (N,2)

    def e_of(params):
        """Discrete blast seeds with optimizable AMPLITUDE and POSITION. Each
        seed's bounded energy is bilinearly deposited at its continuous position
        (differentiable in position) then blurred to a localized Sedov spark."""
        logamp, dpos = params
        amp = e_max * jax.nn.sigmoid(logamp)                  # (N,) bounded
        pos = base_pos + dpos
        px = jnp.clip(pos[:, 0], 0.0, Nx - 1.001)
        py = jnp.clip(pos[:, 1], 0.0, Nx - 1.001)
        ix = jnp.floor(px).astype(jnp.int32); iy = jnp.floor(py).astype(jnp.int32)
        fx = px - ix; fy = py - iy
        field = jnp.zeros((Nx, Nx))
        for ddx, ddy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                            (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            field = field.at[jnp.clip(ix + ddx, 0, Nx - 1),
                             jnp.clip(iy + ddy, 0, Nx - 1)].add(w * amp)
        return e_amb + convolve2d(field, kernel, mode="same")

    def build_U(efield):
        rho = jnp.full((Nx, Nx), rho0); mom = jnp.zeros((Nx, Nx))
        U = jnp.stack([rho, mom, mom, efield], axis=0)
        return jax.lax.with_sharding_constraint(U, sharding)

    per = max(1, int(round(n_steps ** 0.5)))
    nchunk = (n_steps + per - 1) // per

    def simulate(theta):
        U = build_U(e_of(theta))
        def body(U, _): return step_fn(U, dt), None
        @jax.checkpoint
        def chunk(U, _):
            U, _ = jax.lax.scan(body, U, None, length=per); return U, None
        Uf, _ = jax.lax.scan(chunk, U, None, length=nchunk)
        return Uf

    # ---- target ----
    text = os.environ.get("LUNA_BLAST_TARGET", "")
    img = os.environ.get("LUNA_BLAST_IMAGE", "")
    rho_hi = rho0 * float(os.environ.get("LUNA_BLAST_CONTRAST", "2.2"))
    label = os.path.basename(img) if img else (text or "LUNA")
    rho_tgt_np = (_image_density(Nx, img, rho0, rho_hi) if img
                  else _text_density(Nx, text or "LUNA", rho0, rho_hi))
    rho_tgt = jnp.asarray(rho_tgt_np)

    def loss(theta):
        return jnp.mean((simulate(theta)[0] - rho_tgt) ** 2)

    if jax.process_index() == 0:
        print(f"Blast-assembly: Nx={Nx}, {Ns}x{Ns}={Ns*Ns} blast seeds, "
              f"n_steps={n_steps}, target={label}", flush=True)

    # ---- optimize seed AMPLITUDES and POSITIONS (physically-motivated warm start)
    # Strong seeds in the BACKGROUND (dark target) so their expanding shells
    # converge to compress gas INTO the bright letters; positions then migrate so
    # the shell COLLISIONS trace the target. ----
    tgt_seed = np.asarray(rho_tgt_np)[s // 2::s, s // 2::s]
    tgt_seed = (tgt_seed - tgt_seed.min()) / (tgt_seed.max() - tgt_seed.min() + 1e-9)
    logamp = jax.device_put(jnp.asarray((2.5 * (0.5 - tgt_seed)).ravel()), rep)
    dpos = jax.device_put(jnp.zeros((Ns * Ns, 2)), rep)
    params = (logamp, dpos)
    loss_grad = jax.jit(jax.value_and_grad(loss))
    _, g = loss_grad(params)
    if jax.process_index() == 0:
        print(f"  grad finite={bool(all(bool(jnp.all(jnp.isfinite(x))) for x in g))}", flush=True)
    lr_amp = float(os.environ.get("LUNA_BLAST_LR", "1.2e-1"))
    lr_pos = float(os.environ.get("LUNA_BLAST_LRPOS", "0.4"))      # grid cells / iter
    m = jax.tree.map(jnp.zeros_like, params); v = jax.tree.map(jnp.zeros_like, params)
    t0 = time.time()
    for it in range(n_iter):
        L_, g = loss_grad(params)
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        bc1 = 1 - 0.9 ** (it + 1); bc2 = 1 - 0.999 ** (it + 1)
        def _upd(p, mm, vv, l): return p - l * (mm / bc1) / (jnp.sqrt(vv / bc2) + 1e-8)
        params = (_upd(params[0], m[0], v[0], lr_amp),
                  _upd(params[1], m[1], v[1], lr_pos))
        if jax.process_index() == 0 and (it % max(1, n_iter // 12) == 0 or it == n_iter - 1):
            print(f"    it {it:4d}  L={float(L_):.3e}", flush=True)
    if jax.process_index() != 0:
        return
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # ---- time montage: rerun the optimized seeds, save density vs time ----
    fracs = [0.0, 0.12, 0.30, 0.55, 1.0]
    e0 = e_of(params)
    frames = []
    for f in fracs:
        ns = int(round(f * n_steps))
        U = build_U(e0)
        def body(U, _): return step_fn(U, dt), None
        if ns > 0:
            U, _ = jax.lax.scan(body, U, None, length=ns)
        frames.append(np.asarray(U[0]))
    _save_montage(np.asarray(e0), frames, fracs, rho_tgt_np, Nx, label)


def _save_montage(e0, frames, fracs, rho_tgt, Nx, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, FIGSIZE
    apply_style()
    n = len(frames) + 1
    fig, ax = plt.subplots(1, n, figsize=(FIGSIZE[0] * n * 0.62, FIGSIZE[1] * 0.95))
    vmx = float(np.percentile(np.concatenate([f.ravel() for f in frames]), 99.5))
    ax[0].imshow(np.log10(e0).T, origin="lower", cmap="inferno")
    ax[0].text(0.04, 0.96, "blast seeds (IC)", transform=ax[0].transAxes,
               va="top", ha="left", color="white", fontsize=10)
    for a, fr, fc in zip(ax[1:1 + len(frames)], frames, fracs):
        a.imshow(fr.T, origin="lower", cmap="viridis", vmin=rho_tgt.min(), vmax=vmx)
        lab = "merged → target" if fc == 1.0 else f"t = {fc:.2f} t_end"
        a.text(0.04, 0.96, lab, transform=a.transAxes, va="top", ha="left",
               color="white", fontsize=10)
    for a in ax:
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    safe = label.replace(" ", "").replace("/", "_").split(".")[0]
    out = f"figures/autodiff_blast_assembly_{safe}_Nx{Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
