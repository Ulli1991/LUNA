"""Horowitz-style (diffhydro Fig. 12, "Blasts") initial-condition optimization.

Faithful to diffhydro §7.1: the ENTIRE per-cell internal-energy field e(x,0) of
a uniform medium AT REST is the free parameter -- there are no placed blast
seeds.  Reverse-mode AD through the production 2D solver optimizes e(x,0) so
that, after a LONG evolution, the late-time internal energy matches a target
(the word LUNA, or a photo).

Why the optimized IC does NOT spell the target (the whole point):
  In a cold uniform medium, any localized energy peak launches an *outgoing*
  Sedov-Taylor blast -- energy SPREADS.  A target-shaped IC would simply blow
  apart, so it cannot reproduce a sharp target at late time.  The only way to
  match the target after a long evolution is to seat scattered blast sites whose
  expanding shells COLLIDE and CONVERGE into the target.  Hence the converged
  e(x,0) is an ensemble of scattered Sedov sites -- the target emerges purely
  from collide-and-merge dynamics, not from the initial layout.
  (diffhydro §7.1: "localized peaks in the optimized e(x,0) behave like an
   ensemble of impulsive Sedov-Taylor injections ... reorganize into a curved
   network of shocks, rarefactions, and interaction bridges".)

The field is INITIALIZED UNIFORM (a noise floor, NOT the target), so there is no
target imprint to begin with.

Env: LUNA_FIELD_TARGET="LUNA" | LUNA_FIELD_IMAGE=<path>, LUNA_FIELD_NX (512),
     LUNA_FIELD_NSTEPS (300), LUNA_FIELD_NITER (200), LUNA_FIELD_EMAX (50),
     LUNA_FIELD_CFL (0.25), LUNA_FIELD_CONTRAST (4.0), LUNA_FIELD_MATCH
     ("e"=internal energy [default, faithful to paper] | "rho"=density),
     LUNA_FIELD_MESH ("2x2" for 4 GPUs).
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


def main():
    Nx = int(os.environ.get("LUNA_FIELD_NX", "512"))
    n_steps = int(os.environ.get("LUNA_FIELD_NSTEPS", "300"))
    n_iter = int(os.environ.get("LUNA_FIELD_NITER", "200"))
    L, gamma, rho0 = 1.0, 5.0 / 3.0, 1.0

    mesh_env = os.environ.get("LUNA_FIELD_MESH", "")
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
    # rank-2 spatial sharding for the free per-cell field theta (U is rank-3)
    field_sharding = (NamedSharding(mesh, P('x', 'y')) if mesh_shape else rep)

    dx = L / Nx
    e_floor = 1.0
    e_max = e_floor * float(os.environ.get("LUNA_FIELD_EMAX", "50.0"))
    cs_max = float(jnp.sqrt(gamma * (gamma - 1.0) * e_max / rho0))
    cfl = float(os.environ.get("LUNA_FIELD_CFL", "0.25"))
    dt = cfl * dx / cs_max                          # GLOBAL fixed dt, CFL-safe

    def e_of(theta):
        """Free per-cell internal-energy field, sigmoid-bounded to [e_floor, e_max]
        for positivity and CFL stability. The whole field is the free parameter."""
        return e_floor + (e_max - e_floor) * jax.nn.sigmoid(theta)

    def build_U(efield):
        # uniform density at rest; efield is total energy density (v=0 => internal)
        rho = jnp.full((Nx, Nx), rho0); mom = jnp.zeros((Nx, Nx))
        U = jnp.stack([rho, mom, mom, efield], axis=0)
        return jax.lax.with_sharding_constraint(U, sharding)

    def internal_e(U):
        ke = 0.5 * (U[1] ** 2 + U[2] ** 2) / U[0]
        return U[-1] - ke

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

    # ---- target (matched at LATE time) ----
    text = os.environ.get("LUNA_FIELD_TARGET", "")
    img = os.environ.get("LUNA_FIELD_IMAGE", "")
    match = os.environ.get("LUNA_FIELD_MATCH", "e")        # "e" or "rho"
    lo = (rho0 if match == "rho" else e_floor)
    hi = lo * float(os.environ.get("LUNA_FIELD_CONTRAST", "4.0"))
    label = os.path.basename(img) if img else (text or "LUNA")
    tgt_np = (_image_density(Nx, img, lo, hi) if img
              else _text_density(Nx, text or "LUNA", lo, hi))
    tgt = jnp.asarray(tgt_np)

    def observe(U):
        return U[0] if match == "rho" else internal_e(U)

    def loss(theta):
        return jnp.mean((observe(simulate(theta)) - tgt) ** 2)

    if jax.process_index() == 0:
        print(f"Blast-field (Horowitz §7.1): Nx={Nx}, FREE per-cell e(x,0) "
              f"({Nx*Nx} params), n_steps={n_steps}, match={match}, "
              f"target={label}", flush=True)

    # ---- init UNIFORM (noise floor, NOT the target) ----
    # sigmoid(-3) ~ 0.047 => e(x,0) ~ e_floor everywhere; small noise breaks
    # symmetry so the optimizer can grow distinct blast peaks.
    rng = np.random.RandomState(0)
    theta = jax.device_put(
        jnp.asarray(-3.0 + 0.3 * rng.standard_normal((Nx, Nx))), field_sharding)

    loss_grad = jax.jit(jax.value_and_grad(loss))
    _, g = loss_grad(theta)
    if jax.process_index() == 0:
        print(f"  grad finite={bool(jnp.all(jnp.isfinite(g)))}", flush=True)

    lr = float(os.environ.get("LUNA_FIELD_LR", "2.0e-1"))
    m = jnp.zeros_like(theta); v = jnp.zeros_like(theta)
    t0 = time.time()
    for it in range(n_iter):
        L_, g = loss_grad(theta)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        bc1 = 1 - 0.9 ** (it + 1); bc2 = 1 - 0.999 ** (it + 1)
        theta = theta - lr * (m / bc1) / (jnp.sqrt(v / bc2) + 1e-8)
        if jax.process_index() == 0 and (it % max(1, n_iter // 12) == 0 or it == n_iter - 1):
            print(f"    it {it:4d}  L={float(L_):.3e}", flush=True)
    if jax.process_index() != 0:
        return
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # ---- time montage: rerun optimized field, save observable vs time ----
    fracs = [0.0, 0.12, 0.30, 0.55, 1.0]
    e0 = e_of(theta)
    frames = []
    for f in fracs:
        ns = int(round(f * n_steps))
        U = build_U(e0)
        def body(U, _): return step_fn(U, dt), None
        if ns > 0:
            U, _ = jax.lax.scan(body, U, None, length=ns)
        frames.append(np.asarray(observe(U)))
    _save_montage(np.asarray(e0), frames, fracs, tgt_np, Nx, label, match)


def _save_montage(e0, frames, fracs, tgt, Nx, label, match):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, FIGSIZE
    apply_style()
    n = len(frames) + 1
    fig, ax = plt.subplots(1, n, figsize=(FIGSIZE[0] * n * 0.62, FIGSIZE[1] * 0.95))
    vmx = float(np.percentile(np.concatenate([f.ravel() for f in frames]), 99.5))
    ax[0].imshow(np.log10(e0).T, origin="lower", cmap="inferno")
    ax[0].text(0.04, 0.96, "optimized e(x,0)", transform=ax[0].transAxes,
               va="top", ha="left", color="white", fontsize=10)
    for a, fr, fc in zip(ax[1:1 + len(frames)], frames, fracs):
        a.imshow(fr.T, origin="lower", cmap="viridis", vmin=tgt.min(), vmax=vmx)
        lab = "merged -> target" if fc == 1.0 else f"t = {fc:.2f} t_end"
        a.text(0.04, 0.96, lab, transform=a.transAxes, va="top", ha="left",
               color="white", fontsize=10)
    for a in ax:
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    safe = label.replace(" ", "").replace("/", "_").split(".")[0]
    out = f"figures/autodiff_blast_field_{safe}_Nx{Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
