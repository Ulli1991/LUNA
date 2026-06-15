"""Turbulence-driven target emergence — diffhydro Fig.12 second row (§7.2) analog.

Where the blast example (§7.1, see blast_field_2d.py) sculpts a target from an
ensemble of colliding Sedov blasts, this example uses a TURBULENT VELOCITY field
as the driver.  The control variable is the initial velocity field v(x,0) of an
isothermal, initially uniform-density gas (band-limited to large scales, so it
reads as a turbulent stirring field rather than per-cell noise).  Reverse-mode AD
through the production 2D solver optimizes v(x,0) so that the gas it stirs and
compresses has a late-time DENSITY matching a target (the word LUNA).

The density starts uniform; convergent (compressive) regions of the optimized
velocity field pile gas up, and the optimizer arranges those convergence zones so
the target emerges purely from the turbulent advection/compression — the velocity
field itself does not spell anything.  This mirrors diffhydro's turbulent-velocity
reconstruction (Fourier-phase optimization against a projected-density target).

Env: LUNA_TURB_TARGET="LUNA" | LUNA_TURB_IMAGE=<path>, LUNA_TURB_NX (256),
     LUNA_TURB_NC (control grid per side, 32 -> band-limit), LUNA_TURB_NSTEPS (300),
     LUNA_TURB_NITER (200), LUNA_TURB_MACH (3.0), LUNA_TURB_CONTRAST (4.0),
     LUNA_TURB_LR (1.5e-1), LUNA_TURB_MESH ("2x2" for 4 GPUs).
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
    Nx = int(os.environ.get("LUNA_TURB_NX", "256"))
    Nc = int(os.environ.get("LUNA_TURB_NC", "32"))       # coarse control -> band-limit
    n_steps = int(os.environ.get("LUNA_TURB_NSTEPS", "300"))
    n_iter = int(os.environ.get("LUNA_TURB_NITER", "200"))
    L, gamma, rho0 = 1.0, 5.0 / 3.0, 1.0
    cs2 = 1.0                                              # isothermal sound speed^2
    mach = float(os.environ.get("LUNA_TURB_MACH", "3.0"))
    v_max = mach * float(np.sqrt(cs2))

    mesh_env = os.environ.get("LUNA_TURB_MESH", "")
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
    field_sharding = (NamedSharding(mesh, P('x', 'y')) if mesh_shape else rep)

    dx = L / Nx
    cs_max = float(np.sqrt(gamma * cs2 + v_max ** 2))      # bound the signal speed
    dt = 0.20 * dx / cs_max                                # global fixed dt, CFL-safe

    def v_of(theta):
        """Band-limited initial velocity from a coarse control field upsampled to
        the grid (large-scale 'turbulent driver'), bounded to |v| <= v_max."""
        # theta: (2, Nc, Nc) -> bilinear upsample -> (2, Nx, Nx), tanh-bounded
        vx = jax.image.resize(theta[0], (Nx, Nx), method="linear")
        vy = jax.image.resize(theta[1], (Nx, Nx), method="linear")
        return v_max * jnp.tanh(vx), v_max * jnp.tanh(vy)

    def build_U(vx, vy):
        rho = jnp.full((Nx, Nx), rho0)
        mom_x = rho0 * vx
        mom_y = rho0 * vy
        ke = 0.5 * rho0 * (vx ** 2 + vy ** 2)
        E = cs2 * rho0 / (gamma - 1.0) + ke                # isothermal P=cs2*rho
        U = jnp.stack([rho, mom_x, mom_y, E], axis=0)
        return jax.lax.with_sharding_constraint(U, sharding)

    def iso_project(U):
        """Operator-split isothermal thermostat: reset P = cs2 * rho each step."""
        rho = U[0]
        ke = 0.5 * (U[1] ** 2 + U[2] ** 2) / rho
        E = cs2 * rho / (gamma - 1.0) + ke
        return U.at[-1].set(E)

    per = max(1, int(round(n_steps ** 0.5)))
    nchunk = (n_steps + per - 1) // per

    def simulate(theta):
        vx, vy = v_of(theta)
        U = build_U(vx, vy)
        def body(U, _):
            return iso_project(step_fn(U, dt)), None
        @jax.checkpoint
        def chunk(U, _):
            U, _ = jax.lax.scan(body, U, None, length=per); return U, None
        Uf, _ = jax.lax.scan(chunk, U, None, length=nchunk)
        return Uf

    # ---- target (matched in DENSITY at late time) ----
    text = os.environ.get("LUNA_TURB_TARGET", "")
    img = os.environ.get("LUNA_TURB_IMAGE", "")
    rho_hi = rho0 * float(os.environ.get("LUNA_TURB_CONTRAST", "4.0"))
    label = os.path.basename(img) if img else (text or "LUNA")
    tgt_np = (_image_density(Nx, img, rho0, rho_hi) if img
              else _text_density(Nx, text or "LUNA", rho0, rho_hi))
    tgt = jnp.asarray(tgt_np)

    def loss(theta):
        return jnp.mean((simulate(theta)[0] - tgt) ** 2)

    if jax.process_index() == 0:
        print(f"Turb-field (Fig12 row 2): Nx={Nx}, control {Nc}x{Nc} velocity "
              f"({2*Nc*Nc} params), Mach={mach}, n_steps={n_steps}, "
              f"target={label}", flush=True)

    # ---- init: small random velocity (NOT target-shaped) ----
    rng = np.random.RandomState(0)
    theta = jax.device_put(jnp.asarray(0.3 * rng.standard_normal((2, Nc, Nc))), rep)
    loss_grad = jax.jit(jax.value_and_grad(loss))
    _, g = loss_grad(theta)
    if jax.process_index() == 0:
        print(f"  grad finite={bool(jnp.all(jnp.isfinite(g)))}", flush=True)

    lr = float(os.environ.get("LUNA_TURB_LR", "1.5e-1"))
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

    # ---- time montage: rerun the optimized velocity field ----
    fracs = [0.0, 0.12, 0.30, 0.55, 1.0]
    vx0, vy0 = v_of(theta)
    speed0 = np.asarray(jnp.sqrt(vx0 ** 2 + vy0 ** 2))
    frames = []
    for f in fracs:
        ns = int(round(f * n_steps))
        U = build_U(vx0, vy0)
        def body(U, _): return iso_project(step_fn(U, dt)), None
        if ns > 0:
            U, _ = jax.lax.scan(body, U, None, length=ns)
        frames.append(np.asarray(U[0]))
    _save_montage(speed0, frames, fracs, tgt_np, Nx, label)


def _save_montage(speed0, frames, fracs, tgt, Nx, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.figstyle import apply_style, FIGSIZE
    apply_style()
    n = len(frames) + 1
    fig, ax = plt.subplots(1, n, figsize=(FIGSIZE[0] * n * 0.62, FIGSIZE[1] * 0.95))
    vmx = float(np.percentile(np.concatenate([f.ravel() for f in frames]), 99.5))
    ax[0].imshow(speed0.T, origin="lower", cmap="cividis")
    ax[0].text(0.04, 0.96, "optimized v(x,0)", transform=ax[0].transAxes,
               va="top", ha="left", color="white", fontsize=10)
    for a, fr, fc in zip(ax[1:1 + len(frames)], frames, fracs):
        a.imshow(fr.T, origin="lower", cmap="viridis", vmin=tgt.min(), vmax=vmx)
        lab = "stirred -> target" if fc == 1.0 else f"t = {fc:.2f} t_end"
        a.text(0.04, 0.96, lab, transform=a.transAxes, va="top", ha="left",
               color="white", fontsize=10)
    for a in ax:
        a.grid(False); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    safe = label.replace(" ", "").replace("/", "_").split(".")[0]
    out = f"figures/autodiff_turb_field_{safe}_Nx{Nx}.png"
    fig.savefig(out); plt.close(fig)
    print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
