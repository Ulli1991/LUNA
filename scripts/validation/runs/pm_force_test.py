"""PM gravity correctness: (1) two-body force vs Newtonian 1/r^2, and (2) CIC
self-force cancellation. Single GPU, no cosmology -- a static accuracy check of
the deposit -> MG-Poisson -> 4th-order-gradient -> gather force pipeline.

  unit box (Lx=1), G=1, point masses m=1; periodic Poisson with the mean density
  subtracted (k=0 = 0), so the near-image pair force ~ Gm/r^2 for r << box.

Env: LUNA_PMF_NX (256).
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravity.particle_mesh import ParticleMeshGravity
from gravity.poisson_multigrid import solve_poisson_fmg
from template_config import Config

Nx = int(os.environ.get("LUNA_PMF_NX", "256"))
cfg = replace(Config(), SPATIAL_DIMENSION=3, Nx=Nx, Ny=Nx, Nz=Nx,
              Lx=1.0, Ly=1.0, Lz=1.0, GRAVITATIONAL_G=1.0,
              BOUNDARY=0, POISSON_BOUNDARY=0, POISSON_FMG_EXTRA_VCYCLES=4)
pm = ParticleMeshGravity(cfg)
dx = pm.dx
fourpiG = 4.0 * np.pi * 1.0
pkw = dict(n_levels=None, n_extra_vcycles=4, n_pre=2, n_post=2, n_coarsest=50)
from gravity.poisson_multigrid import _max_levels
pkw["n_levels"] = _max_levels(Nx)


@jax.jit
def accel(pos, mass):
    """PM acceleration on each particle in `pos` (M,3) with masses `mass` (M,)."""
    active = jnp.ones(pos.shape[0])
    rho = pm.cic_deposit(pos, mass, active)
    src = rho - jnp.mean(rho)                      # periodic solvability (k=0 -> 0)
    phi = solve_poisson_fmg(src, dx, fourpiG, boundary=0, **pkw)
    g = pm.grad_phi(phi, order=4)                  # -grad(phi), 4th order
    return pm.accel_at_points(g, pos)              # (M,3)


# ---- (2) CIC self-force: ONE particle, scan its sub-cell position ----
print("=== self-force cancellation (1 particle) ===", flush=True)
offs = np.linspace(0.0, 1.0, 17)                   # sub-cell offset in units of dx
self_f = []
c = 0.5                                            # centre of the box
for o in offs:
    p = jnp.asarray([[c + o * dx, c, c]])
    a = np.asarray(accel(p, jnp.ones(1)))[0]
    self_f.append(np.linalg.norm(a))
self_f = np.array(self_f)
# reference scale: Newtonian force at one cell, Gm/dx^2
f_cell = 1.0 / dx ** 2
# also sample the self-force at random positions (cancellation must hold anywhere)
rng = np.random.default_rng(0)
rand_pos = rng.uniform(0.2, 0.8, (64, 3))
self_rand = np.array([np.linalg.norm(np.asarray(accel(jnp.asarray([p]), jnp.ones(1)))[0])
                      for p in rand_pos]) / f_cell
print(f"  max |a_self|/(Gm/dx^2): sub-cell scan {self_f.max()/f_cell:.2e}, "
      f"random {self_rand.max():.2e}  (median {np.median(self_rand):.2e})", flush=True)

# ---- (1) two-body force vs 1/r^2 ----
print("=== two-body force vs Newtonian 1/r^2 ===", flush=True)
rs_cells = np.unique(np.round(np.geomspace(0.5, Nx / 8, 28)).astype(int))
rs_cells = rs_cells[rs_cells >= 1]
r_dx, f_pm, f_newt = [], [], []
for rc in rs_cells:
    r = rc * dx
    pA = [c, c, c]; pB = [c + r, c, c]
    pos = jnp.asarray([pA, pB])
    a = np.asarray(accel(pos, jnp.ones(2)))
    fA = np.linalg.norm(a[0])                       # |accel on A| ~ Gm/r^2 from B
    r_dx.append(rc); f_pm.append(fA); f_newt.append(1.0 / r ** 2)
r_dx = np.array(r_dx); f_pm = np.array(f_pm); f_newt = np.array(f_newt)
ratio = f_pm / f_newt
# effective softening: where PM force drops to half Newtonian
half = np.interp(0.5, ratio[np.argsort(r_dx)], r_dx[np.argsort(r_dx)]) if (ratio.min() < 0.5 < ratio.max()) else np.nan
print(f"  force/Newtonian: at r=2dx {np.interp(2,r_dx,ratio):.3f}, "
      f"r=4dx {np.interp(4,r_dx,ratio):.3f}, r=8dx {np.interp(8,r_dx,ratio):.3f}", flush=True)
print(f"  ~half-force radius (effective softening) ~ {half:.2f} cells", flush=True)

# ---- figure 1: two-body force vs 1/r^2 ----
os.makedirs("code_paper_figs", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(r_dx, f_pm, 'o-', ms=4, label='LUNA PM force')
ax[0].loglog(r_dx, f_newt, 'k--', label=r'Newtonian $Gm/r^2$')
ax[0].axvline(2, color='grey', ls=':', lw=0.8)
ax[0].set_xlabel('separation $r$ [cells]'); ax[0].set_ylabel('$|\\mathbf{F}|$')
ax[0].legend(frameon=False); ax[0].set_title('two-body force vs $1/r^2$')
ax[1].semilogx(r_dx, ratio, 'o-', ms=4); ax[1].axhline(1.0, color='k', ls='--')
ax[1].axvline(2, color='grey', ls=':', lw=0.8)
ax[1].set_xlabel('separation $r$ [cells]'); ax[1].set_ylabel('PM / Newtonian')
ax[1].set_ylim(0, 1.3); ax[1].set_title('force ratio (1 = exact $1/r^2$)')
fig.tight_layout()
out1 = f"code_paper_figs/pm_force_1overr2_Nx{Nx}.png"
fig.savefig(out1, dpi=130); plt.close(fig); print(f"  wrote {out1}", flush=True)

# ---- figure 2: CIC self-force cancellation ----
fig, ax = plt.subplots(figsize=(6, 4))
ax.semilogy(offs, np.maximum(self_f / f_cell, 1e-16), 'o-', ms=4,
            label='single particle, sub-cell scan along $x$')
ax.scatter(rng.uniform(0, 1, self_rand.size), np.maximum(self_rand, 1e-16),
           s=10, c='C1', alpha=0.6, label='64 random positions')
ax.set_xlabel('sub-cell offset [cells]')
ax.set_ylabel(r'$|\mathbf{F}_{\rm self}| \, / \, (Gm/\Delta x^2)$')
ax.set_title('CIC self-force cancellation')
ax.legend(frameon=False); ax.grid(True, which='both', alpha=0.3)
fig.tight_layout()
out2 = f"code_paper_figs/pm_self_force_Nx{Nx}.png"
fig.savefig(out2, dpi=130); plt.close(fig); print(f"  wrote {out2}", flush=True)

np.savez(f"code_paper_figs/pm_force_test_Nx{Nx}.npz", r_dx=r_dx, f_pm=f_pm, f_newt=f_newt,
         self_offsets=offs, self_force=self_f, self_rand=self_rand, f_cell=f_cell)
print("=== done ===", flush=True)
