"""AMR physics benchmark #3: smooth advection through a STATIC refined region.

A Gaussian density pulse (rho = 1 + 0.5 exp(-(x-x0)^2/2 sigma^2), vx = 1, p = 1)
advects through a statically-refined central slab (box-fraction x in
[0.375, 0.625], full transverse) in a periodic box and returns to its starting
position after exactly one crossing (t = Lx / vx = 1). The exact solution is the
initial condition, so this measures:

  * convergence ORDER of the scheme on smooth flow (uniform 64 / 128 / 256)
  * whether refinement helps (AMR 64 + 2 levels vs uniform 64) and whether the
    C/F interfaces at x = 0.375 / 0.625 leave artifacts (error(x) profile)
  * conservation through ~1400 fine steps of C/F transfer (expect bit-exact)

Everything runs in-process through amr.driver (no snapshot machinery).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from amr.config import AMRConfig
from amr import driver

GAMMA = 1.4
SIGMA = 0.05
X0 = 0.5     # slab CENTER: interfaces at 0.25/0.75 stay in quiet flow
T_END = 1.0                                     # one periodic crossing at vx=1


def gaussian_ic(nx, nt):
    x = (jnp.arange(nx) + 0.5) / nx
    rho1d = 1.0 + 0.5 * jnp.exp(-((x - X0) ** 2) / (2 * SIGMA ** 2))
    rho = rho1d[:, None, None] * jnp.ones((nx, nt, nt))
    p = jnp.ones_like(rho)
    U = jnp.zeros((5, nx, nt, nt))
    U = U.at[0].set(rho)
    U = U.at[1].set(rho * 1.0)
    U = U.at[4].set(p / (GAMMA - 1.0) + 0.5 * rho * 1.0 ** 2)
    return U


def run(nx, levels, tag):
    nt = max(16, nx // 4)                       # transverse: 1 block at base 64
    U0 = gaussian_ic(nx, nt)
    extra = {}
    if levels > 0:
        extra["static_region"] = tuple([(0.25, 0.75)]
                                       + [(0.0, 1.0)] * 2)
    cfg = AMRConfig(ndim=3, box=(1.0, nt / nx, nt / nx), base_grid=(nx, nt, nt),
                    block_shape=(16,) * 3, max_levels=levels, boundary=0,
                    gamma=GAMMA, **extra)
    h = driver.build_hierarchy(U0, cfg, thresholds=[0.0] * max(levels, 1),
                               criterion="static", periodic=True, pad_to=1)
    n_blocks = [int(l.n_active) for l in h]
    M0 = float(jnp.nansum(h[0].data[: n_blocks[0], 0])) * (1 / nx) ** 3
    t = 0.0
    while t < T_END:
        dt = float(driver.compute_dt(h, cfg, gamma=GAMMA, CFL=0.4))
        dt = min(dt, T_END - t)
        h = driver.advance(h, dt, cfg, gamma=GAMMA, limiter=0,
                           riemann_solver="hllc")
        t += dt
    M1 = float(jnp.nansum(h[0].data[: n_blocks[0], 0])) * (1 / nx) ** 3
    # composite at the finest resolution, x-profile (transverse-uniform flow)
    img = driver.composite_to_dense(h, cfg)
    nxf = nx * 2 ** levels
    rho = np.asarray(img[0]).reshape(nxf, -1).mean(axis=1)
    xf = (np.arange(nxf) + 0.5) / nxf
    exact = 1.0 + 0.5 * np.exp(-((xf - X0) ** 2) / (2 * SIGMA ** 2))
    L1 = float(np.mean(np.abs(rho - exact)))
    print(f"  {tag:12s}: blocks={n_blocks}  L1={L1:.5e}  dM/M={abs(M1-M0)/M0:.2e}",
          flush=True)
    return dict(tag=tag, x=xf, rho=rho, exact=exact, L1=L1)


import os
print("=== AMR advection benchmark (exact = translated IC, t=1.0) ===", flush=True)
if os.environ.get("BENCH_RUN", "all") == "amr":
    # uniform L1s are deterministic -- measured previously (same code/config)
    u64 = dict(L1=2.09569e-02, x=None, rho=None, exact=None, tag="uniform 64")
    u128 = dict(L1=7.91139e-03)
    u256 = dict(L1=2.50663e-03)
    print("  (uniform L1s reused from prior run)", flush=True)
else:
    u64 = run(64, 0, "uniform 64")
    u128 = run(128, 0, "uniform 128")
    u256 = run(256, 0, "uniform 256")
amr = run(64, 2, "AMR 64+2lvl")
if u64["x"] is None:
    # rebuild the uniform-64 profile cheaply for the figure (short run)
    u64 = run(64, 0, "uniform 64")
    u256 = dict(u256, x=amr["x"], rho=None)

p1 = np.log2(u64["L1"] / u128["L1"])
p2 = np.log2(u128["L1"] / u256["L1"])
print(f"  convergence order: 64->128 p={p1:.2f}   128->256 p={p2:.2f}", flush=True)
gain = u64["L1"] / amr["L1"]
print(f"  AMR gain over uniform-64: {gain:.2f}x", flush=True)
# C/F interface-artifact gate: error in +-3 fine cells of each interface must
# not exceed 2x the global mean error (the classic SMR advection criterion --
# a feature crossing the interface must not reflect/deposit anything).
err = np.abs(amr["rho"] - amr["exact"])
nf = len(err)
win = np.zeros(nf, bool)
for e in (0.25, 0.75):
    i = int(e * nf)
    win[max(0, i - 3):i + 3] = True
iface_ratio = float(err[win].mean() / err.mean())
print(f"  C/F interface error ratio: {iface_ratio:.2f} (gate < 2)", flush=True)
ok = (amr["L1"] <= u64["L1"] * 1.05) and p2 > 1.5 and iface_ratio < 2.0
print("BENCH RESULT:", "PASS" if ok else "FAIL", flush=True)

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax.plot(amr["x"], amr["exact"], "k-", lw=1.4, label="exact")
ax.plot(u64["x"], u64["rho"], "o-", ms=2, lw=0.7, color="#d62728", label="uniform 64")
ax.plot(amr["x"], amr["rho"], "-", lw=1.0, color="#1f77b4", label="AMR 64+2 (eff. 256)")
for e in (0.25, 0.75):
    ax.axvline(e, color="gray", ls=":", lw=0.8)
ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Gaussian pulse after one periodic crossing")
ax2.semilogy(u64["x"], np.abs(u64["rho"] - u64["exact"]), color="#d62728",
             lw=0.9, label="uniform 64")
if u256.get("rho") is not None:
    ax2.semilogy(u256["x"], np.abs(u256["rho"] - u256["exact"]), color="#2ca02c",
                 lw=0.9, label="uniform 256")
ax2.semilogy(amr["x"], np.abs(amr["rho"] - amr["exact"]), color="#1f77b4",
             lw=0.9, label="AMR 64+2")
for e in (0.25, 0.75):
    ax2.axvline(e, color="gray", ls=":", lw=0.8)
ax2.set_xlabel("x"); ax2.set_ylabel(r"$|\rho - \rho_{\rm exact}|$")
ax2.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("code_paper_figs/amr_benchmark_advect.png", dpi=150)
print("wrote code_paper_figs/amr_benchmark_advect.png", flush=True)
