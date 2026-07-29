"""Level-by-level Dirichlet self-gravity on the AMR hierarchy (amr/poisson.py).

Anchors:
  1. Level-0-only hierarchy -> phi is EXACTLY the dense FMG solve (same solver).
  2. Two-level hierarchy on an analytic problem: rho = A sin(kx)sin(ky)sin(kz)
     has phi = -4*pi*G*A*sin*sin*sin / (3 k^2) (periodic, zero-mean). Every
     level's phi must match the analytic solution to truncation accuracy.
  3. The per-level kick reproduces the dense self-gravity kick formula.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from amr.config import AMRConfig
from amr import driver
from amr.poisson import apply_gravity_kick, solve_phi_hierarchy
from amr.state import assemble_base_level

FOURPIG = 4.0 * np.pi
N = 32
BOX = 1.0


def _sine_U(n):
    """Conservative hydro field whose density is the zero-mean sine mode
    (offset to stay positive), on an n^3 grid."""
    x = (jnp.arange(n) + 0.5) * (BOX / n)
    k = 2 * jnp.pi / BOX
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    rho = 1.0 + 0.5 * jnp.sin(k * X) * jnp.sin(k * Y) * jnp.sin(k * Z)
    U = jnp.zeros((5, n, n, n))
    U = U.at[0].set(rho)
    U = U.at[4].set(1.0)                                  # arbitrary energy
    return U


def _phi_analytic(coords_ijk, dx):
    """Analytic phi at the cell centres of index arrays (i,j,k) with spacing dx.
    For rho = 1 + 0.5 sin sin sin (periodic, Jeans-subtracted mean=1):
    lap(phi) = 4piG * 0.5 sin sin sin -> phi = -4piG*0.5/(3k^2) sin sin sin."""
    k = 2 * np.pi / BOX
    i, j, l = coords_ijk
    x, y, z = (i + 0.5) * dx, (j + 0.5) * dx, (l + 0.5) * dx
    amp = -FOURPIG * 0.5 / (3 * k * k)
    return amp * np.sin(k * x) * np.sin(k * y) * np.sin(k * z)


def _hierarchy(max_levels, thresholds):
    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=max_levels, boundary=0)
    h = driver.build_hierarchy(_sine_U(N), cfg, thresholds=thresholds,
                               criterion="density", periodic=True)
    return h, cfg


def test_l0_only_matches_dense_fmg():
    from gravity.poisson_multigrid import solve_poisson_fmg
    h, cfg = _hierarchy(0, [])
    phis = solve_phi_hierarchy(h, cfg, fourpiG=FOURPIG, boundary=0)
    phi_amr = np.asarray(assemble_base_level(phis[0][:8], cfg)[0])
    rho = np.asarray(assemble_base_level(h[0].data[:8, 0:1], cfg)[0])
    # reference does the Jeans subtraction the dense caller does
    phi_ref = np.asarray(solve_poisson_fmg(jnp.asarray(rho - rho.mean()),
                                           BOX / N, FOURPIG, 0))
    assert np.allclose(phi_amr, phi_ref, atol=1e-12)
    # and it must be the physical solution, not shared garbage
    amp = FOURPIG * 0.5 / (3 * (2 * np.pi) ** 2)
    assert 0.5 * amp < np.abs(phi_amr).max() < 2.0 * amp


def test_two_level_phi_matches_analytic():
    # tiny threshold -> everything refines (uniform 2x coverage, exercises the
    # coarse-Dirichlet fill at the level-1 domain edge via the parent's
    # periodic halo)
    h, cfg = _hierarchy(1, [1e-9])
    assert int(h[1].n_active) > 0
    phis = solve_phi_hierarchy(h, cfg, fourpiG=FOURPIG, boundary=0, n_vcycles=3)

    for L, (lvl, phi) in enumerate(zip(h, phis)):
        na = int(lvl.n_active)
        dx = BOX / (N * 2 ** L)
        coords = np.asarray(lvl.coords[:na])
        ph = np.asarray(phi[:na, 0])
        bs = 16
        io = np.arange(bs)
        II, JJ, KK = np.meshgrid(io, io, io, indexing="ij")
        errs = []
        for b in range(na):
            gi = coords[b][:, None, None, None] * bs + np.stack([II, JJ, KK])
            ref = _phi_analytic((gi[0], gi[1], gi[2]), dx)
            errs.append(np.abs(ph[b] - ref).max())
        scale = FOURPIG * 0.5 / (3 * (2 * np.pi) ** 2)   # phi amplitude
        rel = max(errs) / scale
        # truncation of the 7-point Laplacian at 32^3 is ~(k*dx)^2/12 ~ 0.3%;
        # allow headroom for the Dirichlet chain + finite sweeps.
        assert rel < 0.02, f"level {L}: rel phi error {rel:.3e}"


def test_kick_matches_dense_formula():
    h, cfg = _hierarchy(0, [])
    phis = solve_phi_hierarchy(h, cfg, fourpiG=FOURPIG, boundary=0)
    dt = 1e-3
    out = apply_gravity_kick(h, phis, dt, cfg)

    U = np.asarray(assemble_base_level(h[0].data[:8], cfg))
    phi = np.asarray(assemble_base_level(phis[0][:8], cfg)[0])
    dx = BOX / N
    g = [-(np.roll(phi, -1, axis=d) - np.roll(phi, 1, axis=d)) / (2 * dx)
         for d in range(3)]
    rho = U[0]
    for d in range(3):
        m_old = U[1 + d]
        m_new = m_old + dt * rho * g[d]
        got = np.asarray(assemble_base_level(out[0].data[:8], cfg))[1 + d]
        assert np.allclose(got, m_new, atol=1e-12), f"momentum axis {d}"


def test_synchronized_advance_reflux_conserves():
    """Berger-Colella reflux: composite mass and energy are conserved to machine
    precision across a multi-level synchronized advance (periodic box, flowing
    gas, partial refinement -> real coarse-fine faces). Without reflux the C/F
    flux mismatch leaks conserved quantities; with it they telescope exactly."""
    cfg = AMRConfig(ndim=3, box=(BOX,) * 3, base_grid=(N,) * 3,
                    block_shape=(16,) * 3, max_levels=1, boundary=0)
    U = _sine_U(N)
    U = U.at[1].set(0.2 * U[0])                 # vx = 0.2: flow across C/F faces
    U = U.at[4].set(U[4] + 0.5 * 0.2 ** 2 * U[0])
    h = driver.build_hierarchy(U, cfg, thresholds=[1.2], criterion="overdensity",
                               periodic=True)
    na = int(h[1].n_active)
    assert 0 < na < 8 * 8, f"want PARTIAL refinement, got {na} fine blocks"

    def totals(hier):
        comp = driver.composite_to_dense(hier, cfg)
        dv = (BOX / (N * 2)) ** 3
        return np.array([float(jnp.nansum(comp[0])) * dv,
                         float(jnp.nansum(comp[4])) * dv])

    t0 = totals(h)
    step = jax.jit(lambda lv, dt: driver.advance(lv, dt, cfg, gamma=5 / 3,
                                                 limiter=1, riemann_solver="hllc"))
    for _ in range(10):
        dt = 0.5 * float(driver.compute_dt(h, cfg, gamma=5 / 3, CFL=0.4))
        h = step(h, dt)
    t1 = totals(h)
    drift = np.abs(t1 - t0) / np.abs(t0)
    assert drift[0] < 1e-12, f"mass drift {drift[0]:.3e}"
    assert drift[1] < 1e-12, f"energy drift {drift[1]:.3e}"
