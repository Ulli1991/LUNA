"""Agertz et al. (2007) blob test (G2): a chi = 10 cloud at rest in a Mach 2.7 wind, periodic {10, 10, 40} R. The cloud
mass (T < 0.9 T_ext, rho > 0.64 rho_cl) against t / tau_KH, tau_KH = 1.6 t_cc: the grid codes keep it to ~tau_KH
(ablation), then lose all of it by 2.5 tau_KH (their Fig. 6, 8), at 64, 128, 256 cells across the box.
No analytic or converged solution: the bounds are LUNA's own runs (a regression), no AthenaK reference yet."""
import functools

import numpy as np
import pytest

from tests._helpers import accuracy_cfg, check, evolve, gpu_only

FRACS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)    # t / tau_KH
SCHEMES = {"hllc-minmod": 0, "hllc-mc": 1}
NS_CPU, NS_GPU = (32,), (64, 128, 256)                            # cells across the box (Lx = 10 R)
# cloud fraction at FRACS, A100 FP64 (jobs 31344274-80; N = 32 on CPU, to 0.5 tau_KH)
MEASURED = {
    ("hllc-minmod", 32): (0.7331, 0.5382),
    ("hllc-mc", 32): (0.7757, 0.6431),
    ("hllc-minmod", 64): (0.8205, 0.7267, 0.5591, 0.3727, 0.3194, 0.0025, 0.0, 0.0, 0.0, 0.0),
    ("hllc-mc", 64): (0.8379, 0.7366, 0.6097, 0.4753, 0.4340, 0.1901, 0.0076, 0.0, 0.0, 0.0),
    ("hllc-minmod", 128): (0.8305, 0.7729, 0.6807, 0.5087, 0.4828, 0.2912, 0.0018, 0.0, 0.0, 0.0),
    ("hllc-mc", 128): (0.8448, 0.8004, 0.7282, 0.5558, 0.5924, 0.4205, 0.1616, 0.0491, 0.0, 0.0),
    ("hllc-minmod", 256): (0.8303, 0.8034, 0.7398, 0.5658, 0.5899, 0.4766, 0.2162, 0.0456, 0.0193, 0.0148),
    ("hllc-mc", 256): (0.8465, 0.8275, 0.7817, 0.6350, 0.6761, 0.5435, 0.3446, 0.2235, 0.1415, 0.0918),
}


def _bounds(f, m):
    """Ablation (to tau_KH) is deterministic: +-15 %; the break-up after it is chaotic (phase changes with N, as
    in the paper's Fig. 7): unbounded, except that by 2.5 tau_KH the cloud is (nearly) gone, as in every grid code."""
    if f <= 1.0:
        return 0.85 * m, 1.15 * m
    if f == FRACS[-1]:
        return 0.0, max(1.3 * m, 0.02)
    return None


BOUNDS = {k: tuple(_bounds(f, m) for f, m in zip(FRACS, v)) for k, v in MEASURED.items()}


def cfg_for(n, scheme, **kw):
    from luna.ic_generation import agertz_blob as B
    base = dict(test_problem="Agertz-Blob-3D", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=4 * n, Lx=10.0, Ly=10.0, Lz=40.0,
                BOUNDARY=0, gamma=5.0 / 3.0, LIMITER=SCHEMES[scheme], hydro_kernel="auto")
    cfg = accuracy_cfg(**{**base, **kw})
    return cfg, 1.6 * B.t_cc(cfg)


@functools.lru_cache(maxsize=None)
def history(n, scheme, fracs=FRACS):
    """(cloud mass / initial at fracs, steps) through `evolve` (mass, energy, momentum to round-off: periodic)."""
    import jax

    from luna.ic_generation import agertz_blob as B
    from luna.ic_generation.ic import initial_conditions
    cfg, tau = cfg_for(n, scheme)
    U0 = initial_conditions(cfg)
    m0 = float(B.cloud_mass(U0, cfg))
    obs = jax.jit(lambda U: B.cloud_mass(U, cfg))
    masses, steps = evolve(cfg, np.asarray(U0), [f * tau for f in fracs], observe=lambda U: float(obs(U)))
    return tuple(m / m0 for m in masses), steps


def _check(n, scheme, fr):
    for f, m, b in zip(FRACS, fr, BOUNDS[(scheme, n)]):
        print(f"MEASURED blob {scheme} N={n} t/tau_KH={f} cloud fraction {m:.4f}")
        if b is not None:
            check(f"blob {scheme} N={n} t/tau_KH={f} cloud fraction", m, b[0], lower=True)
            check(f"blob {scheme} N={n} t/tau_KH={f} cloud fraction", m, b[1])


def lattice_cloud_mass(n, chi=10.0, s=4, cut=0.64):
    """Independent numpy count: volume fractions from s^3 points per cell, mass of the cells with rho > cut chi."""
    h = 10.0 / n
    off = (np.arange(s) + 0.5) / s
    c = (np.arange(n)[:, None] + off[None, :]) * h - 5.0           # (cell, sub-point) coordinate about the centre
    zc = (np.arange(4 * n)[:, None] + off[None, :]) * 40.0 / (4 * n) - 5.0
    r2 = (c[:, None, None, :, None, None] ** 2 + c[None, :, None, None, :, None] ** 2
          + zc[None, None, :, None, None, :] ** 2)
    f = (r2 < 1.0).mean(axis=(3, 4, 5))
    rho = 1.0 + (chi - 1.0) * f
    return float(rho[rho > cut * chi].sum() * h ** 3)


# The 32-cell lattice's cloud against the sphere chi (4/3) pi R^3: a property of the grid (3.2 cells per R) and the
# 0.64 chi cut, not of the solver; measured 0.0936 (37.968 against 41.888; CPU-8 FP64, job 31655039), bounded at 1.3x
SPHERE_BOUND = 0.122


def test_initial_cloud_is_the_sphere():
    """The criterion's cloud equals an independent numpy count of the IC's cells to round-off, and the sphere's mass."""
    from luna.ic_generation import agertz_blob as B
    from luna.ic_generation.ic import initial_conditions
    cfg, _ = cfg_for(32, "hllc-minmod")
    m = float(B.cloud_mass(initial_conditions(cfg), cfg))
    ref = lattice_cloud_mass(32)
    exact = 10.0 * 4.0 / 3.0 * np.pi
    print(f"MEASURED blob initial cloud mass {m:.10g} lattice count {ref:.10g} sphere {exact:.10g}")
    # ~1e3 cells summed in FP64: round-off ~1e3 eps ~ 2e-13 at worst
    check("blob initial cloud mass vs numpy lattice count", abs(m - ref) / ref, 1e-12)
    check("blob initial cloud mass vs sphere (lattice error)", abs(m - exact) / exact, SPHERE_BOUND)
    # teeth: a criterion cut at 0.8 chi instead of 0.64 chi would pick another mass (0.5 chi picks the same cells)
    check("blob lattice count with a 0.8 chi cut vs the criterion", abs(lattice_cloud_mass(32, cut=0.8) - m) / m,
          1e-12, lower=True)


@pytest.mark.parametrize("scheme", list(SCHEMES))
@pytest.mark.parametrize("n", NS_CPU)
def test_blob_cpu(n, scheme):
    fr, _ = history(n, scheme, FRACS[:2])
    _check(n, scheme, fr)


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
@pytest.mark.parametrize("n", NS_GPU)
def test_blob_gpu(n, scheme):
    fr, _ = history(n, scheme)
    _check(n, scheme, fr)


@gpu_only
@pytest.mark.parametrize("scheme", list(SCHEMES))
def test_blob_keeps_more_cloud_with_resolution_gpu(scheme):
    """The cloud fraction at tau_KH grows with N (the paper's Fig. 8: 'increases a little but converges'), and MC
    keeps more than minmod at every N."""
    at = FRACS.index(1.0)
    fr = [history(n, scheme)[0][at] for n in NS_GPU]
    print(f"MEASURED blob {scheme} cloud fraction at tau_KH, N = {NS_GPU}: {fr}")
    assert fr[0] < fr[1] < fr[2], f"{scheme}: cloud fraction at tau_KH not increasing with N: {fr}"
    if scheme == "hllc-mc":
        mm = [history(n, "hllc-minmod")[0][at] for n in NS_GPU]
        assert all(a > b for a, b in zip(fr, mm)), f"MC {fr} does not keep more than minmod {mm}"
