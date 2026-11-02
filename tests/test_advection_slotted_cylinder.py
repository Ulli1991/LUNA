"""Rotating slotted cylinder by linear advection (Palafoutas, Velasco Romero & Teyssier 2024, sec. 4.4)."""
import numpy as np
import pytest

from tests._helpers import check, gpu_only

# {N: (L1, L2) bound} ~1.3x measured and (order lo, hi) = measured -0.1, +0.3 for L1 then L2
# (CPU, FP64, CFL 0.8, one turn). Discontinuous data: expected orders ~2/3 (L1), ~1/3 (L2).
BOUNDS = {
    0: ({64: (0.234, 0.334), 128: (0.172, 0.281), 256: (0.113, 0.222)}, (0.43, 0.83), (0.19, 0.59)),
    # minmod: measured L1 0.180, 0.132, 0.0870; L2 0.257, 0.216, 0.171; orders 0.526, 0.294
    1: ({64: (0.164, 0.283), 128: (0.108, 0.224), 256: (0.065, 0.170)}, (0.57, 0.97), (0.27, 0.67)),
    # MC: measured L1 0.126, 0.0829, 0.0500; L2 0.217, 0.172, 0.130; orders 0.668, 0.368
    2: ({64: (0.185, 0.298), 128: (0.124, 0.239), 256: (0.0764, 0.184)}, (0.54, 0.94), (0.25, 0.65)),
    # van Leer: measured L1 0.142, 0.0957, 0.0588; L2 0.230, 0.184, 0.142; orders 0.638, 0.347
}
# Maximum principle, (min lower bound, max - 1 upper bound) over the ladder. minmod keeps it as the
# paper's MUSCL-Hancock does; MC over- and undershoots (measured -3.22e-3, +2.68e-3 at 256).
BOUNDS_MP = {0: (0.0, 0.0), 1: (-4.2e-3, 3.5e-3), 2: (-1e-8, 0.0)}



def rotate(n, limiter, turns=1.0):
    """(u after `turns` turns, u0, cell centres) on [-1, 1]^2 with n x n cells."""
    import jax.numpy as jnp
    from luna.hydro.advection import make_advection_2d
    from luna.ic_generation.rotation_slotted_disk import rotation, slotted_cylinder
    c = -1.0 + (np.arange(n) + 0.5) * (2.0 / n)
    X, Y = np.meshgrid(c, c, indexing="ij")
    u0 = jnp.asarray(slotted_cylinder(X, Y))
    compute_dt, step = make_advection_2d(n, -1.0, 1.0, rotation, limiter=limiter)
    T, t, u = turns * 2 * np.pi, 0.0, u0
    while t < T - 1e-12:
        dt = min(float(compute_dt(u)), T - t)
        u, t = step(u, dt), t + dt
    return np.asarray(u), np.asarray(u0), c


def rotate_fixed(n, limiter, turns):
    """(u, u0, steps, wall s): `turns` turns at the CFL-0.8 dt shrunk to T / steps, one fori_loop."""
    import time

    import jax
    import jax.numpy as jnp
    from luna.hydro.advection import make_advection_2d
    from luna.ic_generation.rotation_slotted_disk import rotation, slotted_cylinder
    c = -1.0 + (np.arange(n) + 0.5) * (2.0 / n)
    X, Y = np.meshgrid(c, c, indexing="ij")
    u0 = jnp.asarray(slotted_cylinder(X, Y))
    compute_dt, step = make_advection_2d(n, -1.0, 1.0, rotation, limiter=limiter)
    T = turns * 2 * np.pi
    k = int(np.ceil(T / float(compute_dt(u0)) - 1e-9))    # the velocity is fixed: dt is too
    run = jax.jit(lambda u, dt: jax.lax.fori_loop(0, k, lambda i, v: step(v, dt), u))
    dt = jnp.asarray(T / k)
    run = run.lower(u0, dt).compile()
    t0 = time.perf_counter()
    u = run(u0, dt).block_until_ready()
    return np.asarray(u), np.asarray(u0), k, time.perf_counter() - t0


def norms(u, u0, n):
    """(L1, L2) of u - u0 as integrals over the domain."""
    e, a = u - u0, (2.0 / n) ** 2
    return float(a * np.abs(e).sum()), float(np.sqrt(a * (e ** 2).sum()))


@pytest.mark.parametrize("limiter", list(BOUNDS), ids=["minmod", "mc", "vanleer"])
def test_one_turn_ladder(limiter):
    """L1 and L2 at every resolution, their orders, and the maximum principle."""
    bnd, (l1_lo, l1_hi), (l2_lo, l2_hi) = BOUNDS[limiter]
    lo, hi = BOUNDS_MP[limiter]
    ns, e1, e2 = sorted(bnd), [], []
    for n in ns:
        u, u0, _ = rotate(n, limiter)
        l1, l2 = norms(u, u0, n)
        check(f"lim{limiter} N={n} L1", l1, bnd[n][0])
        check(f"lim{limiter} N={n} L2", l2, bnd[n][1])
        check(f"lim{limiter} N={n} min", float(u.min()), lo, lower=True)
        check(f"lim{limiter} N={n} max-1", float(u.max()) - 1.0, hi)
        e1.append(l1)
        e2.append(l2)
    for name, e, lo_o, hi_o in (("L1", e1, l1_lo, l1_hi), ("L2", e2, l2_lo, l2_hi)):
        order = -np.polyfit(np.log(ns), np.log(e), 1)[0]
        check(f"lim{limiter} {name} order", order, lo_o, lower=True)
        check(f"lim{limiter} {name} order", order, hi_o)


def test_bound_catches_a_short_turn():
    """Teeth: stopped 5% short of the turn (18 degrees), minmod at 128 misses its L1 bound."""
    u, u0, _ = rotate(128, 0, turns=0.95)
    l1 = norms(u, u0, 128)[0]
    assert l1 > BOUNDS[0][0][128][0], f"L1 {l1:.4e} of a short turn passes the bound"


def test_slot_depth_matches_the_paper():
    """MC at 256: the y = 0.5 cut through the slot sits near the ~0.4 of Palafoutas+2024 fig. 11's
    MUSCL-Hancock curve (read off the figure; measured 0.424)."""
    u, _, c = rotate(256, 1)
    slot = u[np.argmin(abs(c)), np.argmin(abs(c - 0.5))]
    check("MC N=256 slot depth", float(slot), 0.35, lower=True)
    check("MC N=256 slot depth", float(slot), 0.50)


# GPU tier: 512-2048 after 1 and 10 turns (the paper runs 10 periods at N = 256, figs. 11-13), fixed
# dt = T / steps (rotate_fixed). {(limiter, turns): bounds}, A100, FP64, CFL 0.8: L1, L2 ~1.3x measured;
# Linf 1.1x (it is O(1) at every N, the discontinuity is never resolved; no convergence claimed);
# orders fitted over 512-2048, measured -0.1, +0.3; mass |dM|/M floored at 1e-12 (round-off), above
# it the outflow through the zero ghosts once the diffused tail reaches the walls (minmod); mp = (min
# lower bound, max - 1 upper bound) over the ladder; slot: u at (0, 0.5) inside the slot, upper bound,
# floored at 1e-3.
NS_GPU = (512, 1024, 2048)
GPU_BOUNDS = {
    (0, 1): dict(L1={512: 0.076, 1024: 0.048, 2048: 0.03}, L2={512: 0.18, 1024: 0.14, 2048: 0.11},
                 Linf={512: 0.89, 1024: 0.88, 2048: 0.88}, L1_order=(0.58, 0.98), L2_order=(0.29, 0.69),
                 mass={512: 3.7e-10, 1024: 1e-12, 2048: 1e-12}, mp=(0.0, 0.0),
                 slot={512: 0.62, 1024: 0.28, 2048: 0.064}),
    # minmod measured L1 0.0581, 0.0364, 0.0227; L2 0.136, 0.101, 0.0788; Linf 0.808, 0.795, 0.794;
    # orders 0.679, 0.392; dM/M 2.82e-10, 3.45e-15, 4.23e-14; min 4.94e-324, max-1 -2.57e-13;
    # slot 0.476, 0.208, 0.0487; steps 4014, 8035, 16078; A100 0.16, 1.28, 9.01 s
    (0, 10): dict(L1={512: 0.15, 1024: 0.11, 2048: 0.066}, L2={512: 0.26, 1024: 0.21, 2048: 0.17},
                  Linf={512: 0.89, 1024: 0.9, 2048: 0.9}, L1_order=(0.47, 0.87), L2_order=(0.23, 0.63),
                  mass={512: 1.7e-4, 1024: 6.2e-7, 2048: 2.9e-11}, mp=(0.0, 0.0),
                  slot={512: 0.99, 1024: 0.76, 2048: 0.49}),
    # minmod measured L1 0.113, 0.0783, 0.0508; L2 0.196, 0.161, 0.124; Linf 0.809, 0.818, 0.813;
    # orders 0.574, 0.329; dM/M 1.29e-4, 4.70e-7, 2.22e-11 (walls); min 9.75e-305, max-1 -2.32e-5;
    # slot 0.756, 0.583, 0.371; steps 40134, 80347, 160772; A100 1.55, 12.6, 89.5 s
    (1, 1): dict(L1={512: 0.038, 1024: 0.023, 2048: 0.014}, L2={512: 0.13, 1024: 0.096, 2048: 0.075},
                 Linf={512: 0.88, 1024: 0.85, 2048: 0.86}, L1_order=(0.62, 1.02), L2_order=(0.26, 0.66),
                 mass={512: 1e-12, 1024: 1e-12, 2048: 1e-12}, mp=(-8.7e-3, 7.2e-3),
                 slot={512: 0.19, 1024: 3.4e-3, 2048: 1e-3}),
    # MC measured L1 0.0289, 0.0174, 0.0107; L2 0.0946, 0.0735, 0.0576; Linf 0.797, 0.771, 0.780;
    # orders 0.716, 0.358; dM/M 7.06e-15, 3.64e-14, 1.08e-13; min -6.62e-3, max-1 5.51e-3 (both at
    # 2048: MC's over/undershoot grows with N); slot 0.141, 0.0026, < 5e-5; steps as minmod;
    # A100 0.16, 1.29, 9.03 s
    (1, 10): dict(L1={512: 0.077, 1024: 0.047, 2048: 0.029}, L2={512: 0.19, 1024: 0.14, 2048: 0.11},
                  Linf={512: 0.94, 1024: 0.91, 2048: 0.9}, L1_order=(0.60, 1.00), L2_order=(0.29, 0.69),
                  mass={512: 4.5e-11, 1024: 1e-12, 2048: 1e-12}, mp=(-5.6e-3, 1.1e-2),
                  slot={512: 0.63, 1024: 0.24, 2048: 0.036}),
    # MC measured L1 0.0587, 0.0360, 0.0223; L2 0.143, 0.107, 0.0836; Linf 0.847, 0.821, 0.811;
    # orders 0.699, 0.388; dM/M 3.41e-11, 8.45e-14, 2.62e-13; min -4.26e-3, max-1 8.26e-3;
    # slot 0.481, 0.179, 0.0275; A100 1.54, 12.8, 90.2 s
    (2, 1): dict(L1={512: 0.046, 1024: 0.027, 2048: 0.017}, L2={512: 0.14, 1024: 0.11, 2048: 0.08},
                 Linf={512: 0.9, 1024: 0.87, 2048: 0.87}, L1_order=(0.65, 1.05), L2_order=(0.28, 0.68),
                 mass={512: 1e-12, 1024: 1e-12, 2048: 1e-12}, mp=(-2e-9, 1.5e-8),
                 slot={512: 0.33, 1024: 0.034, 2048: 1.4e-3}),
    # van Leer measured L1 0.0349, 0.0203, 0.0124; L2 0.104, 0.0780, 0.0612; Linf 0.813, 0.783,
    # 0.790; orders 0.748, 0.382; dM/M 2.57e-15, 3.19e-14, 1.18e-13; min -1.50e-9, max-1 1.08e-8;
    # slot 0.248, 0.026, 0.001; A100 0.16, 1.23, 8.85 s
    (2, 10): dict(L1={512: 0.087, 1024: 0.054, 2048: 0.032}, L2={512: 0.21, 1024: 0.15, 2048: 0.12},
                  Linf={512: 0.92, 1024: 0.92, 2048: 0.9}, L1_order=(0.63, 1.03), L2_order=(0.31, 0.71),
                  mass={512: 1e-12, 1024: 1e-12, 2048: 1e-12}, mp=(-4.3e-12, 0.0),
                  slot={512: 0.75, 1024: 0.36, 2048: 0.073}),
    # van Leer measured L1 0.0669, 0.0412, 0.0245; L2 0.154, 0.115, 0.0870; Linf 0.829, 0.834,
    # 0.815; orders 0.725, 0.412; dM/M 3.81e-14, 5.41e-14, 5.16e-13; min -3.28e-12, max-1 -1.69e-12;
    # slot 0.570, 0.276, 0.0554; A100 1.59, 12.2, 87.8 s
}


@gpu_only
@pytest.mark.parametrize("turns", (1, 10))
@pytest.mark.parametrize("limiter", (0, 1, 2), ids=["minmod", "mc", "vanleer"])
def test_gpu_ladder(limiter, turns):
    """512-2048 after 1 or 10 turns: L1, L2, Linf, orders, mass, maximum principle, slot depth."""
    b = GPU_BOUNDS[(limiter, turns)]
    lo, hi = b["mp"]
    e1, e2 = [], []
    for n in NS_GPU:
        u, u0, k, wall = rotate_fixed(n, limiter, turns)
        name = f"lim{limiter} T={turns} N={n}"
        print(f"COST {name} steps={k} wall={wall:.2f}s")
        l1, l2 = norms(u, u0, n)
        c = -1.0 + (np.arange(n) + 0.5) * (2.0 / n)
        check(f"{name} L1", l1, b["L1"][n])
        check(f"{name} L2", l2, b["L2"][n])
        check(f"{name} Linf", float(np.abs(u - u0).max()), b["Linf"][n])
        check(f"{name} |dM|/M", float(abs(u.sum() - u0.sum()) / u0.sum()), b["mass"][n])
        check(f"{name} min", float(u.min()), lo, lower=True)
        check(f"{name} max-1", float(u.max()) - 1.0, hi)
        check(f"{name} slot", float(u[np.argmin(abs(c)), np.argmin(abs(c - 0.5))]), b["slot"][n])
        e1.append(l1)
        e2.append(l2)
    for key, e in (("L1", e1), ("L2", e2)):
        order = -np.polyfit(np.log(NS_GPU), np.log(e), 1)[0]
        check(f"lim{limiter} T={turns} {key} order", order, b[f"{key}_order"][0], lower=True)
        check(f"lim{limiter} T={turns} {key} order", order, b[f"{key}_order"][1])
