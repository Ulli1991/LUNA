"""Cylindrical (2D) and spherical (3D) Noh (Noh 1987, JCP 72, 78) against the exact solution: L1, orders,
shock radius, plateau, the wall-heating defect at the origin, anisotropy and the closed-box budget.

Cold gas (rho0 = 1, |u0| = 1 radially inward, p0 = 1e-6, gamma = 5/3) converges on the origin. Exact, from
ExactPack `exactpack/solvers/noh/noh1.py` (github.com/lanl/ExactPack, BSD-3, lines 53-68) and Kamm et al.
2008 (LA-14379, Noh section), geometry d = 2, 3: shock at r_s = |u0| t (gamma - 1) / 2 = t / 3; behind it
rho = rho0 ((gamma + 1) / (gamma - 1))^d = 16, 64, u = 0, p = rho0 u0^2 (gamma - 1) / 2 ((gamma + 1) /
(gamma - 1))^d = 16/3, 64/3; ahead of it u = u0, rho = rho0 (1 + |u0| t / r)^(d - 1). d = 1 is the planar
test (`tests/test_hydro_hard_noh_planar.py`); the full-box, thin-3D 2D Noh in
`tests/test_hydro_hard_carbuncle.py` is the HLL-vs-HLLC symmetry check, this file the exact-solution ladder.

The domain is the quadrant / octant [0, L]^d with reflecting faces (BOUNDARY=2): the axis planes are the
symmetry planes. BOUNDARY is one code for every face and LUNA has no time-dependent inflow face
(`src/luna/utils/halo_local.py:3`, `src/luna/ic_generation/planar_riemann_2d.py:49`), so the outer faces
are walls too, and, as in the planar test, the gas is brought to rest towards them by a radial ramp,
|u| = |u0| smoothstep((L - r) / (L - AV)), at rest beyond L (the box corner). Unlike 1D, a cold ramp goes
to negative pressure (measured, minmod, linear ramp from 1.0: L = 1.2 at step 2-3 in 2D and 3D, L = 2 in
3D at step 5), and a ramp warmed only behind its edge leaves a cold rarefaction at r = A - t where MC and
van Leer go negative (2D 256-1024, 3D 64-256, t = 0.01-0.56; minmod survives). So the inflow is warmed
first, while still uniform: c = C_WARM smoothstep((r - A) / (AV - A)), max c' = 1.5 < 1 / T, so no
characteristic from r > A outruns the one from A; the ramp then decelerates warm gas. The inflow inside A
is exact until r = A - (|u0| + c0) t = 0.4 at T; the measured ball r < X_MAX = 0.3 keeps 0.1 from it, and
the inflow there is checked against the exact one. Mass and energy close to round-off (in evolve).
"""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from tests._helpers import accuracy_cfg, check, evolve, gpu_only

G, RHO0, U0, P0 = 5.0 / 3.0, 1.0, -1.0, 1e-6
L, A, AV, C_WARM = 1.7, 1.0, 1.5, 0.5               # box, warming from A, at rest towards L from AV, c there
T, X_MAX = 0.6, 0.3
D = (G - 1.0) / 2.0 * abs(U0)                       # shock speed, 1/3
COMP = (G + 1.0) / (G - 1.0)                        # 4
PROBLEM = {2: "Square-Advected", 3: "Sod-Shock-3D"}  # any registered name of that dimension; U0 is given
SOLVERS = {("hllc", 0): "minmod", ("hllc", 1): "mc", ("hllc", 2): "vanleer", ("hll", 0): "hll-minmod"}


def rho_s(d):
    """Post-shock density rho0 ((gamma + 1) / (gamma - 1))^d."""
    return RHO0 * COMP ** d


def p_s(d):
    """Post-shock pressure rho0 u0^2 (gamma - 1) / 2 ((gamma + 1) / (gamma - 1))^d."""
    return RHO0 * U0 * U0 * (G - 1.0) / 2.0 * COMP ** d


def exact_point(d, r, t):
    """Exact (rho, u_r, p) at radii r and time t."""
    post = r < D * t
    rho_in = RHO0 * (1.0 + abs(U0) * t / np.maximum(r, 1e-300)) ** (d - 1)
    return (np.where(post, rho_s(d), rho_in), np.where(post, 0.0, U0),
            np.where(post, p_s(d), P0 * (rho_in / RHO0) ** G))


def centres(d, n):
    """Cell-centre coordinates (d arrays) and radius of the n^d cells of [0, L]^d."""
    x = (np.arange(n) + 0.5) * L / n
    X = np.meshgrid(*([x] * d), indexing="ij")
    return X, np.sqrt(sum(c * c for c in X))


def smoothstep(s):
    """3 s^2 - 2 s^3 of s clipped to [0, 1]."""
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def initial(d, n):
    """Conserved state on n^d cells of [0, L]^d: inflow at u0, warmed on [A, AV], ramped to rest on [AV, L]."""
    X, r = centres(d, n)
    vr = U0 * smoothstep((L - r) / (L - AV))
    rho = np.full_like(r, RHO0)
    mom = [rho * vr * c / r for c in X]
    p = P0 + rho * (C_WARM * smoothstep((r - A) / (AV - A))) ** 2 / G
    return np.stack([rho, *mom, p / (G - 1.0) + 0.5 * rho * vr * vr])


def exact(d, n, t=T, sub=4):
    """Exact cell averages of (rho, u_r, p) on the cells of the sub-box holding r < X_MAX + h; cells the
    shock cuts averaged over sub^d points, the rest at the centre (smooth there: O(h^2))."""
    h, m = L / n, int(np.ceil(X_MAX * n / L)) + 1
    x = (np.arange(m) + 0.5) * h
    X = np.meshgrid(*([x] * d), indexing="ij")
    r = np.sqrt(sum(c * c for c in X))
    out = [np.array(a, dtype=float) for a in exact_point(d, r, t)]
    cut = np.abs(r - D * t) < 0.5 * h * np.sqrt(d) + 1e-12
    off = (np.arange(sub) + 0.5) / sub - 0.5
    O = np.meshgrid(*([off] * d), indexing="ij")
    rs = np.sqrt(sum((c[cut][:, None] + h * o.ravel()[None, :]) ** 2 for c, o in zip(X, O)))
    for a, v in zip(out, exact_point(d, rs, t)):
        a[cut] = v.mean(axis=1)
    return out, r


# Minmod has no unphysical Hancock face (audit C8); MC and van Leer do, first order there, the count pinned (FP64,
# MI300A, equal in the three runs). 2D MC needs preserve_face_states as well: without it a cold inflow cell near the
# diagonal goes to p < 0 at t = 0.507, N = 256; with it p stays > 0 and every bound holds.
FACES = {(d, lim, n): (c, c) for (d, lim), row in {
    (2, 1): {256: 7868, 512: 32316, 1024: 130702, 2048: 524202},
    (2, 2): {256: 4432, 512: 18224, 1024: 73782, 2048: 296138},
    (3, 1): {64: 4242, 128: 35883, 256: 300579},
    (3, 2): {64: 1908, 128: 18714, 256: 165294}}.items() for n, c in row.items()}


def run(d, n, limiter=0, solver="hllc", t=T, mesh_shape=None):
    """U at t on n^d cells (evolve checks mass and energy to round-off) and U at 0."""
    cfg = accuracy_cfg(test_problem=PROBLEM[d], SPATIAL_DIMENSION=d, Nx=n, Ny=n, Nz=n if d == 3 else 1,
                       Lx=L, Ly=L, Lz=L if d == 3 else 1.0, BOUNDARY=2, gamma=G, LIMITER=limiter,
                       riemann_solver=solver, allow_first_order_faces=limiter != 0,
                       preserve_face_states=(d, limiter) == (2, 1))
    U0_ = initial(d, n)
    faces = FACES[d, limiter, n] if limiter != 0 else None
    return evolve(cfg, U0_, t, mesh_shape=mesh_shape, faces=faces)[0], U0_


def directions(d):
    """Unit rays in the first quadrant / octant: 2D every 1 deg; 3D a 5 deg (theta, phi) grid plus (1,1,1)."""
    if d == 2:
        a = np.radians(np.arange(91.0))
        return np.stack([np.cos(a), np.sin(a)], 1)
    th, ph = np.meshgrid(np.radians(np.arange(0, 91, 5.0)), np.radians(np.arange(0, 91, 5.0)), indexing="ij")
    v = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1).reshape(-1, 3)
    return np.concatenate([v, np.full((1, 3), 3 ** -0.5)])


def on_rays(f, n, dirs, radii):
    """f (cell-centred, mirror-symmetric at the axis planes) linearly interpolated at radii along dirs."""
    pts = dirs[:, None, :] * radii[None, :, None] * n / L - 0.5
    flat = pts.reshape(-1, dirs.shape[1]).T
    return map_coordinates(f, flat, order=1, mode="reflect").reshape(len(dirs), len(radii))


def shock_radii(rho, d, n, t=T):
    """Shock radius along each ray: the outermost crossing of (rho ahead + rho behind) / 2, linear on the ray."""
    dirs = directions(d)
    half = 0.5 * (rho_s(d) + rho_s(d) / COMP)
    radii = np.arange(0.0, X_MAX, L / n / 4.0)
    prof = on_rays(rho, n, dirs, radii)
    out = []
    for p in prof:
        i = int(np.where((p[:-1] >= half) & (p[1:] < half))[0][-1])
        out.append(radii[i] + (radii[i + 1] - radii[i]) * (p[i] - half) / (p[i] - p[i + 1]))
    return np.array(out), dirs


def measure(d, n, limiter=0, solver="hllc", t=T, mesh_shape=None):
    """Every Noh metric of one run (see the test docstrings)."""
    U, _ = run(d, n, limiter, solver, t, mesh_shape)
    h, xs = L / n, D * t
    rho = U[0]
    X, r = centres(d, n)
    vel = U[1:1 + d] / rho
    vr = sum(v * c for v, c in zip(vel, X)) / r
    vt = np.sqrt(np.maximum(sum(v * v for v in vel) - vr * vr, 0.0))
    p = (G - 1.0) * (U[-1] - 0.5 * rho * sum(v * v for v in vel))
    ex, rex = exact(d, n, t)
    m = ex[0].shape[0]
    box = (slice(0, m),) * d
    ball = rex < X_MAX
    l1 = [float(np.abs(a[box] - b)[ball].sum() * h ** d) for a, b in zip((rho, vr, p), ex)]
    l2 = float(np.sqrt(((rho[box] - ex[0])[ball] ** 2).sum() * h ** d))
    rs, dirs = shock_radii(rho, d, n, t)
    axis = np.isclose(dirs.max(1), 1.0)
    diag = np.isclose(dirs.min(1), dirs.max(1))                 # the (1,1) / (1,1,1) body diagonal
    plat = (r > 0.4 * xs) & (r < 0.8 * xs)
    plat_rays = on_rays(rho, n, dirs, np.linspace(0.4 * xs, 0.8 * xs, 9)).mean(1) / rho_s(d) - 1.0
    shells = np.floor(r / h).astype(int)
    k_in = int(0.8 * xs / h)
    prof = np.bincount(shells[r < k_in * h], weights=rho[r < k_in * h]) / np.bincount(shells[r < k_in * h])
    bad = np.abs(prof / rho_s(d) - 1.0) > 0.02
    extent = int(np.argmin(bad)) if not bad.all() else len(prof)   # shells of width h from the origin above 2 %
    ahead = (rex > xs + 0.05) & ball
    sym = max(float(np.abs(rho - np.swapaxes(rho, a, b)).max()) for a in range(d) for b in range(a + 1, d))
    o = (0,) * d
    return dict(
        l1=l1, l2=l2, shock_mean=(rs.mean() - xs) / h, shock_spread=float((rs.max() - rs.min()) / rs.mean()),
        shock_axis=(rs[axis].mean() - xs) / h, shock_diag=(rs[diag].mean() - xs) / h,
        plateau=float(rho[plat].mean() / rho_s(d) - 1.0), plateau_std=float(rho[plat].std() / rho[plat].mean()),
        plateau_axis=float(plat_rays[axis].mean()), plateau_diag=float(plat_rays[diag].mean()),
        origin=float(rho[o] / rho_s(d)), s_origin=float(p[o] / rho[o] ** G / (p_s(d) / rho_s(d) ** G) - 1.0),
        extent=extent, vt=float(vt[r < X_MAX].max() / abs(U0)),
        ahead=float(np.abs(rho[box][ahead] / ex[0][ahead] - 1.0).max()), sym=sym / rho_s(d),
        rho_min=float(rho.min()), p_min=float(p.min()), profile=prof[:12] / rho_s(d))


def report(tag, r):
    """Print one measurement as a MEASURED line (bounds are set from these)."""
    print(f"MEASURED {tag} L1 rho u p = {r['l1'][0]:.4e} {r['l1'][1]:.4e} {r['l1'][2]:.4e}; L2 rho = {r['l2']:.4e}; "
          f"shock (mean, axis, diag) - t/3 in h = {r['shock_mean']:+.3f} {r['shock_axis']:+.3f} {r['shock_diag']:+.3f}; "
          f"shock spread = {r['shock_spread']:.4e}; plateau-1 = {r['plateau']:+.4e} std {r['plateau_std']:.4e} "
          f"axis {r['plateau_axis']:+.4e} diag {r['plateau_diag']:+.4e}; origin rho/rho_s = {r['origin']:.4f} "
          f"s-1 = {r['s_origin']:.3e}; extent = {r['extent']}; vt = {r['vt']:.4e}; ahead = {r['ahead']:.3e}; "
          f"sym = {r['sym']:.3e}; min rho p = {r['rho_min']:.4e} {r['p_min']:.4e}")
    print(f"PROFILE {tag} shell rho/rho_s " + " ".join(f"{v:.4f}" for v in r["profile"]))


# Measured, FP64, default dt (CFL 0.8), per (d, solver, limiter) and N (cells across [0, L]):
# (L1 rho, L1 u_r, L1 p over r < X_MAX; mean shock radius - t/3 in h; shock radius spread over rays; plateau
#  rho / rho_s - 1 on [0.4, 0.8] r_s; its std / mean; rho / rho_s in the origin cell; wall-defect extent in
#  shells of width h (2 %); max |v_t| / |u0| in r < X_MAX; max |rho / exact - 1| of the inflow on [r_s + 0.05, X_MAX];
#  L2 rho, appended below). GPU: one A100, jobs 31386298 (2D), 31386299 (3D).
MEAS_GPU = {
    (2, "hllc", 0): {256: (2.3315e-02, 1.8154e-03, 6.5540e-03, 0.417, 1.4220e-02, -1.7649e-02, 2.3314e-02, 0.7790, 10, 7.5830e-02, 1.977e-03),
                     512: (1.3386e-02, 1.0842e-03, 3.6030e-03, 0.462, 9.2087e-03, -8.9062e-03, 1.6490e-02, 0.7821, 8, 7.8126e-02, 5.195e-04),
                     1024: (8.9784e-03, 7.2562e-04, 2.1635e-03, 0.538, 7.4950e-03, -4.3481e-03, 1.4845e-02, 0.7836, 8, 7.8448e-02, 1.296e-04),
                     2048: (6.2084e-03, 5.0200e-04, 1.3186e-03, 0.626, 5.1137e-03, -2.3852e-03, 1.2160e-02, 0.7845, 7, 9.0204e-02, 3.318e-05)},
    (2, "hllc", 1): {256: (1.8742e-02, 1.5372e-03, 5.3318e-03, 0.307, 1.0605e-02, -1.4769e-02, 2.1329e-02, 0.8896, 7, 8.0304e-02, 4.401e-04),
                     512: (9.8095e-03, 7.9615e-04, 2.7119e-03, 0.333, 6.7068e-03, -8.0729e-03, 1.4260e-02, 0.8925, 2, 9.2893e-02, 1.180e-04),
                     1024: (7.3503e-03, 6.0730e-04, 1.7740e-03, 0.406, 4.7410e-03, -3.6372e-03, 1.2922e-02, 0.8939, 2, 6.4171e-02, 2.996e-05),
                     2048: (5.2279e-03, 4.4039e-04, 1.0975e-03, 0.489, 3.9109e-03, -2.0338e-03, 1.1072e-02, 0.8944, 1, 6.7515e-02, 7.983e-06)},
    (2, "hllc", 2): {256: (1.9681e-02, 1.6181e-03, 5.6172e-03, 0.339, 1.2585e-02, -1.5732e-02, 2.1874e-02, 0.8483, 9, 8.1424e-02, 4.660e-04),
                     512: (1.1648e-02, 9.5724e-04, 3.0962e-03, 0.385, 8.1459e-03, -8.0448e-03, 1.5918e-02, 0.8512, 5, 6.7460e-02, 1.231e-04),
                     1024: (7.9480e-03, 6.6844e-04, 1.9234e-03, 0.451, 5.6961e-03, -4.1084e-03, 1.3994e-02, 0.8525, 3, 7.5013e-02, 3.102e-05),
                     2048: (5.6581e-03, 4.7910e-04, 1.1828e-03, 0.535, 4.2119e-03, -2.2953e-03, 1.1889e-02, 0.8532, 2, 6.9852e-02, 8.216e-06)},
    (2, "hll", 0): {256: (1.8159e-02, 1.2244e-03, 5.2913e-03, 0.354, 4.6158e-03, -1.4688e-02, 2.3117e-03, 0.9409, 12, 4.0389e-02, 1.977e-03),
                    512: (8.8946e-03, 6.2314e-04, 2.5535e-03, 0.365, 2.7560e-03, -7.2298e-03, 1.1836e-03, 0.9550, 11, 3.6786e-02, 5.196e-04),
                    1024: (4.4544e-03, 3.2283e-04, 1.2708e-03, 0.372, 1.5072e-03, -3.6020e-03, 6.0121e-04, 0.9653, 11, 3.4860e-02, 1.296e-04),
                    2048: (2.2312e-03, 1.6542e-04, 6.3873e-04, 0.365, 7.0199e-04, -1.8012e-03, 3.0154e-04, 0.9732, 11, 3.3699e-02, 3.314e-05)},
    (3, "hllc", 0): {64: (7.8981e-02, 1.4144e-03, 2.6247e-02, 0.382, 3.2505e-02, -1.7160e-01, 3.8312e-02, 0.5617, 6, 7.0086e-02, 7.963e-02),
                     128: (4.2841e-02, 7.8733e-04, 1.4342e-02, 0.518, 2.2545e-02, -8.4327e-02, 3.8576e-02, 0.5962, 12, 7.9563e-02, 1.688e-02),
                     256: (2.4798e-02, 5.0123e-04, 8.4116e-03, 0.643, 1.8199e-02, -4.1017e-02, 3.6846e-02, 0.6160, 24, 1.0663e-01, 4.398e-03)},
    (3, "hllc", 1): {64: (5.8687e-02, 1.1497e-03, 2.0520e-02, 0.286, 2.5889e-02, -1.3291e-01, 5.7525e-02, 0.7133, 6, 1.4103e-01, 1.318e-02),
                     128: (3.2353e-02, 6.4297e-04, 1.1145e-02, 0.381, 1.5631e-02, -6.6965e-02, 4.0377e-02, 0.7407, 12, 1.1155e-01, 3.460e-03),
                     256: (1.9928e-02, 4.1311e-04, 6.7021e-03, 0.512, 1.5348e-02, -3.2781e-02, 3.4782e-02, 0.7611, 24, 1.2193e-01, 9.502e-04)},
    (3, "hllc", 2): {64: (6.3443e-02, 1.2413e-03, 2.1875e-02, 0.302, 2.7510e-02, -1.4065e-01, 6.2157e-02, 0.6511, 6, 1.5116e-01, 1.364e-02),
                     128: (3.4865e-02, 6.8441e-04, 1.1910e-02, 0.429, 1.9009e-02, -7.2993e-02, 4.1995e-02, 0.6857, 12, 8.0266e-02, 3.711e-03),
                     256: (2.1113e-02, 4.4227e-04, 7.1314e-03, 0.551, 1.4961e-02, -3.4746e-02, 3.5923e-02, 0.7026, 24, 1.4944e-01, 9.860e-04)},
    (3, "hll", 0): {64: (7.5973e-02, 1.4238e-03, 2.4482e-02, 0.331, 1.6595e-02, -1.6399e-01, 1.9193e-02, 0.7837, 6, 3.4913e-02, 7.480e-02),
                    128: (4.0453e-02, 6.5715e-04, 1.2857e-02, 0.465, 7.9793e-03, -8.3507e-02, 1.1230e-02, 0.8475, 12, 5.2738e-02, 1.688e-02),
                    256: (2.0607e-02, 3.1851e-04, 6.5004e-03, 0.527, 6.0888e-03, -4.0469e-02, 5.1826e-03, 0.8896, 24, 4.4650e-02, 4.398e-03)},
}
# L2 rho over r < X_MAX at the same N, jobs 31386429 (2D), 31386430 (3D); the other metrics reproduced to every printed digit.
L2_GPU = {(2, "hllc", 0): (2.4384e-01, 1.7705e-01, 1.4116e-01, 1.1594e-01),
          (2, "hllc", 1): (1.9370e-01, 1.3163e-01, 1.1515e-01, 9.8908e-02),
          (2, "hllc", 2): (2.0851e-01, 1.5290e-01, 1.2526e-01, 1.0576e-01),
          (2, "hll", 0): (2.0395e-01, 1.3312e-01, 9.3484e-02, 6.5805e-02),
          (3, "hllc", 0): (1.0744e+00, 7.5120e-01, 5.7311e-01),
          (3, "hllc", 1): (8.5391e-01, 6.0168e-01, 4.7777e-01),
          (3, "hllc", 2): (9.1440e-01, 6.4494e-01, 5.0639e-01),
          (3, "hll", 0): (1.0017e+00, 6.8640e-01, 4.7149e-01)}
MEAS_GPU = {k: {n: (*m, l2) for (n, m), l2 in zip(v.items(), L2_GPU[k])} for k, v in MEAS_GPU.items()}
# CPU-8 (in-job JAX_PLATFORMS=cpu), jobs 31386555 (2D), 31386556 (3D); 256 equals the GPU row to every digit.
MEAS_CPU = {
    (2, "hllc", 0): {128: (3.8774e-02, 2.9053e-03, 1.1124e-02, 0.351, 1.9713e-02, -3.2064e-02, 2.3477e-02, 0.7728, 12, 6.2516e-02, 7.687e-03, 3.0670e-01),
                     256: MEAS_GPU[2, "hllc", 0][256]},
    (3, "hllc", 0): {32: (1.2344e-01, 2.8838e-03, 4.0000e-02, -0.012, 6.5414e-02, -2.9976e-01, 4.4995e-02, 0.5195, 3, 6.2169e-02, 7.541e-01, 1.3808e+00)},
}
SLACK = 1.3                # bounds: 1.3x the measured magnitude; orders measured -0.1 .. +0.3
ORIGIN_TOL = 0.01          # |rho / rho_s in the origin cell - measured|
SYM = 1e-11                # max |rho - rho with two axes swapped| / rho_s; measured 0 (2D), <= 2.6e-12 (3D)
KEYS = ("L1 rho", "L1 u_r", "L1 p", "|shock - t/3| / h", "shock spread", "|plateau - 1|", "plateau std",
        "origin", "extent", "max v_t", "inflow error", "L2 rho")


def order(ns, values):
    """Fitted order -d log(value) / d log(N)."""
    return float(-np.polyfit(np.log(ns), np.log(values), 1)[0])


def check_run(tag, d, r, meas):
    """Every metric of one run against SLACK x its measured value; the origin to ORIGIN_TOL."""
    got = (*r["l1"], abs(r["shock_mean"]), r["shock_spread"], abs(r["plateau"]), r["plateau_std"], r["origin"],
           r["extent"], r["vt"], r["ahead"], r["l2"])
    for name, g, m in zip(KEYS, got, meas):
        if name == "origin":
            check(f"{tag} origin rho / rho_s", g, m - ORIGIN_TOL, lower=True)
            check(f"{tag} origin rho / rho_s", g, m + ORIGIN_TOL)
        elif name == "extent":
            if d == 2:                                   # 3D: the plateau is > 2 % low everywhere (first order)
                check(f"{tag} wall-defect extent (cells)", g, float(np.ceil(SLACK * m)))
        else:                                            # the shock offset crosses 0 (3D at 32): + 0.05 h
            check(f"{tag} {name}", g, SLACK * abs(m) + (0.05 if name.startswith("|shock") else 0.0))
    check(f"{tag} diagonal mirror symmetry", r["sym"], SYM)
    check(f"{tag} min p", r["p_min"], 0.0, lower=True)


def ladder(d, solver, limiter, meas):
    """Run the ladder, check every run and the L1 orders; return the measurements."""
    tag, out = f"Noh {d}D {SOLVERS[solver, limiter]}", {}
    for n, m in meas.items():
        out[n] = r = measure(d, n, limiter, solver)
        report(f"{tag} N={n}", r)
        check_run(f"{tag} N={n}", d, r, m)
    ns = list(meas)
    if len(ns) > 1:
        for k, name in ((0, "L1 rho"), (1, "L1 u_r"), (2, "L1 p"), (11, "L2 rho")):
            got = [out[n]["l2"] if k == 11 else out[n]["l1"][k] for n in ns]
            o, om = order(ns, got), order(ns, [meas[n][k] for n in ns])
            check(f"{tag} {name} order", o, om - 0.1, lower=True)
            check(f"{tag} {name} order", o, om + 0.3)
    return out


@pytest.mark.parametrize("d", [2, 3])
def test_noh_cpu(d):
    """CPU, HLLC-minmod, 2D at 128 and 256, 3D at 32: every metric against its bound, mass and energy to round-off."""
    ladder(d, "hllc", 0, MEAS_CPU[d, "hllc", 0])


GPU_IDS = [f"{d}D-{SOLVERS[s, lim]}" for d, s, lim in MEAS_GPU]


@gpu_only
@pytest.mark.parametrize("d,solver,limiter", list(MEAS_GPU), ids=GPU_IDS)
def test_noh_ladder_gpu(d, solver, limiter):
    """GPU ladder (2D 256-2048, 3D 64-256): L1 of rho, u_r, p and their orders, shock radius, plateau, wall
    defect, v_t, the inflow, diagonal symmetry, positivity."""
    ladder(d, solver, limiter, MEAS_GPU[d, solver, limiter])


CARBUNCLE = "open bug: HLLC carbuncles at grid-aligned strong shocks (tests/test_hydro_hard_carbuncle.py)"
HLLC = [k for k in MEAS_GPU if k[1] == "hllc"]


@gpu_only
@pytest.mark.parametrize("d,solver,limiter", HLLC, ids=[i for i, k in zip(GPU_IDS, MEAS_GPU) if k in HLLC])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=CARBUNCLE)
def test_noh_plateau_isotropy_converges_gpu(d, solver, limiter):
    """The plateau's angular std converges at order >= 0.8, as HLL's does (from MEAS_GPU: HLL 1.0 in 2D, 0.9
    in 3D; HLLC 0.3 in 2D, 0.0-0.4 in 3D, where the shock runs ahead along the axes)."""
    ns = list(MEAS_GPU[d, solver, limiter])
    stds = [measure(d, n, limiter, solver)["plateau_std"] for n in (ns[0], ns[-1])]
    check(f"Noh {d}D {SOLVERS[solver, limiter]} plateau std order", order((ns[0], ns[-1]), stds), 0.8, lower=True)


def test_bound_catches_a_short_run():
    """Teeth: stopped at 0.95 T, the 2D density L1 at 256 against the exact state at T misses its bound."""
    U, _ = run(2, 256, t=0.95 * T)
    ex, rex = exact(2, 256)
    m = ex[0].shape[0]
    l1 = float(np.abs(U[0][:m, :m] - ex[0])[rex < X_MAX].sum() * (L / 256) ** 2)
    print(f"MEASURED short run L1 rho = {l1:.4e}")
    assert l1 > SLACK * MEAS_CPU[2, "hllc", 0][256][0], f"L1 rho {l1:.4e} of a short run passes the bound"


def test_exact_solution():
    """The exact state satisfies Rankine-Hugoniot at r_s with the converged upstream rho0 (1 + 3)^(d - 1), the
    shocked ball holds the mass that started inside r_s + |u0| t, the inflow solves the radial continuity
    equation, and d = 1 gives the planar 4, 4/3."""
    def flux(rho, u, p):
        return np.array([rho * u, rho * u * u + p, u * (p / (G - 1.0) + 0.5 * rho * u * u + p)])

    def cons(rho, u, p):
        return np.array([rho, rho * u, p / (G - 1.0) + 0.5 * rho * u * u])

    check("Noh d=1 vs planar (4, 4/3)", max(abs(rho_s(1) - 4.0), abs(p_s(1) - 4.0 / 3.0)), 1e-15)
    for d in (2, 3):
        up, down = (RHO0 * (1.0 + abs(U0) / D) ** (d - 1), U0, 0.0), (rho_s(d), 0.0, p_s(d))
        jump = flux(*down) - flux(*up) - D * (cons(*down) - cons(*up))
        check(f"Noh {d}D Rankine-Hugoniot max |[F] - D [U]|", float(np.abs(jump).max()), 1e-13)
        check(f"Noh {d}D shocked mass / inflow mass - 1", abs(rho_s(d) * D ** d / (D + abs(U0)) ** d - 1.0), 1e-15)
        r, t, e = np.linspace(0.5, 1.0, 11), 0.3, 1e-5
        rho = lambda r, t: exact_point(d, r, t)[0]                           # noqa: E731
        res = ((rho(r, t + e) - rho(r, t - e)) / (2 * e)
               + ((r + e) ** (d - 1) * rho(r + e, t) - (r - e) ** (d - 1) * rho(r - e, t)) * U0 / (2 * e * r ** (d - 1)))
        check(f"Noh {d}D inflow continuity residual", float(np.abs(res).max()), 1e-8)
