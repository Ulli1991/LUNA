"""Exact Riemann solver for the 1D Euler equations (Toro 2009, ch. 4), numpy only.

A reference solution for tests and validation; the hydro step never calls it (it uses HLLC/HLL).
States are primitive (rho, u, p)."""
import numpy as np

# Toro (2009) Table 4.1: (W_L, W_R, x0, t_end) of tests 1-5, gamma = 1.4.
TORO = {
    1: ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.5, 0.25),
    2: ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.5, 0.15),
    3: ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.5, 0.012),
    4: ((1.0, 0.0, 0.01), (1.0, 0.0, 100.0), 0.5, 0.035),
    5: ((5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.0950), 0.5, 0.035),
}
# Toro Table 4.2: (p*, u*, rho*_L, rho*_R) of the same tests.
TORO_STAR = {
    1: (0.30313, 0.92745, 0.42632, 0.26557),
    2: (0.00189, 0.00000, 0.02185, 0.02185),
    3: (460.894, 19.5975, 0.57506, 5.99924),
    4: (46.0950, -6.19633, 5.99242, 0.57511),
    5: (1691.64, 8.68975, 14.2823, 31.0426),
}


def _f(p, rho, pk, g):
    """Toro's pressure function f_K(p) and its derivative for one side."""
    a = np.sqrt(g * pk / rho)
    if p > pk:
        A, B = 2.0 / ((g + 1.0) * rho), (g - 1.0) / (g + 1.0) * pk
        q = np.sqrt(A / (p + B))
        return (p - pk) * q, q * (1.0 - 0.5 * (p - pk) / (p + B))
    r = (p / pk) ** ((g - 1.0) / (2.0 * g))
    return 2.0 * a / (g - 1.0) * (r - 1.0), r / (rho * a) * (pk / p)


def star(WL, WR, g):
    """Star-region (p*, u*) by Newton iteration from the two-rarefaction guess."""
    (rl, ul, pl), (rr, ur, pr) = WL, WR
    al, ar = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)
    if 2.0 * (al + ar) / (g - 1.0) <= ur - ul:
        raise ValueError(f"the data generate vacuum (Toro eq. 4.40): u_R - u_L = {ur - ul} >= 2 (a_L + a_R) / (g - 1)"
                         f" = {2.0 * (al + ar) / (g - 1.0)}; no star region")
    z = (g - 1.0) / (2.0 * g)
    p = ((al + ar - 0.5 * (g - 1.0) * (ur - ul)) / (al / pl ** z + ar / pr ** z)) ** (1.0 / z)
    for _ in range(100):
        fl, dl = _f(p, rl, pl, g)
        fr, dr = _f(p, rr, pr, g)
        p_new = max(p - (fl + fr + ur - ul) / (dl + dr), 1e-14 * p)
        done = abs(p_new - p) < 1e-15 * (p_new + p)
        p = p_new
        if done:
            break
    return p, 0.5 * (ul + ur) + 0.5 * (_f(p, rr, pr, g)[0] - _f(p, rl, pl, g)[0])


def _side(W, ps, us, g, s):
    """Star density and the wave's (head, tail) speeds on side s = -1 (left) or +1 (right)."""
    rho, u, p = W
    a = np.sqrt(g * p / rho)
    if ps > p:
        q = (ps / p + (g - 1.0) / (g + 1.0)) / ((g - 1.0) / (g + 1.0) * ps / p + 1.0)
        S = u + s * a * np.sqrt((g + 1.0) / (2.0 * g) * ps / p + (g - 1.0) / (2.0 * g))
        return rho * q, S, S
    a_s = a * (ps / p) ** ((g - 1.0) / (2.0 * g))
    return rho * (ps / p) ** (1.0 / g), u + s * a, us + s * a_s


def waves(WL, WR, g):
    """Dict of p*, u*, rho*_L, rho*_R and the left/right wave (head, tail) speeds."""
    ps, us = star(WL, WR, g)
    rl, hl, tl = _side(WL, ps, us, g, -1.0)
    rr, hr, tr = _side(WR, ps, us, g, +1.0)
    return dict(p=ps, u=us, rho_L=rl, rho_R=rr, left=(hl, tl), right=(hr, tr))


def _fan(W, xi, g, s):
    """Primitive state inside a rarefaction fan on side s at speed xi."""
    rho, u, p = W
    a = np.sqrt(g * p / rho)
    c = 2.0 / (g + 1.0) - s * (g - 1.0) / ((g + 1.0) * a) * (u - xi)
    return rho * c ** (2.0 / (g - 1.0)), 2.0 / (g + 1.0) * (-s * a + 0.5 * (g - 1.0) * u + xi), \
        p * c ** (2.0 * g / (g - 1.0))


def sample(WL, WR, g, xi):
    """(rho, u, p) arrays of the self-similar solution at speeds xi = (x - x0) / t."""
    xi = np.asarray(xi, dtype=np.float64)
    w = waves(WL, WR, g)
    out = [np.empty_like(xi) for _ in range(3)]
    for s, W, rs, (head, tail) in ((-1.0, WL, w["rho_L"], w["left"]),
                                   (1.0, WR, w["rho_R"], w["right"])):
        side = (xi < w["u"]) if s < 0 else (xi >= w["u"])
        outer = side & (s * (xi - head) > 0)
        inner = side & (s * (xi - tail) <= 0)
        fan = side & ~outer & ~inner
        for k, (v_out, v_in) in enumerate(zip(W, (rs, w["u"], w["p"]))):
            out[k][outer], out[k][inner] = v_out, v_in
        if fan.any():
            f = _fan(W, xi[fan], g, s)
            for k in range(3):
                out[k][fan] = f[k]
    return tuple(out)


def cell_average(WL, WR, g, x_edges, x0, t, sub=256):
    """(rho, u, p) of the exact solution averaged (conservatively) over cells with these edges."""
    x_edges = np.asarray(x_edges, dtype=np.float64)
    f = (np.arange(sub) + 0.5) / sub
    x = x_edges[:-1, None] + (x_edges[1:] - x_edges[:-1])[:, None] * f[None, :]
    rho, u, p = sample(WL, WR, g, (x - x0) / t)
    m, mom = rho.mean(1), (rho * u).mean(1)
    E = (p / (g - 1.0) + 0.5 * rho * u * u).mean(1)
    return m, mom / m, (g - 1.0) * (E - 0.5 * mom * mom / m)


def double_tube(WL, WR, g, phase, t, scale=1.0):
    """(rho, u, p) of the periodic double tube R | L | R, L on phase [0.25, 0.75), at time t.

    Two mirrored copies of the WL | WR problem, interfaces at phase 0.25 and 0.75; u is along
    +phase. scale converts phase to distance (sqrt(m) for a phase n . x summed over m axes)."""
    first = phase < 0.5
    xi = np.where(first, -(phase - 0.25), phase - 0.75) / scale / t
    rho, u, p = sample(WL, WR, g, xi)
    return rho, np.where(first, -u, u), p


def godunov_flux(WL, WR, g, vt_L=(0.0, 0.0), vt_R=(0.0, 0.0)):
    """Exact (Godunov) flux (F_rho, F_mn, F_mt1, F_mt2, F_E) at x/t = 0; transverse v rides the contact."""
    rho, u, p = (float(v[0]) for v in sample(WL, WR, g, np.zeros(1)))
    vt = vt_L if 0.0 < star(WL, WR, g)[1] else vt_R
    E = p / (g - 1.0) + 0.5 * rho * (u * u + vt[0] ** 2 + vt[1] ** 2)
    return np.array([rho * u, rho * u * u + p, rho * u * vt[0], rho * u * vt[1], u * (E + p)])
