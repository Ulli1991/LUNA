"""T and gamma from e/(k n_tot): Glover's table lookup (calc_temp.F, eos="table") or a Newton
solve of ekn(T; He/H, x_H2) with a He-free e_H2(T) table (eos="newton"). Needs jax_enable_x64.
"""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from luna.chemistry import tables as tb

KBOLTZ = tb.KBOLTZ
ABHE = tb.ABHE
_F01 = tb.f32(0.1)                 # the undecorated 0.1 in lookup_temp_gamma


def _table():
    _, temp, gamma = tb.energy_table()
    # Fortran reads index 50000*k + idx + 1 for idx = 50000 (ekn = 5e4 exactly),
    # one past the block; that weight is (ekn - idx) = 0, so pad with a copy.
    pad = lambda a: np.concatenate([a.ravel(), a.ravel()[-1:]])  # noqa: E731
    return jnp.asarray(pad(temp)), jnp.asarray(pad(gamma))


_TEMP, _GAMMA = None, None


def _tables():
    global _TEMP, _GAMMA
    if _TEMP is None:
        with jax.ensure_compile_time_eval():     # a concrete array, even if first built under jit
            _TEMP, _GAMMA = _table()
    return tb.device_table("energy", lambda: (_TEMP, _GAMMA))


def n_total(x_h2, x_e, n_h):
    """Particles per cm^3 as rate_eq counts them: (1 + x_He - x_H2 + x_e) n_H."""
    return (1.0 + ABHE - x_h2 + x_e) * n_h


def temperature_gamma(x_h2, ekn, abhe=ABHE, eos="table"):
    """T [K] and gamma from ekn = e / (k n_tot), x_H2 and He/H: eos "table" (He/H = 0.1 only) or "newton"."""
    if eos == "newton":
        return temperature_gamma_newton(x_h2, ekn, abhe)
    if eos != "table":
        raise ValueError(f"eos={eos!r}")
    return temperature_gamma_table(x_h2, ekn)


def temperature_gamma_table(x_h2, ekn):
    """T [K] and gamma from ekn = e / (k n_tot) and x_H2 (Glover's table, calc_temp.F:167-217)."""
    temp_t, gamma_t = _tables()
    ekn = jnp.asarray(ekn, jnp.float64)
    x_h2 = jnp.asarray(x_h2, jnp.float64)
    atomic = (ekn < 1.0) | (ekn > 5e4) | (x_h2 == 0.0)
    e_safe = jnp.clip(ekn, 1.0, 5e4)
    idx_en = jnp.floor(e_safe)
    diff_en = e_safe - idx_en
    idx_en = idx_en.astype(jnp.int32)

    # x_H2 bin; the Fortran divides by and multiplies with the REAL*4 0.1,
    # the product 0.1*idx itself in single precision
    idx_h2 = jnp.floor(x_h2 / _F01).astype(jnp.int32)
    idx_h2 = jnp.clip(idx_h2, 0, 4)
    lo = (_F01 * idx_h2.astype(jnp.float32)).astype(jnp.float64)
    diff_h2 = (x_h2 - lo) / _F01

    def row(block):
        i = 50000 * block + idx_en - 1                 # Fortran 1-based -> 0-based
        t = temp_t[i] + (temp_t[i + 1] - temp_t[i]) * diff_en
        g = gamma_t[i] + (gamma_t[i + 1] - gamma_t[i]) * diff_en
        return t, g

    t_hi, g_hi = row(idx_h2)
    t_lo, g_lo = row(jnp.maximum(idx_h2 - 1, 0))
    t_lo = jnp.where(idx_h2 == 0, (2.0 / 3.0) * e_safe, t_lo)
    g_lo = jnp.where(idx_h2 == 0, 5.0 / 3.0, g_lo)
    temp = t_lo + (t_hi - t_lo) * diff_h2
    gamma = g_lo + (g_hi - g_lo) * diff_h2

    t_half, g_half = row(4)                              # x_H2 = 0.5 exactly
    temp = jnp.where(x_h2 == 0.5, t_half, temp)
    gamma = jnp.where(x_h2 == 0.5, g_half, gamma)
    temp = jnp.where(atomic, (2.0 / 3.0) * ekn, temp)
    gamma = jnp.where(atomic, 5.0 / 3.0, gamma)
    return temp, gamma


def ekn_from_energy(energy, x_h2, x_e, n_h):
    """e / (k n_tot) [K], rate_eq_simple.F:124."""
    return energy / (n_total(x_h2, x_e, n_h) * KBOLTZ)


# --------------------------------------------------------------------------
# Newton solve of ekn(T; a, x) = [x e_H2(T) + (1+a-2x) 1.5 T] / (1+a-x)
# --------------------------------------------------------------------------

N_NEWTON = 6                       # stopped-gradient iterations; one differentiable step follows


@lru_cache(maxsize=None)
def _eh2():
    """(ln T0, d ln T, Hermite rows, phi bounds) of $LUNA_DATA/chemistry/eh2_table.npz, in numpy."""
    u0, du, rows = _load_eh2()
    return u0, du, rows, _phi_bounds(rows)


def _load_eh2():
    """phi = e_H2/T on the ln T nodes, as Hermite rows (phi_i, phi_i+1, m_i, m_i+1), m per cell."""
    with np.load(tb.data_path("eh2_table.npz")) as z:
        u0, du, e, de = float(z["ln_t0"]), float(z["dln_t"]), z["e_h2"], z["de_h2"]
    t = np.exp(u0 + du * np.arange(e.size))
    phi = e / t
    m = (de - phi) * du                # d phi / d ln T, per node spacing
    rows = np.stack([phi[:-1], phi[1:], m[:-1], m[1:]], axis=1)
    return u0, du, rows


def _hermite(rows, s):
    """phi and d phi / d(cell coordinate) at s; constant (1.5, 3) outside the nodes."""
    n = rows.shape[0]
    i = jnp.clip(jnp.floor(s), 0, n - 1).astype(jnp.int32)
    t = jnp.clip(s - i, 0.0, 1.0)
    r = jnp.asarray(rows)[i]
    p0, p1, m0, m1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    t2 = t * t
    phi = (p0 + t2 * (3.0 - 2.0 * t) * (p1 - p0) + t * (1.0 - t) * (1.0 - t) * m0
           + t2 * (t - 1.0) * m1)
    dphi = 6.0 * t * (1.0 - t) * (p1 - p0) + (1.0 - t) * (1.0 - 3.0 * t) * m0 + t * (3.0 * t - 2.0) * m1
    inside = (s >= 0.0) & (s <= n)
    return phi, jnp.where(inside, dphi, 0.0)


def eh2_over_t(lnt):
    """e_H2 / T and its derivative in ln T (He-free H2 internal energy per k)."""
    u0, du, rows, _ = _eh2()
    phi, dphi = _hermite(rows, (lnt - u0) / du)
    return phi, dphi / du


def _phi_bounds(rows):
    """Range of e_H2/T over the Hermite table (dense in each cell), widened slightly: the Newton bracket."""
    s = (np.arange(rows.shape[0])[:, None] + np.linspace(0, 1, 33)[None, :]).ravel()
    i = np.minimum(np.floor(s).astype(int), rows.shape[0] - 1)
    t = s - i
    p0, p1, m0, m1 = rows[i].T
    phi = p0 + t * t * (3 - 2 * t) * (p1 - p0) + t * (1 - t) ** 2 * m0 + t * t * (t - 1) * m1
    return phi.min() * (1 - 1e-6), phi.max() * (1 + 1e-6)


def _residual(u, lnr, x, c1):
    """G(u) = ln(x e_H2 + c1 T) - ln r at u = ln T, and dG/du."""
    phi, dphi = eh2_over_t(u)
    s = x * phi + c1
    return u + jnp.log(s) - lnr, 1.0 + x * dphi / s


def temperature_gamma_newton(x_h2, ekn, abhe):
    """T [K] and gamma solving ekn(T; abhe, x_H2) = ekn by safeguarded Newton in ln T (atomic for ekn <= 0)."""
    x = jnp.asarray(x_h2, jnp.float64)
    ekn = jnp.asarray(ekn, jnp.float64)
    a = jnp.asarray(abhe, jnp.float64)
    q = 1.0 + a - x
    c1 = 1.5 * (1.0 + a - 2.0 * x)
    pos = ekn > 0.0
    lnr = jnp.log(jnp.where(pos, ekn, 1.0) * q)
    sg = jax.lax.stop_gradient
    xs, lnrs, c1s = sg(x), sg(lnr), sg(c1)
    phi_lo, phi_hi = _eh2()[3]
    lo = lnrs - jnp.log(xs * phi_hi + c1s)
    hi = lnrs - jnp.log(xs * phi_lo + c1s)
    u = lnrs - jnp.log(xs * eh2_over_t(hi)[0] + c1s)
    for _ in range(N_NEWTON):
        g, dg = _residual(u, lnrs, xs, c1s)
        lo = jnp.where(g < 0.0, u, lo)
        hi = jnp.where(g > 0.0, u, hi)
        u = jnp.clip(u - g / dg, lo, hi)
    u = sg(u)
    g, dg = _residual(u, lnr, x, c1)          # the implicit-function derivative rides on this step
    u = u - g / dg
    temp = jnp.exp(u)
    phi, dphi = eh2_over_t(u)
    gamma = 1.0 + q / (x * (phi + dphi) + c1)
    return jnp.where(pos, temp, (2.0 / 3.0) * ekn), jnp.where(pos, gamma, 5.0 / 3.0)
