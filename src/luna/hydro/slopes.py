"""Slope limiters and limited slopes from ghost-padded arrays for MUSCL-Hancock."""

from contextlib import contextmanager
from dataclasses import dataclass
import jax
import jax.numpy as jnp

def minmod(a, b):
    """Minmod limiter: smaller-magnitude of a, b when same sign, else 0."""
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)


def minmod3(a, b, c):
    """Three-argument minmod: smallest magnitude when all share a sign, else 0."""
    s = jnp.sign(a)
    return jnp.where(
        (a * b > 0) & (a * c > 0),
        s * jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c))),
        0.0,
    )

def moncen(dl, dr):
    """Monotonized-central limiter: minmod3(2 dl, 2 dr, (dl+dr)/2)."""
    return minmod3(2 * dl, 2 * dr, 0.5 * (dl + dr))

def vanleer(dl, dr):
    """Van Leer harmonic limiter: 2 dl dr / (dl + dr) when dl*dr > 0, else 0."""
    # Safe denominator and a normal-float (not 0) threshold: otherwise the VJP is NaN
    # (0/0) in flat regions and for subnormal slopes.
    tiny = jnp.finfo(jnp.result_type(dl, dr)).tiny
    same_sign = dl * dr > tiny
    denom = jnp.where(same_sign, dl + dr, 1.0)
    return jnp.where(same_sign, 2 * dl * dr / denom, 0.0)

def unphysical_face(Wf, ip):
    """True where the face state Wf (rho row 0, p row ip; array or list of rows) has rho <= 0 or p <= 0 (audit C8)."""
    return (Wf[0] <= 0.0) | (Wf[ip] <= 0.0)


def first_order_if_unphysical(Wf, Wc, allow=True):
    """Face states Wf (rho, v.., p first on axis 0), the cell-centre Wc where Wf is unphysical (audit C8); allow=False: NaN there (refused)."""
    bad = unphysical_face(Wf, Wf.ndim)         # p after rho and one velocity per spatial axis
    return jnp.where(bad, Wc.astype(Wf.dtype) if allow else jnp.nan, Wf)


def first_order_if_unphysical_rows(Wf, Wc, ip, allow=True):
    """As first_order_if_unphysical on lists of rows (Pallas tiles), p at row ip."""
    bad = unphysical_face(Wf, ip)
    return [jnp.where(bad, c.astype(f.dtype) if allow else jnp.asarray(jnp.nan, f.dtype), f)
            for f, c in zip(Wf, Wc)]


_RECORDERS = []


@contextmanager
def recording_faces():
    """Collect (first-order mask, floor mask or None) of every XLA face_guard_fn call while open, per-cell arrays."""
    rec = []
    _RECORDERS.append(rec)
    try:
        yield rec
    finally:
        _RECORDERS.pop()


@dataclass(frozen=True)
class SmoothTangentMinmod:
    """Limiter spec: exact minmod forward, tangent of a minmod smoothed over ``eps`` x cell scale."""
    eps: float

def smooth_minmod(a, b, e):
    """Minmod with sign and min smoothed over the scale e > 0."""
    A = jnp.sqrt(a * a + e * e)
    B = jnp.sqrt(b * b + e * e)
    soft_min = 0.5 * (A + B - jnp.sqrt((A - B) ** 2 + e * e) + e)
    return 0.5 * (a / A + b / B) * soft_min

@jax.custom_jvp
def minmod_smooth_tangent(a, b, e):
    """minmod(a, b) exactly; its tangent is smooth_minmod's."""
    return minmod(a, b)

@minmod_smooth_tangent.defjvp
def _minmod_smooth_tangent_jvp(primals, tangents):
    a, b, e = primals
    da, db, _ = tangents
    e = jax.lax.stop_gradient(e)
    t = jax.jvp(lambda a, b: smooth_minmod(a, b, e), (a, b), (da, db))[1]
    return minmod(a, b), t

def _tangent_scale(Qc, eps):
    """Smoothing scale eps x (rho, sqrt(p/rho) per velocity, p) of a primitive block."""
    rho, p = jnp.abs(Qc[0]), jnp.abs(Qc[-1])
    cs = jnp.sqrt(p / rho)
    return eps * jnp.stack([rho] + [cs] * (Qc.shape[0] - 2) + [p])

def slope_limiter(dl, dr, limiter, Qc=None):
    """Apply the static ``limiter``: 0 minmod, 1 moncen, 2 vanleer, or SmoothTangentMinmod (needs Qc)."""
    if isinstance(limiter, SmoothTangentMinmod):
        return minmod_smooth_tangent(dl, dr, _tangent_scale(Qc, limiter.eps))
    if limiter == 0:
        return minmod(dl, dr)
    if limiter == 1:
        return moncen(dl, dr)
    if limiter == 2:
        return vanleer(dl, dr)
    raise ValueError(f"Unknown limiter: {limiter}")

def compute_slopes_from_padded(U_pad, limiter):
    """Limited interior slopes (dUx, dUy, dUz) from a one-ghost pad; unused axes are None."""
    ndim = U_pad.ndim - 1

    if ndim == 1:
        dx_L = U_pad[:, 1:-1] - U_pad[:, :-2]
        dx_R = U_pad[:, 2:] - U_pad[:, 1:-1]
        dUx = slope_limiter(dx_L, dx_R, limiter, U_pad[:, 1:-1])
        dUy = None
        dUz = None
        return dUx, dUy, dUz

    if ndim == 2:
        Qc = U_pad[:, 1:-1, 1:-1]

        dx_L = Qc - U_pad[:, :-2, 1:-1]
        dx_R = U_pad[:, 2:, 1:-1] - Qc
        dy_D = Qc - U_pad[:, 1:-1, :-2]
        dy_U = U_pad[:, 1:-1, 2:] - Qc

        dUx = slope_limiter(dx_L, dx_R, limiter, Qc)
        dUy = slope_limiter(dy_D, dy_U, limiter, Qc)
        dUz = None
        return dUx, dUy, dUz

    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]

    dx_L = Qc - U_pad[:, :-2, 1:-1, 1:-1]
    dx_R = U_pad[:, 2:, 1:-1, 1:-1] - Qc
    dy_D = Qc - U_pad[:, 1:-1, :-2, 1:-1]
    dy_U = U_pad[:, 1:-1, 2:, 1:-1] - Qc
    dz_B = Qc - U_pad[:, 1:-1, 1:-1, :-2]
    dz_F = U_pad[:, 1:-1, 1:-1, 2:] - Qc

    dUx = slope_limiter(dx_L, dx_R, limiter, Qc)
    dUy = slope_limiter(dy_D, dy_U, limiter, Qc)
    dUz = slope_limiter(dz_B, dz_F, limiter, Qc)

    return dUx, dUy, dUz

def _compute_slope_3d_axis(U_pad, limiter, d):
    """Limited slope along axis ``d`` (0=x, 1=y, 2=z) of a 3D padded array."""
    Qc = U_pad[:, 1:-1, 1:-1, 1:-1]
    lo = [slice(1, -1)] * 3
    hi = [slice(1, -1)] * 3
    lo[d] = slice(None, -2)
    hi[d] = slice(2, None)
    d_L = Qc - U_pad[(slice(None), *lo)]
    d_R = U_pad[(slice(None), *hi)] - Qc
    return slope_limiter(d_L, d_R, limiter, Qc)

def compute_slope_x_3d_from_padded(U_pad, limiter):
    """Limited x-slope of a 3D padded array."""
    return _compute_slope_3d_axis(U_pad, limiter, 0)

def compute_slope_y_3d_from_padded(U_pad, limiter):
    """Limited y-slope of a 3D padded array."""
    return _compute_slope_3d_axis(U_pad, limiter, 1)

def compute_slope_z_3d_from_padded(U_pad, limiter):
    """Limited z-slope of a 3D padded array."""
    return _compute_slope_3d_axis(U_pad, limiter, 2)


def _with(Wf, ip, rho, p):
    """Wf with density and pressure replaced (a list, or an array stacked on axis 0)."""
    if isinstance(Wf, list):
        return [rho] + Wf[1:ip] + [p] + Wf[ip + 1:]
    return jnp.concatenate([rho[None], Wf[1:ip], p[None], Wf[ip + 1:]])


def ramses_face(Wf, Wc, ip, smallr, smallc, gamma):
    """RAMSES face floors: trace guard (umuscl.f90:422-441), then the Riemann inputs (godunov_utils.f90:370-383)."""
    rho = jnp.where(Wf[0] < smallr, Wc[0], Wf[0])
    p = jnp.where(Wf[ip] < smallr * smallc ** 2, Wc[ip], Wf[ip])
    rho = jnp.maximum(rho, smallr)
    return _with(Wf, ip, rho, jnp.maximum(p, rho * (smallc ** 2 / gamma)))


def face_spec(floors=None, gamma=None, allow_first_order=False):
    """The face_guard spec: (allow_first_order, (smallr, smallc, gamma) or None); hashable (Pallas cache key)."""
    return (bool(allow_first_order), (float(floors[0]), float(floors[1]), float(gamma)) if floors else None)


def _parse_spec(spec):
    """(allow, floors) of a face_guard spec: False (refuse, no floors), a (smallr, smallc, gamma) triple (refuse, floors) or face_spec."""
    if not spec:
        return False, None
    if len(spec) == 3:
        return False, tuple(spec)
    return bool(spec[0]), spec[1]


def ramses_replaced(Wf, ip, smallr, smallc):
    """True where ramses_face takes the cell-centre rho or p (the trace guard)."""
    return (Wf[0] < smallr) | (Wf[ip] < smallr * smallc ** 2)


def face_rule(spec):
    """(Wf, Wc, ip) -> (face state, first-order mask, floor mask or None): audit C8 (allowed: first order, else NaN, or with floors none), then the RAMSES floors."""
    allow, floors = _parse_spec(spec)
    rows = None

    def rule(Wf, Wc, ip):
        bad = unphysical_face(Wf, ip)
        if floors is not None and not allow:
            # the floors alone: rho < smallr and p < smallr smallc^2 take the cell's value, so no face stays
            # unphysical and none is first order by C8; they are counted as floored
            smallr, smallc, gamma = floors
            return (ramses_face(Wf, Wc, ip, smallr, smallc, gamma), bad & False,
                    ramses_replaced(Wf, ip, smallr, smallc))
        W = (first_order_if_unphysical_rows(Wf, Wc, ip, allow) if isinstance(Wf, list)
             else jnp.where(bad, Wc.astype(Wf.dtype) if allow else jnp.nan, Wf))
        if floors is None:
            return W, bad, rows
        smallr, smallc, gamma = floors
        return ramses_face(W, Wc, ip, smallr, smallc, gamma), bad, ramses_replaced(W, ip, smallr, smallc)
    return rule


def face_guard_fn(spec):
    """(Wf, Wc, ip) -> face state under face_rule(spec); records the masks while recording_faces is open."""
    rule = face_rule(spec)

    def guard(Wf, Wc, ip):
        W, bad, floored = rule(Wf, Wc, ip)
        if _RECORDERS:
            _RECORDERS[-1].append((bad, floored))
        return W
    return guard
