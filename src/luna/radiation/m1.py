"""Grey and multigroup M1 transport: Levermore closure, GLF flux, reduced speed of light.

R is (G, 1+ndim, N...) or grey (1+ndim, N...); c everywhere is c_hat.
See docs/RADIATION.md.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

E_FLOOR = 1e-30


def _closure(E, Fvec, chat):
    """Return (s, 1/|F|), s = sqrt(4 - 3 f^2), f = |F| / (c_hat E); Levermore chi = (5 - 2s) / 3."""
    # softened inside the sqrt: d|F|/dF is infinite at F=0 and reverse mode gives NaN
    Fmag = jnp.sqrt(sum(Fi * Fi for Fi in Fvec) + E_FLOOR * E_FLOOR)
    Em = jnp.maximum(E, E_FLOOR)
    Fm = jnp.maximum(Fmag, E_FLOOR)
    q = 1.0 / (chat * Em * Fm)
    f = jnp.clip(Fmag * Fm * q, 0.0, 1.0 - 1e-10)              # |F| / (c_hat E)
    return jnp.sqrt(4.0 - 3.0 * f * f), chat * Em * q          # 1/|F|; isotropic fallback where |F| ~ 0


def _pressure_row(E, Fvec, d, chat, ndim):
    """Return row d of the M1 pressure tensor, P_dj for j=0..ndim-1."""
    s, inv = _closure(E, Fvec, chat)
    n = [Fi * inv for Fi in Fvec]
    iso = (s - 1.0) * (1.0 / 3.0)                              # (1 - chi) / 2
    ani = 2.0 - s                                              # (3 chi - 1) / 2
    return [E * (iso * (1.0 if j == d else 0.0) + ani * n[d] * n[j])
            for j in range(ndim)]


def _phys_flux_axis(R, d, chat, ndim):
    """Physical flux of (E, F_0..F_{ndim-1}) along axis d, one group."""
    E, Fvec = R[0], [R[1 + j] for j in range(ndim)]
    P_row = _pressure_row(E, Fvec, d, chat, ndim)
    return jnp.stack([Fvec[d]] + [chat * chat * P for P in P_row])


def _minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a),
                                                          jnp.abs(b)), 0.0)


def ghosts(order):
    """Return ghost cells per side: 2 for MUSCL, 1 for donor cell."""
    return 2 if order == 2 else 1


def _pad_local(R, g, boundary):
    """Pad all spatial axes locally (single device)."""
    mode = "wrap" if boundary == 0 else "edge"
    return jnp.pad(R, [(0, 0)] + [(g, g)] * (R.ndim - 1), mode=mode)


def _transport_from_padded(R, Rp, dt, dx, *, chat, order=2):
    """One explicit GLF transport step for one group from a padded copy.

    R: (1+ndim, N...); Rp: R with `ghosts(order)` cells each side.
    order=1: donor cell; order=2: minmod-MUSCL.
    """
    ndim = R.ndim - 1
    g = ghosts(order)
    dR = jnp.zeros_like(R)
    for d in range(ndim):
        ax = 1 + d
        if d > 0:
            # barrier on the inputs too, else XLA builds all axes' face states at once
            dR, Rp = jax.lax.optimization_barrier((dR, Rp))

        def sl(a, b):
            s = [slice(g, -g)] * ndim
            s[d] = slice(a, b if b != 0 else None)
            return (slice(None),) + tuple(s)

        if order == 2:
            # cells d-index i-1..i+n (n+2 cells) around the n+1 faces
            C = Rp[sl(g - 2, -(g - 2))]                    # (.., n+4, ..)
            axsl = lambda a, b: tuple(                     # noqa: E731
                slice(a, b if b != 0 else None) if k == ax else slice(None)
                for k in range(ndim + 1))
            dl = C[axsl(1, -1)] - C[axsl(0, -2)]
            dr = C[axsl(2, 0)] - C[axsl(1, -1)]
            slope = _minmod(dl, dr)                        # (.., n+2, ..)
            Rc = C[axsl(1, -1)]
            RL = (Rc + 0.5 * slope)[axsl(0, -1)]           # left of each face
            RR = (Rc - 0.5 * slope)[axsl(1, 0)]            # right of each face
        else:                                              # donor cell (g == 1)
            RL = Rp[sl(0, -1)]
            RR = Rp[sl(1, 0)]
        fL = _phys_flux_axis(RL, d, chat, ndim)
        fR = _phys_flux_axis(RR, d, chat, ndim)
        # materialised: fused, XLA recomputes the closure once per output component
        fL, fR = jax.lax.optimization_barrier((fL, fR))
        face = 0.5 * (fL + fR) - 0.5 * chat * (RR - RL)   # GLF
        take_r = [slice(None)] * (ndim + 1)
        take_l = [slice(None)] * (ndim + 1)
        take_r[ax] = slice(1, None)
        take_l[ax] = slice(0, -1)
        dR = dR - dt / dx * (face[tuple(take_r)] - face[tuple(take_l)])
    return R + dR


def _grouped(R):
    """Return (R with a group axis, was_grey).

    Layouts are told apart by axis 1 having length R.ndim - 1, so a grey grid
    with x extent == ndim is misread.
    """
    grouped = R.ndim >= 3 and R.shape[1] == R.ndim - 1
    return (R if grouped else R[None]), not grouped


def m1_transport_step(R, dt, dx, *, chat, boundary=0, order=2, pad=None):
    """One explicit transport step for every group (vmap over groups).

    `pad(A, g)` maps (G*(1+ndim), N...) -> (G*(1+ndim), N+2g...); None pads
    locally, the sharded driver passes its halo exchange.
    """
    R, grey = _grouped(R)
    g = ghosts(order)
    if pad is None:
        Rp = jax.vmap(lambda r: _pad_local(r, g, boundary))(R)
    else:
        flat = pad(R.reshape((-1,) + R.shape[2:]), g)
        Rp = flat.reshape(R.shape[:2] + flat.shape[1:])
    out = jax.vmap(lambda r, rp: _transport_from_padded(
        r, rp, dt, dx, chat=chat, order=order))(R, Rp)
    return out[0] if grey else out


def _pad_axis_local(R, g, boundary, d):
    """Pad spatial axis d only, locally."""
    mode = "wrap" if boundary == 0 else "edge"
    width = [(0, 0)] * R.ndim
    width[1 + d] = (g, g)
    return jnp.pad(R, width, mode=mode)


def _hancock_sweep(R, Rp, dt, dx, d, *, chat):
    """One MUSCL-Hancock GLF step along axis d for one group; Rp has 2 ghosts along d."""
    ndim = R.ndim - 1
    ax = 1 + d

    def s(a, b):
        return tuple(slice(a, b if b != 0 else None) if k == ax else slice(None)
                     for k in range(ndim + 1))

    def edges():
        dl = Rp[s(1, -1)] - Rp[s(0, -2)]
        dr = Rp[s(2, 0)] - Rp[s(1, -1)]
        slope = _minmod(dl, dr)                            # cells -1..n
        Rc = Rp[s(1, -1)]
        return Rc - 0.5 * slope, Rc + 0.5 * slope          # (minus, plus) edges

    Rm, Rq = edges()
    # no barrier on these fluxes or in the closure: -16 % on an A100 (CODE_NOTES.md)
    Fq, Fm = _phys_flux_axis(Rq, d, chat, ndim), _phys_flux_axis(Rm, d, chat, ndim)
    inc = (0.5 * dt / dx) * (Fq - Fm)
    RL = (Rq - inc)[s(0, -1)]                              # left of each face
    RR = (Rm - inc)[s(1, 0)]                               # right of each face
    RL, RR = jax.lax.optimization_barrier((RL, RR))
    fL, fR = jax.lax.optimization_barrier((_phys_flux_axis(RL, d, chat, ndim),
                                           _phys_flux_axis(RR, d, chat, ndim)))
    face = 0.5 * (fL + fR) - 0.5 * chat * (RR - RL)
    return R - dt / dx * (face[s(1, 0)] - face[s(0, -1)])


def m1_split_step(R, dt, dx, *, chat, boundary=0, pad=None):
    """One transport step as split MUSCL-Hancock sweeps x y z z y x at dt/2 each.

    Stable for dt/2 <= cfl dx / c_hat (vs dx / (ndim c_hat) unsplit).
    """
    R, grey = _grouped(R)
    ndim = R.ndim - 2
    g = 2
    from luna.radiation import m1_pallas
    if m1_pallas.use_pallas(R):
        # fused sweeps, no XLA pad on axes this device holds whole (m1_pallas.split_step)
        local = tuple(range(ndim)) if pad is None else getattr(pad, "local_axes", ())
        R = m1_pallas.split_step(R, dt, dx, chat=chat, boundary=boundary, pad=pad, local=local,
                                 interpret=m1_pallas.interpret_mode())
        return R[0] if grey else R
    for d in list(range(ndim)) + list(range(ndim))[::-1]:
        if pad is None:
            Rp = jax.vmap(lambda r: _pad_axis_local(r, g, boundary, d))(R)
        else:
            flat = pad(R.reshape((-1,) + R.shape[2:]), g, axes=(d,))
            Rp = flat.reshape(R.shape[:2] + flat.shape[1:])
        R = jax.vmap(lambda r, rp: _hancock_sweep(r, rp, 0.5 * dt, dx, d, chat=chat))(R, Rp)
        # floor after every sweep: predictor overshoots at fronts compound otherwise
        R = m1_floor(R, chat)
        R = jax.lax.optimization_barrier(R)        # one sweep's scratch at a time
    return R[0] if grey else R


def m1_floor(R, chat):
    """Clip to the closure's domain: E >= 0 and |F| <= c_hat E (F rescaled)."""
    R, grey = _grouped(R)
    E = jnp.maximum(R[:, 0], 0.0)
    Fmag = jnp.sqrt(jnp.sum(R[:, 1:] ** 2, axis=1) + E_FLOOR * E_FLOOR)
    scale = jnp.minimum(1.0, chat * E / Fmag)
    out = jnp.concatenate([E[:, None], R[:, 1:] * scale[:, None]], axis=1)   # one pass, not two updates
    return out[0] if grey else out


def m1_step(R, dt, dx, *, chat, boundary=0, order=2, pad=None, floor=True,
            scheme="rk2"):
    """One transport step: scheme "rk2" (unsplit, SSP-RK2 at order 2) or "hancock_split"."""
    if scheme == "hancock_split":
        # every sweep already floors; a last floor only moves cells with c_hat E < E_FLOOR (CODE_NOTES.md)
        return m1_split_step(R, dt, dx, chat=chat, boundary=boundary, pad=pad)
    if scheme != "rk2":
        raise ValueError(f"rad_scheme={scheme!r}: 'rk2' or 'hancock_split'")
    R1 = m1_transport_step(R, dt, dx, chat=chat, boundary=boundary, order=order,
                           pad=pad)
    out = R1 if order == 1 else 0.5 * (R + m1_transport_step(
        R1, dt, dx, chat=chat, boundary=boundary, order=order, pad=pad))
    return m1_floor(out, chat) if floor else out


def m1_damp(R, dt, *, chat, kappa):
    """Damp the flux only, F *= exp(-c kappa dt); absorption is `m1_exchange`."""
    R, grey = _grouped(R)
    damp = jnp.exp(-chat * _as_group_field(kappa, R) * dt)
    out = R.at[:, 1:].multiply(damp[:, None])
    return out[0] if grey else out


def _as_group_field(kappa, R):
    """Broadcast a scalar, per-group or per-cell kappa to (G, N...)."""
    k = jnp.asarray(kappa)
    if k.ndim == 0:
        return jnp.broadcast_to(k, (R.shape[0],) + R.shape[2:])
    if k.ndim == 1:                                   # one per group
        return jnp.broadcast_to(k[(slice(None),) + (None,) * (R.ndim - 2)],
                                (R.shape[0],) + R.shape[2:])
    return jnp.broadcast_to(k, (R.shape[0],) + R.shape[2:])


def m1_exchange(R, e_gas, dt, *, chat, kappa_a, source):
    """Exact absorption/emission E_new = S + (E - S) exp(-c kappa_a dt); return (R, e_gas).

    The gas loses what the radiation gains. `source` S is in units of E.
    """
    R, grey = _grouped(R)
    ka = _as_group_field(kappa_a, R)
    S = _as_group_field(source, R)
    E = R[:, 0]
    E_new = S + (E - S) * jnp.exp(-chat * ka * dt)
    dE = jnp.sum(E_new - E, axis=0)
    out = R.at[:, 0].set(E_new)
    return (out[0] if grey else out), e_gas - dE


def m1_substeps(dt_hydro, dx, *, chat, ndim, cfl=0.4):
    """Return the substep count for one hydro dt as a Python int (dt must be concrete).

    Unsplit bound: dt_sub <= cfl dx / (ndim c_hat).
    """
    import math
    # relative slack so a ratio like 75.00000000000001 does not ceil to 76
    x = ndim * chat * float(dt_hydro) / (cfl * dx)
    return max(1, int(math.ceil(x - 1e-12 * max(1.0, x))))


def m1_advance(R, dt_hydro, dx, *, chat, kappa, boundary=0, cfl=0.4, order=2,
               n_sub=None, pad=None):
    """Advance radiation over one hydro dt in transport + flux-damping substeps (no gas)."""
    ndim = _grouped(R)[0].ndim - 2
    n = m1_substeps(dt_hydro, dx, chat=chat, ndim=ndim, cfl=cfl) if n_sub is None else int(n_sub)
    dt = dt_hydro / n

    def body(Rc, _):
        Rc = m1_step(Rc, dt, dx, chat=chat, boundary=boundary, order=order,
                     pad=pad)
        return m1_damp(Rc, dt, chat=chat, kappa=kappa), None

    R, _ = jax.lax.scan(body, R, None, length=n)
    return R
