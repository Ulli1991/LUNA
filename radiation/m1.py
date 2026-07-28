"""Grey M1 radiation transport: first-order Godunov + GLF, RSLA, exp damping.

The 3D generalization of Ulli's 1D M1 notebook (Levermore closure, global
Lax-Friedrichs face flux, opacity damping), the RAMSES-RT recipe (Rosdahl et
al. 2013) in grey (frequency-integrated) form:

    d_t E   + div F           = 0          (transport; sources separate)
    d_t F_i + c^2 d_j P_ij    = -c kappa F (flux damping by opacity)
    P_ij = E [ (1-chi)/2 delta_ij + (3chi-1)/2 n_i n_j ],  n = F/|F|
    chi(f) = (3 + 4 f^2) / (5 + 2 sqrt(4 - 3 f^2)),  f = |F| / (c E)

c here is the REDUCED speed of light c_hat (RSLA); the caller picks it
(>~ 10x max signal speed). The damping is applied as the exact integral
F *= exp(-c kappa dt): unconditionally stable at any optical depth, unlike
the notebook's explicit form. State R has shape (4, Nx[, Ny[, Nz]]):
(E, F_x[, F_y[, F_z]]) -- the module is dimension-agnostic like the hydro.

First-order in space by design (standard for M1; the closure's own diffusion
dominates any reconstruction gain) => single ghost cell, GLF dissipation
|jump| * c_hat / 2.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

E_FLOOR = 1e-30


def eddington_chi(f):
    """Levermore M1 Eddington factor; chi(0)=1/3, chi(1)=1."""
    f = jnp.clip(f, 0.0, 1.0 - 1e-10)
    return (3.0 + 4.0 * f * f) / (5.0 + 2.0 * jnp.sqrt(4.0 - 3.0 * f * f))


def _pressure_row(E, Fvec, d, chat, ndim):
    """Row d of the M1 pressure tensor: P_dj for j=0..ndim-1."""
    Fmag = jnp.sqrt(sum(Fi * Fi for Fi in Fvec))
    f = Fmag / (chat * jnp.maximum(E, E_FLOOR))
    chi = eddington_chi(f)
    # unit direction; isotropic fallback where |F| ~ 0
    inv = 1.0 / jnp.maximum(Fmag, E_FLOOR)
    n = [Fi * inv for Fi in Fvec]
    iso = 0.5 * (1.0 - chi)
    ani = 0.5 * (3.0 * chi - 1.0)
    return [E * (iso * (1.0 if j == d else 0.0) + ani * n[d] * n[j])
            for j in range(ndim)]


def _phys_flux_axis(R, d, chat, ndim):
    """Physical flux of (E, F_0..F_{ndim-1}) along axis d."""
    E, Fvec = R[0], [R[1 + j] for j in range(ndim)]
    P_row = _pressure_row(E, Fvec, d, chat, ndim)
    return jnp.stack([Fvec[d]] + [chat * chat * P for P in P_row])


def _pad(R, boundary, ndim):
    """One ghost cell per side: periodic wrap (0) or zero-gradient (else)."""
    mode = "wrap" if boundary == 0 else "edge"
    return jnp.pad(R, [(0, 0)] + [(1, 1)] * ndim, mode=mode)


def _minmod(a, b):
    return jnp.where(a * b > 0, jnp.sign(a) * jnp.minimum(jnp.abs(a),
                                                          jnp.abs(b)), 0.0)


def m1_transport_step(R, dt, dx, *, chat, boundary=0, order=2):
    """One explicit GLF transport step of the grey M1 system.

    R: (1+ndim, N...) = (E, F...). Requires the RT CFL dt <= CFL * dx / chat.
    order=1: donor cell. order=2: minmod-MUSCL face reconstruction -- kills
    the O(chat*dx) GLF diffusion floor that otherwise swamps the physical
    D = chat/(3 kappa) whenever kappa*dx is not << 1.
    """
    ndim = R.ndim - 1
    g = 2 if order == 2 else 1
    mode = "wrap" if boundary == 0 else "edge"
    Rp = jnp.pad(R, [(0, 0)] + [(g, g)] * ndim, mode=mode)
    dR = jnp.zeros_like(R)
    for d in range(ndim):
        ax = 1 + d
        n = R.shape[ax]

        def sl(a, b):
            s = [slice(g, -g)] * ndim
            s[d] = slice(a, b if b != 0 else None)
            return (slice(None),) + tuple(s)

        if order == 2:
            # cells d-index i-1..i+n (n+2 cells) around the n+1 faces
            C = Rp[sl(g - 2, -(g - 2))]                    # (.., n+4, ..)
            axsl = lambda a, b: tuple(
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
        face = 0.5 * (fL + fR) - 0.5 * chat * (RR - RL)   # GLF
        take_r = [slice(None)] * (ndim + 1)
        take_l = [slice(None)] * (ndim + 1)
        take_r[ax] = slice(1, None)
        take_l[ax] = slice(0, -1)
        dR = dR - dt / dx * (face[tuple(take_r)] - face[tuple(take_l)])
    return R + dR


def m1_step(R, dt, dx, *, chat, boundary=0, order=2):
    """One transport step; order=2 uses SSP-RK2 (Heun) -- MUSCL reconstruction
    with forward Euler alone is not advection-stable."""
    R1 = m1_transport_step(R, dt, dx, chat=chat, boundary=boundary, order=order)
    if order == 1:
        return R1
    R2 = m1_transport_step(R1, dt, dx, chat=chat, boundary=boundary, order=order)
    return 0.5 * (R + R2)


def m1_damp(R, dt, *, chat, kappa):
    """Exact opacity damping of the flux: F *= exp(-c kappa dt).
    kappa may be a scalar or a per-cell field (grey absorption+scattering)."""
    damp = jnp.exp(-chat * kappa * dt)
    return R.at[1:].multiply(damp)


def m1_substep(R, dt_hydro, dx, *, chat, kappa, boundary=0, cfl=0.4, order=2):
    """Advance radiation over one hydro dt with RSLA substeps (lax.scan).

    The update is UNSPLIT (all-axis fluxes from one time level, like the
    hydro), so the stable step is cfl * dx / (ndim * chat) -- the multi-D
    CFL divides by ndim, it is not the 1D bound."""
    ndim = R.ndim - 1
    n_sub = jnp.maximum(1, jnp.ceil(
        ndim * chat * dt_hydro / (cfl * dx))).astype(int)
    # static substep count is required under jit: caller passes concrete dt
    # (fixed-grid loop) or wraps with a padded max count. Host-side here:
    n_sub = int(n_sub)
    dt = dt_hydro / n_sub

    def body(Rc, _):
        Rc = m1_step(Rc, dt, dx, chat=chat, boundary=boundary, order=order)
        return m1_damp(Rc, dt, chat=chat, kappa=kappa), None

    R, _ = jax.lax.scan(body, R, None, length=n_sub)
    return R
