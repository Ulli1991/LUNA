"""STARBENCH hydrogen ionization on the M1 field, read as photon number (Bisbas+2015).

Code units throughout; `starbench_units` converts the benchmark's cgs numbers.
See docs/RADIATION.md.
"""
import jax
import jax.numpy as jnp
import numpy as np

# sigma is not fixed by STARBENCH: the H photoionization cross-section at 13.6 eV
CASE_B_CGS = 2.7e-13          # cm^3 s^-1, case-B recombination at 1e4 K
SIGMA_HI_CGS = 6.3e-18        # cm^2, photoionization at 13.6 eV
PROTON_MASS_CGS = 1.6726219e-24
PC_CM = 3.0856775814913673e18
MYR_S = 3.1556952e13


def ionize(R, x, rho, dt, *, chat, sigma, alpha_B, m_H):
    """Exact absorption (capped by n_HI, photon-conserving) then exact recombination; return (R, x)."""
    n_H = rho / m_H
    n_HI = n_H * (1.0 - x)
    N = R[0, 0]                                   # photon number density
    absorbed = N * -jnp.expm1(-chat * sigma * n_HI * dt)
    dx = jnp.minimum(absorbed / jnp.maximum(n_H, 1e-300), 1.0 - x)
    N_new = jnp.maximum(N - dx * n_H, 0.0)        # photon-conserving
    # F scales with N, not by exp(-c kappa dt): see docs/RADIATION.md (ionization)
    keep = N_new / jnp.maximum(N, 1e-300)
    R = R.at[0, 0].set(N_new).at[0, 1:].multiply(keep)
    x = x + dx
    x = x / (1.0 + alpha_B * n_H * x * dt)        # exact recombination
    return R, jnp.clip(x, 0.0, 1.0)


def ionize_implicit(R, x, rho, dt, *, chat, sigma, alpha_B, m_H, iters=60):
    """Backward-Euler absorption + recombination per cell; return (R, x).

    Newton on the neutral fraction z = 1 - x in a log-z bracket (equation in docs/RADIATION.md).
    """
    from luna.radiation import m1_pallas
    if (m1_pallas.use_pallas(R) and R.shape[0] == 1
            and all(isinstance(v, (int, float)) for v in (chat, sigma, alpha_B, m_H))):
        return m1_pallas.ionize(R, x, rho, dt, chat=chat, sigma=sigma, alpha_B=alpha_B,
                                m_H=m_H, iters=iters, early=True)
    n_H = rho / m_H
    N0 = R[0, 0]
    P = N0 / jnp.maximum(n_H, 1e-300)
    a = chat * sigma * n_H * dt
    b = alpha_B * n_H * dt
    x0 = jnp.clip(x, 0.0, 1.0)

    def G(z):                                      # decreasing in z
        u = a * z
        y = 1.0 - z
        r = 1.0 / (1.0 + u)                        # one division for g and g'
        Pr = P * r
        return (y - x0 - Pr * u + b * y * y,
                -1.0 - Pr * a * r - 2.0 * b * y)

    def body(_, c):
        z, lo, hi = c
        gz, dg = G(z)
        lo = jnp.where(gz > 0.0, z, lo)            # root above
        hi = jnp.where(gz <= 0.0, z, hi)
        zn = z - gz / dg
        zn = jnp.where((zn > lo) & (zn < hi), zn, jnp.sqrt(lo * hi))
        # an exact root stays put (e.g. a dark neutral cell, z = 1)
        z = jnp.where(gz == 0.0, z, zn)
        return z, lo, hi

    Z_MIN = 1e-300                                 # G(Z_MIN) > 0 unless x0 = 1
    z0 = jnp.clip(1.0 - x0, Z_MIN, 1.0)
    # unrolled, not fori_loop: one elementwise kernel instead of one per iteration
    c = (z0, jnp.full_like(x0, Z_MIN), jnp.ones_like(x0))
    for i in range(iters):
        c = body(i, c)
    z = c[0]
    N1 = N0 / (1.0 + a * z)
    keep = N1 / jnp.maximum(N0, 1e-300)
    R = R.at[0, 0].set(N1).at[0, 1:].multiply(keep)
    # NaN photons bisect to z ~ 0 (x = 1, a silently ionised cell): pass the NaN on to x instead
    return R, jnp.where(jnp.abs(N0) < jnp.inf, 1.0 - jnp.clip(z, 0.0, 1.0), jnp.nan)


def opacity(x, rho, *, sigma, m_H):
    """Return kappa = sigma n_HI shaped (1, N...) (diagnostic; not used by `ionize`)."""
    return (sigma * (rho / m_H) * (1.0 - x))[None]


def transport_and_ionize(R, x, rho, dt, dx, *, n_sub, chat, sigma, alpha_B,
                         m_H, weight=None, rate=0.0, boundary=1, order=2,
                         pad=None, scheme="rk2", ion_sub=1, ion_solver="split",
                         closure="m1", params=None, kernel="auto"):
    """Run `n_sub` substeps of (inject, transport, ionize); return (R, x).

    `ion_sub` subcycles the split ionize inside each substep.
    """
    from luna.radiation.m1 import m1_step

    h = dt / n_sub

    def body(carry, _):
        R, x = carry
        if weight is not None:
            R = inject(R, weight, rate, h)
        if closure == "m1":
            R = m1_step(R, h, dx, chat=chat, boundary=boundary, order=order, pad=pad,
                        scheme=scheme)
        else:
            from luna.radiation import m2_stage
            R = m2_stage.transport(R, h, dx, chat=chat, boundary=boundary, pad=pad,
                                   closure=closure, params=params, kernel=kernel)
        if ion_solver == "implicit":
            R, x = ionize_implicit(R, x, rho, h, chat=chat, sigma=sigma,
                                   alpha_B=alpha_B, m_H=m_H)
        else:
            for _ in range(int(ion_sub)):
                R, x = ionize(R, x, rho, h / ion_sub, chat=chat, sigma=sigma,
                              alpha_B=alpha_B, m_H=m_H)
        return (R, x), None

    (R, x), _ = jax.lax.scan(body, (R, x), None, length=n_sub)
    return R, x


def inject(R, weight, rate, dt):
    """Add rate * dt photons isotropically, spread over `weight` (per unit volume, sum(weight dV) = 1)."""
    from luna.radiation.m2_stage import add_isotropic, is_m2
    if is_m2(R):
        return add_isotropic(R, jnp.zeros_like(R[:, 0]).at[0].add(rate * weight * dt))
    return R.at[0, 0].add(rate * weight * dt)


def source_weights(shape, dx, centre, radius):
    """Return a top-hat over cells within `radius` of `centre`, normalised so sum(weight dV) = 1."""
    axes = [(jnp.arange(n) + 0.5) * d - c for n, d, c in zip(shape, dx, centre)]
    r2 = sum(a.reshape([-1 if i == k else 1 for i in range(len(shape))]) ** 2
             for k, a in enumerate(axes))
    inside = (r2 <= radius ** 2).astype(jnp.float64)
    volume = jnp.sum(inside) * dx[0] * dx[1] * dx[2]
    return inside / volume


def deposit_updates(shape, dx, positions, rates, boundary=1):
    """`deposit_sources` as host-side point updates: (idx (8M, 3), vals (8M,)) for rate > 0."""
    Nx, Ny, Nz = (int(n) for n in shape)
    dV = dx[0] * dx[1] * dx[2]
    pos = np.asarray(positions, float).reshape(-1, 3)
    rates = np.asarray(rates, float).ravel()
    on = rates > 0.0
    pos, w = pos[on], rates[on] / dV
    if len(w) == 0:
        return np.zeros((0, 3), np.int64), np.zeros(0)
    f = [pos[:, k] / dx[k] - 0.5 for k in range(3)]
    i0 = [np.floor(fk).astype(np.int64) for fk in f]
    t = [fk - ik for fk, ik in zip(f, i0)]
    if boundary != 0:
        hi = (Nx - 2, Ny - 2, Nz - 2)
        i0 = [np.clip(ik, 0, h) for ik, h in zip(i0, hi)]
        t = [np.clip(fk - ik, 0.0, 1.0) for fk, ik in zip(f, i0)]
    idx, vals = [], []
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                wt = w * ((t[0] if di else 1 - t[0]) * (t[1] if dj else 1 - t[1])
                          * (t[2] if dk else 1 - t[2]))
                ijk = [i0[0] + di, i0[1] + dj, i0[2] + dk]
                if boundary == 0:
                    ijk = [np.mod(ix, n) for ix, n in zip(ijk, (Nx, Ny, Nz))]
                idx.append(np.stack(ijk, axis=1))
                vals.append(wt)
    return np.concatenate(idx), np.concatenate(vals)


def deposit_sources(shape, dx, positions, rates, active=None, boundary=1):
    """Return the photon rate per unit volume from point sources by CIC deposit.

    sum(field) dV = sum of active rates; same clamp/wrap rule as the gravity CIC.
    positions (N, 3), rates (N,), active (N,) gates empty slots; boundary 0 = periodic.
    """
    Nx, Ny, Nz = (int(n) for n in shape)
    dV = dx[0] * dx[1] * dx[2]
    rates = jnp.asarray(rates)
    if active is not None:
        rates = rates * jnp.asarray(active)
    w = rates / dV

    f = [jnp.asarray(positions)[:, k] / dx[k] - 0.5 for k in range(3)]
    i0 = [jnp.floor(fk).astype(jnp.int32) for fk in f]
    t = [fk - ik for fk, ik in zip(f, i0)]
    if boundary != 0:                       # non-periodic: clamp to the faces
        hi = (Nx - 2, Ny - 2, Nz - 2)
        i0 = [jnp.clip(ik, 0, h) for ik, h in zip(i0, hi)]
        t = [jnp.clip(fk - ik, 0.0, 1.0) for fk, ik in zip(f, i0)]

    field = jnp.zeros((Nx, Ny, Nz), dtype=w.dtype)
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                wt = w * ((t[0] if di else 1 - t[0])
                          * (t[1] if dj else 1 - t[1])
                          * (t[2] if dk else 1 - t[2]))
                idx = [i0[0] + di, i0[1] + dj, i0[2] + dk]
                if boundary == 0:           # periodic wrap
                    idx = [jnp.mod(ix, n) for ix, n in zip(idx, (Nx, Ny, Nz))]
                field = field.at[idx[0], idx[1], idx[2]].add(wt)
    return field


def two_temperature(U, x, *, cs2_neutral, cs2_ionized, gamma, neutral_table=None):
    """Set internal energy to P = rho cs^2(x), cs^2 = cs_o^2 + (cs_i^2 - cs_o^2) x.

    `neutral_table` = (ln rho, ln cs_o^2) makes cs_o density dependent (log-log interp).
    """
    rho = U[0]
    kinetic = 0.5 * jnp.sum(U[1:4] ** 2, axis=0) / rho
    if neutral_table is not None:
        lr, lc = neutral_table
        cs2_neutral = jnp.exp(jnp.interp(jnp.log(rho), lr, lc))
    cs2 = cs2_neutral + (cs2_ionized - cs2_neutral) * x
    return U.at[4].set(rho * cs2 / (gamma - 1.0) + kinetic)


def load_neutral_table(path):
    """Return (ln rho, ln cs^2) from a two-column text file (code units, rho increasing)."""
    import numpy as np
    t = np.loadtxt(path, ndmin=2)
    if t.shape[1] != 2 or np.any(np.diff(t[:, 0]) <= 0) or np.any(t <= 0):
        raise ValueError(f"{path}: two positive columns (rho, cs^2), rho increasing")
    return jnp.asarray(np.log(t[:, 0])), jnp.asarray(np.log(t[:, 1]))


def ionized_radius(x, dx, threshold=0.5):
    """Return the volume-equivalent radius of the region x > threshold."""
    volume = jnp.sum((x > threshold).astype(x.dtype)) * dx[0] * dx[1] * dx[2]
    return (3.0 * volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)


def starbench_units(phase="early", unit_length_cm=PC_CM, unit_time_s=MYR_S):
    """Return the STARBENCH parameters (Bisbas+2015 Table 1) in code units, with rho_0 = 1."""
    rho0_cgs = 5.21e-21
    ndot = 1.0e49
    ci_cgs = 12.85e5
    co_cgs = 0.91e5 if phase == "early" else 2.87e5
    t_end_myr = 0.141 if phase == "early" else 3.0

    L, T = float(unit_length_cm), float(unit_time_s)
    V = L / T                                    # velocity unit
    M = rho0_cgs * L ** 3                        # so that rho_0 = 1
    n_H_cgs = rho0_cgs / PROTON_MASS_CGS
    R_St_cm = (3.0 * ndot / (4.0 * jnp.pi * CASE_B_CGS * n_H_cgs ** 2)) ** (1 / 3)
    return {
        "rho0": 1.0,
        "m_H": PROTON_MASS_CGS / M,              # mass per H nucleus, code
        "n_H": float(n_H_cgs * L ** 3),          # per code volume
        "cs2_neutral": (co_cgs / V) ** 2,
        "cs2_ionized": (ci_cgs / V) ** 2,
        "c_i_kms": ci_cgs / 1e5,
        "photon_rate": ndot * T,                 # photons per code time
        "sigma": SIGMA_HI_CGS / L ** 2,
        "alpha_B": CASE_B_CGS * T / L ** 3,
        "R_St": float(R_St_cm) / L,
        "t_end": t_end_myr * MYR_S / T,
        "unit_length_cm": L,
        "unit_time_s": T,
        "unit_mass_g": M,
    }


def spitzer_radius(t, R_St, c_i):
    """R_Sp = R_St (1 + 7 c_i t / (4 R_St))^(4/7)  (Bisbas+15 eq. 9)."""
    return R_St * (1.0 + 1.75 * c_i * t / R_St) ** (4.0 / 7.0)


def hosokawa_inutsuka_radius(t, R_St, c_i):
    """R_HI = R_St (1 + 7 sqrt(4/3) c_i t / (4 R_St))^(4/7)  (eq. 13)."""
    return R_St * (1.0 + 1.75 * (4.0 / 3.0) ** 0.5 * c_i * t / R_St) ** (4.0 / 7.0)
