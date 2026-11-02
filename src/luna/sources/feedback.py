"""Stellar feedback (SN energy, ejecta, winds) deposited into the gas state U, code units.

SeBa decides when and how much; this applies it. See docs/TORCH.md."""
import jax.numpy as jnp
import numpy as np


def _grid_coords(cfg):
    dx = cfg.Lx / cfg.Nx
    xc = (np.arange(cfg.Nx) + 0.5) * dx
    yc = (np.arange(cfg.Ny) + 0.5) * (cfg.Ly / cfg.Ny)
    zc = (np.arange(cfg.Nz) + 0.5) * (cfg.Lz / cfg.Nz)
    X, Y, Z = jnp.meshgrid(jnp.asarray(xc), jnp.asarray(yc), jnp.asarray(zc), indexing="ij")
    return X, Y, Z, dx ** 3


def _offsets(cfg, x, y, z):
    """Cell centres minus (x, y, z), to the nearest image in a periodic box."""
    X, Y, Z, dV = _grid_coords(cfg)
    d = [X - x, Y - y, Z - z]
    if int(getattr(cfg, "BOUNDARY", 1)) == 0:
        d = [q - L * jnp.floor(q / L + 0.5) for q, L in zip(d, (cfg.Lx, cfg.Ly, cfg.Lz))]
    return d, dV


def simpson_kinetic_fraction(cfg, bg_rho, energy):
    """FLASH-Torch's SN kinetic fraction (Simpson et al. 2015) at mean injection density `bg_rho`."""
    u = cfg.units
    if not u.is_enabled:
        raise ValueError("a negative f_kin needs physical units (cfg.units)")
    n = u.to_cgs_density(bg_rho) / (1.3 * 1.6726e-24)           # cm^-3
    e51 = u.to_cgs_energy(energy) / 1e51
    t_pds = 26.5 * e51 ** (3.0 / 14.0) * n ** (-4.0 / 7.0)      # kyr
    r_pds = 18.5 * e51 ** (2.0 / 7.0) * n ** (-3.0 / 7.0)       # pc
    dx_cm = u.to_cgs_length(min(cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny, cfg.Lz / cfg.Nz))
    if 4.5 * dx_cm < r_pds * 3.0856775814913673e18:
        return 0.0
    return 3.97e-6 * n * r_pds ** 7 * t_pds ** -2 * (dx_cm / 3.0856775814913673e18) ** -2 / e51


def inject_energy(U, cfg, x, y, z, energy, mass=0.0, radius=None, f_kin=0.0):
    """Deposit `energy` and `mass` within `radius`; `f_kin` (<0: Simpson) as a radial kick."""
    (DX, DY, DZ), dV = _offsets(cfg, x, y, z)
    if radius is None:
        radius = float(getattr(cfg, "feedback_radius", 0.0) or 0.0) or 3.0 * (cfg.Lx / cfg.Nx)
    r2 = DX ** 2 + DY ** 2 + DZ ** 2
    inside = (r2 < radius * radius)
    vol = jnp.maximum(jnp.sum(inside) * dV, 1e-30)
    msk = inside.astype(U.dtype)
    if f_kin < 0.0:
        f_kin = simpson_kinetic_fraction(cfg, float(jnp.sum(msk * U[0]) * dV / vol), energy)

    U = U.at[0].add(msk * (mass / vol))

    U = U.at[4].add(msk * ((1.0 - f_kin) * energy / vol))

    if f_kin > 0.0:
        rho = jnp.maximum(U[0], 1e-30)
        ke_dens = f_kin * energy / vol
        r = jnp.sqrt(r2) + 1e-30
        ex, ey, ez = DX / r, DY / r, DZ / r
        p = jnp.sqrt(2.0 * rho * ke_dens) * msk
        U = (U.at[1].add(p * ex).at[2].add(p * ey).at[3].add(p * ez)
             .at[4].add(msk * ke_dens))
    return U


def _keep_internal(U_old, U_new):
    """U_new with U_old's internal energy: E changes only by the bulk kinetic energy."""
    def ke(U):
        return 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
    return U_new.at[4].set(U_old[4] + (ke(U_new) - ke(U_old)))


def _zero_net(e):
    """Directions e (M, 3) minus their mean, rescaled to keep sum |e|^2: sum e = 0 (docs/TORCH.md)."""
    s0 = float((e * e).sum())
    e = e - e.mean(axis=0)
    s1 = float((e * e).sum())
    if s0 > 0.0 and not s1 > 0.0:
        raise ValueError(f"a {len(e)}-cell wind window has no radial pattern of zero net "
                         "momentum; enlarge the wind radius")
    return e * np.sqrt(s0 / s1) if s0 > 0.0 else e


def _zero_net_grid(ex, ey, ez, msk):
    """`_zero_net` on the cells of the 0/1 mask `msk` (0 outside), differentiable."""
    n = jnp.maximum(jnp.sum(msk), 1.0)
    e = [msk * (c - jnp.sum(msk * c) / n) for c in (ex, ey, ez)]
    s0 = jnp.sum(msk * (ex * ex + ey * ey + ez * ez))
    s1 = jnp.sum(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
    ok = s1 > 0.0
    scale = jnp.where(ok, jnp.sqrt(s0 / jnp.where(ok, s1, 1.0)), 0.0)
    return tuple(c * scale for c in e)


def inject_wind(U, cfg, x, y, z, dm, v_term, radius=None, v_star=(0.0, 0.0, 0.0),
                add_heat=False):
    """Add wind mass `dm` at `v_term` (+ `v_star`); internal energy kept unless `add_heat`."""
    (DX, DY, DZ), dV = _offsets(cfg, x, y, z)
    if radius is None:
        radius = float(getattr(cfg, "feedback_radius", 0.0) or 0.0) or 3.0 * (cfg.Lx / cfg.Nx)
    r2 = DX ** 2 + DY ** 2 + DZ ** 2
    msk = (r2 < radius * radius).astype(U.dtype)
    vol = jnp.maximum(jnp.sum(msk) * dV, 1e-30)
    drho = (dm / vol) * msk
    rr = jnp.sqrt(r2) + 1e-30
    ex, ey, ez = _zero_net_grid(DX / rr, DY / rr, DZ / rr, msk)
    vex = v_star[0] + v_term * ex
    vey = v_star[1] + v_term * ey
    vez = v_star[2] + v_term * ez
    U0 = U
    U = U.at[0].add(drho)
    U = U.at[1].add(drho * vex).at[2].add(drho * vey).at[3].add(drho * vez)
    if add_heat:
        return U.at[4].add(0.5 * drho * (vex ** 2 + vey ** 2 + vez ** 2))
    return _keep_internal(U0, U)


def mass_load_factor(v_term, v_load):
    """Wind mass loading factor f = max(v_term / v_load, 1); v_load <= 0 is off (f = 1)."""
    v_term = np.asarray(v_term, float)
    if not v_load or v_load <= 0.0:
        return np.ones_like(v_term)
    return np.maximum(v_term / v_load, 1.0)


def wind_updates(pos, vel, dm, v_term, N3, dxyz, periodic, radius, v_load=0.0):
    """Host-side wind cells and increments of every star: (idx (M, 3), vals (5, M))."""
    pos, vel = np.asarray(pos, float).reshape(-1, 3), np.asarray(vel, float).reshape(-1, 3)
    dm, v_term = np.asarray(dm, float).ravel(), np.asarray(v_term, float).ravel()
    # energy the sweep-up dissipates, per unit ejecta mass: (1 - 1/f) v_term^2 / 2
    loss = 0.5 * v_term ** 2 * (1.0 - 1.0 / mass_load_factor(v_term, v_load))
    d = np.asarray(dxyz, float)
    dV = float(np.prod(d))
    n_ax = np.asarray(N3)
    R = int(np.ceil(radius / d.min())) + 1
    off = np.arange(-R, R + 1)
    O = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    idx_all, vals_all = [], []
    for p, v, m, vt, q in zip(pos, vel, dm, v_term, loss):
        cell = np.floor(p / d).astype(np.int64) + O
        dr = (cell + 0.5) * d - p
        r2 = (dr * dr).sum(axis=1)
        inside = r2 < radius * radius
        if not periodic:
            inside &= np.all((cell >= 0) & (cell < n_ax), axis=1)
        if not inside.any():
            continue
        cell, dr, r2 = cell[inside], dr[inside], r2[inside]
        idx = np.mod(cell, n_ax) if periodic else np.clip(cell, 0, n_ax - 1)
        drho = m / (len(idx) * dV)
        e = _zero_net(dr / (np.sqrt(r2) + 1e-30)[:, None])
        ve = v[None, :] + vt * e
        idx_all.append(idx)
        vals_all.append(np.stack([np.full(len(idx), drho), drho * ve[:, 0], drho * ve[:, 1],
                                  drho * ve[:, 2], 0.5 * drho * (ve * ve).sum(axis=1) - drho * q]))
    if not idx_all:
        return np.zeros((0, 3), np.int64), np.zeros((5, 0))
    return np.concatenate(idx_all), np.concatenate(vals_all, axis=1)


def make_wind_injector(N3, dxyz, periodic, radius=None, add_heat=False, v_load=0.0,
                       sharding=None):
    """Build `inject(U, pos, vel, dm, v_term, active) -> (U, n, dm_sum, E_sum)`, unloaded sums."""
    from luna.utils.grid_scatter import make_grid_scatter
    radius = 3.0 * float(dxyz[0]) if radius is None else float(radius)
    R = int(np.ceil(radius / min(dxyz))) + 1
    if 2 * R + 1 > min(N3):
        raise ValueError(f"wind radius {radius} needs a {2 * R + 1}-cell window; "
                         f"the grid is {N3}")
    scatter = make_grid_scatter(sharding, n_comp=5,
                                after=None if add_heat else _keep_internal)

    def inject(U, pos, vel, dm, v_term, active):
        pos, vel, dm, v_term, active = (np.asarray(jax_get(a)) for a in (pos, vel, dm, v_term, active))
        emit = active.astype(bool) & (dm > 0.0)
        n = int(emit.sum())
        if n == 0:
            return U, 0, 0.0, 0.0
        idx, vals = wind_updates(pos[emit], vel[emit], dm[emit], v_term[emit], N3, dxyz,
                                 periodic, radius, v_load)
        if idx.shape[0]:
            U = scatter(U, idx, vals)
        return U, n, float(dm[emit].sum()), float((0.5 * dm[emit] * v_term[emit] ** 2).sum())

    return inject


def jax_get(a):
    """A device array (or anything) as host numpy."""
    import jax
    return jax.device_get(a)


# -- FLASH-Torch's inject_direct.F90 (wind_mass_load_mode "torch"/"sink"); docs/TORCH.md

TORCH_REF_RADIUS_CELLS = 3.5 * np.sqrt(3.0)     # FLASH ref_radius = -1
WEAVER_COEFF = 0.74296                          # Weaver et al. 1977, eq. 12
LOAD_MODES = ("conservative", "torch", "sink")


def torch_load(v_term, v_ref, cons_quant="momentum"):
    """inject_direct's (F, v_inj): F dm injected at v_inj; no floor at F = 1, as in FLASH."""
    v_term = np.asarray(v_term, float)
    if cons_quant == "momentum":
        F = 1.0 + (v_term / v_ref - 1.0)
        return F, v_term / F
    if cons_quant == "energy":
        F = 1.0 + (v_term ** 2 / v_ref ** 2 - 1.0)
        return F, v_term / np.sqrt(F)
    raise ValueError(f"wind_cons_quant {cons_quant!r}: 'momentum' or 'energy'")


def weaver_radius(mdot, bg_rho, v_term, t_wind):
    """Weaver's R_1 = 0.74296 (Mdot/rho0)^0.3 v_w^0.1 t_w^0.4, any consistent units."""
    return (WEAVER_COEFF * (np.asarray(mdot, float) / bg_rho) ** 0.3
            * np.asarray(v_term, float) ** 0.1 * np.asarray(t_wind, float) ** 0.4)


def torch_wind_radius(mdot, bg_rho, v_term, t_wind, dx, r_ref, var_radius):
    """inject_direct's radius: r_ref, or R_1 clipped to [sqrt(3) dx, r_ref] once bg_rho > 0."""
    bg_rho = np.asarray(bg_rho, float)
    r = np.full(bg_rho.shape, float(r_ref))
    if var_radius:
        known = bg_rho > 0.0
        r1 = weaver_radius(mdot, np.where(known, bg_rho, 1.0), v_term, t_wind)
        r = np.where(known, np.minimum(r1, r_ref), r)
    return np.maximum(r, np.sqrt(3.0) * dx)


def flash_overlap_kernel(r, R):
    """FLASH's overlap weight of a point at distance r from the centre of a sphere of radius R."""
    s2, s3 = np.sqrt(2.0), np.sqrt(3.0)
    d = R / s3
    ratio = r / d
    a = (3.0 / s2 - 2.0) / (s3 - s2) ** 2
    mid = 3.0 * d / jnp.maximum(r, 1e-300) - 2.0
    return jnp.where(ratio <= 1.0, 1.0,
                     jnp.where(ratio <= s2, mid,
                               jnp.where(r <= R, a * (s3 - ratio) ** 2, 0.0)))


def flash_overlap(rel, R, d, nsteps=10):
    """FLASH's overlap(): the kernel averaged over nsteps^3 points of each cell (rel = centre - star)."""
    s = (jnp.arange(nsteps) + 0.5) / nsteps - 0.5
    q = rel[..., None, :] + s[:, None] * jnp.asarray(d)       # (..., nsteps, 3)
    x2, y2, z2 = q[..., 0] ** 2, q[..., 1] ** 2, q[..., 2] ** 2
    r = jnp.sqrt(x2[..., :, None, None] + y2[..., None, :, None] + z2[..., None, None, :])
    R = jnp.asarray(R)[..., None, None, None]
    return jnp.mean(flash_overlap_kernel(r, R), axis=(-3, -2, -1))


def _torch_energy_mix(old, new):
    """inject_direct's energy mode on touched cells; `new` is `old` + (dDens, dDens u|u|, 0)."""
    rho_o, rho_n = old[0], new[0]
    v_o = old[1:4] / rho_o
    S = rho_o * v_o * jnp.abs(v_o) + (new[1:4] - old[1:4])
    v_n = jnp.sign(S) * jnp.sqrt(jnp.abs(S) / rho_n)
    e_spec = (old[4] - 0.5 * rho_o * jnp.sum(v_o * v_o, axis=0)) / rho_o
    E_n = rho_n * (e_spec + 0.5 * jnp.sum(v_n * v_n, axis=0))
    mixed = jnp.concatenate([rho_n[None], rho_n * v_n, E_n[None]], axis=0)
    return jnp.where(rho_n > rho_o, mixed, old)


def make_torch_wind_injector(N3, dxyz, periodic, *, v_ref, cons_quant="momentum",
                             add_therm_e=False, ref_radius=None, var_radius=False,
                             source="torch", sharding=None):
    """Build FLASH-Torch's inject_direct for all stars: `inject(U, pos, mdot, ...) -> (U, info)`.

    `sinks` = host (pos, vel, mass, active), mass reduced in place; bg_rho 0 = not yet measured."""
    import jax
    from luna.utils.grid_scatter import make_grid_gather, make_grid_scatter
    if source not in ("torch", "sink"):
        raise ValueError(f"source {source!r}: 'torch' or 'sink'")
    if cons_quant not in ("momentum", "energy"):
        raise ValueError(f"wind_cons_quant {cons_quant!r}: 'momentum' or 'energy'")
    if source == "sink" and cons_quant != "momentum":
        raise ValueError("wind_mass_load_mode 'sink' needs wind_cons_quant 'momentum'")
    if not v_ref > 0.0:
        raise ValueError("FLASH-Torch's wind needs its loading on (wind_target_temp_K > 0); "
                         "the unloaded solid-angle weighting is not ported")
    d = np.asarray(dxyz, float)
    dx = float(d.min())
    dV = float(np.prod(d))
    r_ref = float(ref_radius) if ref_radius else TORCH_REF_RADIUS_CELLS * dx
    K = int(np.ceil(r_ref / dx)) + 1
    n_ax = np.asarray(N3)
    if 2 * K + 1 > int(n_ax.min()):
        raise ValueError(f"wind radius {r_ref} needs a {2 * K + 1}-cell window; the grid is {N3}")
    off = np.arange(-K, K + 1)
    O = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    after = _torch_energy_mix if cons_quant == "energy" else (None if add_therm_e else _keep_internal)
    scatter = make_grid_scatter(sharding, n_comp=5, after=after)
    gather = make_grid_gather(sharding)

    @jax.jit
    def weights(rel, R):
        # one star at a time: nsteps^3 points per window cell
        return jax.lax.map(lambda a: flash_overlap(a[0], a[1], d), (rel, R))

    @jax.jit
    def increments(rel, w, M, v_inj, p_bulk, e_add):
        # (5, n, W): the conservative increments of n stars' windows
        rr = jnp.sqrt(jnp.sum(rel * rel, axis=-1))
        e = jnp.where(rr[..., None] > 0.0, rel / jnp.maximum(rr, 1e-300)[..., None], 0.0)
        drho = w * (M[:, None] / dV)
        u = v_inj[:, None, None] * e
        if cons_quant == "energy":
            mom = drho[..., None] * u * jnp.abs(u)
        else:
            mom = drho[..., None] * u + w[..., None] * (p_bulk[:, None, :] / dV)
        en = w * (e_add[:, None] / dV)
        return jnp.concatenate([drho[None], jnp.moveaxis(mom, -1, 0), en[None]], axis=0)

    def inject(U, pos, mdot, v_term, active, bg_rho, t_wind, dt, sinks=None,
               sink_periodic=None):
        pos = np.asarray(pos, float).reshape(-1, 3)
        mdot, v_term = np.asarray(mdot, float).ravel(), np.asarray(v_term, float).ravel()
        bg_rho = np.array(bg_rho, float).ravel()
        t_wind = np.asarray(t_wind, float).ravel()
        dm = mdot * dt
        # inject_direct returns at once for a massless or still wind
        emit = np.asarray(active).astype(bool) & (dm > 0.0) & (v_term > 0.0)
        info = dict(n=int(emit.sum()), dm=0.0, E=0.0, dm_created=0.0, dm_from_sinks=0.0,
                    n_capped=0, bg_rho=bg_rho)
        if not emit.any():
            return U, info
        sel = np.where(emit)[0]
        p, m, vt = pos[sel], dm[sel], v_term[sel]
        F, v_inj = torch_load(vt, v_ref, cons_quant)
        cell = np.floor(p / d).astype(np.int64)[:, None, :] + O[None]     # (n, W, 3)
        rel = (cell + 0.5) * d - p[:, None, :]
        if periodic:
            L = n_ax * d
            rel = rel - L * np.round(rel / L)
            idx = np.mod(cell, n_ax)
            inside = np.ones(cell.shape[:2])
        else:
            inside = np.all((cell >= 0) & (cell < n_ax), axis=-1).astype(float)
            idx = np.clip(cell, 0, n_ax - 1)
        # the background density, once per star, over r_ref (FLASH's first call)
        first = bool(var_radius) & (bg_rho[sel] == 0.0)
        if first.any():
            wf = np.asarray(weights(jnp.asarray(rel[first]),
                                    jnp.full(int(first.sum()), r_ref))) * inside[first]
            rho = np.asarray(gather(U, idx[first].reshape(-1, 3))).reshape(wf.shape)
            bg_rho[sel[first]] = (wf * rho).sum(1) / wf.sum(1)
        R = torch_wind_radius(mdot[sel], np.where(first, 0.0, bg_rho[sel]), vt, t_wind[sel],
                              dx, r_ref, var_radius)
        # normalised over the cells in the box (an outflow face cuts the sphere)
        w = np.asarray(weights(jnp.asarray(rel), jnp.asarray(R))) * inside
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-300)
        M, p_bulk, e_bulk = F * m, np.zeros((len(sel), 3)), np.zeros(len(sel))
        if source == "sink":
            M, v_inj, p_bulk, e_bulk, taken, capped = _take_from_sinks(
                p, m, vt, F, sinks, periodic if sink_periodic is None else sink_periodic,
                n_ax * d)
            info.update(dm_from_sinks=float(taken.sum()), n_capped=int(capped.sum()))
        else:
            info["dm_created"] = float((M - m).sum())
        e_add = (0.5 * m * vt ** 2 + e_bulk) if add_therm_e else np.zeros(len(sel))
        vals = np.asarray(increments(jnp.asarray(rel), jnp.asarray(w), jnp.asarray(M),
                                     jnp.asarray(v_inj), jnp.asarray(p_bulk), jnp.asarray(e_add)))
        keep = (w > 0.0).ravel()
        U = scatter(U, idx.reshape(-1, 3)[keep], vals.reshape(5, -1)[:, keep])
        info.update(dm=float(m.sum()), E=float((0.5 * m * vt ** 2).sum()))
        return U, info

    return inject


def _take_from_sinks(p, dm, v_term, F, sinks, periodic, L):
    """Each star's (F - 1) dm from its nearest active sink, capped at what it holds.

    Returns per star (M, v_inj, p_bulk, e_bulk, taken, capped); the moved mass keeps v_sink."""
    need = np.maximum(F - 1.0, 0.0) * dm
    taken = np.zeros_like(dm)
    p_bulk = np.zeros((len(dm), 3))
    e_bulk = np.zeros(len(dm))
    if sinks is not None:
        s_pos, s_vel, s_mass, s_act = sinks
        live = np.asarray(s_act) > 0.5
        for i in range(len(dm)):
            if need[i] <= 0.0 or not live.any():
                continue
            r = np.asarray(s_pos, float) - p[i]
            if periodic:
                r = r - L * np.round(r / L)
            k = int(np.argmin(np.where(live, np.sum(r * r, axis=1), np.inf)))
            taken[i] = min(need[i], max(float(s_mass[k]), 0.0))
            s_mass[k] = float(s_mass[k]) - taken[i]
            v = np.asarray(s_vel[k], float)
            p_bulk[i] = taken[i] * v
            e_bulk[i] = 0.5 * taken[i] * float(v @ v)
    M = dm + taken
    return M, v_term * dm / M, p_bulk, e_bulk, taken, taken < need
