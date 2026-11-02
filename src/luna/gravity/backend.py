"""The gravity backends behind one interface: `LocalGravity` (one device) and `ShardedGravity` (a block mesh).

See docs/ARCHITECTURE.md.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

DELTA = "delta"       # lap(psi) = rho/<rho> - 1
PROPER = "proper"     # lap(psi) = 4 pi G (rho - <rho>)   (rho alone if not periodic)


def _fused_sink_kick(U, extra, kernel):
    """(sinks, pos, mass, active) when the kick can be the fused Pallas kernel, else None."""
    from luna.gravity import kick_pallas
    if kernel == "xla" or not kick_pallas.select(kernel):
        return None
    if getattr(U, "ndim", 0) != 4 or U.dtype != jnp.float64:
        return None
    return kick_pallas.sink_args(extra)


def _plain_kick_fn(kernel, adjoint):
    """kick_gas_body, or (gravity_kernel pallas/auto on a GPU, no autodiff) the bitwise-equal Pallas kick."""
    from luna.gravity import kick_pallas
    if not kick_pallas.select(kernel, autodiff=adjoint):
        return kick_gas_body

    def kick(U, g, s, dt):
        if kick_pallas.applies_plain(U, g):
            return kick_pallas.kick_plain(U, g, s, dt)
        if kernel == "pallas":
            raise ValueError(f"gravity_kernel='pallas': the Pallas kick needs 5 gas rows and g in U's dtype; "
                             f"got U {U.shape} {U.dtype}, g {[str(q.dtype) for q in g]}")
        return kick_gas_body(U, g, s, dt)
    return kick


@dataclass(frozen=True)
class PoissonOpts:
    """Multigrid settings, read once from the config."""
    n_levels: object = None
    n_vcycles_per_level: int = 1
    n_extra_vcycles: int = 0
    n_pre: int = 2
    n_post: int = 2
    n_coarsest: object = 50     # None: coarsest_sweeps decides by grid and boundary
    warm_cycles: int = 0        # 0 = cold FMG every solve; k = k V-cycles on the carried psi
    warm_n_pre: int = 1
    warm_n_post: int = 1
    mg_ca: str = "auto"         # sharded solve: communication-avoiding smoother 'auto' | '0' | '1'
    mg_smoother: str = "rbgs"   # sharded solve: 'rbgs' | 'cheb'
    mg_agg_n: int = 128
    warm_tol: float = 0.0       # > 0: warm V-cycles until rms residual <= warm_tol rms source
    warm_max_cycles: int = 8
    extrapolate: bool = False   # seed with psi extrapolated in time from the last two solves
    dt_ratio_max: float = 4.0

    @classmethod
    def from_config(cls, cfg, warm_cycles):
        return cls(n_levels=cfg.POISSON_N_LEVELS,
                   n_vcycles_per_level=int(cfg.POISSON_FMG_VCYCLES_PER_LEVEL),
                   n_extra_vcycles=int(cfg.POISSON_FMG_EXTRA_VCYCLES),
                   n_pre=int(cfg.POISSON_N_PRE), n_post=int(cfg.POISSON_N_POST),
                   n_coarsest=(None if cfg.POISSON_N_COARSEST is None
                               else int(cfg.POISSON_N_COARSEST)),
                   warm_cycles=int(warm_cycles),
                   warm_n_pre=int(cfg.dm_warm_n_pre), warm_n_post=int(cfg.dm_warm_n_post),
                   mg_ca=str(cfg.gravity_mg_ca), mg_smoother=str(cfg.gravity_mg_smoother),
                   mg_agg_n=int(cfg.gravity_mg_agg_n),
                   warm_tol=float(cfg.gravity_warm_tol),
                   warm_max_cycles=int(cfg.gravity_warm_max_cycles),
                   extrapolate=bool(cfg.gravity_warm_extrapolate),
                   dt_ratio_max=float(cfg.gravity_warm_dt_ratio_max))


def supported_opts(opts, adjoint):
    """opts minus what the solve lacks: reverse-mode AD through the adaptive cycle loop."""
    from dataclasses import replace
    if adjoint and opts.warm_tol > 0:
        opts = replace(opts, warm_tol=0.0)          # a while_loop has no reverse-mode derivative
    return opts


def _dense_halos(src, dx, boundary):
    """(top halo, error-subproblem halo or None) of the one-device solve; isolated faces from the source."""
    from luna.gravity.poisson_bc import make_halo_3d
    if boundary != 3:
        return make_halo_3d(boundary), None
    from luna.gravity.poisson_bc import (make_linear_extrap_dirichlet_halo_3d,
                                         make_odd_dirichlet_halo_3d,
                                         multipole_dirichlet_face_values)
    top = make_linear_extrap_dirichlet_halo_3d(multipole_dirichlet_face_values(src, dx, 1.0))
    return top, make_odd_dirichlet_halo_3d()


def kick_gas_body(U, g, s, dt, extra=None):
    """Momentum kick rho v += dt rho g with the time-centred energy work.

    g: the unscaled -grad psi; s: the kick scale; extra: a force field added unscaled.
    """
    rho = U[0]
    gx, gy, gz = g[0] * s, g[1] * s, g[2] * s
    if extra is not None:
        gx, gy, gz = gx + extra[0], gy + extra[1], gz + extra[2]
    mx, my, mz = U[1], U[2], U[3]
    nmx = mx + dt * rho * gx
    nmy = my + dt * rho * gy
    nmz = mz + dt * rho * gz
    E = U[-1] + dt * (0.5 * (mx + nmx) * gx + 0.5 * (my + nmy) * gy
                      + 0.5 * (mz + nmz) * gz)
    return jnp.stack([rho, nmx, nmy, nmz, E])


def kick_gas_work_body(kick, U, g, s, dt):
    """(U after `kick`, the energy it added): the opening half-kick of gravity_energy='flux'."""
    out = kick(U, g, s, dt)
    return out, out[-1] - U[-1]


def step_potential(psi0, psi1, s0, s1, phi_ext=None, comoving=False):
    """The step's cell potential Phibar: (s0 psi0 + s1 psi1) / 2, or comoving (s0 + s1)/2 (psi0 + psi1)/2; + phi_ext.

    Comoving, with W~ = 1/2 s sum rho~ psi and psi linear in rho~ (one symmetric operator), W~1 - W~0 =
    sum (rho~1 - rho~0) sbar psibar dV + (ds / 2)(W~0 / s0 + W~1 / s1) exactly: the face work on sbar psibar leaves
    d(E~ + W~) = (ds / 2)(W~0 / s0 + W~1 / s1), the trapezoid of Layzer-Irvine's W~ d ln a (s ~ a). phi_ext: a
    cell-centred potential not in psi, already step-averaged; None adds nothing (the expression is unchanged).
    """
    if comoving:
        phi = (0.5 * (s0 + s1)) * (0.5 * (psi0 + psi1))
    else:
        phi = 0.5 * (s0 * psi0 + s1 * psi1)
    return phi if phi_ext is None else phi + phi_ext


def flux_work_body(psi0, psi1, s0, s1, F, dt, inv_dx, pad, donor=None, **potential):
    """Energy the gas gains from the potential over a step, from the face mass fluxes (Mullen et al. 2020).

    Each face's work -dt/dx F (Phi_R - Phi_L), Phi the `step_potential` (keywords `potential`), is shared half to
    each side, or all to the donor cell (the one the mass leaves) on a face next to a cell flagged in `donor`
    (padded like psi, > 0.5 = flagged); either way the sum is minus the hydro's change of sum rho Phi. F[d] has
    N_d + 1 faces along d; pad adds a one-cell halo.
    """
    pp = pad(step_potential(psi0, psi1, s0, s1, **potential))
    acc = None
    for d in range(3):
        L, R = [slice(1, -1)] * 3, [slice(1, -1)] * 3
        L[d], R[d] = slice(0, -1), slice(1, None)
        w = F[d] * (pp[tuple(R)] - pp[tuple(L)])          # minus the face's work, per dt / dx
        if donor is None:
            to_l = to_r = 0.5 * w
        else:
            flag = (donor[tuple(L)] > 0.5) | (donor[tuple(R)] > 0.5)
            to_l = jnp.where(flag, jnp.where(F[d] > 0.0, w, 0.0), 0.5 * w)
            to_r = w - to_l
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[d], hi[d] = slice(0, -1), slice(1, None)
        term = to_r[tuple(lo)] + to_l[tuple(hi)]          # cell i: right side of face i, left side of face i + 1
        acc = term if acc is None else acc + term
    return (-dt * inv_dx) * acc


def kick_gas_flux_body(U, g, s, dt, dE):
    """Momentum kick rho v += dt rho g (as `kick_gas_body`) with the energy changed by dE instead of the work."""
    rho = U[0]
    gx, gy, gz = g[0] * s, g[1] * s, g[2] * s
    nmx = U[1] + dt * rho * gx
    nmy = U[2] + dt * rho * gy
    nmz = U[3] + dt * rho * gz
    return jnp.stack([rho, nmx, nmy, nmz, U[-1] + dE])


FLUX_ETH_KEEP = 0.5    # the flux coupling may take at most 1 - this of a cell's internal energy in one step:
                       # donor-cell faces around a cell the half split would cool further, then the limiter


def _thermal_change(U, g, s, h, dE):
    """(internal energy after the hydro, its change if the closing kick changes the energy by dE)."""
    rho = U[0]
    gx, gy, gz = g[0] * s, g[1] * s, g[2] * s
    mx, my, mz = U[1], U[2], U[3]
    nmx = mx + h * rho * gx
    nmy = my + h * rho * gy
    nmz = mz + h * rho * gz
    ke0 = 0.5 * (mx * mx + my * my + mz * mz) / rho
    ke1 = 0.5 * (nmx * nmx + nmy * nmy + nmz * nmz) / rho
    eth0 = U[-1] - ke0
    return eth0, dE - (ke1 - ke0)


def flux_positive_body(U, g, s, h, dE, total):
    """(dE limited so no cell keeps less than FLUX_ETH_KEEP of its internal energy, the energy that took back).

    Where the face work minus the kick's change of kinetic energy would cool a cell below FLUX_ETH_KEEP of its
    internal energy, the cell keeps that fraction; the sum of what was kept back is removed from every cell in
    proportion to its internal energy, so the total (`total`: the sum over the whole grid) is unchanged. With
    nothing to limit both corrections are zero and dE comes back unchanged.
    """
    eth0, de = _thermal_change(U, g, s, h, dE)
    eth1 = eth0 + de
    keep = jnp.maximum(eth1, FLUX_ETH_KEEP * eth0)
    excess = keep - eth1
    moved = total(excess)
    weight = jnp.maximum(keep, 0.0)
    return dE + (excess - moved * (weight / total(weight))), moved


def close_flux_body(kick, pad, inv_dx, total, U, g, s, h, psi0, psi1, s0, F, W1, dt, phi_ext=None, comoving=False):
    """The closing half-kick of gravity_energy='flux': momentum by h rho g, energy by the flux work minus W1.

    The face work is split half / half, except next to a cell the split would leave with less than FLUX_ETH_KEEP of
    its internal energy, where it goes to the donor cell; then positivity-limited (`flux_positive_body`).
    phi_ext, comoving: as `step_potential` (static: the proper, external-free call is the original expression).
    Returns (U, the energy density the limiter moved, summed).
    """
    pot = {k: v for k, v in (("phi_ext", phi_ext), ("comoving", comoving)) if v is not None and v is not False}
    half = flux_work_body(psi0, psi1, s0, s, F, dt, inv_dx, pad, **pot) - W1
    eth0, de = _thermal_change(U, g, s, h, half)
    flagged = (eth0 + de < FLUX_ETH_KEEP * eth0).astype(eth0.dtype)
    # with nothing flagged the donor pass is the half split bit for bit, and the limiter adds zero: skip both
    dE = jax.lax.cond(
        total(flagged) > 0,
        lambda: flux_work_body(psi0, psi1, s0, s, F, dt, inv_dx, pad, donor=pad(flagged), **pot) - W1,
        lambda: half)
    eth1 = eth0 + _thermal_change(U, g, s, h, dE)[1]
    dE, moved = jax.lax.cond(
        total((eth1 < FLUX_ETH_KEEP * eth0).astype(eth0.dtype)) > 0,
        lambda: flux_positive_body(U, g, s, h, dE, total),
        lambda: (dE, jnp.zeros((), dE.dtype)))
    return kick(U, g, s, h, dE), moved


def booked_work_body(total, periodic, psi0, psi1, s0, s1, drho, phi_ext=None, comoving=False):
    """sum drho Phibar (per unit volume): what a mass change outside the hydro's fluxes did to W, booked.

    periodic: the solve removes the mean density, so W = 1/2 sum (rho - <rho>) Phi and drho enters less its mean
    (Phibar's free constant drops out)."""
    if periodic:
        drho = drho - total(drho) / total(jnp.ones_like(drho))
    return total(drho * step_potential(psi0, psi1, s0, s1, phi_ext, comoving))


def _flux_kick_fn(kernel, adjoint):
    """kick_gas_flux_body, or (gravity_kernel pallas/auto on a GPU, no autodiff) the bitwise-equal Pallas kick."""
    from luna.gravity import kick_pallas
    if not kick_pallas.select(kernel, autodiff=adjoint):
        return kick_gas_flux_body

    def kick(U, g, s, dt, dE):
        if kick_pallas.applies_plain(U, g) and dE.dtype == U.dtype:
            return kick_pallas.kick_flux(U, g, s, dt, dE)
        if kernel == "pallas":
            raise ValueError(f"gravity_kernel='pallas': the Pallas kick needs 5 gas rows and g in U's dtype; "
                             f"got U {U.shape} {U.dtype}, g {[str(q.dtype) for q in g]}")
        return kick_gas_flux_body(U, g, s, dt, dE)
    return kick


def kick_gas_fields_body(U, g, s, dt, fns, args):
    """`kick_gas_body` with the species' fields `fns[i](*args[i])` evaluated in the same program."""
    extra = None
    for fn, a in zip(fns, args):
        f = fn(*a)
        extra = f if extra is None else tuple(x + y for x, y in zip(extra, f))
    return kick_gas_body(U, g, s, dt, extra=extra)


def kick_drift_body(pos, p, g, dtau, box):
    p_half = p + g * (0.5 * dtau)
    return jnp.mod(pos + p_half * dtau, box), p_half


def close_kick_body(graw, p_half, active, dtau, scale, allmax=lambda x: x):
    """Scale the gathered force and close the kick: (p_new, g, max |g|^2 over live particles)."""
    g = graw * scale
    p_new = p_half + g * (0.5 * dtau)
    g2 = allmax(jnp.max(jnp.where(active > 0, jnp.sum(g ** 2, axis=1), 0.0)))
    return p_new, g, g2


def source_body(rho, mean, mode, fourpiG, periodic):
    if mode == DELTA:
        return rho / mean - 1.0
    return fourpiG * (rho - mean) if periodic else fourpiG * rho


class LocalGravity:
    """Whole grid on one device."""
    sharded = False

    def __init__(self, pm, box, opts, prec, mode, fourpiG, adjoint=False, kernel="auto"):
        """pm: a ParticleMeshGravity; adjoint: implicit-adjoint FMG (poisson_diff); kernel: sink kick 'auto'|'xla'|'pallas'."""
        from luna.gravity import kick_pallas
        from luna.gravity.poisson_diff import solve_poisson_fmg_diff
        from luna.gravity.poisson_multigrid import (_resolve_n_levels, cycles_to_tol,
                                                    extrapolated_seed, laplacian_7pt, solve_poisson,
                                                    solve_poisson_fmg)
        kick_pallas.select(kernel)               # rejects a bad value
        self.kernel = str(kernel)
        self.pm = pm
        self.dx = float(pm.dx)
        self.box = float(box)
        self.boundary = int(pm.boundary)
        self.opts = opts = supported_opts(opts, adjoint)
        self.prec = prec
        self.mode = mode
        self.fourpiG = float(fourpiG)
        o = opts
        dx, bnd = self.dx, self.boundary

        if adjoint:
            self._cold = jax.jit(lambda src: solve_poisson_fmg_diff(
                src, dx, 1.0, bnd, o.n_vcycles_per_level, o.n_extra_vcycles,
                o.n_levels, o.n_pre, o.n_post, o.n_coarsest))
        else:
            self._cold = jax.jit(lambda src: solve_poisson_fmg(
                src, dx, 1.0, boundary=bnd, n_levels=o.n_levels,
                n_vcycles_per_level=o.n_vcycles_per_level,
                n_extra_vcycles=o.n_extra_vcycles, n_pre=o.n_pre, n_post=o.n_post,
                n_coarsest=o.n_coarsest))
        halo = pm.halo_fn
        if o.warm_tol > 0 or (prec.cycle_mixed and bnd == 3):
            # to a tolerance, or fixed mixed cycles with the isolated faces on the FP64 residual only
            n_max = o.warm_max_cycles if o.warm_tol > 0 else max(o.warm_cycles, 1)

            def _warm(src, seed, prev, q):
                top, inner = _dense_halos(src, dx, bnd)
                return cycles_to_tol(
                    extrapolated_seed(seed, prev, q), src, dx, top,
                    _resolve_n_levels(src, o.n_levels) - 1, o.warm_n_pre, o.warm_n_post,
                    o.n_coarsest, o.warm_tol, n_max, mixed=prec.cycle_mixed, halo_inner=inner)
        elif prec.cycle_mixed:
            # FP64 residual of the carried psi, FP32 V-cycles on the correction
            def _warm(src, seed, prev, q):
                seed = extrapolated_seed(seed, prev, q)
                rhs = (src - laplacian_7pt(seed, dx, halo)).astype(jnp.float32)
                e = solve_poisson(rhs, dx, 1.0, bnd, n_levels=o.n_levels,
                                  n_cycles=max(o.warm_cycles, 1), n_pre=o.warm_n_pre,
                                  n_post=o.warm_n_post)
                return seed + e
        else:
            def _warm(src, seed, prev, q):
                return solve_poisson(src, dx, 1.0, bnd, n_levels=o.n_levels,
                                     n_cycles=max(o.warm_cycles, 1),
                                     phi_init=extrapolated_seed(seed, prev, q),
                                     n_pre=o.warm_n_pre, n_post=o.warm_n_post)
        self._warm = jax.jit(_warm)
        self._adaptive = o.warm_tol > 0 or (prec.cycle_mixed and bnd == 3)
        self.last_info = None
        self._grad = {2: jax.jit(lambda psi: pm.grad_phi(psi, order=2)),
                      4: jax.jit(lambda psi: pm.grad_phi(psi, order=4))}
        self._gather = jax.jit(pm.accel_at_points)
        self._kick = jax.jit(_plain_kick_fn(self.kernel, adjoint))
        self._kick_work = jax.jit(partial(kick_gas_work_body, _plain_kick_fn(self.kernel, adjoint)))
        if bnd == 3:
            from luna.gravity.isolated import extrap_pad
            pad1 = partial(extrap_pad, width=1)
        else:
            pad1 = halo
        comoving = mode == DELTA
        self._close_flux = jax.jit(partial(close_flux_body, _flux_kick_fn(self.kernel, adjoint), pad1,
                                           1.0 / dx, jnp.sum, comoving=comoving))
        self._booked_work = jax.jit(partial(booked_work_body, jnp.sum, bnd == 0, comoving=comoving))
        self._kick_extra = jax.jit(kick_gas_fields_body, static_argnums=(4,))
        self._kick_drift = jax.jit(partial(kick_drift_body, box=self.box))
        self._close = jax.jit(close_kick_body)
        self._source = {}
        self._deposits = {}

    def source(self, U, rho_dm, p, mass, active):
        """(src, stats): stats are 0-d arrays delta_max, delta_min, p2_max, n_active, mass, rho_gas_max or None."""
        key = (U is not None, rho_dm is not None, p is not None)
        fn = self._source.get(key)
        if fn is None:
            fn = self._source[key] = jax.jit(partial(self._source_body, *key))
        return fn(U, rho_dm, p, mass, active)

    def _source_body(self, has_gas, has_dm, has_p, U, rho_dm, p, mass, active):
        rho_gas = U[0] if has_gas else None
        if has_gas and has_dm:
            rho = rho_gas.astype(rho_dm.dtype) + rho_dm
        else:
            rho = rho_gas if has_gas else rho_dm
        mean = jnp.mean(rho)
        src = source_body(rho, mean, self.mode, self.fourpiG, self.boundary == 0)
        # diagnostics: no gradient through them (the adjoint runner differentiates this program)
        rho_d = jax.lax.stop_gradient(rho)
        delta = rho_d / jax.lax.stop_gradient(mean) - 1.0
        st = {"delta_max": jnp.max(delta), "delta_min": jnp.min(delta),
              "rho_gas_max": jnp.max(jax.lax.stop_gradient(rho_gas)) if has_gas else None,
              "p2_max": (jnp.max(jnp.where(active > 0, jnp.sum(p ** 2, axis=1), 0.0))
                         if has_p else None),
              "n_active": jnp.sum((active > 0).astype(jnp.int32)) if has_p else None,
              "mass": jnp.sum(mass * (active > 0)) if has_p else None}
        return src.astype(self.prec.phi), st

    def solve(self, src, seed, prev=None, q=0.0):
        """psi: a cold FMG without a seed, else warm cycles on the seed (extrapolated by q when prev is given)."""
        self.last_info = None
        if seed is None or self.opts.warm_cycles <= 0:
            return self._cold(src)
        out = self._warm(src, seed, prev, q)
        if self._adaptive:
            out, k, rel = out
            if self.opts.warm_tol > 0:
                self.last_info = (k, rel)
        return out

    def grad(self, psi, order):
        return self._grad[order](psi)

    def make_deposit(self, mass_unit):
        prec = self.prec
        fixed = prec.fixed_int if prec.fixed_deposit else None
        return jax.jit(lambda pos, mass, active: self.pm.cic_deposit(
            pos, mass, active, dtype=prec.deposit, fixed=fixed,
            fixed_bits=prec.fixed_bits, mass_unit=mass_unit))

    def gather(self, g, pos):
        return self._gather(g, pos)

    def kick_gas(self, U, g, s, dt, extra=None):
        """Kick the gas; extra: the species' field programs, a tuple of (fn, args)."""
        if not extra:
            return self._kick(U, g, s, dt)
        fused = _fused_sink_kick(U, extra, self.kernel)
        if fused is not None:
            sk, pos, mass, active = fused
            if getattr(self, "_kick_fused", None) is None:
                from luna.gravity import kick_pallas
                self._kick_fused = jax.jit(lambda U, g, s, dt, pos, mass, active, sk=sk: kick_pallas.kick(
                    U, g, s, dt, pos, mass, active, origin=(0, 0, 0), dxyz=sk.dxyz, G=sk.G, eps=sk.eps,
                    box=sk.image_box))
            return self._kick_fused(U, tuple(g), s, dt, pos, mass, active)
        return self._kick_extra(U, g, s, dt, tuple(f for f, _ in extra),
                                tuple(a for _, a in extra))

    def kick_gas_work(self, U, g, s, dt):
        """(U after the plain kick, the energy it added per cell)."""
        return self._kick_work(U, tuple(g), s, dt)

    def close_flux(self, U, g, s, h, psi0, psi1, s0, F, W1, dt, phi_ext=None):
        """(U after the closing half-kick of gravity_energy='flux', the energy its positivity limit moved)."""
        if phi_ext is None:
            U, moved = self._close_flux(U, tuple(g), s, h, psi0, psi1, s0, tuple(F), W1, dt)
        else:
            U, moved = self._close_flux(U, tuple(g), s, h, psi0, psi1, s0, tuple(F), W1, dt, phi_ext)
        return U, moved * self.dx ** 3

    def booked_work(self, psi0, psi1, s0, s1, drho, phi_ext=None):
        """sum drho Phibar dV: the change of W from mass the hydro's fluxes did not move."""
        return self._booked_work(psi0, psi1, s0, s1, drho, phi_ext) * self.dx ** 3

    def kick_drift(self, pos, p, g, dtau):
        return self._kick_drift(pos, p, g, dtau)

    def close_kick(self, graw, p_half, active, dtau, s):
        return self._close(graw, p_half, active, dtau, s)

    def assemble_particles(self, pos, p, mass, ids, n_total=None, capacity_factor=1.5):
        dt = self.prec.particles
        pos = jnp.asarray(pos, dt)
        return (pos, jnp.asarray(p, dt), jnp.asarray(mass, dt),
                jnp.ones(pos.shape[0], dt), None if ids is None else jnp.asarray(ids))

    def exchange(self, pos, p, mass, active, ids, face_capacity_factor=None):
        return pos, p, mass, active, ids, 0

    def settle(self, pos, p, mass, active, ids, pos_host):
        return pos, p, mass, active, ids, 0

    def resort(self, pos, p, mass, active, g, ids):
        return pos, p, mass, active, g, ids

    def component(self, U, k):
        """One conserved field of the grid."""
        return U[k]

    def grid_max(self, field):
        return jnp.max(field)

    def grid_sum(self, field):
        return jnp.sum(field)

    def grid_mean(self, field):
        return jnp.mean(field)

    def block_box(self):
        """Index box this process holds of the grid: whole grid."""
        return None, None

    def local_rows(self, arr):
        return np.asarray(arr)


class ShardedGravity:
    """The (px, py, pz) block mesh: shard_map programs with explicit collectives."""
    sharded = True

    def __init__(self, mesh, N3, dx, box, boundary, opts, prec, mode, fourpiG, adjoint=False,
                 kernel="auto"):
        """adjoint: the cold solve carries the implicit adjoint of poisson_diff; kernel as LocalGravity."""
        from luna.gravity import kick_pallas
        from jax.sharding import PartitionSpec as P
        from luna.utils.shmap import shard_map
        if boundary == 3 and adjoint:
            raise NotImplementedError("isolated gravity (POISSON_BOUNDARY=3) on a mesh "
                                      "has no implicit adjoint; autodiff runs periodic")
        kick_pallas.select(kernel)               # rejects a bad value
        self.kernel = str(kernel)
        self.mesh = mesh
        self.N3 = tuple(int(n) for n in N3)
        self.dx = float(dx)
        self.box = float(box)
        self.boundary = int(boundary)
        self.opts = opts = supported_opts(opts, adjoint)
        self.prec = prec
        self.mode = mode
        self.fourpiG = float(fourpiG)
        self.last_info = None
        self.pdims = tuple(int(mesh.shape[a]) for a in ("x", "y", "z"))
        if any(n % p for n, p in zip(self.N3, self.pdims)):
            raise ValueError(f"block decomposition {self.pdims} must divide {self.N3}")
        S = ("x", "y", "z")
        Pp, Ps, Pg, PU = P(S, None), P(S), P("x", "y", "z"), P(None, "x", "y", "z")
        sm = partial(shard_map, mesh=mesh)
        box_ = self.box
        self._kick = jax.jit(sm(_plain_kick_fn(self.kernel, adjoint),
                                in_specs=(PU, (Pg, Pg, Pg), P(), P()), out_specs=PU))
        self._kick_work = jax.jit(sm(partial(kick_gas_work_body, _plain_kick_fn(self.kernel, adjoint)),
                                     in_specs=(PU, (Pg, Pg, Pg), P(), P()), out_specs=(PU, Pg)))
        bnd_ = self.boundary
        if bnd_ == 0:
            from luna.gravity.particle_mesh_blocks import _halo
            pad1 = lambda phi: _halo(phi, mesh, 1)            # noqa: E731
        else:
            from luna.gravity.isolated import pad_sharded
            pad1 = lambda phi: pad_sharded(phi, mesh, 1, bnd_)   # noqa: E731
        comoving = mode == DELTA
        gsum = lambda a: jax.lax.psum(jnp.sum(a), S)          # noqa: E731
        close = partial(close_flux_body, _flux_kick_fn(self.kernel, adjoint), pad1, 1.0 / self.dx, gsum,
                        comoving=comoving)
        close_specs = (PU, (Pg, Pg, Pg), P(), P(), Pg, Pg, P(), (Pg, Pg, Pg), Pg, P())
        self._close_flux = jax.jit(sm(close, in_specs=close_specs, out_specs=(PU, P())))
        self._close_flux_ext = jax.jit(sm(close, in_specs=close_specs + (Pg,), out_specs=(PU, P())))
        booked = partial(booked_work_body, gsum, bnd_ == 0, comoving=comoving)
        self._booked_work = jax.jit(sm(booked, in_specs=(Pg, Pg, P(), P(), Pg), out_specs=P()))
        self._booked_work_ext = jax.jit(sm(booked, in_specs=(Pg, Pg, P(), P(), Pg, Pg), out_specs=P()))
        self._kick_drift = jax.jit(sm(partial(kick_drift_body, box=box_),
                                      in_specs=(Pp, Pp, Pp, P()), out_specs=(Pp, Pp)),
                                   donate_argnums=(0, 1, 2))
        self._close = jax.jit(sm(partial(close_kick_body,
                                         allmax=lambda x: jax.lax.pmax(x, S)),
                                 in_specs=(Pp, Pp, Ps, P(), P()),
                                 out_specs=(Pp, Pp, P())), donate_argnums=(0, 1))
        self._specs = dict(Pp=Pp, Ps=Ps, Pg=Pg)
        self._source = {}
        self._components = {}
        self._kick_fields = {}
        self._cold_diff = self._implicit_adjoint() if adjoint else None

    def source(self, U, rho_dm, p, mass, active):
        """(src, stats), as `LocalGravity.source`."""
        key = (U is not None, rho_dm is not None, p is not None)
        fn = self._source.get(key)
        if fn is None:
            fn = self._source[key] = self._build_source(*key)
        args = [x for x, k in ((U, key[0]), (rho_dm, key[1])) if k]
        if key[2]:
            args += [p, mass, active]
        src, *st = fn(*args)
        names = ["delta_max", "delta_min"] + (["rho_gas_max"] if key[0] else []) \
            + (["p2_max", "n_active", "mass"] if key[2] else [])
        stats = dict(zip(names, st))
        for k in ("rho_gas_max", "p2_max", "n_active", "mass"):
            stats.setdefault(k, None)
        return src, stats

    def _build_source(self, has_gas, has_dm, has_p):
        from jax.sharding import PartitionSpec as P
        from luna.utils.shmap import shard_map
        S = ("x", "y", "z")
        Pp, Ps, Pg = self._specs["Pp"], self._specs["Ps"], self._specs["Pg"]
        PU = P(None, "x", "y", "z")
        in_specs = tuple(([PU] if has_gas else []) + ([Pg] if has_dm else [])
                         + ([Pp, Ps, Ps] if has_p else []))
        n_out = 3 + int(has_gas) + 3 * int(has_p)
        out_specs = (Pg,) + (P(),) * (n_out - 1)
        mode, fourpiG, periodic, phi_dt = self.mode, self.fourpiG, self.boundary == 0, self.prec.phi

        def body(*args):
            args = list(args)
            rho_gas = args.pop(0)[0] if has_gas else None
            rho_dm = args.pop(0) if has_dm else None
            if has_gas and has_dm:
                rho_l = rho_gas.astype(rho_dm.dtype) + rho_dm
            else:
                rho_l = rho_gas if has_gas else rho_dm
            ssum = jax.lax.psum(jnp.sum(rho_l), S)
            cnt = jax.lax.psum(jnp.asarray(rho_l.size, rho_l.dtype), S)
            mean = ssum / cnt
            src_l = source_body(rho_l, mean, mode, fourpiG, periodic)
            # stopped copy: pmax/pmin have no differentiation rule
            rho_d = jax.lax.stop_gradient(rho_l)
            delta_l = rho_d / jax.lax.stop_gradient(mean) - 1.0
            out = [src_l.astype(phi_dt),
                   jax.lax.pmax(jnp.max(delta_l), S),
                   jax.lax.pmin(jnp.min(delta_l), S)]
            if has_gas:
                out.append(jax.lax.pmax(jnp.max(jax.lax.stop_gradient(rho_gas)), S))
            if has_p:
                p_l, mass_l, act_l = args
                out.append(jax.lax.pmax(jnp.max(jnp.where(
                    act_l > 0, jnp.sum(p_l ** 2, axis=1), 0.0)), S))
                out.append(jax.lax.psum(jnp.sum((act_l > 0).astype(jnp.int32)), S))
                out.append(jax.lax.psum(jnp.sum(mass_l * (act_l > 0)), S))
            return tuple(out)

        return jax.jit(shard_map(body, mesh=self.mesh, in_specs=in_specs,
                                 out_specs=out_specs))

    def _implicit_adjoint(self):
        """The cold sharded FMG as a custom_vjp with the implicit adjoint (as poisson_diff)."""
        from luna.gravity.poisson_multigrid import solve_poisson_fmg_shardmap
        from luna.utils.mesh_reduce import grid_mean
        o, mesh, dx, bnd = self.opts, self.mesh, self.dx, self.boundary
        nvpl = max(int(o.n_vcycles_per_level), 2)
        nev = max(int(o.n_extra_vcycles), 12 if bnd == 0 else 4)
        mg = self._mg_kwargs()

        def solve(rhs):
            phi = solve_poisson_fmg_shardmap(
                rhs, dx, 1.0, boundary=bnd, mesh=mesh, n_levels=o.n_levels,
                n_vcycles_per_level=nvpl, n_extra_vcycles=nev, n_pre=o.n_pre,
                n_post=o.n_post, n_coarsest=o.n_coarsest, subtract_mean=False, **mg)
            return phi - grid_mean(mesh, phi) if bnd == 0 else phi

        @jax.custom_vjp
        def solve_diff(src):
            return solve(src)

        def fwd(src):
            return solve_diff(src), None

        def bwd(_, phi_bar):
            if bnd == 0:
                phi_bar = phi_bar - grid_mean(mesh, phi_bar)
            return (solve(phi_bar),)

        solve_diff.defvjp(fwd, bwd)
        return solve_diff

    def _mg_kwargs(self):
        """This backend's sharded-solve options, passed into every solve."""
        return dict(mixed=bool(self.prec.cycle_mixed), ca=self.opts.mg_ca,
                    smoother=self.opts.mg_smoother, agg_n=self.opts.mg_agg_n)

    def solve(self, src, seed, prev=None, q=0.0):
        """As `LocalGravity.solve`."""
        from luna.gravity.poisson_multigrid import solve_poisson_fmg_shardmap
        o = self.opts
        mg = self._mg_kwargs()
        self.last_info = None
        if self._cold_diff is not None and (seed is None or o.warm_cycles <= 0):
            return self._cold_diff(src)
        if seed is None or o.warm_cycles <= 0:
            return solve_poisson_fmg_shardmap(
                src, self.dx, 1.0, boundary=self.boundary, mesh=self.mesh,
                n_levels=o.n_levels, n_vcycles_per_level=o.n_vcycles_per_level,
                n_extra_vcycles=o.n_extra_vcycles, n_pre=o.n_pre, n_post=o.n_post,
                n_coarsest=o.n_coarsest, subtract_mean=False, **mg)
        out = solve_poisson_fmg_shardmap(
            src, self.dx, 1.0, boundary=self.boundary, mesh=self.mesh,
            n_levels=o.n_levels, n_coarsest=o.n_coarsest, subtract_mean=False,
            phi_init=seed, warm_cycles=max(o.warm_cycles, 1),
            n_pre=o.warm_n_pre, n_post=o.warm_n_post, phi_prev=prev, q=q,
            tol=o.warm_tol, max_cycles=o.warm_max_cycles, **mg)
        if o.warm_tol > 0:
            out, k, rel = out
            self.last_info = (k, rel)
        return out

    def grad(self, psi, order):
        from luna.gravity.particle_mesh_blocks import grad_phi_blocks
        return grad_phi_blocks(psi, self.dx, self.mesh, order=order, boundary=self.boundary)

    def make_deposit(self, mass_unit):
        from luna.gravity.particle_mesh_blocks import cic_deposit_blocks
        prec = self.prec
        fixed = prec.fixed_int if prec.fixed_deposit else None
        return lambda pos, mass, active: cic_deposit_blocks(
            pos, mass, active, self.N3, self.dx, self.mesh, dtype=prec.deposit,
            fixed=fixed, fixed_bits=prec.fixed_bits, mass_unit=mass_unit)

    def gather(self, g, pos):
        from luna.gravity.particle_mesh_blocks import gather_accel_blocks
        return gather_accel_blocks(g, pos, self.N3, self.dx, self.mesh)

    def kick_gas(self, U, g, s, dt, extra=None):
        """Kick the gas; extra: (fn, args) field programs, each evaluated on the block at its origin."""
        if not extra:
            return self._kick(U, g, s, dt)
        fns = tuple(f for f, _ in extra)
        prog = self._kick_fields.get(fns)
        if prog is None:
            from luna.gravity.particle_mesh_blocks import _block_origin
            from luna.utils.shmap import shard_map
            from jax.sharding import PartitionSpec as P
            Pg, PU = P("x", "y", "z"), P(None, "x", "y", "z")
            nloc = tuple(n // p for n, p in zip(self.N3, self.pdims))
            pdims = self.pdims

            fused = _fused_sink_kick(U, extra, self.kernel)

            def body(U, g, s, dt, *args):
                org = _block_origin(pdims, nloc)
                if fused is not None:
                    from luna.gravity import kick_pallas
                    sk = fused[0]
                    return kick_pallas.kick(U, g, s, dt, *args[0], origin=org, dxyz=sk.dxyz,
                                            G=sk.G, eps=sk.eps, box=sk.image_box)
                fields = [fn(*a, origin=org) for fn, a in zip(fns, args)]
                ex = fields[0]
                for f in fields[1:]:
                    ex = tuple(x + y for x, y in zip(ex, f))
                return kick_gas_body(U, g, s, dt, extra=ex)

            n_args = [len(a) for _, a in extra]
            in_specs = (PU, (Pg, Pg, Pg), P(), P()) + tuple(tuple(P() for _ in range(k))
                                                            for k in n_args)
            prog = self._kick_fields[fns] = jax.jit(shard_map(
                body, mesh=self.mesh, in_specs=in_specs, out_specs=PU))
        return prog(U, g, s, dt, *(tuple(a) for _, a in extra))

    def kick_gas_work(self, U, g, s, dt):
        """As `LocalGravity.kick_gas_work`."""
        return self._kick_work(U, tuple(g), s, dt)

    def close_flux(self, U, g, s, h, psi0, psi1, s0, F, W1, dt, phi_ext=None):
        """As `LocalGravity.close_flux`; F per shard has N_d + 1 faces along d."""
        if phi_ext is None:
            U, moved = self._close_flux(U, tuple(g), s, h, psi0, psi1, s0, tuple(F), W1, dt)
        else:
            U, moved = self._close_flux_ext(U, tuple(g), s, h, psi0, psi1, s0, tuple(F), W1, dt, phi_ext)
        return U, moved * self.dx ** 3

    def booked_work(self, psi0, psi1, s0, s1, drho, phi_ext=None):
        """As `LocalGravity.booked_work`."""
        if phi_ext is None:
            return self._booked_work(psi0, psi1, s0, s1, drho) * self.dx ** 3
        return self._booked_work_ext(psi0, psi1, s0, s1, drho, phi_ext) * self.dx ** 3

    def kick_drift(self, pos, p, g, dtau):
        return self._kick_drift(pos, p, g, dtau)

    def close_kick(self, graw, p_half, active, dtau, s):
        return self._close(graw, p_half, active, dtau, s)

    def assemble_particles(self, pos, p, mass, ids, n_total=None, capacity_factor=1.5):
        """Per-block fixed-capacity buffers; n_total: this process holds only its block's rows (same on every rank)."""
        from luna.gravity.particle_mesh_blocks import (AXES, assemble_blocks_local,
                                                  assemble_blocks_sharded)
        from jax.sharding import PartitionSpec as P
        pos, p, mass = np.asarray(pos), np.asarray(p), np.asarray(mass)
        idsS = None
        if n_total is not None:
            cols, specs = [pos, p, mass], [P(AXES, None), P(AXES, None), P(AXES)]
            if ids is not None:
                cols.append(np.asarray(ids))
                specs.append(P(AXES))
            bufs, actS, C = assemble_blocks_local(cols, specs, n_total, self.mesh,
                                                  capacity_factor)
            posS, pS, massS = bufs[:3]
            if ids is not None:
                idsS = bufs[3]
        else:
            posS, pS, massS, actS, idsS, C = assemble_blocks_sharded(
                pos, p, mass, self.N3, self.pdims, self.dx, self.box, self.mesh,
                capacity_factor, ids=None if ids is None else np.asarray(ids))
        self.capacity = int(C)
        if self.prec.particles is jnp.float32:
            posS, pS, massS, actS = (x.astype(jnp.float32) for x in (posS, pS, massS, actS))
        return posS, pS, massS, actS, idsS

    def exchange(self, pos, p, mass, active, ids, face_capacity_factor=None):
        from luna.gravity.particle_mesh_blocks import exchange_particles_blocks
        return exchange_particles_blocks(pos, p, mass, active, self.N3, self.dx,
                                         self.box, self.mesh,
                                         face_capacity_factor=face_capacity_factor, ids=ids)

    def settle(self, pos, p, mass, active, ids, pos_host):
        """One exchange sized by counting the movers (the IC displacement exceeds the per-step face buffers)."""
        from luna.gravity.particle_mesh_blocks import (block_of_process,
                                                  exact_settle_face_capacity,
                                                  exchange_particles_blocks)
        cap = exact_settle_face_capacity(np.asarray(pos_host), self.N3, self.pdims,
                                         self.dx, self.box, block_of_process(self.mesh))
        return exchange_particles_blocks(pos, p, mass, active, self.N3, self.dx,
                                         self.box, self.mesh, face_capacity=cap, ids=ids)

    def resort(self, pos, p, mass, active, g, ids):
        from luna.gravity.particle_mesh_blocks import resort_blocks
        return resort_blocks(pos, p, mass, active, g, self.N3, self.dx, self.mesh, ids=ids)

    def component(self, U, k):
        from jax.sharding import PartitionSpec as P
        from luna.utils.shmap import shard_map
        fn = self._components.get(k)
        if fn is None:
            fn = self._components[k] = jax.jit(shard_map(
                lambda u: u[k], mesh=self.mesh, in_specs=(P(None, "x", "y", "z"),),
                out_specs=P("x", "y", "z")))
        return fn(U)

    def grid_max(self, field):
        from luna.utils.mesh_reduce import grid_reduce
        return grid_reduce(self.mesh, "max", field)

    def grid_sum(self, field):
        from luna.utils.mesh_reduce import grid_reduce
        return grid_reduce(self.mesh, "sum", field)

    def grid_mean(self, field):
        from luna.utils.mesh_reduce import grid_mean
        return grid_mean(self.mesh, field)

    def block_box(self):
        """(lo, hi) index box of the block this process writes; None on one process."""
        if jax.process_count() == 1:
            return None, None
        from luna.gravity.particle_mesh_blocks import block_of_process
        blk = block_of_process(self.mesh)
        nloc = tuple(n // q for n, q in zip(self.N3, self.pdims))
        return (tuple(i * m for i, m in zip(blk, nloc)),
                tuple((i + 1) * m for i, m in zip(blk, nloc)))

    def local_rows(self, arr):
        """This process's rows of a flat particle buffer."""
        return np.concatenate([np.asarray(s.data) for s in arr.addressable_shards], axis=0)
