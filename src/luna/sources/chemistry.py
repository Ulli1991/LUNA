"""Prototype coupling of chemcool network 5 as a stage: every cell integrated over the hydro dt.

Abundances ride through the hydro as passive scalars S = rho x (state.sources["chemistry"]["S"])."""
import functools
import math

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from luna.ode import rodas3 as ig

from luna.chemistry import networks
from luna.chemistry import rates_net5
from luna.chemistry import tables as tb

NAME = "chemistry"
SPECIES = ("x_H2", "x_H+", "x_CO")
ABUNDC, ABUNDO = 1.41e-4, 3.16e-4          # solar C and O per H, as in the port's references
SKIP_T = 5e5                              # evolve_abundances' hot-gas bypass [K]
T_FLOOR = 5.0                             # internal energy floor, as 1.5 n k T of atomic gas [K]

NET = networks.NET5._replace(rhs=functools.partial(rates_net5.rhs, lvg=False))   # no LVG: divv = 0
DUST_ITERS = 4                            # the hybrid's dust Newton: this many steps, no loop


class ChemConsts:
    """Unit conversions and network parameters the stage closes over."""

    def __init__(self, cfg):
        u = cfg.units
        if not u.is_enabled:
            raise ValueError("chemistry needs cfg.units (physical units)")
        self.gamma = float(cfg.gamma)
        self.rho_cgs = u.UnitDensity_in_cgs
        self.p_cgs = u.UnitPressure_in_cgs
        self.t_s = u.UnitTime_in_s
        self.box_cm = float(cfg.Lx) * u.UnitLength_in_cm
        self.mu_h = (1.0 + 4.0 * tb.ABHE) * tb.MH       # gas mass per H nucleus [g]
        self.g0 = float(cfg.chem_g0)
        self.zeta = float(cfg.chem_zeta)
        self.chunk = int(cfg.chem_chunk_log2)
        self.dust_rtol = float(getattr(cfg, "chem_dust_rtol", 1e-2))
        self.solver = "hybrid"            # "adaptive": integrate() with error control (set before make_apply)
        self.opt = ig.Options(chunk=2 ** self.chunk)

    @property
    def net(self):
        """Network 5 without LVG, the dust Newton stopping at dust_rtol."""
        return NET._replace(converge=(("dust_rtol", self.dust_rtol),))

    @property
    def net_fixed(self):
        """The hybrid's network 5: the generated f and exact J, DUST_ITERS dust Newton steps."""
        return fixed_net(self.dust_rtol)


@functools.lru_cache(maxsize=None)
def fixed_net(dust_rtol=1e-2, dust_iters=DUST_ITERS):
    """Network 5 for integrate_fixed: net5_core (no LVG, one pixel) as lists, straight-line dust."""
    from luna.chemistry import generated
    return NET._replace(rhs=functools.partial(generated.rhs_nolvg, dust_iters=dust_iters),
                        lists=generated.Core("implicit", dust_iters), ymax=networks._net5_ymax_list,
                        converge=(("dust_rtol", dust_rtol),))


def cell_inputs(U, S, c):
    """(y0 per cell (..., 4), n_H, local columns N_H, N_H2, N_CO, kinetic energy, floored)."""
    rho = U[0]
    ke = 0.5 * (U[1] * U[1] + U[2] * U[2] + U[3] * U[3]) / rho
    x = S / rho
    rho_cgs = rho * c.rho_cgs
    n_h = rho_cgs / c.mu_h
    e_floor = energy_floor(n_h)
    e_hydro = (U[4] - ke) * c.p_cgs
    floored = ~(e_hydro >= e_floor)                  # also catches NaN
    e_cgs = jnp.where(floored, e_floor, e_hydro)
    cs2 = c.gamma * (c.gamma - 1.0) * e_cgs / rho_cgs
    l_j = jnp.minimum(jnp.sqrt(math.pi * cs2 / (tb.GRAVITY * rho_cgs)), c.box_cm)
    n_col = n_h * l_j
    y0 = jnp.stack([x[0], x[1], x[2], e_cgs], -1)
    return y0, n_h, (n_col, x[0] * n_col, x[2] * n_col), ke, floored


def energy_floor(n_h):
    """The internal energy density [erg cm^-3] of atomic gas at T_FLOOR."""
    return 1.5 * (1.0 + tb.ABHE) * n_h * tb.KBOLTZ * T_FLOOR


def params(c, n_h, cols, tdust, per_cell=True):
    """Network-5 Params for a batch of cells, one TreeCol pixel holding the local column."""
    z = 0.0 * n_h                         # every field per cell: varying over the mesh axes
    base = rates_net5.Params(
        n_h=n_h, n_h_col=cols[0][:, None], n_h2_col=cols[1][:, None],
        n_co_col=cols[2][:, None], g0=c.g0, g0_dust=c.g0, zeta=c.zeta, dgr=1.0,
        abundc=ABUNDC, abundo=ABUNDO, tdust=tdust)
    if not per_cell:                      # constants shared by all cells (integrate_fixed)
        return rates_net5.Params(*[jnp.asarray(v, jnp.float64) for v in base])
    return rates_net5.Params(*[v if jnp.ndim(v) else z + v for v in base])


def integrate_cells(c, y0, n_h, cols, tdust, h0, dt_s, opt=ig.Options()):
    """integrate() over flat cells in chunks of 2^chunk (lax.map): Result fields as a tuple."""
    n = y0.shape[0]
    m = min(2 ** c.chunk, n)
    k = n // m

    def one(a):
        y, nh, ch, ch2, cco, td, h, dts = a
        r = ig.integrate(y, params(c, nh, (ch, ch2, cco), td), dts, h0=h, net=c.net, opt=opt)
        return r.y, r.params.tdust, r.h, r.n_steps, r.n_rejected, r.failed

    args = (y0, n_h, *cols, tdust, h0, jnp.broadcast_to(dt_s, (n,)))
    out = jax.lax.map(one, tuple(a.reshape((k, m) + a.shape[1:]) for a in args))
    return tuple(a.reshape((n,) + a.shape[2:]) for a in out)


def solve_fixed(c, y0, n_h, cols, tdust, dt_s, kernel=None):
    """integrate_fixed over flat cells with the stage's network and options (kernel: the fused one)."""
    p = params(c, n_h, cols, tdust, per_cell=False)
    return ig.integrate_fixed(y0, p, dt_s, net=c.net_fixed, opt=c.opt, solve=kernel)


def fused_kernel():
    """The fused network-5 kernel on a GPU, else None (the plain-JAX hybrid)."""
    if jax.default_backend() not in ("gpu", "rocm", "cuda"):
        return None
    from luna.chemistry import fused
    return fused.solve


def make_apply(c, mesh):
    """The jitted per-shard update (U, carry, dt) -> (U, carry)."""
    if c.solver == "hybrid":
        return make_apply_fixed(c, mesh)
    spec, cell = P(None, "x", "y", "z"), P("x", "y", "z")

    def local(U, S, tdust, h, dt):
        shape = U.shape[1:]
        n = U[0].size
        m = min(2 ** c.chunk, n)
        U2, S2 = U.reshape(5, n), S.reshape(3, n)
        td2, h2 = tdust.reshape(n), h.reshape(n)

        def one(i):                       # one chunk, sliced from U: no full-size input copies
            sl = lambda a: jax.lax.dynamic_slice_in_dim(a, i * m, m, axis=a.ndim - 1)  # noqa: E731
            Uc, Sc = sl(U2), sl(S2)
            y0, n_h, cols, ke, lo = cell_inputs(Uc, Sc, c)
            r = ig.integrate(y0, params(c, n_h, cols, sl(td2)), dt * c.t_s, h0=sl(h2), net=c.net)
            y = jnp.where(jnp.all(jnp.isfinite(r.y), -1, keepdims=True), r.y, y0)
            low = y[:, 3] < energy_floor(n_h)            # a failed cell can stop at e = 0
            e = jnp.where(low, energy_floor(n_h), y[:, 3])
            rho = Uc[0]
            return (ke + e / c.p_cgs, rho * y[:, 0], rho * y[:, 1], rho * y[:, 2],
                    r.params.tdust, r.h, r.n_steps, r.n_rejected, r.failed, lo | low)

        out = jax.lax.map(one, jnp.arange(n // m))
        E, s0, s1, s2, td, hn, ns, nr, fl, lo = (a.reshape(shape) for a in out)
        return U.at[4].set(E), jnp.stack([s0, s1, s2]), td, hn, ns, nr, fl, lo

    f = shard_map_jit(local, mesh, (spec, spec, cell, cell, P()),
                      (spec, spec, cell, cell, cell, cell, cell, cell))

    def apply(U, carry, dt):
        U, S, td, h, ns, nr, fl, lo = f(U, carry["S"], carry["tdust"], carry["h"], dt)
        return U, {"S": S, "tdust": td, "h": h, "n_steps": ns, "n_rejected": nr, "failed": fl,
                   "floored": lo}
    return apply


def make_apply_fixed(c, mesh, kernel="auto"):
    """make_apply for the hybrid solver: every cell of the shard in one integrate_fixed call."""
    spec, cell = P(None, "x", "y", "z"), P("x", "y", "z")
    kern = fused_kernel() if kernel == "auto" else kernel

    def local(U, S, tdust, h, dt):
        shape = U.shape[1:]
        n = U[0].size
        U2, S2 = U.reshape(5, n), S.reshape(3, n)
        y0, n_h, cols, ke, lo = cell_inputs(U2, S2, c)
        r = solve_fixed(c, y0, n_h, cols, tdust.reshape(n), dt * c.t_s, kern)
        ok = jnp.all(jnp.isfinite(r.y), -1)
        y = jnp.where(ok[:, None], r.y, y0)
        low = y[:, 3] < energy_floor(n_h)            # an invalid cell can stop at e = 0
        e = jnp.where(low, energy_floor(n_h), y[:, 3])
        rho = U2[0]
        steps = jnp.where(r.flagged, c.opt.k1 + c.opt.k2, c.opt.k1).astype(jnp.int32)
        out = (ke + e / c.p_cgs, rho * y[:, 0], rho * y[:, 1], rho * y[:, 2], r.params.tdust,
               steps, 0 * steps, ~ok, lo | low)
        E, s0, s1, s2, td, ns, nr, fl, lo = (a.reshape(shape) for a in out)
        return U.at[4].set(E), jnp.stack([s0, s1, s2]), td, h, ns, nr, fl, lo

    f = shard_map_jit(local, mesh, (spec, spec, cell, cell, P()),
                      (spec, spec, cell, cell, cell, cell, cell, cell))

    def apply(U, carry, dt):
        U, S, td, h, ns, nr, fl, lo = f(U, carry["S"], carry["tdust"], carry["h"], dt)
        return U, {"S": S, "tdust": td, "h": h, "n_steps": ns, "n_rejected": nr, "failed": fl,
                   "floored": lo}
    return apply


def shard_map_jit(fn, mesh, in_specs, out_specs):
    """jit(shard_map(fn)) on LUNA's dense mesh."""
    from luna.utils.shmap import shard_map
    return jax.jit(shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs))


def initial_carry(U, x0, tdust0, sharding):
    """S = rho x0 for uniform abundances x0, T_dust = tdust0, h = 0 (first-step estimate)."""
    cell = NamedSharding(sharding.mesh, P("x", "y", "z"))

    @jax.jit
    def build(U):
        S = jnp.stack([U[0] * float(v) for v in x0])
        z = jnp.zeros(U.shape[1:], U.dtype)
        zi = jnp.zeros(U.shape[1:], jnp.int32)
        return {"S": S, "tdust": z + float(tdust0), "h": z, "n_steps": zi, "n_rejected": zi,
                "failed": zi > 0, "floored": zi > 0}
    out = build(U)
    return {k: jax.device_put(v, sharding if k == "S" else cell) for k, v in out.items()}


class Chemistry:
    """Network 5 as the outermost stage: nothing before the hydro, the full dt after it."""
    name = NAME

    def __init__(self, cfg, mesh, U0, sharding):
        self.consts = ChemConsts(cfg)
        self._apply = make_apply(self.consts, mesh)
        self.carry0 = initial_carry(U0, cfg.chem_x0, cfg.chem_tdust0, sharding)

    def before(self, state, h, dt):
        return state

    def after(self, state, h, dt):
        U, carry = self._apply(state.U, state.sources[self.name], dt)
        return state.update(U=U).with_source(self.name, carry)


def setup_chemistry(cfg, hydro, init, is_main=True):
    """The Chemistry stage a config asks for, or None."""
    if getattr(cfg, "chemistry", "") != "net5":
        return None
    if cfg.cooling_enabled:
        raise ValueError("chemistry='net5' replaces the cooling stage: set cooling_enabled=False")
    if not hydro.scalars:
        raise RuntimeError("chemistry needs the hydro built with passive scalars")
    jax.config.update("jax_enable_x64", True)
    stage = Chemistry(cfg, hydro.sharding.mesh, init.U, hydro.sharding)
    if is_main:
        print(f"✓ Chemistry: network 5 (prototype stage), G0={cfg.chem_g0}, "
              f"zeta={cfg.chem_zeta:.2e}, x0={tuple(cfg.chem_x0)}", flush=True)
    return stage
