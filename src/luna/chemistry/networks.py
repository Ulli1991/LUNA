"""chemcool networks 5 and 17 as Rodas3 Networks: tolerances, bounds and first-step floors."""
import functools

import jax.numpy as jnp

from luna.chemistry import rates_net5, rates_net17
from luna.ode.rodas3 import Network


def _net5_ymax_list(p):
    return [0.5 + 0 * p.n_h, 1.0 + 0 * p.n_h, jnp.minimum(p.abundc, p.abundo) + 0 * p.n_h,
            jnp.inf + 0 * p.n_h]


def _net5_ymax(p):
    return jnp.stack(_net5_ymax_list(p))


def _net5_floor(p):
    co = jnp.maximum(1e-14, 1e-6 * jnp.minimum(p.abundc, p.abundo))
    return jnp.stack([1e-7 + 0 * p.n_h, 1e-12 + 0 * p.n_h, co, 0 * p.n_h])


# chemcool_consts.h: RTOL, ATOL_H2, ATOL_HP, ATOL_CO, ATOL_TMP; evolve_abundances'
# safety (0.1, DTCOOL_SCALE_FACTOR = 0.3 for e) and pool floor for CO
NET5 = Network(rhs=rates_net5.rhs, rtol=1e-4, atol=(1e-7, 1e-12, 1e-14, 0.0),
               safety=(0.1, 0.1, 0.1, 0.3), ymax=_net5_ymax, ctrl_floor=_net5_floor,
               hidden=("tdust",), converge=(("dust_rtol", 1e-12),))
NET5_CIE = NET5._replace(rhs=functools.partial(rates_net5.rhs, cie_cool=True))   # WSS_CIE_COOL


def _net17_ymax(p):
    z, he = 0 * p.n_h, rates_net17.ABHE
    return jnp.stack([0.5 + z, 1.0 + z, jnp.minimum(p.abundc, p.abundo), he + z, he + z,
                      p.abund_d + z, p.abund_d + z, jnp.inf + z])


def _net17_floor(p):
    z, eps = 0 * p.n_h, 1e-6
    co = jnp.maximum(1e-14, eps * jnp.minimum(p.abundc, p.abundo))
    he = jnp.maximum(1e-14, eps * rates_net17.ABHE) + z
    return jnp.stack([1e-7 + z, 1e-12 + z, co, he, he, jnp.maximum(1e-14, eps * p.abund_d) + z,
                      jnp.maximum(1e-12, eps * p.abund_d) + z, z])


# chemcool_consts.h ATOL_H2, _HP, _CO, _HEP, _HEPP, _DP, _HD, _TMP; check_abundances' pools;
# evolve_abundances' ctrl_floor (CO, He+, He++, D+, HD against their pools)
NET17 = Network(rhs=rates_net17.rhs, rtol=1e-4,
                atol=(1e-7, 1e-12, 1e-14, 1e-14, 1e-14, 1e-14, 1e-12, 0.0),
                safety=(0.1,) * 7 + (0.3,), ymax=_net17_ymax, ctrl_floor=_net17_floor,
                hidden=("tdust",), converge=(("dust_rtol", 1e-12),))
NET17_FIXED = NET17._replace(rhs=functools.partial(rates_net17.rhs, fixes17="fixed"))
NET5_NEWTON = NET5._replace(rhs=functools.partial(rates_net5.rhs, eos="newton"))
