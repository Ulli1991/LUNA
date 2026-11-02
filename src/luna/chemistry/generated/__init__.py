"""SymPy-generated network-5 RHS and exact Jacobian (tools/chem_sympy/gen_net5.py); rates_net5.rhs stays the default.
dust="implicit": T_dust from the Newton solve, its implicit derivative; "frozen": p.tdust. net5_core: one pixel, lists.
"""
import importlib
from typing import NamedTuple

import jax.numpy as jnp

NPIX = (1, 12)


def _module(p):
    npix = p.n_h2_col.shape[-1]
    if npix not in NPIX:
        raise ValueError(f"no generated network 5 for n_pix = {npix} (have {NPIX})")
    return importlib.import_module(f"luna.chemistry.generated.net5_p{npix}")


def rhs(y, p, dust="implicit"):
    """(f, aux) of network 5 from the generated code."""
    return getattr(_module(p), f"rhs_{dust}")(y, p)


def rhs_and_jac(y, p, dust="implicit"):
    """(f, J, aux) of network 5 with J[..., i, j] = d f_i / d y_j, from the generated code."""
    return getattr(_module(p), f"rhs_and_jac_{dust}")(y, p)


def core_q(p):
    """Params for net5_core: the one TreeCol pixel's columns as plain per-cell values."""
    if p.n_h2_col.shape[-1] != 1:
        raise ValueError("net5_core is network 5 with one TreeCol pixel")
    return p._replace(**{k: getattr(p, k)[..., 0] for k in ("n_h_col", "n_h2_col", "n_co_col")})


class Core(NamedTuple):
    """net5_core on (list of components, Params): Network.lists; squeezed: the columns have no pixel axis."""
    dust: str = "implicit"
    dust_iters: int = None
    squeezed: bool = False

    def _q(self, p):
        return p if self.squeezed else core_q(p)

    def rhs(self, ys, p):
        """(f, aux) as lists."""
        from luna.chemistry.generated import net5_core
        return getattr(net5_core, f"rhs_{self.dust}")(*ys, self._q(p), self.dust_iters)

    def rhs_and_jac(self, ys, p):
        """(f, J, aux) as lists, J[i][j] = d f_i / d y_j."""
        from luna.chemistry.generated import net5_core
        return getattr(net5_core, f"rhs_and_jac_{self.dust}")(*ys, self._q(p), self.dust_iters)


def rhs_nolvg(y, p, dust="implicit", dust_iters=None):
    """(f, aux) of net5_core with y (..., 4) stacked: rates_net5.rhs(lvg=False), generated."""
    f, aux = Core(dust, dust_iters).rhs([y[..., i] for i in range(4)], p)
    return jnp.stack(f, -1), aux


def rhs_and_jac_nolvg(y, p, dust="implicit", dust_iters=None):
    """(f, J, aux) of net5_core with y (..., 4) stacked."""
    f, J, aux = Core(dust, dust_iters).rhs_and_jac([y[..., i] for i in range(4)], p)
    return jnp.stack(f, -1), jnp.stack([jnp.stack(r, -1) for r in J], -2), aux
