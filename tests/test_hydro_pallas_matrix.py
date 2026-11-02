"""Every (dimension, limiter, Riemann solver, precision) of the Pallas hydro kernels against XLA.

The XLA reference is the wide step of muscl_hancock_divergence (1-D/2-D face fluxes, 3-D
_muscl_hancock_3d); both use slopes.slope_limiter, jacobian.py's rows and the fluxes cores, so
under Triton the fused and split kernels equal XLA bitwise (A100, 2026-10-12). The CPU interpreter
rounds a few cells differently (XLA CPU contracts FMAs per expression), hence round-off bounds
there. Non-cubic blocks. recon32 is 3-D only, in XLA and Pallas alike. Apart from
tests/test_hydro_pallas.py because 42 combinations x 3 compiles take minutes, not seconds."""
import numpy as np
import pytest

import jax.numpy as jnp

from tests.test_hydro_pallas import (GPU, LIMITERS, _SHAPES, _agreement, _imbalance,
                                     _pallas_nd, _prims_nd, _same_arithmetic, _xla_nd)


def _combos():
    out = []
    for nd in (1, 2, 3):
        for mode in ("fp64", "fp32", "mixed"):
            if mode == "mixed" and nd != 3:
                continue
            for lim in LIMITERS:
                for solver in ("hllc", "hll"):
                    marks = [] if mode == "fp64" else [pytest.mark.precision(mode)]
                    out.append(pytest.param(nd, lim, solver, mode, marks=marks,
                                            id=f"{nd}d-{LIMITERS[lim]}-{solver}-{mode}"))
    return out


# CPU interpreter, these shapes (2026-10-14): FP64 <= 2.1e-16, FP32 <= 5.6e-8 of max|dU|.
_AGREE = {"fp64": 1e-12, "fp32": 1e-6, "mixed": 1e-6}


@pytest.mark.parametrize("nd,limiter,solver,mode", _combos())
def test_every_dim_limiter_solver_precision_is_xla(nd, limiter, solver, mode):
    """Fused and split kernels against the XLA wide step: bitwise on a GPU; split == fused."""
    shape = _SHAPES[nd]
    W = _prims_nd(shape)
    if mode == "fp32":
        W = W.astype(jnp.float32)
    Wp = jnp.pad(W, [(0, 0)] + [(2, 2)] * nd, mode="wrap")
    dt = 0.2 / max(shape) / 3.0
    a = _xla_nd(Wp, dt, limiter, solver, recon32=mode == "mixed")
    f = _pallas_nd(Wp, dt, limiter, solver, recon32=mode == "mixed", split=False)
    s = _pallas_nd(Wp, dt, limiter, solver, recon32=mode == "mixed", split=True)
    assert a.dtype == f.dtype == s.dtype == (np.float32 if mode == "fp32" else np.float64)
    if GPU:
        np.testing.assert_array_equal(f, a)
        np.testing.assert_array_equal(s, a)
    assert _agreement(a, f) < _AGREE[mode]
    _same_arithmetic(f, s, mode)
    if mode == "fp64":                    # periodic: the faces telescope
        assert _imbalance(f) < 1e-13
    if limiter:                           # it is this limiter, not minmod
        assert _agreement(_xla_nd(Wp, dt, 0, solver, mode == "mixed"), f) > 1e-6
