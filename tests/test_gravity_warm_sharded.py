"""The sharded warm-started gravity solve gives the one-device potentials, step by step, periodic and isolated."""
import jax
import numpy as np
import pytest

from tests import _gravity_warm as gw
from tests.test_gravity_warm import CASES, N_CPU
from tests.test_gravity_warm_events import CASES as EVENT_CASES

C_D = dict(gravity_warm_tol=3e-3, gravity_warm_extrapolate=True, gravity_proper_mixed=True)
OFF = dict(gravity_warm_tol=0.0, gravity_warm_extrapolate=False, gravity_proper_mixed=False)   # fixed FP64 cycles


def _mesh():
    n = jax.device_count()
    if n < 2:
        pytest.skip("needs 2 or more devices")
    from luna.gravity.particle_mesh_blocks import make_block_mesh
    pdims = (2, 2, 2) if n >= 8 else (2, 1, 1)
    return make_block_mesh(jax.devices()[:int(np.prod(pdims))], pdims)


@pytest.mark.parametrize("name,over", [
    ("turbulence_fast", OFF), ("turbulence_fast", {**OFF, "gravity_mg_agg_n": 8}), ("turbulence_fast", {}),
    ("turbulence_fast", C_D), ("turbulence_fast", {**C_D, "gravity_mg_agg_n": 8}),
    ("collapse_isolated", OFF), ("collapse_isolated", {}), ("collapse_isolated", C_D), ("collapse_isolated", {**C_D, "gravity_mg_agg_n": 8})])
def test_sharded_warm_solve_matches_one_device(name, over):
    """agg_n 8: sharded levels down to an 8^3 root grid; {}: the defaults; OFF: fixed FP64 cycles; C_D: tol 3e-3."""
    make, boundary, _, _ = {**CASES, **EVENT_CASES}[name][:4]
    steps = make(N_CPU)
    cfg = gw.config(N_CPU, boundary, **over)
    dense, sharded, info_d, info_s = [], [], [], []
    err_d = gw.run(gw.backend(cfg), steps, boundary, keep=dense, info=info_d)
    err_s = gw.run(gw.backend(cfg, _mesh()), steps, boundary, keep=sharded, info=info_s)
    if gw.tight_bounds():
        assert err_s[1:].max() <= gw.TIGHT[(name, N_CPU)]
    else:
        assert err_s[:, 1].max() <= gw.bounds(cfg)[(name, N_CPU)]
    # FP32 cycles sum in another order on each layout: FP32 rounding
    same = 1e-6 if cfg.gravity_proper_mixed else 1e-10
    for d, s in zip(dense, sharded):
        if boundary == 0:
            d, s = d - d.mean(), s - s.mean()
        assert np.sqrt(np.mean((d - s) ** 2) / np.mean(d ** 2)) < same
    assert np.allclose(err_d, err_s, rtol=1e-6 if same < 1e-6 else 1e-3, atol=1e-12)
    if cfg.gravity_warm_tol > 0:
        assert [i and i[0] for i in info_d] == [i and i[0] for i in info_s]
        assert any(i is not None for i in info_s)
