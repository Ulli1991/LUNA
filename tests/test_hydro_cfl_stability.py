"""Unsplit MUSCL-Hancock (minmod, HLLC, FP64) vs the hydro dt rule: grid-scale noise decays at the default
(RAMSES, CFL 0.8) and grows above the measured limit (docs/HYDRO_TIMESTEP.md)."""
import numpy as np
import pytest

G = 1.4
N = {1: 64, 2: 32, 3: 16}
STEPS = 300



def growth(u, kind="noise", CFL=None):
    """std(rho) after STEPS steps over its start: 1e-8 noise or a diagonal grid-scale wave on velocity u (c = 1)."""
    import jax
    import jax.numpy as jnp
    from luna.hydro.conversions import prim_to_cons
    from luna.template_config import Config
    from luna.time_integration.integrators import create_muscl_hancock_modular_step
    from tests._helpers import check_conserved
    nd = len(u)
    n = N[nd]
    dx = 1.0 / n
    CFL = Config().CFL if CFL is None else CFL
    cdt, step = create_muscl_hancock_modular_step(None, G, 1, 0, dx, dx, dx, n, n, n, 0, nd, CFL,
                                                  riemann_solver="hllc")
    x = (np.arange(n) + 0.5) / n
    X = np.meshgrid(*([x] * nd), indexing="ij")
    rng = np.random.default_rng(1)
    d = 1e-8 * (rng.standard_normal((n,) * nd) if kind == "noise" else np.sin(2 * np.pi * (n // 4) * sum(X)))
    W = np.stack([1.0 + d] + [v + 0 * d for v in u] + [(1.0 + G * d) / G])
    U0 = prim_to_cons(jnp.asarray(W), G)
    U = jax.jit(lambda U: jax.lax.fori_loop(0, STEPS, lambda _, U: step(U, cdt(U)), U))(U0)
    if np.isfinite(np.asarray(U)).all():                 # periodic: mass, momentum, energy to round-off
        check_conserved(f"growth {nd}D u={u} {kind} CFL={CFL}", U0, U)
    r = float(np.std(np.asarray(U[0])) / np.std(1.0 + d))
    print(f"MEASURED growth {nd}D u={u} {kind} CFL={CFL}: {r:.3e}")
    return r


# (u, kind, bound): bound ~1.3x the growth measured on CPU-8 and one MI300A (equal to 4 digits, wave 1.3e-7 / 1.1e-7);
# the 3-D wave at rest is the round-off floor (std rho ~1e-15): 6.4e-7 since HLLC reuses reciprocals (2026-10-11)
DEFAULT = [((0.0,), "noise", 0.15), ((5.0,), "noise", 0.18),
           ((0.0, 0.0), "noise", 0.062), ((5.0, 0.0), "noise", 0.09), ((2.0, 2.0), "noise", 0.058),
           ((0.0, 0.0, 0.0), "noise", 0.028), ((0.0, 0.0, 0.0), "wave", 8.3e-7), ((5.0, 0.0, 0.0), "noise", 0.022),
           ((1.0, 1.0, 1.0), "noise", 0.059), ((5.0, 5.0, 5.0), "noise", 0.0134)]


@pytest.mark.parametrize("u,kind,bound", DEFAULT)
def test_default_time_step_damps_grid_noise(u, kind, bound):
    """The default damps grid-scale noise at rest and in fast axis / diagonal flow, in 1D, 2D and 3D."""
    assert growth(u, kind) < bound


@pytest.mark.parametrize("u", [(0.0,), (0.0, 0.0), (0.0, 0.0, 0.0), (5.0, 5.0, 5.0)])
def test_above_the_limit_is_unstable(u):
    """Teeth: CFL 1.2, above the measured limit 1, grows noise in every dimension and in diagonal flow."""
    r = growth(u, CFL=1.2)
    assert not np.isfinite(r) or r > 1e3


def test_the_rule():
    """dt = CFL / sum_d (|v_d|+c)/dx_d per cell, in 1, 2 and 3 dimensions."""
    import jax.numpy as jnp
    from luna.time_integration.integrators import cell_dt
    rng = np.random.default_rng(0)
    v = [jnp.asarray(rng.normal(size=64)) for _ in range(3)]
    c, dx = jnp.asarray(rng.uniform(0.5, 2.0, 64)), (0.1, 0.2, 0.3)
    for nd in (1, 2, 3):
        want = 0.8 / sum((np.abs(vi) + c) / d for vi, d in zip(v[:nd], dx[:nd]))
        assert np.allclose(cell_dt(v[:nd], c, dx[:nd], 0.8), want, rtol=1e-15)
    assert np.array_equal(cell_dt(v[:1], c, dx[:1], 0.8), 0.8 / ((jnp.abs(v[0]) + c) / dx[0]))


def test_validate_refuses_unstable_courant_numbers():
    """validate(): CFL above 1 is an error; the retired cfl_rule names what replaced it."""
    import warnings
    from dataclasses import replace
    from luna.template_config import Config
    base = replace(Config(), test_problem="Sod-Shock-3D", SPATIAL_DIMENSION=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        base.validate()
        replace(base, CFL=1.0).validate()
    with pytest.raises(ValueError, match="CFL=1.05: the unsplit step is unstable above 1"):
        replace(base, CFL=1.05).validate()
    from luna.setup.cli import overrides
    with pytest.raises(ValueError, match="'cfl_rule' is retired: removed: CFL is always the RAMSES"):
        overrides({"cfl_rule": "per_axis"})
