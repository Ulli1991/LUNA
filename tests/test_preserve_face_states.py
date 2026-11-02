"""preserve_face_states (RAMSES floors): near-vacuum states, floors firing below smallr, a no-op on smooth states, finite Riemann VJPs."""
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from luna.hydro.conversions import prim_to_cons
from luna.time_integration.integrators import create_muscl_hancock_modular_step

G, N = 5.0 / 3.0, 32
FLOORS = (1e-10, 1e-10)


def _state(kind, dim):
    """Primitive (dim + 2, N, [4, 4]) state varying along x: an advected 4x-per-cell ramp into 1e-8, or a 1 : 1e-6 contact kicked at v = 5."""
    x = (np.arange(N) + 0.5) / N
    if kind == "ramp":
        rho, v = 1e-8 * 4.0 ** np.minimum(np.abs(np.arange(N) - N // 2), 12), np.full(N, 50.0)
    else:
        rho, v = np.where(x < 0.5, 1.0, 1e-6), np.where(np.abs(x - 0.5) < 0.1, 5.0, 0.0)
    sh = (N,) if dim == 1 else (N, 4, 4)
    b = lambda a: np.broadcast_to(a.reshape((N,) + (1,) * (dim - 1)), sh)  # noqa: E731
    return np.stack([b(rho), b(v), *[np.zeros(sh)] * (dim - 1), np.full(sh, 1e-6)])


def _run(W, dim, floors, allow, steps=40):
    """(minimum conserved density, -inf once non-finite; first-order faces; floored faces) over `steps` CFL-0.8
    MC/HLLC steps (periodic); allow: the first-order face state (audit C8), else such a face is NaN (refused)."""
    cdt, step = create_muscl_hancock_modular_step(None, G, 1, 1, 1 / N, 1 / 4, 1 / 4, N, 4, 4, 0, dim, 0.8,
                                                  floors=floors, allow_first_order=allow, face_report=True)
    U, lo, n, nf = prim_to_cons(jnp.asarray(W), G), np.inf, 0, 0
    for _ in range(steps):
        U, rep = step(U, cdt(U))
        n, nf = n + int(np.sum(rep["n"])), nf + int(np.sum(rep["n_floor"]))
        if not bool(jnp.isfinite(U).all()):
            return -np.inf, n, nf
        lo = min(lo, float(jnp.min(U[0])))
    return lo, n, nf


# The faces MC makes non-physical here over 40 steps (CPU FP64, exact: integer counts of a deterministic run),
# FACES[(kind, dim)] = (first order with C8 allowed, floored with the floors alone), from the jobs below.
FACES = {("ramp", 1): (321, 321), ("ramp", 3): (5152, 5152), ("jump", 1): (73, 71),
         ("jump", 3): (1184, 1168)}   # job 31654813


@pytest.mark.parametrize("kind,dim", [("ramp", 1), ("ramp", 3), ("jump", 1), ("jump", 3)])
def test_density_stays_positive(kind, dim):
    """Neither fix: MC's faces go non-physical and the run is refused (NaN). The first-order face fix alone (C8,
    allowed) keeps rho > 0 for 40 steps, and so do the floors alone, and both; each count is asserted."""
    W = _state(kind, dim)
    lo, n, _ = _run(W, dim, None, False)
    assert lo == -np.inf and n > 0
    lo, n, _ = _run(W, dim, None, True)
    lo_f, n_f, nf_f = _run(W, dim, FLOORS, False)
    lo_b, n_b, nf_b = _run(W, dim, FLOORS, True)
    print(f"FACES {kind} {dim}-D: C8 {n} (rho min {lo:.3g}); floors alone {nf_f} floored (rho min {lo_f:.3g}); "
          f"both {n_b} first order + {nf_b} floored")
    assert lo > 0.0 and lo_f > 0.0 and lo_b > 0.0 and n_f == 0
    assert (n, nf_f) == FACES[(kind, dim)]


def test_floors_fire_below_smallr():
    """Cells at 0 < rho < smallr: the face fix leaves them, the floors change the step, both stay finite and positive."""
    x = (np.arange(N) + 0.5) / N
    W = np.stack([np.where(np.abs(x - 0.5) < 0.2, 1e-12, 1.0), np.full(N, 0.5), np.ones(N)])
    outs = []
    for fl in (None, FLOORS):
        cdt, step = create_muscl_hancock_modular_step(None, G, 1, 1, 1 / N, 1, 1, N, 1, 1, 0, 1, 0.8, floors=fl)
        U = prim_to_cons(jnp.asarray(W), G)
        for _ in range(10):
            U = step(U, cdt(U))
        outs.append(np.asarray(U))
    for U in outs:
        assert np.isfinite(U).all() and U[0].min() > 0.0
    assert not np.array_equal(*outs)


def test_floors_are_bitwise_noop_on_smooth_state():
    """No floor fires on a smooth subsonic state: the step equals the default bit for bit."""
    x = (np.arange(N) + 0.5) / N
    W = np.stack([1 + 0.3 * np.sin(2 * np.pi * x), 0.2 * np.cos(2 * np.pi * x), np.ones(N)])
    outs = []
    for fl in (None, FLOORS):
        cdt, step = create_muscl_hancock_modular_step(None, G, 1, 1, 1 / N, 1, 1, N, 1, 1, 0, 1, 0.8, floors=fl)
        U = prim_to_cons(jnp.asarray(W), G)
        for _ in range(10):
            U = step(U, cdt(U))
        outs.append(np.asarray(U))
    assert np.array_equal(*outs)


@pytest.mark.parametrize("solver", ["hllc", "hll"])
def test_riemann_vjp_finite_with_negative_face_pressure(solver):
    """A face with c^2 < 0: the sqrt guard keeps the VJP finite (it was NaN)."""
    from luna.hydro.fluxes import hll_flux_primitive, hllc_flux_primitive
    fn = hllc_flux_primitive if solver == "hllc" else hll_flux_primitive
    WL = jnp.array([[1.0, 1.0], [0.5, 0.5], [0.0, 0.0], [0.0, 0.0], [-0.1, 1.0]])   # face 0: P_L < 0
    WR = jnp.array([[1.0, 1.0], [0.4, 0.4], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    gL, gR = jax.grad(lambda a, b: jnp.sum(fn(a, b, 0, G)), argnums=(0, 1))(WL, WR)
    assert np.isfinite(np.asarray(gL)).all() and np.isfinite(np.asarray(gR)).all()


def test_validate_refuses_bad_floors():
    """Config.validate: floors must be > 0, and the dense ORDER=1 step only."""
    from luna.template_config import Config
    with pytest.raises(ValueError, match="hydro_smallr"):
        replace(Config(), preserve_face_states=True, hydro_smallr=0.0).validate()
    with pytest.raises(ValueError, match="preserve_face_states: the dense"):
        replace(Config(), preserve_face_states=True, ORDER=0).validate()
