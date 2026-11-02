"""Binomial (revolve) checkpointing: the schedule, the gradient, the cost.

  1. the schedule reverses every step from the right state, and its advance
     count is Griewank's closed form, which is the optimum of the recursion
  2. the revolve gradient equals the per-step-checkpoint gradient of the
     production step (hydro, hydro + FMG gravity), 1 device and a 2x2x2 mesh,
     for several snapshot counts
  3. the steps actually executed are the schedule's count
  4. the compiled reverse pass needs less memory with fewer snapshots
"""
from functools import lru_cache

import numpy as np
import pytest

import jax
import jax.numpy as jnp                                    # noqa: E402

from luna.time_integration.revolve import (binomial_advances,  # noqa: E402
                                           revolve_rollout, schedule)

NX = 16


# ---------------------------------------------------------------------------
# 1. the schedule
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _dp(l, c):
    """Minimal advances by brute recursion: the definition of optimal."""
    if l <= 1:
        return 0
    if c == 1:
        return l * (l - 1) // 2
    return min(m + _dp(l - m, c - 1) + _dp(m, c) for m in range(1, l))


def _run_symbolic(n, c):
    """Execute the table on step indices; return the advances it made."""
    sch = schedule(n, c)
    buf, cur, adv = {0: 0}, 0, 0
    for a, sl in zip(sch["sweep_adv"], sch["sweep_slot"]):
        cur, adv = cur + a, adv + a
        buf[int(sl)] = cur
    for g in range(n):
        cur = buf[int(sch["restore"][g])]
        for j in range(sch["npairs"][g]):
            cur, adv = cur + sch["adv"][g, j], adv + sch["adv"][g, j]
            buf[int(sch["slot"][g, j])] = cur
        assert cur == n - 1 - g, f"n={n} c={c}: VJP of step {n - 1 - g} sees {cur}"
    assert max(buf) <= c                   # c snapshots + the scratch slot
    return int(adv)


def test_closed_form_is_the_optimum():
    for l in range(1, 45):
        for c in range(1, 7):
            assert binomial_advances(l, c) == _dp(l, c), (l, c)


# c <= n: more snapshots than steps is not a schedule
@pytest.mark.parametrize("n,c", [(n, c) for c in (1, 2, 3, 8, 32) for n in (1, 2, 7, 33, 120, 1000) if c <= n])
def test_schedule_reverses_every_step(n, c):
    adv = _run_symbolic(n, c)
    assert adv == schedule(n, c)["advances"] == binomial_advances(n, c)


# ---------------------------------------------------------------------------
# 2-4. the production step
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _problem(case, mesh, n):
    """The production rollout (tools/bench_autodiff.py's build) on the first
    prod(mesh) of the virtual devices."""
    from luna.utils import distributed
    from tools.bench_autodiff import _build
    devs = jax.devices()[:int(np.prod(mesh))]
    real = jax.devices, distributed.mesh_utils.create_device_mesh
    jax.devices = lambda *a, **k: devs
    distributed.mesh_utils.create_device_mesh = \
        lambda shape, *a, **k: np.array(devs).reshape(shape)
    try:
        sim, theta, U0, info = _build(case, NX, n, mesh, "step", 32)
    finally:
        jax.devices, distributed.mesh_utils.create_device_mesh = real
    return sim, theta, U0, info["step_once"], info["carry0"]


def _revolve_sim(case, mesh, n, c, step_wrap=None):
    _, theta, U0, step_once, carry0 = _problem(case, mesh, n)
    step = step_wrap(step_once) if step_wrap else step_once
    roll = revolve_rollout(step, n, c)
    return lambda U: roll(theta, carry0(U)).U


def _loss(sim):
    return lambda U: jnp.sum(sim(U) ** 2)


N_GRAD = 10


@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 2, 2)], ids=["1dev", "2x2x2"])
@pytest.mark.parametrize("case", ["hydro", "gravity"])
def test_gradient_equals_per_step(case, mesh):
    sim, theta, U0, _, _ = _problem(case, mesh, N_GRAD)
    g_ref = np.asarray(jax.jit(jax.grad(_loss(lambda U: sim(theta, U))))(U0))
    assert np.all(np.isfinite(g_ref)) and np.max(np.abs(g_ref)) > 0
    for c in (1, 3, N_GRAD):
        g = np.asarray(jax.jit(jax.grad(_loss(_revolve_sim(case, mesh, N_GRAD, c))))(U0))
        np.testing.assert_array_equal(g, g_ref, err_msg=f"{case} {mesh} c={c}")


N_COUNT = 12


@pytest.mark.parametrize("c", [1, 2, 3, 5, N_COUNT])
def test_recompute_count_matches_schedule(c):
    """Every executed step, counted on the host: the forward sweep and the
    recomputation (binomial_advances + 1) plus one linearization per VJP."""
    count = [0]

    def wrap(step):
        def counted(state, theta):
            jax.debug.callback(lambda: count.__setitem__(0, count[0] + 1))
            return step(state, theta)
        return counted

    _, _, U0, _, _ = _problem("hydro", (1, 1, 1), N_COUNT)
    grad = jax.jit(jax.grad(_loss(_revolve_sim("hydro", (1, 1, 1), N_COUNT, c, wrap))))
    jax.block_until_ready(grad(U0))
    count[0] = 0
    jax.block_until_ready(grad(U0))
    assert count[0] == binomial_advances(N_COUNT, c) + 1 + N_COUNT


N_MEM = 24


def test_memory_falls_with_fewer_snapshots():
    """XLA's temp allocation of the compiled value-and-grad: per-step
    checkpointing holds N_MEM states; revolve holds c + 1 slots (the scratch
    slot included) plus the rollout's output and the state of step n-1."""
    sim, theta, U0, _, carry0 = _problem("hydro", (1, 1, 1), N_MEM)
    state = sum(x.size * x.dtype.itemsize
                for x in jax.tree.leaves(jax.eval_shape(carry0, U0)))

    def temp(loss):
        return (jax.jit(jax.value_and_grad(loss)).lower(U0).compile()
                .memory_analysis().temp_size_in_bytes)

    ref = temp(_loss(lambda U: sim(theta, U)))
    got = {c: temp(_loss(_revolve_sim("hydro", (1, 1, 1), N_MEM, c)))
           for c in (2, 4, 8)}
    assert got[2] < got[4] < got[8] < ref, (got, ref)
    # each snapshot is a state (measured on CPU: exactly 6.0 from c=2 to c=8)
    assert got[8] - got[2] >= 6 * state, (got, state)
    assert ref - got[8] >= (N_MEM - (8 + 3)) * state, (ref, got, state)


def test_snapshot_budget_is_per_device():
    """autodiff_snapshot_gib counts the state per device, scratch slot included:
    1 GiB over 5 MiB per device (10 MiB on a 2x1x1 mesh) is 204 slots."""
    from dataclasses import replace
    from luna.autodiff.rollout import snapshots_for
    from luna.template_config import Config
    state = [jax.ShapeDtypeStruct((5, 64, 64, 64), jnp.float64)]      # 10 MiB
    cfg = replace(Config(), autodiff_snapshot_gib=1.0, MESH_SHAPE=(2, 1, 1))
    assert snapshots_for(cfg, state) == 203
    assert snapshots_for(replace(cfg, autodiff_snapshot_gib=0.0,
                                 autodiff_snapshots=7), state) == 7
