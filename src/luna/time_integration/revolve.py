"""Binomial ("revolve") checkpointing of a fixed-length rollout (Griewank & Walther 2000).

Host-built schedule tables drive one device loop; reverse mode only. See docs/AUTODIFF.md.
"""
from functools import lru_cache
from math import comb

import numpy as np
import jax
import jax.numpy as jnp


def repetitions(n_steps, snapshots):
    """Smallest t with C(snapshots + t, t) >= n_steps."""
    t = 0
    while comb(snapshots + t, t) < n_steps:
        t += 1
    return t


def binomial_advances(n_steps, snapshots):
    """Minimal forward steps t*l - C(s+t, t-1) to reverse `n_steps` with `snapshots` slots."""
    if n_steps <= 1:
        return 0
    s = snapshots
    t = repetitions(n_steps, s)
    return t * n_steps - comb(s + t, t - 1)


@lru_cache(maxsize=None)
def _cost_row(snapshots, n_max):
    """binomial_advances(l, snapshots) for l in 0..n_max."""
    return np.array([binomial_advances(l, snapshots) for l in range(n_max + 1)],
                    dtype=np.int64)


def _split(length, snapshots, n_max):
    """Optimal first advance m = argmin m + G(length - m, s - 1) + G(m, s)."""
    right = _cost_row(snapshots - 1, n_max)
    left = _cost_row(snapshots, n_max)
    m = np.arange(1, length)
    cost = m + right[length - m] + left[m]
    return int(m[np.argmin(cost)])


def schedule(n_steps, snapshots):
    """Action tables of the binomial schedule, as a dict of int32 arrays.

    sweep_adv/sweep_slot: forward sweep (advance, store) pairs; restore [n], npairs [n],
    adv [n, P], slot [n, P]: per reversed step, last first. Also `advances` and `scratch`.
    """
    n, s = int(n_steps), int(snapshots)
    if n < 1:
        raise ValueError(f"n_steps={n}: at least one step")
    if s < 1:
        raise ValueError(f"snapshots={s}: at least one (the start)")
    scratch = s
    ops = [("store", 0)]                 # the start sits in slot 0
    stack = [("rev", 0, n, s, 0)]        # reverse steps [lo, hi) from slot k
    while stack:
        _, lo, hi, c, k = stack.pop()
        if hi - lo == 1:
            ops += [("restore", k), ("rev", lo)]
        elif c == 1:
            for i in range(hi - 1, lo - 1, -1):
                ops += [("restore", k)]
                if i > lo:
                    ops += [("adv", i - lo)]
                ops += [("rev", i)]
        else:
            m = lo + _split(hi - lo, c, n)
            ops += [("restore", k), ("adv", m - lo), ("store", k + 1)]
            # LIFO: the right part [m, hi) is reversed first
            stack.append(("rev", lo, m, c, k))
            stack.append(("rev", m, hi, c - 1, k + 1))

    # Group by reversed step: [restore] (adv, store)* [adv] rev.
    groups, cur, first = [], {"restore": None, "pairs": []}, True
    last = None
    for op in ops:
        kind = op[0]
        if kind == "restore":
            if not (last is not None and last[0] == "store" and last[1] == op[1]):
                cur["restore"] = op[1]
        elif kind == "adv":
            cur["pairs"].append([op[1], scratch])
        elif kind == "store":
            if cur["pairs"] and cur["pairs"][-1][1] == scratch:
                cur["pairs"][-1][1] = op[1]
            elif not first or op[1] != 0:
                raise AssertionError(f"store without advance: {op}")
        elif kind == "rev":
            if first and (not cur["pairs"] or cur["pairs"][-1][1] != scratch):
                cur["pairs"].append([0, scratch])     # step n-1's VJP reads it
            cur["step"] = op[1]
            groups.append(cur)
            cur, first = {"restore": None, "pairs": []}, False
        last = op
    assert [g["step"] for g in groups] == list(range(n - 1, -1, -1))

    # The first group's pairs run in the forward pass; its reversal restores the scratch slot.
    sweep = groups[0]["pairs"]
    groups[0] = {"restore": scratch, "pairs": [], "step": n - 1}
    for g in groups[1:]:
        if g["restore"] is None:
            raise AssertionError("a reversal without a restore")
    P = max(1, max(len(g["pairs"]) for g in groups))
    adv = np.zeros((n, P), np.int32)
    slot = np.full((n, P), scratch, np.int32)
    npairs = np.zeros(n, np.int32)
    for i, g in enumerate(groups):
        for j, (a, sl) in enumerate(g["pairs"]):
            adv[i, j], slot[i, j] = a, sl
        npairs[i] = len(g["pairs"])
    sweep_adv = np.array([a for a, _ in sweep], np.int32)
    sweep_slot = np.array([sl for _, sl in sweep], np.int32)
    advances = int(sweep_adv.sum() + adv.sum())
    return {"sweep_adv": sweep_adv, "sweep_slot": sweep_slot,
            "restore": np.array([g["restore"] for g in groups], np.int32),
            "npairs": npairs, "adv": adv, "slot": slot,
            "advances": advances, "scratch": scratch}


def _opaque(k):
    """Return k as a loop bound XLA cannot read."""
    # ROCm segfaults on long while loops with a known trip count (docs/AUTODIFF.md).
    return jax.lax.optimization_barrier(jnp.asarray(k, jnp.int32))


def _advance(step, cur, theta, k):
    return jax.lax.fori_loop(0, k, lambda _, c: step(c, theta), cur)


def _store(buf, cur, i):
    return jax.tree.map(
        lambda b, x: jax.lax.dynamic_update_index_in_dim(b, x, i, 0), buf, cur)


def _load(buf, i):
    return jax.tree.map(lambda b: jax.lax.dynamic_index_in_dim(b, i, 0, False), buf)


def _pairs(step, cur, buf, theta, adv, slot, npairs):
    """Advance adv[j] steps and store in slot[j], for j < npairs."""
    def body(j, cb):
        c, b = cb
        c = _advance(step, c, theta, adv[j])
        return c, _store(b, c, slot[j])
    return jax.lax.fori_loop(0, _opaque(npairs), body, (cur, buf))


def revolve_rollout(step, n_steps, snapshots, aux=False):
    """Return rollout(theta, carry) = `step` applied `n_steps` times, reversed with `snapshots` states.

    aux=True: rollout(theta, carry, aux) of step(carry, theta, aux); aux is not differentiated.
    """
    n = int(n_steps)
    s = max(1, min(int(snapshots), n))       # more slots than steps buy nothing
    sch = schedule(n, s)
    step3 = step if aux else (lambda c, th, _a: step(c, th))

    def plain(theta, carry, a):
        return jax.lax.fori_loop(0, _opaque(n), lambda _, c: step3(c, theta, a), carry)

    @jax.custom_vjp
    def _rollout(theta, carry, a):
        return plain(theta, carry, a)

    # Always jitted: an eager fori_loop would expose the trip count `_opaque` hides.
    _jitted = jax.jit(lambda theta, carry, a: _rollout(
        theta, jax.tree.map(jnp.asarray, carry), a))

    def rollout(theta, carry, a=None):
        return _jitted(theta, carry, a)

    def fwd(theta, carry, a):
        buf = jax.tree.map(
            lambda x: jnp.broadcast_to(x[None], (s + 1,) + jnp.shape(x)), carry)
        cur, buf = _pairs(lambda c, th: step3(c, th, a), carry, buf, theta,
                          jnp.asarray(sch["sweep_adv"]), jnp.asarray(sch["sweep_slot"]),
                          len(sch["sweep_adv"]))
        return step3(cur, theta, a), (buf, theta, a)

    def bwd(res, ct_out):
        buf, theta, a = res
        step = lambda c, th: step3(c, th, a)          # noqa: E731
        leaves, tdef = jax.tree.flatten(ct_out)
        fmask = [x.dtype != jax.dtypes.float0 for x in leaves]   # int leaves: float0
        ct_f = [x for x, f in zip(leaves, fmask) if f]

        def vjp_one(cur, ct_f, theta):
            cl, _ = jax.tree.flatten(cur)
            other = [x for x, f in zip(cl, fmask) if not f]

            def f(fl, th):
                it_f, it_o = iter(fl), iter(other)
                st = jax.tree.unflatten(tdef, [next(it_f) if m else next(it_o)
                                               for m in fmask])
                out = jax.tree.flatten(step(st, th))[0]
                return [x for x, m in zip(out, fmask) if m]

            fl = [x for x, m in zip(cl, fmask) if m]
            _, pull = jax.vjp(f, fl, theta)
            return pull(ct_f)

        rows = tuple(jnp.asarray(sch[k]) for k in ("restore", "npairs", "adv", "slot"))

        def group(g, c):
            buf, ct_f, ct_th = c
            restore, npairs, adv, slot = (r[g] for r in rows)
            cur = _load(buf, restore)
            cur, buf = _pairs(step, cur, buf, theta, adv, slot, npairs)
            ct_f, dth = vjp_one(cur, ct_f, theta)
            ct_th = jax.tree.map(jnp.add, ct_th, dth)
            return buf, ct_f, ct_th

        ct_th0 = jax.tree.map(jnp.zeros_like, theta)
        _, ct_f, ct_th = jax.lax.fori_loop(0, _opaque(n), group, (buf, ct_f, ct_th0))
        it_f = iter(ct_f)
        ct_carry = [next(it_f) if m else np.zeros(np.shape(x), jax.dtypes.float0)
                    for x, m in zip(leaves, fmask)]
        ct_a = jax.tree.map(
            lambda x: (jnp.zeros_like(x) if jnp.issubdtype(x.dtype, jnp.inexact)
                       else np.zeros(np.shape(x), jax.dtypes.float0)), a)
        return ct_th, jax.tree.unflatten(tdef, ct_carry), ct_a

    _rollout.defvjp(fwd, bwd)
    rollout.schedule = sch
    rollout.snapshots = s
    return rollout
