"""Train the input-convex entropy closure, then say how accurate it is and
what it costs against M1 and against the exact M2 solve.

    python -m scripts.validation.runs.m2_learned_train
    python -m scripts.validation.runs.m2_learned_train --steps 4000 --loop 300

Three stages, each measured:

  pointwise    Sobolev training against the exact entropy solve: match the
               convex dual and its gradient on sampled realizable states
  in the loop  fine-tune against a whole M2 rollout, differentiating through
               the transport -- the closure then answers for the trajectory
               it produces and not for a pointwise error
  report       closure error over the realizable set, rollout error against
               the exact closure, and the per-step cost of M1, exact M2 and
               learned M2 on the same grid

Everything is slab geometry, where the exact answer exists to check against.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np


import jax                                                    # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp                                       # noqa: E402

from luna.radiation import m2, m2_learned                     # noqa: E402
from luna.sources.ml_corrector import adam_init, adam_step    # noqa: E402

CHAT = 1.0


def exact_targets(fg):
    """(m3*, h*, a*) from the entropy solve: the closure itself and the two
    Sobolev targets."""
    ab = jax.vmap(m2.solve_multipliers)(fg)
    logm0 = jax.vmap(lambda a: m2.normalised_moments(a)[1])(ab)
    m3 = jax.vmap(lambda a: m2.normalised_moments(a)[0][2])(ab)
    return m3, jnp.sum(ab * fg, axis=1) - logm0, ab


def pointwise_loss(params, fg, m3_star, h_star, a_star, w=0.01):
    """What is reported is the closure error, so that is what is trained,
    with a light pull towards the true dual and multipliers to keep the
    network near the function it is supposed to be.

    The Sobolev terms are scaled: near the edge of the realizable set the
    multipliers run to tens, and an unscaled gradient term spends the whole
    capacity of the network there while the closure -- which saturates --
    barely moves. Measured with the unscaled version: closure rms 2.0e-1.
    """
    m3 = jax.vmap(lambda p: m2_learned.closure(params, p))(fg)
    h = jax.vmap(lambda p: m2_learned.potential(params, p))(fg)
    a = jax.vmap(lambda p: m2_learned.multipliers(params, p))(fg)
    return (jnp.mean((m3 - m3_star) ** 2)
            + w * jnp.mean((h - h_star) ** 2 / (1.0 + h_star ** 2))
            + w * jnp.mean(jnp.sum((a - a_star) ** 2, axis=1)
                           / (1.0 + jnp.sum(a_star ** 2, axis=1))))


def train_pointwise(params, key, steps, batch, lr, resample=250):
    """Adam with a fresh batch every `resample` steps and a step decay."""
    state = adam_init(params)
    lg = jax.jit(jax.value_and_grad(pointwise_loss), static_argnums=())
    first, loss = None, jnp.inf
    fg = m3_star = h_star = a_star = None
    for i in range(steps):
        if i % resample == 0:
            key, k = jax.random.split(key)
            fg = m2_learned.sample_states(k, batch)
            m3_star, h_star, a_star = exact_targets(fg)
        step_lr = lr * (0.1 if i > 0.8 * steps else 0.3 if i > 0.5 * steps else 1.0)
        loss, g = lg(params, fg, m3_star, h_star, a_star)
        params, state = adam_step(g, params, state, step_lr)
        first = float(loss) if first is None else first
        if (i + 1) % max(1, steps // 8) == 0:
            print(f"  pointwise {i + 1:6d}  loss {float(loss):.4e}", flush=True)
    return params, first, float(loss)


def _rollout(U, dt, dx, steps, closure_fn):
    def body(U, _):
        return m2.step(U, dt, dx, chat=CHAT, boundary=1, order=2,
                       closure_fn=closure_fn), None
    return jax.lax.scan(body, U, None, length=steps)[0]


def _initial_states(n_cells, n_cases=3):
    """A beam, two beams, and something in between."""
    x = (np.arange(n_cells) + 0.5) / n_cells - 0.5
    e = np.exp(-0.5 * (x / 0.07) ** 2)
    return jnp.asarray(np.stack([
        np.stack([e, 0.8 * e, 0.75 * e]),           # a beam
        np.stack([e, 0.0 * e, 0.95 * e]),           # two beams
        np.stack([e, 0.2 * e, 0.4 * e]),            # in between
    ][:n_cases]))


def train_in_the_loop(params, steps, lr, n_cells=96, n_steps=12):
    """Fine-tune on trajectories: the reference is the exact-closure rollout
    and the gradient goes through the transport."""
    U0 = _initial_states(n_cells)
    dx = 1.0 / n_cells
    dt = 0.4 * dx / CHAT
    ref = jax.jit(lambda u: jax.vmap(
        lambda v: _rollout(v, dt, dx, n_steps, None))(u))(U0)

    def loss(p):
        fn = m2_learned.closure_fn(p)
        out = jax.vmap(lambda u: _rollout(u, dt, dx, n_steps, fn))(U0)
        return jnp.mean((out - ref) ** 2) / jnp.mean(ref ** 2)

    state = adam_init(params)
    lg = jax.jit(jax.value_and_grad(loss))
    first = None
    for i in range(steps):
        v, g = lg(params)
        params, state = adam_step(g, params, state, lr)
        first = float(v) if first is None else first
        if (i + 1) % max(1, steps // 6) == 0:
            print(f"  in-the-loop {i + 1:5d}  relative rollout mse "
                  f"{float(v):.4e}", flush=True)
    return params, first, float(v)


def closure_error(params, key, n=4000, margin=0.02):
    """Max and RMS error of m3 over the realizable set, against the exact
    entropy solve."""
    fg = m2_learned.sample_states(key, n, margin=margin)
    exact = jax.vmap(m2.closure)(fg)
    learned = jax.vmap(lambda p: m2_learned.closure(params, p))(fg)
    d = np.asarray(learned - exact)
    return float(np.max(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))


def rollout_error(params, n_cells=96, n_steps=40):
    U0 = _initial_states(n_cells)
    dx = 1.0 / n_cells
    dt = 0.4 * dx / CHAT
    ref = jax.jit(lambda u: jax.vmap(lambda v: _rollout(v, dt, dx, n_steps,
                                                        None))(u))(U0)
    fn = m2_learned.closure_fn(params)
    got = jax.jit(lambda u: jax.vmap(lambda v: _rollout(v, dt, dx, n_steps,
                                                        fn))(u))(U0)
    d = np.asarray(got - ref)
    return float(np.max(np.abs(d)) / np.max(np.abs(np.asarray(ref)))), \
        float(np.sqrt(np.mean(d ** 2) / np.mean(np.asarray(ref) ** 2)))


def timings(params, n_cells=4096, reps=5):
    """Per-step wall time of the three schemes on the same grid."""
    from luna.radiation.m1 import m1_step
    x = (np.arange(n_cells) + 0.5) / n_cells - 0.5
    e = np.exp(-0.5 * (x / 0.07) ** 2)
    dx = 1.0 / n_cells
    dt = 0.4 * dx / CHAT
    U2 = jnp.asarray(np.stack([e, 0.3 * e, 0.5 * e]))
    R1 = jnp.asarray(np.stack([e, 0.3 * e]))
    fns = {
        "M1": (jax.jit(lambda r: m1_step(r, dt, dx, chat=CHAT, boundary=1,
                                         order=2)), R1),
        "M2 exact": (jax.jit(lambda u: m2.step(u, dt, dx, chat=CHAT,
                                               boundary=1, order=2)), U2),
        "M2 learned": (jax.jit(lambda u: m2.step(
            u, dt, dx, chat=CHAT, boundary=1, order=2,
            closure_fn=m2_learned.closure_fn(params))), U2),
    }
    out = {}
    for name, (f, arg) in fns.items():
        jax.block_until_ready(f(arg))
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            jax.block_until_ready(f(arg))
            ts.append(time.perf_counter() - t0)
        out[name] = statistics.median(ts)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--loop", type=int, default=200)
    ap.add_argument("--loop-lr", type=float, default=3e-4)
    ap.add_argument("--cells", type=int, default=4096)
    ap.add_argument("--out", default="m2_learned")
    ap.add_argument("--load", default="",
                    help="skip training, load these parameters and report")
    ap.add_argument("--skip-accuracy", action="store_true",
                    help="timings only: the accuracy is backend independent "
                         "and the exact solve over a large sample is one of "
                         "the programs that crashes on ROCm")
    a = ap.parse_args(argv)

    key = jax.random.PRNGKey(0)
    key, k_init, k_data, k_err1, k_err2 = jax.random.split(key, 5)
    params = m2_learned.init(k_init)

    if a.load:
        # Report only: accuracy and cost of parameters trained elsewhere.
        # Training needs a backend that survives it -- see docs/RADIATION.md
        # on the ROCm segfault -- while the measurement wants the GPU.
        z = np.load(a.load)
        flat = [jnp.asarray(z[k]) for k in sorted(z.files,
                                                  key=lambda n: int(n[4:]))]
        params = jax.tree.unflatten(jax.tree.structure(params), flat)
        e_max = e_rms = r_max = r_rms = float("nan")
        if not a.skip_accuracy:
            e_max, e_rms = closure_error(params, k_err1)
            r_max, r_rms = rollout_error(params)
            print(f"closure error  max {e_max:.3e}  rms {e_rms:.3e}")
            print(f"rollout error  max {r_max:.3e}  rms {r_rms:.3e}")
        ts = timings(params, a.cells)
        print(f"\nper step at {a.cells} cells:")
        for k, v in ts.items():
            print(f"  {k:11s} {v * 1e3:8.3f} ms   x M1 = {v / ts['M1']:6.2f}")
        with open(a.load.replace(".npz", "") + "_report.json", "w") as f:
            json.dump({"closure_error": [e_max, e_rms],
                       "rollout_error": [r_max, r_rms],
                       "seconds_per_step": ts, "cells": a.cells}, f, indent=1)
        return 0

    t0 = time.perf_counter()
    params, l0, l1 = train_pointwise(params, k_data, a.steps, a.batch, a.lr)
    t_point = time.perf_counter() - t0
    e_max, e_rms = closure_error(params, k_err1)
    r_max, r_rms = rollout_error(params)
    print(f"after pointwise: closure max {e_max:.3e} rms {e_rms:.3e}, "
          f"rollout max {r_max:.3e} rms {r_rms:.3e}", flush=True)

    t0 = time.perf_counter()
    params, s0, s1 = train_in_the_loop(params, a.loop, a.loop_lr)
    t_loop = time.perf_counter() - t0
    e_max2, e_rms2 = closure_error(params, k_err2)
    r_max2, r_rms2 = rollout_error(params)
    print(f"after in-the-loop: closure max {e_max2:.3e} rms {e_rms2:.3e}, "
          f"rollout max {r_max2:.3e} rms {r_rms2:.3e}", flush=True)

    ts = timings(params, a.cells)
    print(f"\nper step at {a.cells} cells:")
    for k, v in ts.items():
        print(f"  {k:11s} {v * 1e3:8.3f} ms   x M1 = {v / ts['M1']:6.2f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    flat, tree = jax.tree.flatten(params)
    np.savez(a.out + ".npz", *[np.asarray(x) for x in flat])
    with open(a.out + ".json", "w") as f:
        json.dump({"pointwise": {"loss_first": l0, "loss_last": l1,
                                 "seconds": t_point, "steps": a.steps},
                   "in_the_loop": {"first": s0, "last": s1,
                                   "seconds": t_loop, "steps": a.loop},
                   "closure_error": {"after_pointwise": [e_max, e_rms],
                                     "after_loop": [e_max2, e_rms2]},
                   "rollout_error": {"after_pointwise": [r_max, r_rms],
                                     "after_loop": [r_max2, r_rms2]},
                   "seconds_per_step": ts, "cells": a.cells,
                   "widths": list(m2_learned.WIDTHS)}, f, indent=1)
    print(f"\nparameters -> {a.out}.npz, report -> {a.out}.json "
          f"({len(tree.flatten_up_to(params))} leaves)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
