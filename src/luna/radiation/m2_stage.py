"""M2 state for the radiation stage (`rad_closure`): groups (G, 10, N, N, N), m0 = E, c m1 = F.

See docs/RADIATION.md ("M2 as a stage").
"""
import functools
import os

import jax.numpy as jnp
import numpy as np

CLOSURES = ("m1", "m2_learned", "m2_entropy")
ENTROPY_CHUNK = 32768       # states per entropy-solve block: 15.4 GiB peak at 128^3 on an A100
PARAMS_FILE = os.path.join(os.path.dirname(__file__), "m2_closure_w32.npz")


def load_params(path=""):
    """Return [(W1, b1), (W2, b2), (W3, b3)] from an .npz; "" loads the shipped network."""
    z = np.load(path or PARAMS_FILE)
    keys = ["W1", "b1", "W2", "b2", "W3", "b3"]
    if not all(k in z.files for k in keys):
        keys = [f"arr_{k}" for k in range(6)]
    a = [jnp.asarray(z[k], jnp.float64) for k in keys]
    return [(a[0], a[1]), (a[2], a[3]), (a[4], a[5])]


def is_m2(R):
    """Return whether R is an M2 stage state (G, 10, N, N, N)."""
    return R.ndim == 5 and R.shape[1] == 10


def isotropic(E):
    """Return the ten moments of isotropic radiation with energy density E."""
    z = jnp.zeros_like(E)
    t = E / 3.0
    return jnp.stack([E, z, z, z, t, t, t, z, z, z], axis=-4)


def add_isotropic(R, dE):
    """Add dE (G, N...) to m0 and dE/3 to each m2_ii of R (G, 10, N...)."""
    return R.at[:, 0].add(dE).at[:, 4:7].add((dE / 3.0)[:, None])


def transport(R, dt, dx, *, chat, boundary, pad, closure, params=None, kernel="auto"):
    """One split transport step of every group of R (G, 10, N, N, N)."""
    from luna.radiation import m2_3d, m2_split
    closure_fn = None
    if closure == "m2_entropy":
        # in blocks: whole 128^3 faces at once asked XLA for 110-163 GiB (CODE_NOTES.md, 2026-10-11)
        closure_fn = functools.partial(m2_3d.third_moment, chunk=ENTROPY_CHUNK)
    elif closure != "m2_learned":
        raise ValueError(f"rad_closure={closure!r}: the M2 closures are "
                         "'m2_learned' and 'm2_entropy'")
    return jnp.stack([m2_split.split_step(R[g], dt, dx, chat=chat, boundary=boundary,
                                          closure_fn=closure_fn, params=params, pad=pad,
                                          kernel=kernel)
                      for g in range(R.shape[0])])


def damp(R, dt, *, chat, kappa):
    """Damp m1 and trace-free m2 by exp(-c kappa dt); m0 and trace(m2) untouched."""
    from luna.radiation.m1 import _as_group_field
    f = jnp.exp(-chat * _as_group_field(kappa, R) * dt)
    tr3 = (R[:, 4] + R[:, 5] + R[:, 6]) / 3.0
    diag = tr3[:, None] + f[:, None] * (R[:, 4:7] - tr3[:, None])
    return (R.at[:, 1:4].multiply(f[:, None]).at[:, 4:7].set(diag)
            .at[:, 7:10].multiply(f[:, None]))
