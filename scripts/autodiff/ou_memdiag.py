"""Compile-only memory diagnosis for the OU inverse gradient (#40).

The 128^3 gradient OOMed with a single 53.4 GiB temp allocation despite
chunked remat. Instead of guessing, ask XLA: lower and compile the
loss-gradient for each (Nx, chunk_len) WITHOUT running it, and print the
compiler's own temp/argument/output memory budget. The Nx scaling tells us
whether the blob is per-step trajectory storage (scales with n_steps x Nx^3)
or something else; the chunk_len scaling tells us whether remat is engaged
at all.
"""
from __future__ import annotations

import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, "/raven/u/uli/jax-hydro")
jax.config.update("jax_enable_x64", True)

from scripts.autodiff.ou_inverse import _make_simulate
from hydro.turbulence import OUDriver, OUDriverConfig


def analyze(Nx, chunk_len, n_steps=250):
    L, gamma = 1.0, 5.0 / 3.0
    rho_0, p_0 = 1.0, 1.0
    cs = float(np.sqrt(gamma * p_0 / rho_0))
    dt = 0.3 * (L / Nx) / (cs + 2.0)

    ou_cfg = OUDriverConfig(Lx=L, Ly=L, Lz=L, k_min=1.0, k_max=4.0,
                             rms_acceleration=2.0, autocorrelation_time=0.5,
                             solenoidal_weight=1.0)
    ou = OUDriver(cfg=ou_cfg)
    ou_a0 = ou.init_state(jax.random.PRNGKey(0))
    ou_keys = jax.random.split(jax.random.PRNGKey(1), n_steps)
    coord = (jnp.arange(Nx, dtype=jnp.float64) + 0.5) * (L / Nx)
    X, Y, Z = jnp.meshgrid(coord, coord, coord, indexing="ij")

    simulate = _make_simulate(Nx, L, gamma, rho_0, p_0, n_steps, dt,
                               order=2, limiter=0, ou_driver=ou,
                               X=X, Y=Y, Z=Z, ou_keys=ou_keys,
                               chunk_len=chunk_len)

    def loss(dr, U_obs):
        return jnp.mean((simulate(dr, ou_a0) - U_obs) ** 2)

    g = jax.grad(loss)
    dr_spec = jax.ShapeDtypeStruct((Nx, Nx, Nx), jnp.float64)
    obs_spec = jax.ShapeDtypeStruct((5, Nx, Nx, Nx), jnp.float64)
    compiled = jax.jit(g).lower(dr_spec, obs_spec).compile()
    ma = compiled.memory_analysis()
    GiB = 1024 ** 3
    print(f"  Nx={Nx:4d} chunk={chunk_len:3d}  "
          f"temp={ma.temp_size_in_bytes / GiB:8.2f} GiB  "
          f"args={ma.argument_size_in_bytes / GiB:6.2f} GiB  "
          f"out={ma.output_size_in_bytes / GiB:6.2f} GiB", flush=True)


def main():
    print("XLA memory budget for grad(loss), n_steps=250, f64")
    for Nx in (64, 96, 128):
        for chunk in (10, 25, 50):
            try:
                analyze(Nx, chunk)
            except Exception as e:
                print(f"  Nx={Nx:4d} chunk={chunk:3d}  FAILED: {type(e).__name__}: {e}",
                      flush=True)
    print("done.")


if __name__ == "__main__":
    main()
