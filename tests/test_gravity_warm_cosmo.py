"""The comoving warm-started solve (gas + CIC-deposited dark matter, mixed precision) against a converged solve."""
import os

import jax.numpy as jnp
import numpy as np
import pytest

from tests import _gravity_warm as gw
from tests.test_gravity_warm import N_CPU, N_GPU


def zeldovich(N, a_steps, amp=0.12, k_max=4, seed=2):
    """(gas U, DM positions) per scale factor: a Zel'dovich field growing as D = a."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(N, 1.0 / N)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    dk = np.fft.fftn(rng.standard_normal((N, N, N))) * (K2 <= k_max ** 2) * (K2 > 0)
    delta0 = np.real(np.fft.ifftn(dk))
    delta0 *= amp / np.abs(delta0).max()
    phik = -np.fft.fftn(delta0) / np.where(K2 > 0, (2 * np.pi) ** 2 * K2, 1.0)
    disp = [-np.real(np.fft.ifftn(2j * np.pi * Kc * phik)) for Kc in (KX, KY, KZ)]
    q = (np.arange(N) + 0.5) / N
    Q = np.stack(np.meshgrid(q, q, q, indexing="ij"))
    out = []
    for a in a_steps:
        rho = np.clip(1.0 + a * delta0 / 0.1, 0.05, None)
        U = np.stack([rho, 0 * rho, 0 * rho, 0 * rho, rho])
        pos = np.mod(Q + (a / 0.1) * np.stack(disp), 1.0).reshape(3, -1).T
        out.append((U, pos))
    return out


def run_cosmo(N, broken=None):
    cfg = gw.config(N, 0, CosmologicalIntegration=True)
    be = gw.backend(cfg, comoving=True)
    a_steps = np.exp(np.linspace(np.log(0.1), np.log(0.6), 24))
    fields = zeldovich(N, a_steps)
    n_p = N ** 3
    mass = jnp.full((n_p,), 1.0 / n_p)
    active = jnp.ones((n_p,))
    deposit = be.make_deposit(1.0)
    steps = []
    for U, pos in fields:
        rho_dm = deposit(jnp.asarray(pos), mass, active)
        src, _ = be.source(jnp.asarray(U), rho_dm, None, None, None)
        steps.append((np.asarray(src, np.float64), float(np.log(a_steps[1] / a_steps[0]))))
    return gw.run(be, steps, 0, broken=broken)


def check(N):
    gw.assert_tracks("cosmo", N, run_cosmo(N), run_cosmo(N, broken=("skip", 12)))


def test_comoving_warm_solve_tracks_the_converged_solve():
    check(N_CPU)


@pytest.mark.gpu
def test_comoving_warm_solve_tracks_the_converged_solve_large():
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        pytest.skip("the large case runs on a GPU")
    check(N_GPU)
