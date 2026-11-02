"""What the gas a sink swallowed was doing.

A sink->star prescription seeds the star from the kinematics of the gas that
built the sink -- Torch's `make_stars_from_sinks` wants a mean velocity, a
dispersion, a sound speed and a specific angular momentum. Measuring the gas
that happens to be near the sink NOW is a different quantity: the sink may
have swallowed the interesting gas a free-fall time ago.

So `Sinks._accrete` accumulates the sums as it takes the gas, and the tests
below are built around flows whose answer is known in advance, because a
running sum is exactly the kind of thing that looks plausible while being
wrong by a factor of the timestep.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax.numpy as jnp                                       # noqa: E402

from luna.species.sinks import Sinks, accreted_gas_moments     # noqa: E402

pytestmark = pytest.mark.fast

N = 16
BOX = 1.0
GAMMA = 5.0 / 3.0


def _sinks(n_max=2, r_acc=0.25, rho_sink=0.25):
    """`creation_density` is ALSO the accretion threshold: `_accrete` is
    SEREN-style and takes only the gas above rho_sink, leaving the cell at it.
    Setting it to infinity to switch creation off therefore switches accretion
    off too -- which is how these tests first failed. It is kept below the gas
    density here; creation cannot fire anyway because the tests call
    `_accrete` directly, not `_accrete_and_create`."""
    return Sinks(N3=(N, N, N), box3=(BOX, BOX, BOX), G=1.0, boundary=1,
                 n_max=n_max, softening=0.0, accretion_radius=r_acc,
                 creation_density=rho_sink, creation_separation=0.0,
                 create_frac=0.0, gamma=GAMMA, is_main=False)


def _uniform(rho=1.0, vx=0.0, vy=0.0, vz=0.0, P=1.0):
    ones = jnp.ones((N, N, N))
    ke = 0.5 * rho * (vx ** 2 + vy ** 2 + vz ** 2)
    return jnp.stack([rho * ones, rho * vx * ones, rho * vy * ones,
                      rho * vz * ones, (P / (GAMMA - 1.0) + ke) * ones])


def _one_sink(pos=(0.5, 0.5, 0.5)):
    p = jnp.zeros((2, 3)).at[0].set(jnp.asarray(pos))
    return p, jnp.zeros((2, 3)), jnp.zeros(2), jnp.zeros(2).at[0].set(1.0)


def test_the_accreted_mass_is_the_mass_that_left_the_gas():
    """macc must equal what the grid lost. If the sum and the sink disagree,
    one of them is wrong and mass is being invented or dropped."""
    sk = _sinks()
    U = _uniform(rho=2.0)
    pos, vel, mass, active = _one_sink()
    mom = sk.zero_moments()
    before = float(jnp.sum(U[0])) * sk.dV

    U2, pos, vel, mass2, active, mom = sk._accrete(U, pos, vel, mass, active, mom)

    lost = before - float(jnp.sum(U2[0])) * sk.dV
    assert lost > 0.0
    assert abs(float(mom["macc"][0]) - lost) < 1e-10
    assert abs(float(mass2[0]) - lost) < 1e-10


def test_the_mean_velocity_is_the_flow_it_ate():
    """Uniform flow in: the mean accreted velocity is the flow's, and the
    dispersion about it is zero."""
    sk = _sinks()
    U = _uniform(rho=1.0, vx=0.3, vy=-0.2)
    pos, vel, mass, active = _one_sink()
    out = sk._accrete(U, pos, vel, mass, active, sk.zero_moments())
    mom = out[5]

    m = accreted_gas_moments((out[1], out[2], out[3], out[4], mom))
    np.testing.assert_allclose(m["mean_velocity"][0], [0.3, -0.2, 0.0],
                               rtol=0, atol=1e-10)
    # sigma^2 = <v^2> - <v>^2, and a uniform flow has none
    np.testing.assert_allclose(m["var_velocity"][0], [0.0, 0.0, 0.0],
                               rtol=0, atol=1e-10)


def test_the_sound_speed_is_the_gas_it_ate():
    """cs = sqrt(P / rho) -- the isothermal sound speed Torch seeds its stars'
    velocity dispersion with -- known exactly for a uniform state."""
    rho, P = 2.0, 3.0
    sk = _sinks()
    U = _uniform(rho=rho, P=P)
    pos, vel, mass, active = _one_sink()
    out = sk._accrete(U, pos, vel, mass, active, sk.zero_moments())
    m = accreted_gas_moments((out[1], out[2], out[3], out[4], out[5]))
    assert abs(float(m["mean_cs"][0]) - np.sqrt(P / rho)) < 1e-10


def test_solid_body_rotation_gives_the_angular_momentum_it_should():
    """v = omega x r about the sink: the specific angular momentum is along
    omega and positive. A sign or index slip here would seed every star's
    spin backwards, which nothing downstream would flag."""
    sk = _sinks()
    omega = 0.7
    c = 0.5
    ax = (jnp.arange(N) + 0.5) * (BOX / N)
    X, Y, Z = jnp.meshgrid(ax, ax, ax, indexing="ij")
    vx, vy = -omega * (Y - c), omega * (X - c)
    rho = jnp.ones((N, N, N))
    ke = 0.5 * rho * (vx ** 2 + vy ** 2)
    U = jnp.stack([rho, rho * vx, rho * vy, jnp.zeros_like(rho),
                   1.0 / (GAMMA - 1.0) + ke])

    pos, vel, mass, active = _one_sink((c, c, c))
    out = sk._accrete(U, pos, vel, mass, active, sk.zero_moments())
    m = accreted_gas_moments((out[1], out[2], out[3], out[4], out[5]))
    L = m["angular_momentum"][0]
    assert L[2] > 0.0                                  # along +z, with omega
    assert abs(L[0]) < 1e-10 and abs(L[1]) < 1e-10


def test_nothing_accreted_leaves_the_moments_finite():
    """An empty sink divides by its accreted mass. It must give zeros, not
    NaN -- a NaN here would poison every star it later spawns."""
    sk = _sinks()
    pos, vel, mass, active = _one_sink((0.5, 0.5, 0.5))
    m = accreted_gas_moments((pos, vel, mass, active, sk.zero_moments()))
    for k in ("mean_cs", "mean_velocity", "var_velocity", "angular_momentum"):
        assert np.all(np.isfinite(m[k])), k


def test_the_sums_accumulate_across_steps():
    """They are RUNNING sums: two accretion steps must give more mass than
    one, and the same answer as one step of twice the gas would not."""
    sk = _sinks()
    pos, vel, mass, active = _one_sink()
    mom = sk.zero_moments()
    total = 0.0
    for _ in range(2):
        U = _uniform(rho=1.0)                     # fresh gas each time
        out = sk._accrete(U, pos, vel, mass, active, mom)
        _, pos, vel, mass, active, mom = out
        total = float(mom["macc"][0])
    one = _sinks()
    o1 = one._accrete(_uniform(rho=1.0), *_one_sink(), one.zero_moments())
    assert total > float(o1[5]["macc"][0]) * 1.5
