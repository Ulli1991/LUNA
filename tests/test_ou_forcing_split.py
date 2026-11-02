"""The OU driver Strang-splits with ONE draw per step.

cdd5c3b left the forcing unsplit with the reason "two half draws is not one
draw halved". Nobody proposed drawing twice. setup_ou_forcing separates
`evolve` (advance the state, one draw) from `kick` (apply the force over a
time), so the split evolves once per step and applies the impulse in two
halves, one either side of the hydro.

That does change the realised forcing statistics, and by a KNOWN amount. The
step applies h*F(a_n) + h*F(a_n+1) instead of dt*F(a_n+1), and F is linear in
a, so the effective acceleration is the average of two OU samples one step
apart. For a process with correlation rho = exp(-dt/T_ac) between them,

    rms(split) / rms(endpoint) = sqrt((1 + rho) / 2)

which tends to 1 as dt/T_ac -> 0. This gate holds the code to that number
rather than to a hand-waved "small". If anyone ever does make it draw twice,
rho becomes 0 and the ratio drops to 0.707, far outside the window.

GPU test.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp



N, L, T_AC, RMS_A = 16, 1.0, 1.0, 1.0
N_STEPS = 400


def _driver(dt_unused=None):
    from dataclasses import replace

    from luna.hydro.turbulence import OUDriverConfig
    from luna.sources.ou_forcing import setup_ou_forcing
    from luna.template_config import Config

    from jax.sharding import Mesh, PartitionSpec as P
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))

    class _Sh:
        pass

    sh = _Sh()
    sh.mesh = mesh
    sh.spec = P(None, "x", "y", "z")
    cfg = replace(Config(), SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N,
                  Lx=L, Ly=L, Lz=L, ou_seed=7,
                  ou_config=OUDriverConfig(k_min=1.0, k_max=2.0,
                                           autocorrelation_time=T_AC,
                                           rms_acceleration=RMS_A,
                                           Lx=L, Ly=L, Lz=L))
    return setup_ou_forcing(cfg, sh, is_main_process=False), mesh


def _force_rms(ou, mesh, a):
    """|F| rms over the box, read off a unit kick on unit-density gas."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    U0 = jnp.zeros((5, N, N, N)).at[0].set(1.0).at[4].set(1.0)
    U0 = jax.device_put(U0, NamedSharding(mesh, P(None, "x", "y", "z")))
    U1 = ou["kick"](U0, a, 1.0)            # dm = rho * F * 1
    F2 = sum(np.asarray(U1[i] - U0[i]) ** 2 for i in (1, 2, 3))
    return float(np.sqrt(np.mean(F2)))


@pytest.mark.parametrize("dt", [0.05, 0.2])
def test_split_forcing_rms_matches_the_trapezoid_prediction(dt):
    ou, mesh = _driver()
    a, key = ou["a"], ou["key"]
    rms_end, rms_split = [], []
    for _ in range(N_STEPS):
        a_prev = a
        a, key = ou["evolve"](a, key, dt)
        rms_end.append(_force_rms(ou, mesh, a))              # dt * F(a_n+1)
        rms_split.append(_force_rms(ou, mesh, 0.5 * (a_prev + a)))
    # discard the burn-in while the OU state reaches its stationary amplitude
    keep = N_STEPS // 4
    r_end = float(np.mean(rms_end[keep:]))
    r_split = float(np.mean(rms_split[keep:]))
    rho = float(np.exp(-dt / T_AC))
    predicted = float(np.sqrt(0.5 * (1.0 + rho)))
    measured = r_split / r_end
    print(f"[ou dt={dt}] rms endpoint={r_end:.4f} split={r_split:.4f} "
          f"ratio={measured:.4f} predicted={predicted:.4f} "
          f"(shift {100*(measured-1):.2f}%)")
    assert abs(measured / predicted - 1.0) < 0.02, (
        f"dt={dt}: rms ratio {measured:.4f} vs trapezoid prediction "
        f"{predicted:.4f} -- the split is not one draw applied in two halves")


def test_the_split_still_draws_once_per_step():
    """Same seed, same number of evolve() calls -> the OU trajectory is
    UNCHANGED by the split. Only the impulse is halved."""
    ou, _ = _driver()
    dt = 0.1
    a, key = ou["a"], ou["key"]
    traj = []
    for _ in range(20):
        a, key = ou["evolve"](a, key, dt)
        traj.append(np.asarray(a).copy())
    ou2, _ = _driver()
    a2, key2 = ou2["a"], ou2["key"]
    for i in range(20):
        a2, key2 = ou2["evolve"](a2, key2, dt)
        assert np.allclose(np.asarray(a2), traj[i]), f"OU trajectory diverged at {i}"
