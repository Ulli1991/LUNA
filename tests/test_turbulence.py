"""OU turbulent driver normalisation (hydro/turbulence.py).

Audit item 11: the steady-state <|F(x)|^2>_x must equal rms_acceleration^2. The
amplitudes are a_k = sigma_k P_k z_k, so <|a_k|^2> = sigma_k^2 tr(P_k^2); the
per-draw spatial mean square is 2 sum_k |a_k|^2 (Parseval on the periodic grid,
no k=0 mode), whose expectation is 2 sum_k sigma_k^2 tr(P_k^2). Averaging over
draws must recover rms^2. Pre-fix (normalised with tr(P_k)) this was
tr(P^2)/tr(P) = 0.5 rms^2 at zeta = 0.5. zeta in {0, 1} have tr(P)=tr(P^2), so
they pass either way and anchor the test.
"""
import numpy as np
import pytest

import jax


pytestmark = pytest.mark.fast

from luna.hydro.turbulence import OUDriver, OUDriverConfig


@pytest.mark.parametrize("zeta", [0.0, 0.5, 1.0])
def test_realized_rms_matches_target(zeta):
    rms = 2.5
    cfg = OUDriverConfig(k_min=1.0, k_max=3.0, solenoidal_weight=zeta,
                         rms_acceleration=rms, seed=0)
    drv = OUDriver(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 400)
    # per-draw <|F|^2>_x = 2 sum_k |a_k|^2 (Parseval); average over draws.
    msq = np.mean([2.0 * float(np.sum(np.abs(np.asarray(drv.init_state(k))) ** 2))
                   for k in keys])
    assert abs(msq / rms**2 - 1.0) < 0.05, f"zeta={zeta}: <|F|^2>={msq:.4f} vs rms^2={rms**2:.4f}"


def test_forcing_is_computed_in_the_state_precision():
    """In a differentiated FP64 run the OU force is the float64 sum
    2 Re sum_k a_k e^{ik.x}. Wave vectors or cell coordinates in FP32 put
    ~1e-7 into it, which made a sharded gradient differ from the 1-device
    one (docs/AUTODIFF.md, "Across nodes"). Plain runs keep FP32: cheaper."""
    from dataclasses import replace
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.sources.ou_forcing import setup_ou_forcing
    from luna.template_config import Config
    from luna.hydro.turbulence import _enumerate_modes
    n, L = 16, 1.0
    ou = OUDriverConfig(k_min=1.0, k_max=3.0, solenoidal_weight=0.5,
                        rms_acceleration=1.0, seed=0, Lx=L, Ly=L, Lz=L)
    cfg = replace(Config(), SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n,
                  Lx=L, Ly=L, Lz=L, ou_config=ou, autodiff_mode=True)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1, 1), ("x", "y", "z"))
    drive = setup_ou_forcing(cfg, NamedSharding(mesh, P(None, "x", "y", "z")), False)
    U = jax.numpy.zeros((5, n, n, n)).at[0].set(1.0)
    F = np.asarray(drive["kick"](U, drive["a"], 1.0))[1:4]     # rho = h = 1: dm = F

    _, k, _, _ = _enumerate_modes(ou)                             # float64
    a = np.asarray(drive["a"]).astype(np.complex128)
    x = (np.arange(n) + 0.5) * (L / n)
    X = np.stack(np.meshgrid(x, x, x, indexing="ij"))
    phase = np.exp(1j * np.einsum("mi,ixyz->mxyz", k, X))
    ref = 2.0 * np.real(np.einsum("mc,mxyz->cxyz", a, phase))
    err = np.max(np.abs(F - ref)) / np.max(np.abs(ref))
    assert err < 1e-12, f"OU force vs float64 reference: {err:.2e}"


def test_kick_keeps_the_state_dtype_and_matches_the_at_adds():
    """The kick returns U's dtype for a weakly typed FP64 state (jnp.full with a Python float, as
    driven_turbulence_3d builds U; weak FP64 times the FP32 force is FP32), and on a strong FP64 state it
    equals the four .at[].add of before bitwise. setup/evolve.initial_state makes the run's state strong."""
    from luna.setup.evolve import strong
    jnp = jax.numpy
    n = 8
    drv = OUDriver(OUDriverConfig(solenoidal_weight=0.5, rms_acceleration=3.0, seed=0), dtype=jnp.float32)
    a = drv.init_state(jax.random.PRNGKey(1))
    x = (jnp.arange(n, dtype=jnp.float32) + 0.5) / n
    Uw = jnp.stack([jnp.full((n, n, n), v) for v in (1.0, 0.3, -0.2, 0.1, 2.0)])
    Us = strong(Uw)
    assert Uw.dtype == jnp.float64 and Uw.weak_type and not Us.weak_type

    def at_adds(U, a, h):
        F = drv.force_field(a, x, x, x).reshape((3, n, n, n))
        dE = (U[1] * F[0] + U[2] * F[1] + U[3] * F[2]) * h + 0.5 * U[0] * (
            F[0] * F[0] + F[1] * F[1] + F[2] * F[2]) * h * h
        return (U.at[1].add(U[0] * F[0] * h).at[2].add(U[0] * F[1] * h).at[3].add(U[0] * F[2] * h)
                .at[4].add(dE))

    kick = jax.jit(lambda U, a: drv.apply(U, a, 1e-3, x, x, x))
    out_w = kick(Uw, a)
    assert out_w.dtype == jnp.float64, f"kick turned the weakly typed FP64 state into {out_w.dtype}"
    out_s, ref_s = kick(Us, a), jax.jit(lambda U, a: at_adds(U, a, 1e-3))(Us, a)
    assert out_s.dtype == jnp.float64 and not out_s.weak_type
    assert np.array_equal(np.asarray(out_s), np.asarray(ref_s))


def _direct_force(drv, a, x, y, z):
    """2 Re sum_k a_k exp(i k.x) summed mode by mode in float64, the formula the tables factorise."""
    from luna.hydro.turbulence import _enumerate_modes
    _, k, _, _ = _enumerate_modes(drv.cfg)
    X = np.stack(np.meshgrid(x, y, z, indexing="ij"))
    phase = np.exp(1j * np.einsum("mi,ixyz->mxyz", k, X))
    return 2.0 * np.real(np.einsum("mc,mxyz->cxyz", np.asarray(a, np.complex128), phase))


def test_axis_tables_equal_the_direct_mode_sum():
    """The per-axis tables give the plane-wave sum to round-off on a non-cubic box and grid."""
    cfg = OUDriverConfig(k_min=1.0, k_max=3.0, solenoidal_weight=0.5, rms_acceleration=1.0,
                         seed=0, Lx=1.0, Ly=0.5, Lz=0.75)
    drv = OUDriver(cfg, dtype=float)
    a = drv.init_state(jax.random.PRNGKey(3))
    x, y, z = [(np.arange(n) + 0.5) * L / n for n, L in ((16, 1.0), (8, 0.5), (12, 0.75))]
    got = np.asarray(drv.force_field(a, jax.numpy.asarray(x), jax.numpy.asarray(y),
                                     jax.numpy.asarray(z)))
    ref = _direct_force(drv, a, x, y, z)
    err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    assert err < 1e-12, f"axis tables vs direct sum: {err:.2e}"


def test_two_dimensional_kick_uses_z_zero():
    """A 2D run drives with the full 3D mode set at z = 0, as the meshgrid version did."""
    from dataclasses import replace
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from luna.sources.ou_forcing import setup_ou_forcing
    from luna.template_config import Config
    n, L = 16, 1.0
    ou = OUDriverConfig(k_min=1.0, k_max=3.0, solenoidal_weight=0.5, rms_acceleration=1.0,
                        seed=0, Lx=L, Ly=L, Lz=L)
    cfg = replace(Config(), SPATIAL_DIMENSION=2, Nx=n, Ny=n, Nz=1, Lx=L, Ly=L, Lz=L,
                  ou_config=ou, autodiff_mode=True)
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    drive = setup_ou_forcing(cfg, NamedSharding(mesh, P(None, "x", "y")), False)
    U = jax.numpy.zeros((5, n, n)).at[0].set(1.0)
    F = np.asarray(drive["kick"](U, drive["a"], 1.0))[1:4]       # rho = h = 1: dm = F
    x = (np.arange(n) + 0.5) * (L / n)
    ref = _direct_force(OUDriver(ou, dtype=float), drive["a"], x, x, np.zeros(1))[..., 0]
    err = np.max(np.abs(F - ref)) / np.max(np.abs(ref))
    assert err < 1e-12, f"2D kick vs direct sum at z = 0: {err:.2e}"


@pytest.mark.parametrize("k_min, k_max, spectrum, zeta", [
    (1.0, 3.0, "parabolic", 0.0), (1.0, 3.0, "parabolic", 1.0), (1.0, 2.0, "flat", 0.5),
    (2.0, 4.0, "parabolic", 0.5), (1.0, 1.0, "flat", 0.3)])
def test_axis_tables_for_every_driving_setup(k_min, k_max, spectrum, zeta):
    """Mode sets with and without nx, ny or nz = 0, both spectra, all drivings, against the direct sum."""
    cfg = OUDriverConfig(k_min=k_min, k_max=k_max, solenoidal_weight=zeta, spectrum=spectrum,
                         rms_acceleration=1.0, seed=0, Lx=1.0, Ly=1.0, Lz=1.0)
    drv = OUDriver(cfg, dtype=float)
    a = drv.init_state(jax.random.PRNGKey(7))
    x = (np.arange(12) + 0.5) / 12
    got = np.asarray(drv.force_field(a, *(jax.numpy.asarray(x),) * 3))
    ref = _direct_force(drv, a, x, x, x)
    err = np.max(np.abs(got - ref)) / np.max(np.abs(ref))
    assert err < 1e-12, f"k in [{k_min}, {k_max}], {spectrum}, zeta={zeta}: {err:.2e}"


def test_single_precision_force_is_at_single_precision_round_off():
    """The FP32 production force differs from the FP64 one by FP32 round-off only."""
    cfg = OUDriverConfig(k_min=1.0, k_max=3.0, solenoidal_weight=0.5, rms_acceleration=1.0, seed=0)
    d32, d64 = OUDriver(cfg, dtype=jax.numpy.float32), OUDriver(cfg, dtype=float)
    a = d64.init_state(jax.random.PRNGKey(1))
    x = (np.arange(32) + 0.5) / 32
    f32 = np.asarray(d32.force_field(a, *(jax.numpy.asarray(x, jax.numpy.float32),) * 3))
    f64 = np.asarray(d64.force_field(a, *(jax.numpy.asarray(x),) * 3))
    err = np.max(np.abs(f32 - f64)) / np.max(np.abs(f64))
    assert err < 5e-6, f"FP32 vs FP64 force: {err:.2e}"


def test_force_field_mean_square_is_twice_the_mode_power():
    """On the grid, <|F|^2>_x = 2 sum_k |a_k|^2 (Parseval), the identity the normalisation uses."""
    cfg = OUDriverConfig(k_min=1.0, k_max=3.0, solenoidal_weight=0.5, rms_acceleration=1.0, seed=0)
    drv = OUDriver(cfg, dtype=float)
    a = drv.init_state(jax.random.PRNGKey(2))
    x = jax.numpy.asarray((np.arange(16) + 0.5) / 16)
    F = np.asarray(drv.force_field(a, x, x, x))
    msq = float(np.mean(np.sum(F ** 2, axis=0)))
    expect = 2.0 * float(np.sum(np.abs(np.asarray(a)) ** 2))
    assert abs(msq / expect - 1.0) < 1e-12, f"<|F|^2> = {msq:.6e} vs 2 sum |a|^2 = {expect:.6e}"
