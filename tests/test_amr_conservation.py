"""Conservation of the AMR advance — the core invariant the whole reflux +
fine-authoritative-restrict machinery exists to guarantee.

- A block-uniform (max_levels=0) periodic advance is a plain conservative FV
  scheme: total mass, energy and momentum are conserved to machine precision.
- With refinement + reflux, the coarse level (fully overwritten by restrict) sees
  exactly the transverse-averaged fine fluxes, so its totals (and the composite's)
  telescope and stay conserved across a coarse-fine interface.
"""
import numpy as np
import jax
import jax.numpy as jnp

from luna.amr import driver
from tests._helpers import GAMMA, blob_U, cfg3d, check_conserved


def _blob(n):
    # nonzero flow so fluxes are real
    return blob_U(n, vx_amp=0.2)


def _base(level):
    """The active level-0 blocks as (5, n_blocks * bx, by, bz): rows first, three axes, as check_conserved sums them."""
    d = np.moveaxis(np.asarray(jax.device_get(level.data))[: int(level.n_active)], 1, 0)
    return d.reshape(d.shape[0], -1, *d.shape[3:])


def _composite(h, cfg):
    """The composite on the finest grid (5, *grid): fine data where refined, conservative prolongs elsewhere."""
    return np.asarray(driver.composite_to_dense(h, cfg))


def test_block_uniform_conserves_mass_periodic():
    """max_levels=0 periodic advance conserves total mass to machine precision."""
    n = 32
    cfg = cfg3d(n, max_levels=0, boundary=0, gamma=GAMMA)
    h = driver.build_hierarchy(_blob(n), cfg, thresholds=[0.0], criterion="static",
                               periodic=True, pad_to=1)
    U0 = _base(h[0])
    step = jax.jit(lambda hh: driver.advance(hh, jnp.asarray(2e-3), cfg, gamma=GAMMA))
    for _ in range(5):
        h = step(h)
    check_conserved("amr block-uniform 32^3 5 steps", U0, _base(h[0]))


def test_two_level_reflux_conserves_coarse_mass_periodic():
    """2-level periodic advance with reflux: total mass on the (restrict-covered)
    coarse level is conserved -> the C/F fluxes telescope."""
    n = 32
    cfg = cfg3d(n, max_levels=1, boundary=0, gamma=GAMMA,
                static_region=((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)))
    h = driver.build_hierarchy(_blob(n), cfg, thresholds=[0.0], criterion="static",
                               periodic=True, pad_to=1)
    U0, C0 = _base(h[0]), _composite(h, cfg)
    step = jax.jit(lambda hh: driver.advance(hh, jnp.asarray(2e-3), cfg, gamma=GAMMA,
                                             reflux=True))
    for _ in range(5):
        h = step(h)
    check_conserved("amr 2-level reflux coarse level", U0, _base(h[0]))
    check_conserved("amr 2-level reflux composite", C0, _composite(h, cfg))


def test_subcycle_default_conserves_mass_periodic():
    """The subcycled advance called with DEFAULT args (exactly how the runner
    drives a plain hydro run) must conserve mass — i.e. reflux must default ON.
    Regression guard: with reflux=False the composite mass drifts ~3e-7 at the
    coarse-fine seam; with reflux on it returns to machine precision."""
    n = 32
    cfg = cfg3d(n, max_levels=1, boundary=0, gamma=GAMMA,
                static_region=((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)))
    h = driver.build_hierarchy(_blob(n), cfg, thresholds=[0.0], criterion="static",
                               periodic=True, pad_to=1)
    C0 = _composite(h, cfg)
    # NB: no reflux= kwarg -> exercises the default (must be True).
    step = jax.jit(lambda hh: driver.advance_subcycle(hh, jnp.asarray(2e-3), cfg,
                                                      gamma=GAMMA))
    for _ in range(10):
        h = step(h)
    check_conserved("amr subcycle default composite", C0, _composite(h, cfg))


def test_relative_floor_scales_and_resets():
    """Auto-scaling positivity floor: floor_from_fraction derives absolute
    (rho_floor, p_floor) = frac * base-level mean, and floor_positivity uses it to
    quiescent-reset an injected unphysical cell while leaving good cells and the
    NaN sentinel untouched. This is the deep-hierarchy robustness knob made usable
    without a hand-picked absolute density scale."""
    n = 32
    cfg = cfg3d(n, max_levels=1, boundary=0, gamma=GAMMA,
                static_region=((0.25, 0.75), (0.25, 0.75), (0.25, 0.75)))
    h = driver.build_hierarchy(_blob(n), cfg, thresholds=[0.0], criterion="static",
                               periodic=True, pad_to=1)
    assert int(h[1].n_active) > 0
    frac = 1e-6
    rho_f, p_f = driver.floor_from_fraction(h, cfg, gamma=GAMMA, frac=frac)
    rho_mean = float(jnp.nanmean(h[0].data[:, 0]))
    assert rho_f > 0.0 and p_f > 0.0
    assert abs(rho_f - frac * rho_mean) / (frac * rho_mean) < 1e-10

    # inject a negative-density cell into an active fine block, keep a good one
    d1 = np.array(jax.device_get(h[1].data))   # writable copy
    good_ref = float(d1[0, 0, 1, 1, 1])
    d1[0, 0, 0, 0, 0] = -5.0
    hi = h[:1] + (h[1]._replace(data=jnp.asarray(d1)),) + h[2:]
    hf = driver.floor_positivity(hi, cfg, gamma=GAMMA, rho_floor=rho_f, p_floor=p_f)

    assert float(hf[1].data[0, 0, 0, 0, 0]) == rho_f          # bad cell reset to floor
    assert float(hf[1].data[0, 0, 1, 1, 1]) == good_ref       # good cell untouched
    assert bool(jnp.all(jnp.isnan(hf[1].data[-1])))           # sentinel stays NaN
