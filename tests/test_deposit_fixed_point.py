"""Fixed-point CIC deposit (cfg.gravity_deposit_fixed, PRECISION_SURVEY R3):
exact mass, order invariance, and agreement with the float deposit."""
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from gravity.particle_mesh import ParticleMeshGravity
from setup.cosmo_common import gravity_precision_from_config
from template_config import Config

N = 16
BOX = 1.0
NP = 4000


def _pm(**kw):
    cfg = replace(Config(), SPATIAL_DIMENSION=3, Nx=N, Ny=N, Nz=N,
                  Lx=BOX, Ly=BOX, Lz=BOX, BOUNDARY=0, **kw)
    return ParticleMeshGravity(cfg), cfg


def _particles(seed=3):
    rng = np.random.default_rng(seed)
    pos = jnp.asarray(rng.uniform(0.0, BOX, size=(NP, 3)))
    mass = jnp.asarray(rng.uniform(0.5, 1.5, size=(NP,)))
    return pos, mass, jnp.ones(NP)


def test_fixed_point_conserves_mass_exactly():
    """Integer weights sum to the particle's weight -> no mass is lost."""
    pm, _ = _pm()
    pos, mass, act = _particles()
    rho = pm.cic_deposit(pos, mass, act, fixed=jnp.int64, fixed_bits=40)
    tot = float(jnp.sum(rho)) * pm.dV
    ref = float(jnp.sum(mass * act))
    assert abs(tot - ref) / ref < 1e-13, f"mass {tot} vs {ref}"


def test_fixed_point_is_order_invariant():
    """Integer addition is associative: reordering the particles cannot move
    a single bit of the density field (a float deposit does not have this)."""
    pm, _ = _pm()
    pos, mass, act = _particles()
    perm = np.random.default_rng(11).permutation(NP)
    a = pm.cic_deposit(pos, mass, act, fixed=jnp.int32, fixed_bits=17)
    b = pm.cic_deposit(pos[perm], mass[perm], act[perm], fixed=jnp.int32,
                       fixed_bits=17)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_fixed_point_matches_float_deposit():
    """int32 at S=2^17 tracks the float64 deposit to the quantization bound."""
    pm, _ = _pm()
    pos, mass, act = _particles()
    ref = pm.cic_deposit(pos, mass, act)
    fix = pm.cic_deposit(pos, mass, act, fixed=jnp.int32, fixed_bits=17)
    scale = float(jnp.max(ref))
    err = float(jnp.max(jnp.abs(fix.astype(jnp.float64) - ref))) / scale
    assert err < 1e-4, f"fixed-point deposit off by {err:.3e}"


def test_gravity_precision_modes():
    """The one switch resolves to the documented dtype table, and the
    deprecated aliases still map onto it."""
    for mode, part, dep, mix in (("double", jnp.float64, None, False),
                                 ("mixed", jnp.float64, jnp.float32, True),
                                 ("single", jnp.float32, jnp.float32, True)):
        _, cfg = _pm(gravity_precision=mode)
        p = gravity_precision_from_config(cfg)
        assert (p.particles, p.deposit, p.cycle_mixed) == (part, dep, mix)
    _, cfg = _pm(dm_fp32=True)
    assert gravity_precision_from_config(cfg).mode == "single"
    _, cfg = _pm(dm_mixed_precision=False)
    assert gravity_precision_from_config(cfg).mode == "double"
