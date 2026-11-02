"""The rotating slotted cylinder: a linear-advection benchmark, not an Euler problem.

Setup of Palafoutas, Velasco Romero & Teyssier 2024 (arXiv:2404.18037, sec. 4.4, eq. 50): u = 1 in
x² + (y - 0.5)² < 0.3² except the slot |x| < 0.025, y < 0.7; 0 elsewhere on [-1, 1]², ghosts held
at 0, prescribed v = (-y, x), one turn in 2π (the paper also runs 10 turns; figs. 11-13 at N = 256).
Eq. 50 as printed joins "disk, |x| > 0.025, y > 0.7" by commas, which read as "and" would leave only
the cap above y = 0.7; the figures show the slotted disk, i.e. "|x| > 0.025 or y > 0.7", used here.
Zalesak's test is pure advection by a given velocity; `luna.hydro.advection` runs it with LUNA's
slopes and limiters (tests/test_advection_slotted_cylinder.py).

Why the Euler version was removed. Run as Euler, the gas has to carry the rotation:
- an off-centre density contrast has no equilibrium in rigid rotation (dp/dr = rho Ω² r needs rho
  constant on circles); the old IC set p from the local rho and was discontinuous at the disk edge;
- with the disk as a small contrast (a dye) on uniform density, rigid rotation in a walled box is
  spoiled by the zero-gradient walls (background density drift 19% in one turn at 64²);
- a vortex brought to rest before the walls has zero net circulation, a shielded vortex, which is
  barotropically unstable: euler_3d (pseudo-spectral, incompressible) gives growth rates 0.63 / 0.44
  / 0.33 for fall-offs 0.6-0.9 / 0.45-1.0 / 0.45-1.45 (N = 256-384), 8-50x per turn, consistent
  with the background noise LUNA showed growing with resolution (1.3% at 128², 3.7% at 256²).
A stable Euler rotation needs rigid rotation to the walls with ghosts held at the exact flow, which
LUNA's boundaries do not offer yet (docs/TEST_PLAN.md).
"""
import jax.numpy as jnp


def slotted_cylinder(x, y):
    """The initial u of eq. 50: 1 inside the slotted cylinder, 0 outside."""
    disk = x ** 2 + (y - 0.5) ** 2 < 0.3 ** 2
    slot = (jnp.abs(x) < 0.025) & (y < 0.7)
    return jnp.where(disk & ~slot, 1.0, 0.0)


def rotation(x, y):
    """Rigid rotation (vx, vy) = (-y, x), counter-clockwise, period 2π."""
    return -y + 0.0 * x, x + 0.0 * y


def rotation_slotted_disk(cfg, x_slice=None, y_slice=None):
    """Refused: the rotating slotted cylinder is an advection test, see the module docstring."""
    raise ValueError("Rotate-Slotted-Disk is a linear-advection benchmark with a prescribed velocity, "
                     "not an Euler problem (no stable Euler setup in a walled box; see "
                     "luna/ic_generation/rotation_slotted_disk.py). Run it with luna.hydro.advection.")
