# ic.py
import jax.numpy as jnp
from .sod import sod_shock
from .woodward_colella import woodward_colella
from .shu_osher import shu_osher
from .soundwave import soundwave
from .soundwave_advected import soundwave_advected
from .einfeldt_strong_rarefaction import einfeldt_strong_rarefaction
from .einfeldt_shear_rarefaction import einfeldt_shear_rarefaction
from .einfeldt_transverse_rarefaction import einfeldt_transverse_rarefaction
from .advection_square import advection_square
from .rotation_slotted_disk import rotation_slotted_disk
from .soundwave_advected2d import soundwave_advected2d
from .kh import kelvin_helmholtz
from .double_mach_reflection import double_mach_reflection
from .sod3d import sod_shock_3d 

def initial_conditions(cfg):
    """
    Select initial conditions based on cfg.test_problem
    """
    if cfg.test_problem == "Sod-Shock":
        return sod_shock(cfg)
    elif cfg.test_problem == "Woodward-Colella":
        return woodward_colella(cfg)
    elif cfg.test_problem == "Shu-Osher":
        return shu_osher(cfg)
    elif cfg.test_problem == "Soundwave":
        return soundwave(cfg)
    elif cfg.test_problem == "Soundwave-Advected":
        return soundwave_advected(cfg)
    elif cfg.test_problem == "Einfeldt-Strong":
        return einfeldt_strong_rarefaction(cfg)
    elif cfg.test_problem == "Einfeldt-Shear":
        return einfeldt_shear_rarefaction(cfg)
    elif cfg.test_problem == "Einfeldt-Transverse":
        return einfeldt_transverse_rarefaction(cfg)
    elif cfg.test_problem == "Square-Advected":
        return advection_square(cfg)
    elif cfg.test_problem == "Rotate-Slotted-Disk":
        return rotation_slotted_disk(cfg)
    elif cfg.test_problem == "Soundwave-Advected-2D":
        return soundwave_advected2d(cfg)
    elif cfg.test_problem == "Kelvin-Helmholtz":
        return kelvin_helmholtz(cfg)
    elif cfg.test_problem == "Double-Mach-Reflection":
        return double_mach_reflection(cfg)
    elif cfg.test_problem == "Sod-Shock-3D":
        return sod_shock_3d(cfg)
    else:
        raise ValueError(
            f"Unknown test problem '{cfg.test_problem}'.\n"
            "Available options:\n"
            "  1D: Sod-Shock, Woodward-Colella, Shu-Osher,\n"
            "      Soundwave, Soundwave-Advected,\n"
            "      Einfeldt-Strong, Einfeldt-Shear, Einfeldt-Transverse\n"
            "  2D: Square-Advected, Rotate-Slotted-Disk, Soundwave-Advected-2D,\n"
            "      Kelvin-Helmholtz, Double-Mach-Reflection\n"
            "  3D: Sod-Shock-3D"
        )
