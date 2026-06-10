# ic.py
from .advection_square import advection_square
from .cc85_wind_3d import cc85_wind_3d
from .double_mach_reflection import double_mach_reflection
from .driven_turbulence_3d import driven_turbulence_3d
from .einfeldt_shear_rarefaction import einfeldt_shear_rarefaction
from .einfeldt_strong_rarefaction import einfeldt_strong_rarefaction
from .einfeldt_transverse_rarefaction import einfeldt_transverse_rarefaction
from .bonnor_ebert3d import bonnor_ebert_3d
from .evrard3d import evrard_3d
from .gresho_vortex import gresho_vortex
from .jeans3d import jeans_3d
from .kh import kelvin_helmholtz
from .lane_emden3d import lane_emden_3d
from .pulsating_polytrope3d import pulsating_polytrope_3d
from .maclaurin3d import maclaurin_3d
from .boss_bodenheimer3d import boss_bodenheimer_3d
from .kh3d import kelvin_helmholtz_3d
from .m82_disk_3d import m82_disk_3d
from .rotation_slotted_disk import rotation_slotted_disk
from .sedov3d import sedov_blast_3d
from .shu_osher import shu_osher
from .sod import sod_shock
from .sod3d import sod_shock_3d
from .solar_convection_2d import (
    solar_convection_2d,
    solar_convection_2d_equilibrium,
)
from .solar_convection_3d import (
    solar_convection_3d,
    solar_convection_3d_equilibrium,
)
from .solar_convection_vt_2d import (
    solar_convection_vt_2d,
    solar_convection_vt_2d_equilibrium,
)
from .soundwave import soundwave
from .soundwave3d import soundwave_3d
from .soundwave_advected import soundwave_advected
from .soundwave_advected2d import soundwave_advected2d
from .woodward_colella import woodward_colella

# test_problem name -> (ic_function, ndim).
# Every IC function accepts the slice kwargs matching its dimensionality
# (audited 2026-05-17); slices for higher dimensions are dropped here so the
# IC functions don't have to know about them.
_IC_REGISTRY = {
    # 1D
    "Sod-Shock": (sod_shock, 1),
    "Woodward-Colella": (woodward_colella, 1),
    "Shu-Osher": (shu_osher, 1),
    "Soundwave": (soundwave, 1),
    "Soundwave-Advected": (soundwave_advected, 1),
    "Einfeldt-Strong": (einfeldt_strong_rarefaction, 1),
    "Einfeldt-Shear": (einfeldt_shear_rarefaction, 1),
    "Einfeldt-Transverse": (einfeldt_transverse_rarefaction, 1),
    # 2D
    "Square-Advected": (advection_square, 2),
    "Rotate-Slotted-Disk": (rotation_slotted_disk, 2),
    "Soundwave-Advected-2D": (soundwave_advected2d, 2),
    "Kelvin-Helmholtz": (kelvin_helmholtz, 2),
    "Double-Mach-Reflection": (double_mach_reflection, 2),
    # 3D
    "Sod-Shock-3D": (sod_shock_3d, 3),
    "Sedov-Blast-3D": (sedov_blast_3d, 3),
    "Kelvin-Helmholtz-3D": (kelvin_helmholtz_3d, 3),
    "Soundwave-3D": (soundwave_3d, 3),
    "Driven-Turbulence-3D": (driven_turbulence_3d, 3),
    "Solar-Convection-3D": (solar_convection_3d, 3),
    "Solar-Convection-2D": (solar_convection_2d, 2),
    "Solar-Convection-VT-2D": (solar_convection_vt_2d, 2),
    "CC85-Wind-3D": (cc85_wind_3d, 3),
    "M82-Disk-3D": (m82_disk_3d, 3),
    "Gresho-Vortex-2D": (gresho_vortex, 2),
    "Jeans-Instability-3D": (jeans_3d, 3),
    "Evrard-Collapse-3D": (evrard_3d, 3),
    "Bonnor-Ebert-3D": (bonnor_ebert_3d, 3),
    "Lane-Emden-3D": (lane_emden_3d, 3),
    "Pulsating-Polytrope-3D": (pulsating_polytrope_3d, 3),
    "Maclaurin-Spheroid-3D": (maclaurin_3d, 3),
    "Boss-Bodenheimer-3D": (boss_bodenheimer_3d, 3),
}


# Equilibrium-state registry for problems that use the well-balanced
# integrator. Keys are the test_problem name; values are callables with the
# same signature as the IC function but returning U_eq (hydrostatic
# equilibrium state). Problems that don't define HSE simply aren't listed.
_EQ_REGISTRY = {
    "Solar-Convection-3D": solar_convection_3d_equilibrium,
    "Solar-Convection-2D": solar_convection_2d_equilibrium,
    "Solar-Convection-VT-2D": solar_convection_vt_2d_equilibrium,
}


def available_test_problems():
    """Return the list of supported test_problem names."""
    return list(_IC_REGISTRY.keys())


def equilibrium_state(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Return hydrostatic-equilibrium state U_eq for the configured problem.

    Only defined for problems registered in _EQ_REGISTRY. Raises KeyError
    otherwise. Used by the well-balanced integrator path.
    """
    if cfg.test_problem not in _EQ_REGISTRY:
        raise KeyError(
            f"test_problem '{cfg.test_problem}' has no equilibrium_state "
            "registered. Add an entry to _EQ_REGISTRY in ic.py if this "
            "problem supports the well-balanced integrator."
        )
    fn = _EQ_REGISTRY[cfg.test_problem]
    return fn(cfg, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)


def initial_conditions(cfg, x_slice=None, y_slice=None, z_slice=None):
    """
    Build initial conditions for cfg.test_problem.

    If x_slice / y_slice / z_slice are provided, the underlying IC function
    returns only the local portion of the global domain — used for
    distributed/sharded initialization. Slices beyond the problem's
    dimensionality are ignored.

    Returns the conservative state U with shape:
        (3, Nx)            for 1D
        (4, Nx, Ny)        for 2D
        (5, Nx, Ny, Nz)    for 3D
    or the corresponding local-slice shape when slices are passed.
    """
    if cfg.test_problem not in _IC_REGISTRY:
        raise ValueError(
            f"Unknown test problem '{cfg.test_problem}'.\n"
            "Available options:\n"
            "  1D: Sod-Shock, Woodward-Colella, Shu-Osher,\n"
            "      Soundwave, Soundwave-Advected,\n"
            "      Einfeldt-Strong, Einfeldt-Shear, Einfeldt-Transverse\n"
            "  2D: Square-Advected, Rotate-Slotted-Disk, Soundwave-Advected-2D,\n"
            "      Kelvin-Helmholtz, Double-Mach-Reflection\n"
            "  3D: Sod-Shock-3D, Sedov-Blast-3D, Kelvin-Helmholtz-3D, Soundwave-3D"
        )

    ic_fn, ndim = _IC_REGISTRY[cfg.test_problem]

    if ndim == 1:
        return ic_fn(cfg, x_slice=x_slice)
    elif ndim == 2:
        return ic_fn(cfg, x_slice=x_slice, y_slice=y_slice)
    else:  # ndim == 3
        return ic_fn(cfg, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)
