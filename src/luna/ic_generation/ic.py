"""Registry from cfg.test_problem to its IC builder."""
from .advection_square import advection_square
from .agertz_blob import agertz_blob
from .cc85_wind_3d import cc85_wind_3d
from .planar_riemann_2d import double_mach_reflection, planar_riemann_2d
from .driven_turbulence_3d import driven_turbulence_3d
from .einfeldt import (
    einfeldt_shear_rarefaction,
    einfeldt_strong_rarefaction,
    einfeldt_transverse_rarefaction,
)
from .bonnor_ebert3d import bonnor_ebert_3d
from .evrard3d import evrard_3d
from .gresho_vortex import gresho_vortex
from .implosion import implosion
from .hii_region_3d import hii_region_3d
from .jeans3d import jeans_3d
from .kh import kelvin_helmholtz
from .kt_riemann_2d import kt_riemann_2d
from .kh_mcnally import kh_mcnally
from .lane_emden3d import lane_emden_3d
from .pulsating_polytrope3d import pulsating_polytrope_3d
from .maclaurin3d import maclaurin_3d
from .boss_bodenheimer3d import boss_bodenheimer_3d
from .kh3d import kelvin_helmholtz_3d
from . import kh3d, sod3d
from .m82_disk_3d import m82_disk_3d
from .rotation_slotted_disk import rotation_slotted_disk
from .sedov3d import sedov_blast_3d
from .leblanc import leblanc
from .shu_osher import shu_osher
from .titarev_toro import titarev_toro
from .sod import sod_shock
from .sod3d import sod_shock_3d, sod_shock_3d_amr
from .soundwave import soundwave
from .soundwave3d import soundwave_3d
from .soundwave_advected import soundwave_advected
from .soundwave_advected2d import soundwave_advected2d
from .woodward_colella import woodward_colella
from .disc_planet_2d import disc_planet_2d
from .keplerian_disc_2d import keplerian_disc_2d
from .zeldovich import zeldovich_pancake_3d
from .turbulent_sphere_3d import turbulent_sphere_3d

# test_problem -> (ic_function, ndim); each function takes the slice kwargs of its ndim.
_IC_REGISTRY = {
    # 1D
    "Sod-Shock": (sod_shock, 1),
    "Woodward-Colella": (woodward_colella, 1),
    "Shu-Osher": (shu_osher, 1),
    "LeBlanc": (leblanc, 1),
    "Titarev-Toro": (titarev_toro, 1),
    "Soundwave": (soundwave, 1),
    "Soundwave-Advected": (soundwave_advected, 1),
    "Einfeldt-Strong": (einfeldt_strong_rarefaction, 1),
    "Einfeldt-Shear": (einfeldt_shear_rarefaction, 1),
    "Einfeldt-Transverse": (einfeldt_transverse_rarefaction, 2),
    # 2D
    "Square-Advected": (advection_square, 2),
    "Rotate-Slotted-Disk": (rotation_slotted_disk, 2),
    "Soundwave-Advected-2D": (soundwave_advected2d, 2),
    "Kelvin-Helmholtz": (kelvin_helmholtz, 2),
    "KH-McNally-2D": (kh_mcnally, 2),
    "Double-Mach-Reflection": (double_mach_reflection, 2),
    "Planar-Riemann-2D": (planar_riemann_2d, 2),
    "KT-Riemann-2D": (kt_riemann_2d, 2),
    "Disc-Planet-2D": (disc_planet_2d, 2),
    "Keplerian-Disc-2D": (keplerian_disc_2d, 2),
    # 3D
    "Sod-Shock-3D": (sod_shock_3d, 3),
    "Sod-Shock-3D-AMR": (sod_shock_3d_amr, 3),
    "Sedov-Blast-3D": (sedov_blast_3d, 3),
    "Kelvin-Helmholtz-3D": (kelvin_helmholtz_3d, 3),
    "Soundwave-3D": (soundwave_3d, 3),
    "Turbulent-Sphere-3D": (turbulent_sphere_3d, 3),
    "Driven-Turbulence-3D": (driven_turbulence_3d, 3),
    "CC85-Wind-3D": (cc85_wind_3d, 3),
    "M82-Disk-3D": (m82_disk_3d, 3),
    "Gresho-Vortex-2D": (gresho_vortex, 2),
    "LW-Implosion": (implosion, 2),
    "Jeans-Instability-3D": (jeans_3d, 3),
    "Evrard-Collapse-3D": (evrard_3d, 3),
    "Bonnor-Ebert-3D": (bonnor_ebert_3d, 3),
    "Lane-Emden-3D": (lane_emden_3d, 3),
    "Pulsating-Polytrope-3D": (pulsating_polytrope_3d, 3),
    "Maclaurin-Spheroid-3D": (maclaurin_3d, 3),
    "Boss-Bodenheimer-3D": (boss_bodenheimer_3d, 3),
    "Zeldovich-Pancake": (zeldovich_pancake_3d, 3),
    "HII-Region-3D": (hii_region_3d, 3),
    "Agertz-Blob-3D": (agertz_blob, 3),
}


def params_class(test_problem):
    """The Params dataclass of test_problem's IC module, or None if it has none."""
    import sys
    entry = _IC_REGISTRY.get(test_problem)
    return None if entry is None else getattr(sys.modules[entry[0].__module__], "Params", None)


def problem_params(test_problem, given=None):
    """`given`, else test_problem's default Params; another problem's untouched default is swapped."""
    cls = params_class(test_problem)
    if cls is not None and isinstance(given, cls):
        return given
    if given is None or given == type(given)():
        return cls() if cls is not None else None
    where = f"{cls.__module__}.Params" if cls else "nothing: it has no parameters"
    raise ValueError(f"problem={given!r} is not test_problem={test_problem!r}'s; "
                     f"its parameters are {where}")


# Problems that require comoving integration (main.run refuses them otherwise).
_COSMOLOGICAL_PROBLEMS = frozenset({
    "Zeldovich-Pancake",
})


def is_cosmological(test_problem):
    """True if test_problem requires comoving (cosmological) integration."""
    return test_problem in _COSMOLOGICAL_PROBLEMS


# Problems that choose their own transverse grid (a thin tube or slab).
_RESOLVE_REGISTRY = {
    "Sod-Shock-3D": sod3d.transverse,
    "Kelvin-Helmholtz-3D": kh3d.transverse,
}


def resolve(cfg):
    """cfg with the grid fields this problem derives filled in."""
    from dataclasses import replace
    fn = _RESOLVE_REGISTRY.get(cfg.test_problem)
    return replace(cfg, **fn(cfg)) if fn else cfg


def initial_conditions(cfg, x_slice=None, y_slice=None, z_slice=None, user=None):
    """U for cfg.test_problem on the local shard (or the UserIC `user`, or read from cfg.ic_dir).

    Shape (3, Nx) 1D, (4, Nx, Ny) 2D, (5, Nx, Ny, Nz) 3D; extra slices are ignored.
    """
    if user is not None:
        return user.local(cfg, x_slice, y_slice, z_slice)
    if cfg.ic_dir:
        from luna.ic_generation.ic_file import read_ic
        return read_ic(cfg, x_slice, y_slice, z_slice)
    if cfg.test_problem not in _IC_REGISTRY:
        raise ValueError(
            f"Unknown test problem '{cfg.test_problem}'.\n"
            "Available options:\n"
            "  1D: Sod-Shock, Woodward-Colella, Shu-Osher, LeBlanc, Titarev-Toro,\n"
            "      Soundwave, Soundwave-Advected,\n"
            "      Einfeldt-Strong, Einfeldt-Shear\n"
            "  2D: Einfeldt-Transverse, Square-Advected, Rotate-Slotted-Disk, Soundwave-Advected-2D,\n"
            "      Kelvin-Helmholtz, KH-McNally-2D, Planar-Riemann-2D, KT-Riemann-2D,\n"
            "      Double-Mach-Reflection (refused: not yet implemented)\n"
            "  3D: Sod-Shock-3D, Sedov-Blast-3D, Kelvin-Helmholtz-3D, Soundwave-3D"
        )

    ic_fn, ndim = _IC_REGISTRY[cfg.test_problem]
    if cfg.SPATIAL_DIMENSION != ndim:
        raise ValueError(f"{cfg.test_problem} is a {ndim}D problem; got SPATIAL_DIMENSION={cfg.SPATIAL_DIMENSION}")

    if ndim == 1:
        return ic_fn(cfg, x_slice=x_slice)
    elif ndim == 2:
        return ic_fn(cfg, x_slice=x_slice, y_slice=y_slice)
    else:  # ndim == 3
        return ic_fn(cfg, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)
