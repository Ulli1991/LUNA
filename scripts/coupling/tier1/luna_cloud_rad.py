"""The Tier-1 cloud with photoionisation: LUNA's M1 driven by Torch's stars.

Ionisation sets P = rho cs^2(x) in place of barotropic; per-star rates via set_particle_nion."""
import os
import runpy
from dataclasses import replace

_cloud = runpy.run_path(
    __file__.replace("luna_cloud_rad.py", "luna_cloud.py"))
_base = _cloud["Config"]

PC_CM = 3.085677581e18
MSUN_G = 1.98892e33
KMS_CMS = 1.0e5
T_CODE_S = PC_CM / KMS_CMS              # 1 pc / (km/s) = 0.978 Myr

K_B = 1.380649e-16
M_P = 1.6726219e-24
SIGMA_HI_CGS = 6.3e-18                  # cm^2 at 13.6 eV
ALPHA_B_CGS = 2.7e-13                   # cm^3/s, case B at 1e4 K

# mass per hydrogen nucleus (H plus its share of He), not the mean molecular mass
M_H_CODE = 1.4 * M_P / MSUN_G

SIGMA_CODE = SIGMA_HI_CGS / PC_CM ** 2
ALPHA_B_CODE = ALPHA_B_CGS * T_CODE_S / PC_CM ** 3


def _cs2(T_K, mu):
    """Isothermal sound speed squared in (km/s)^2."""
    return (K_B * T_K / (mu * M_P)) / KMS_CMS ** 2


CS2_NEUTRAL = _cs2(_base.problem.T_K, 2.3)        # 15 K molecular
CS2_IONISED = _cs2(1.0e4, 0.6)                  # 1e4 K ionised, ~137 (km/s)^2

# reduced c must beat the ionised sound speed; transport substep is dx / (3 c_hat)
C_HAT = 150.0

Config = replace(
    _base,
    Nx=256, Ny=256, Nz=256,
    # sink accretion memory scales as sink_max_count x N^3
    sink_max_count=32,
    sink_creation_density=_cloud["RHO_CREATE"] * 4.0,   # Truelove goes as 1/dx^2

    # --- the thermal state is the ionisation state -----------------------
    barotropic_eos=False,
    cooling_enabled=False,
    radiation_enabled=True,
    rad_ionization=True,
    rad_chat=C_HAT,
    rad_substeps=16,
    rad_sigma=SIGMA_CODE,
    rad_alpha_B=ALPHA_B_CODE,
    rad_m_H=M_H_CODE,
    rad_cs2_neutral=CS2_NEUTRAL,
    rad_cs2_ionized=CS2_IONISED,
    rad_x0=0.0,                       # the cloud starts neutral
    rad_photon_rate=0.0,              # the stars carry it, via set_particle_nion

    t_end=4.0 * _cloud["T_FF"],
    time_bet_snaps=_cloud["T_FF"] / 10.0,
    output_dir=os.environ.get("LUNA_OUT", "snapshots/tier1") + "_rad",
)
