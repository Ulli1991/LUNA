"""Tier-1 coupled run: LUNA hydro + sinks, PeTar N-body, SeBa, Torch's IMF; no feedback.

Builds on Torch's `torch_user.py` and overrides only what differs; see docs/TORCH.md."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TORCH_ROOT = os.environ["TORCH_ROOT"]

sys.path.insert(0, os.path.join(TORCH_ROOT, "src"))
sys.path.insert(0, TORCH_ROOT)

from amuse.units import units                                 # noqa: E402

import torch_user as base                                     # noqa: E402
from torch_mainloop import run_torch                          # noqa: E402

# code units come from the LUNA Config so the AMUSE converter cannot disagree
LUNA_CONFIG = os.environ.get(
    "LUNA_CONFIG", os.path.join(HERE, "luna_cloud.py"))
_cfg = runpy.run_path(LUNA_CONFIG)["Config"]
_u = _cfg.units


def user_initial_conditions(state, hydro):
    """No-op: the gas comes from $LUNA_CONFIG and stars form from sinks."""
    return


def user_parameters():
    p = base.user_parameters()

    p["hydro_code"] = "luna"
    p["hydro_units"] = (
        _u.UnitLength_in_cm | units.cm,
        _u.UnitMass_in_g | units.g,
        _u.UnitVelocity_in_cm_per_s | units.cm / units.s,
    )

    p["with_petar"] = True
    p["with_ph4"] = False
    p["with_multiples"] = False
    p["with_se"] = True             # SeBa: torch_sf wants stellar properties

    p["with_lyc"] = False
    p["with_pe_heat"] = False
    p["with_sn"] = False
    p["with_winds"] = False

    # LUNA's sink radius, not FLASH's: torch_sf scatters new stars inside it
    dx = _cfg.Lx / _cfg.Nx
    p["sink_rad"] = 2.0 * dx * _u.UnitLength_in_cm | units.cm

    p["num_hy_workers"] = 1
    p["num_grav_workers"] = 1
    p["evolve_async"] = False

    return p


if __name__ == "__main__":
    run_torch(user_initial_conditions, user_parameters)
