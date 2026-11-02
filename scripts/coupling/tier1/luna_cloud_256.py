"""The Tier-1 cloud at 256^3, with winds and supernovae (driven by `torch_user_long.py`)."""
import os
import runpy
from dataclasses import replace

_cloud = runpy.run_path(
    __file__.replace("luna_cloud_256.py", "luna_cloud.py"))

# Truelove density goes as 1/dx^2: 4x luna_cloud.py's 128^3 value
Config = replace(
    _cloud["Config"],
    Nx=256, Ny=256, Nz=256,
    # sink accretion memory scales as sink_max_count x N^3
    sink_max_count=32,
    sink_creation_density=_cloud["RHO_CREATE"] * 4.0,
    t_end=4.0 * _cloud["T_FF"],
    time_bet_snaps=_cloud["T_FF"] / 10.0,
    output_dir=os.environ.get("LUNA_OUT", "snapshots/tier1") + "_256",
)
