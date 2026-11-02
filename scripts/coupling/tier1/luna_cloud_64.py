"""The Tier-1 cloud at 64^3, run to 1.5 free-fall times so stars appear."""
import os
import runpy
from dataclasses import replace

_cloud = runpy.run_path(
    __file__.replace("luna_cloud_64.py", "luna_cloud.py"))

Config = replace(
    _cloud["Config"],
    Nx=64, Ny=64, Nz=64,
    # twice the cell size: the Truelove density is 4x lower
    sink_creation_density=_cloud["RHO_CREATE"] / 4.0,
    t_end=1.5 * _cloud["T_FF"],
    time_bet_snaps=_cloud["T_FF"] / 20.0,
    output_dir=os.environ.get("LUNA_OUT", "snapshots/tier1") + "_64",
)
