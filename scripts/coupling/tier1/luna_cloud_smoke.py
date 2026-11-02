"""The Tier-1 cloud on a small grid and a short run: a plumbing check, not science."""
import os
import runpy
from dataclasses import replace

_cloud = runpy.run_path(__file__.replace("luna_cloud_smoke.py", "luna_cloud.py"))

Config = replace(
    _cloud["Config"],
    Nx=64, Ny=64, Nz=64,
    t_end=0.02 * _cloud["T_FF"],
    time_bet_snaps=0.01 * _cloud["T_FF"],
    output_dir=os.environ.get("LUNA_OUT", "snapshots/tier1") + "_smoke",
    # 64^3 over the same box is twice the cell size, so the density at which a
    # cell stops resolving its own collapse is four times lower.
    sink_creation_density=_cloud["RHO_CREATE"] / 4.0,
)
