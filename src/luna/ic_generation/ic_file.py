"""Initial conditions written once to disk (U.npy + meta.json), read back one slab per rank.

A run whose config disagrees with meta.json is refused. See docs/TORCH.md.
"""
import json
import os

import numpy as np

from luna.ic_generation._grid import local_range
from luna.ic_generation.user_ic import state_shape

# config values an IC depends on; the cloud's parameters are added to these
_KEYS = ("test_problem", "Nx", "Ny", "Nz", "Lx", "Ly", "Lz", "gamma")


def _fingerprint(cfg):
    from dataclasses import asdict
    from luna.ic_generation.turbulent_sphere_3d import Params as Cloud
    fp = {k: getattr(cfg, k) for k in _KEYS}
    # under their old cloud_ names, so an IC written before still matches
    cloud = cfg.problem if isinstance(cfg.problem, Cloud) else Cloud()
    fp.update({f"cloud_{k}": v for k, v in asdict(cloud).items()})
    fp.update(cloud_mu=cfg.sink_star_mu, cloud_seed=cfg.seed)
    u = cfg.units
    fp["units"] = [u.UnitLength_in_cm, u.UnitMass_in_g, u.UnitVelocity_in_cm_per_s]
    # marks the shell-spectrum convention so ICs written with per-mode power are refused
    fp["turb_slope_is"] = "shell E(k)"
    return json.loads(json.dumps(fp))            # tuples -> lists, as read back


def write_ic(cfg, out_dir, slab=64, U0=None):
    """Write cfg's initial conditions (or the conserved state U0, refused unless cfg's shape) to out_dir
    (the cloud slab by slab); (3, Nx) 1D, (4, Nx, Ny) 2D, (5, Nx, Ny, Nz) 3D."""
    from numpy.lib.format import open_memmap
    from luna.ic_generation.user_ic import user_ic
    user = user_ic(cfg, U0=U0) if U0 is not None else None     # the shape, refused before a file is made
    os.makedirs(out_dir, exist_ok=True)
    U = open_memmap(os.path.join(out_dir, "U.npy"), mode="w+", dtype=np.float64, shape=state_shape(cfg))
    if user is not None:
        U[:] = np.asarray(U0, np.float64)
    elif cfg.test_problem == "Turbulent-Sphere-3D":
        from luna.ic_generation.turbulent_sphere_3d import (cloud_params, cloud_slab,
                                                           velocity_field)
        p = cloud_params(cfg)
        v_unit = velocity_field(p)
        for x0 in range(0, cfg.Nx, slab):
            x1 = min(x0 + slab, cfg.Nx)
            U[:, x0:x1] = cloud_slab(p, v_unit, ((x0, x1), (0, cfg.Ny), (0, cfg.Nz)))
        del v_unit
    else:
        from dataclasses import replace
        from luna.ic_generation.ic import initial_conditions
        U[:] = initial_conditions(replace(cfg, ic_dir=""))
    U.flush()
    del U
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(_fingerprint(cfg), f, indent=1)


def read_ic(cfg, x_slice=None, y_slice=None, z_slice=None):
    """This rank's cells of the IC in cfg.ic_dir (all of it without slices)."""
    d = cfg.ic_dir
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    want = _fingerprint(cfg)
    bad = [k for k in sorted(set(meta) | set(want)) if meta.get(k) != want.get(k)]
    if bad:
        raise ValueError(f"ic_dir {d} was written for a different config: "
                         + ", ".join(f"{k}: file {meta.get(k)!r} vs run {want.get(k)!r}"
                                     for k in bad))
    U = np.load(os.path.join(d, "U.npy"), mmap_mode="r")
    want = state_shape(cfg)
    if U.shape != want:
        raise ValueError(f"ic_dir {d}: U.npy has shape {U.shape}; SPATIAL_DIMENSION="
                         f"{cfg.SPATIAL_DIMENSION} on this grid needs {want}")
    dims = cfg.SPATIAL_DIMENSION
    cut = (local_range(n, sl) for n, sl in zip((cfg.Nx, cfg.Ny, cfg.Nz)[:dims], (x_slice, y_slice, z_slice)[:dims]))
    return np.array(U[(slice(None), *(slice(lo, hi) for lo, hi in cut))])
