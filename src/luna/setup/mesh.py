"""How a run's data is laid across devices.

`decomposition(cfg)` names the layout, `build_mesh(cfg)` builds it. One
setting, `cfg.decomposition`, chooses; "auto" derives it from the problem.
Initial conditions are read after the mesh exists, since each rank reads only
the part it owns.

Three layouts. `dense`: the gas grid split over an (x, y, z) device mesh the
way the hydro wants it, which on one device is the whole grid. `blocks`: the
same (x, y, z) mesh with the process grid chosen for particles (`dm_pdims`
or balanced), every device owning its block's particles. `amr_blocks`: one
block axis, topology replicated. A run on one device is `dense` (measured on
l9, 2026-09-20: the local backend 48.6 ms/step against 81.3 on a (1,1,1)
block mesh); any particle run on more than one device is `blocks`. To put
the block layout on one device (a test on a one-GPU node), name
`decomposition="blocks"` outright.
"""
import sys

import jax
import numpy as np
from jax.sharding import Mesh

from luna.utils.distributed import create_device_mesh

DENSE = "dense"            # (x,y,z) over the gas grid; the whole grid on one device
BLOCKS = "blocks"          # (px,py,pz) meshblocks, particles owned per block
AMR_BLOCKS = "amr_blocks"  # one block axis, topology replicated

ALL = (DENSE, BLOCKS, AMR_BLOCKS)


def decomposition(cfg, kind=None, n_devices=None, n_processes=None):
    """The layout `cfg` asks for, as a name. Touches no devices.

    `kind` is the run kind (luna/setup/run_kind.py); a runner that already knows
    its own passes it, since its direct callers need not set the flags the
    derivation reads. None derives it from cfg.
    """
    from luna.setup.run_kind import COUPLED, DM, run_kind
    if cfg.decomposition != "auto":
        if cfg.decomposition not in ALL:
            raise ValueError(f"decomposition={cfg.decomposition!r}: "
                             f"expected 'auto' or one of {ALL}")
        return cfg.decomposition

    kind = run_kind(cfg) if kind is None else kind
    n_dev = jax.device_count() if n_devices is None else n_devices
    one_device = n_dev == 1

    if cfg.AMR:
        return AMR_BLOCKS
    if kind in (DM, COUPLED):
        return DENSE if one_device else BLOCKS
    return DENSE


def balanced_pdims(n_devices):
    """Factor `n_devices` into a balanced (px, py, pz), largest factors first."""
    pdims = [1, 1, 1]
    n, axis = n_devices, 0
    while n > 1:
        f = 2
        while n % f:
            f += 1
        pdims[axis % 3] *= f
        n //= f
        axis += 1
    return tuple(sorted(pdims, reverse=True))


def block_dims(cfg, n_devices=None):
    """(px, py, pz) for the block layouts, from cfg or balanced. A named
    `dm_pdims` must use every device."""
    nd = jax.device_count() if n_devices is None else n_devices
    pd = getattr(cfg, "dm_pdims", None)
    if not pd:
        return balanced_pdims(nd)
    pd = tuple(int(v) for v in pd)
    if len(pd) != 3 or int(np.prod(pd)) != nd:
        raise ValueError(f"dm_pdims {pd} needs {int(np.prod(pd))} devices, found {nd}")
    return pd


def build_mesh(cfg, layout=None, kind=None):
    """The device mesh for this run, and the layout name it was built for.
    A dark-matter-only run on one device gets the trivial (1,1,1) grid mesh
    without the hydro's grid checks, since it has no grid to check."""
    from luna.setup.run_kind import DM
    layout = decomposition(cfg, kind) if layout is None else layout
    devs = np.asarray(jax.devices())
    if layout == DENSE:
        if kind == DM:
            return Mesh(devs[:1].reshape(1, 1, 1), ("x", "y", "z")), DENSE
        return create_distributed_mesh(cfg)[0], DENSE      # also validates cfg vs grid
    if layout == BLOCKS:
        return Mesh(devs.reshape(*block_dims(cfg)), ("x", "y", "z")), BLOCKS
    if layout == AMR_BLOCKS:
        return Mesh(devs, ("blocks",)), AMR_BLOCKS
    raise ValueError(f"unknown decomposition {layout!r}")


def decomposition_name(grid_shape):
    """How the process grid cuts the domain: none, slab (one axis), pencil (two of three) or block (all)."""
    cut = [a for a, n in zip("xyz", grid_shape) if n > 1]
    if not cut:
        return "none (one device)"
    kind = "slab" if len(cut) == 1 else "block" if len(cut) == len(grid_shape) else "pencil"
    return f"{kind} (cut along {', '.join(cut)})"


def create_distributed_mesh(cfg):
    """The dense layout's (x, y, z) device mesh over the gas grid, and its
    process grid shape; checks the grid divides over the devices."""
    is_main_process = jax.process_index() == 0

    n_devices = len(jax.devices())
    if n_devices < 1:
        print("ERROR: No devices available!")
        sys.exit(1)

    if is_main_process:
        print("=" * 60)
        print(f"DOMAIN DECOMPOSITION ({n_devices} devices)")
        print("=" * 60)

    mesh, grid_shape = create_device_mesh(
        n_devices,
        cfg.SPATIAL_DIMENSION,
        Nx=cfg.Nx,
        Ny=cfg.Ny,
        mesh_shape=cfg.MESH_SHAPE,
    )

    if is_main_process:
        print(f"Decomposition strategy: {'MESH_SHAPE' if cfg.MESH_SHAPE else 'auto'}")
        print(f"Process grid: {grid_shape}")
        print(f"Actual decomposition: {decomposition_name(grid_shape)}")
        print()

    return mesh, grid_shape
