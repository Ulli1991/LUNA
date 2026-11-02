"""Initial conditions, read after the mesh so each rank reads only its part.

`read_ics(cfg, mesh, kind, layout)` returns an `Initial`: the gas grid and/or
this rank's dark matter as host `Particles` with canonical momenta, the
global counts, and the config, which a MUSIC header rewrites (box,
cosmology, a_init). One chunk rule serves both particle kinds: a block's
x-slab is a contiguous range of the x-major MUSIC file, then the (y,z) rows
of that range are filtered to the block.

Restart or ICs is decided here for the dense gas grid (`gas_grid`): on a
restart the grid, t and step come from the checkpoint instead, read straight
onto the mesh, so the grid is never held twice. Everything else a restart
restores is Outputs.restore's (luna/setup/outputs.py).
"""
import sys
from dataclasses import dataclass
from typing import Any, Optional

import jax
import numpy as np

from luna.setup.mesh import AMR_BLOCKS, BLOCKS
from luna.setup.run_kind import COUPLED, DM, GAS
from luna.setup.state import Particles
from luna.utils.memory_utils import print_memory_usage


@dataclass
class Initial:
    cfg: Any                        # rewritten from the MUSIC header when there is one
    U: Any = None                   # gas grid, sharded; None for dark matter only
    dm: Optional[Particles] = None  # this rank's dark matter rows (host, canonical p)
    dm_ic: Any = None               # the MUSIC header record (h, masses) for writers
    n_dm: Optional[int] = None      # global count when chunked, else None
    n_gas: Optional[int] = None
    t_start: float = 0.0            # dense gas path only (restart-aware)
    step_start: int = 0


def music_chunk(path, part_type, mesh, cfg, chunked):
    """(lo, hi, n_total) of the file range this rank's block covers; the whole
    file and None when not chunked."""
    from luna.cosmology.music_ic import music_part_count
    if not chunked:
        return None, None, None
    from luna.gravity.particle_mesh_blocks import block_of_process
    from luna.setup.mesh import block_dims
    n_total = music_part_count(path, part_type)
    n_chunks, i_chunk = block_dims(cfg)[0], block_of_process(mesh)[0]
    lo = i_chunk * n_total // n_chunks
    hi = n_total if i_chunk == n_chunks - 1 else (i_chunk + 1) * n_total // n_chunks
    return lo, hi, n_total


def _block_rows(ic, cfg, mesh):
    """Keep the rows of this rank's block: a chunk is an x-slab, the block is a
    (y,z) part of it."""
    from luna.cosmology.music_ic import ic_rows
    from luna.gravity.particle_mesh_blocks import block_of_process, block_rows
    from luna.setup.mesh import block_dims
    N3 = (int(cfg.Nx), int(cfg.Ny), int(cfg.Nz))
    box = float(cfg.Lx)
    return ic_rows(ic, block_rows(np.asarray(ic.pos), N3, block_dims(cfg),
                                  box / N3[0], box, block_of_process(mesh)))


def _seed_gas_temperature(gas, cfg, is_main):
    """MUSIC may write no InternalEnergy; gas at P=0 has no sound speed."""
    from luna.units import KB_OVER_MP_KMS2_PER_K
    if float(np.max(gas.u_int)) > 0.0:
        return gas
    a = float(cfg.a_init)
    T_init = float(cfg.cosmo_T_init) or 2.725 * (1.0 / a)        # CMB at z_init
    u = KB_OVER_MP_KMS2_PER_K * T_init / ((float(cfg.gamma) - 1.0) * float(cfg.cosmo_mu))
    gas.u_int = np.full(gas.n_part, u, dtype=np.float64)
    if is_main:
        print(f"[ics] seeding gas T_init={T_init:.1f} K -> u={u:.3e} (km/s)^2 "
              "(MUSIC wrote no InternalEnergy)", flush=True)
    return gas


def canonical_particles(ic, a):
    """Host `Particles` from a MUSIC record: canonical momentum p = a^2 x_dot
    = sqrt(a) * (Gadget velocity) * a; the IDs as read."""
    return Particles(np.asarray(ic.pos), np.asarray(ic.vel) * (a ** 1.5), np.asarray(ic.mass),
                     ids=ic.ids)


def read_ics(cfg, mesh, kind, layout, is_main=True, user=None):
    """The initial state for a run of `kind` on `mesh`, laid out as `layout`; `user` a UserIC (gas only)."""
    from luna.cosmology.music_ic import music_run_config, read_music_dm_ic, read_music_gas_ic

    if kind == GAS:
        if layout == AMR_BLOCKS:
            return Initial(cfg=cfg)        # the refinement runner builds its hierarchy
        U, t0, s0 = gas_grid(cfg, mesh, user)
        return Initial(cfg=cfg, U=U, t_start=t0, step_start=s0)

    path = cfg.music_ic_path
    chunked = jax.process_count() > 1 and layout == BLOCKS
    dlo, dhi, n_dm = music_chunk(path, 1, mesh, cfg, chunked)
    dm = read_music_dm_ic(path, 1, dlo, dhi)
    gas = n_gas = None
    if kind == COUPLED:
        glo, ghi, n_gas = music_chunk(path, 0, mesh, cfg, chunked)
        gas = read_music_gas_ic(path, glo, ghi)
    if chunked:
        print(f"[ics] rank {jax.process_index()}/{jax.process_count()}: "
              f"dm[{dlo}:{dhi}]" + (f" gas[{glo}:{ghi}]" if gas is not None else "")
              + " (chunked read, this block's x-slab)", flush=True)

    cfg = music_run_config(dm, cfg.Nx, base=cfg)        # box + cosmology from the header
    if chunked:
        dm = _block_rows(dm, cfg, mesh)
        if gas is not None:
            gas = _block_rows(gas, cfg, mesh)
    a = float(cfg.a_init)

    if kind == DM:
        if is_main:
            print(f"[ics] MUSIC DM IC: {n_dm or dm.n_part} particles, box={cfg.Lx:.0f} kpc/h, "
                  f"z={1/a-1:.1f}{' (chunked)' if chunked else ''}", flush=True)
        return Initial(cfg=cfg, dm=canonical_particles(dm, a), dm_ic=dm, n_dm=n_dm)

    gas = _seed_gas_temperature(gas, cfg, is_main)
    N3 = (int(cfg.Nx), int(cfg.Ny), int(cfg.Nz))
    box = float(cfg.Lx)
    if layout == BLOCKS:
        from luna.cosmology.mesh_gas import mesh_gas_to_grid_blocks
        U = mesh_gas_to_grid_blocks(gas, mesh, N3, box / N3[0], box, a, n_total_gas=n_gas)
    else:
        from luna.cosmology.mesh_gas import mesh_gas_to_grid
        from luna.gravity.particle_mesh import ParticleMeshGravity
        U = mesh_gas_to_grid(gas, ParticleMeshGravity(cfg), a)
    if is_main:
        print(f"[ics] MUSIC baryon IC: {n_gas or gas.n_part} gas + {n_dm or dm.n_part} DM, "
              f"box={cfg.Lx:.0f} kpc/h, z={1/a-1:.1f}"
              f"{' (chunked)' if chunked else ''}", flush=True)
    return Initial(cfg=cfg, U=U, dm=canonical_particles(dm, a), dm_ic=dm, n_dm=n_dm, n_gas=n_gas)


def gas_grid(cfg, mesh, user=None):
    """(U, t_start, step_start) of the dense gas grid, sharded on `mesh`: the
    checkpoint's on a restart, else the initial conditions. Read here, where
    the sharding exists, so the grid is never held twice."""
    is_main_process = jax.process_index() == 0
    if is_main_process:
        print("=" * 60)
    if cfg.restart:
        U_sharded, t_start, step_start = grid_from_checkpoint(cfg, mesh)
    else:
        U_sharded, t_start, step_start = grid_from_ics(cfg, mesh, user), cfg.t_start, 0
    if is_main_process:
        print(f"✓ Array sharded across {len(jax.devices())} devices")
        print("  Each process keeps only its local shard")
        print()
        print_memory_usage("After initialization: ")
        print()
    return U_sharded, t_start, step_start


def grid_from_checkpoint(cfg, mesh):
    """(U, t, step) from the grid checkpoint set (cfg.restart_file, else the
    newest complete one); every process assembles its own devices' blocks,
    whatever decomposition wrote it. The rest of that checkpoint (the OU
    forcing state) is restored in Outputs.restore (luna/setup/outputs.py)."""
    from luna.file_io.checkpoint_utils import checkpoint_dir, validate_checkpoint_compatibility
    from luna.file_io.read_checkpoint import find_latest_checkpoint, load_checkpoint_sharded
    is_main_process = jax.process_index() == 0
    if is_main_process:
        print("RESTART FROM CHECKPOINT")
        print("=" * 60)

    if cfg.restart_file is not None:
        checkpoint_file = cfg.restart_file
    else:
        checkpoint_file = find_latest_checkpoint(checkpoint_dir(cfg))
        if checkpoint_file is None:
            sys.exit(f"ERROR: No checkpoint found in {checkpoint_dir(cfg)}")

    U_sharded, checkpoint_data = load_checkpoint_sharded(
        checkpoint_file, mesh, cfg.SPATIAL_DIMENSION)
    validate_checkpoint_compatibility(checkpoint_data, cfg)
    t_start = checkpoint_data["time"]
    step_start = checkpoint_data["step"]

    if is_main_process:
        print(f"✓ Restarting from t = {t_start:.5f}, step = {step_start}")
        print()
    return U_sharded, t_start, step_start


def grid_from_ics(cfg, mesh, user=None):
    """The initial gas grid of cfg.test_problem (or of the UserIC `user`), sharded on
    `mesh`. With several processes each generates only its own slice."""
    from luna.hydro.physics_checks import validate_state
    from luna.ic_generation.ic import initial_conditions
    from luna.utils.distributed import (compute_local_slice, create_sharded_array_from_local,
                                        shard_array_to_devices)
    is_main_process = jax.process_index() == 0
    process_id = jax.process_index()
    if is_main_process:
        print("INITIAL CONDITIONS")
        print("=" * 60)

    if jax.process_count() > 1:
        if is_main_process:
            print("Multi-process mode: using local IC generation")
            print("Each process generates only its local portion (memory efficient)")

        if cfg.SPATIAL_DIMENSION == 3:
            x_slice, y_slice, z_slice = compute_local_slice(
                cfg.Nx, cfg.Ny, mesh, process_id, cfg.Nz
            )
        else:
            x_slice, y_slice, z_slice = compute_local_slice(
                cfg.Nx, cfg.Ny if cfg.SPATIAL_DIMENSION >= 2 else 1, mesh, process_id
            )

        # every IC function takes the slice kwargs of its dimensionality
        try:
            U_local = initial_conditions(cfg, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice,
                                         user=user)
        except ValueError as e:
            if is_main_process:
                print(f"\n{'='*60}")
                print(f"ERROR: {e}")
                print(f"{'='*60}\n")
            sys.exit(1)

        if is_main_process:
            print("Local shape per process: varies by decomposition")

        if cfg.SPATIAL_DIMENSION == 1:
            global_shape = (3, cfg.Nx)  # [rho, rho*u, E]
        elif cfg.SPATIAL_DIMENSION == 2:
            global_shape = (4, cfg.Nx, cfg.Ny)  # [rho, rho*ux, rho*uy, E]
        else:  # 3D
            global_shape = (5, cfg.Nx, cfg.Ny, cfg.Nz)  # [rho, rho*ux, rho*uy, rho*uz, E]

        U_sharded = create_sharded_array_from_local(
            U_local, global_shape, mesh, cfg.SPATIAL_DIMENSION
        )

        if is_main_process:
            print("✓ Sharded array created from local pieces (no validation for large arrays)")
            print()
        return U_sharded

    if is_main_process:
        print("Single-process mode: generating global initial conditions...")

    U_global = initial_conditions(cfg, user=user)

    if is_main_process:
        print(f"Global shape: {U_global.shape}")
        print("Validating initial conditions...")
        validate_state(U_global, cfg.gamma, step=0, time=cfg.t_start)
        print("✓ Initial conditions validated successfully")
        print()
        print("Sharding global array to all devices...")

    return shard_array_to_devices(U_global, mesh, cfg.SPATIAL_DIMENSION)
