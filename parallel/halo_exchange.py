"""
Halo exchange (ghost cell communication) between patches.

This module handles:
- Copying boundary data between neighbor patches
- Applying boundary conditions at domain edges
- Multi-GPU communication (using JAX primitives)

Design:
- Single-GPU: Direct memory copy between patches
- Multi-GPU: Device-to-device transfer via JAX
- Multi-node: MPI communication (future)
"""

from typing import List, Optional
import jax
import jax.numpy as jnp
from grid.patch import GridPatch
from grid.domain import Domain


def exchange_halos(
    domain: Domain,
    devices: Optional[List[jax.Device]] = None
) -> None:
    """
    Exchange ghost cells between all neighboring patches.

    This function updates the ghost cell regions of each patch with data
    from neighboring patches' interior cells.

    Parameters
    ----------
    domain : Domain
        The domain containing all patches
    devices : List[jax.Device], optional
        List of devices (GPUs) to use. If None, uses single device.

    Notes
    -----
    Communication pattern:
    1. For each patch, extract boundary data to send
    2. Transfer data to neighbor patches' ghost regions
    3. Handle this in batches to overlap communication

    For single-GPU: this is just memory copying
    For multi-GPU: this uses JAX device transfers
    """
    if devices is None:
        # Single-GPU case: direct memory operations
        _exchange_halos_single_gpu(domain)
    else:
        # Multi-GPU case: device-to-device transfers
        _exchange_halos_multi_gpu(domain, devices)


def _exchange_halos_single_gpu(domain: Domain) -> None:
    """
    Exchange halos for single-GPU case (direct memory copy).

    Parameters
    ----------
    domain : Domain
        The domain containing all patches
    """
    for patch in domain.patches:
        for direction, neighbors in patch.neighbors.items():
            for neighbor in neighbors:
                # Extract data to send (interior cells adjacent to boundary)
                send_slice = patch.get_ghost_slice(direction, send=True)
                send_data = patch.data[send_slice]

                # Get opposite direction for receiving
                opposite = _get_opposite_direction(direction)

                # Get slice for receiving ghost cells
                recv_slice = neighbor.get_ghost_slice(opposite, send=False)

                # Copy data
                # Need to update neighbor.data in-place
                neighbor.data = neighbor.data.at[recv_slice].set(send_data)


def _exchange_halos_multi_gpu(domain: Domain, devices: List[jax.Device]) -> None:
    """
    Exchange halos for multi-GPU case (device-to-device transfers).

    Parameters
    ----------
    domain : Domain
        The domain containing all patches (local to this process)
    devices : List[jax.Device]
        List of devices being used

    Notes
    -----
    In multi-GPU setup, each patch should already be placed on its
    designated device. This function transfers boundary data between devices.
    """
    # Build list of all transfers to perform
    transfers = []

    for patch in domain.patches:
        for direction, neighbors in patch.neighbors.items():
            for neighbor in neighbors:
                # Extract data to send
                send_slice = patch.get_ghost_slice(direction, send=True)
                send_data = patch.data[send_slice]

                # Get device for neighbor
                neighbor_device = devices[neighbor.rank] if neighbor.rank < len(devices) else devices[0]

                # Transfer to neighbor device
                send_data_on_device = jax.device_put(send_data, neighbor_device)

                opposite = _get_opposite_direction(direction)
                recv_slice = neighbor.get_ghost_slice(opposite, send=False)

                transfers.append((neighbor, recv_slice, send_data_on_device))

    # Apply all transfers
    for neighbor, recv_slice, data in transfers:
        neighbor.data = neighbor.data.at[recv_slice].set(data)


def apply_boundary_conditions(
    patch: GridPatch,
    boundary_type: int
) -> None:
    """
    Apply boundary conditions to ghost cells at domain boundaries.

    This should be called AFTER halo exchange, only for patches at domain edges.

    Parameters
    ----------
    patch : GridPatch
        The patch to apply boundary conditions to
    boundary_type : int
        Type of boundary condition:
        0 = Periodic (handled by halo exchange with wrapping)
        1 = Zero-gradient (copy interior values to ghost cells)
        2 = Reflective (mirror interior values, flip velocity)

    Notes
    -----
    For interior patches (no domain boundaries), this does nothing.
    For patches at domain edges, fills ghost cells appropriately.
    """
    if patch.data is None:
        raise ValueError("Patch data not allocated")

    ndim = patch.ndim
    ng = patch.n_ghost

    # Apply boundary conditions for each face that is a domain boundary
    for direction, bc_type in patch.boundary_types.items():
        if bc_type is None:
            continue  # Not a boundary

        if bc_type == 0:
            # Periodic: handled by halo exchange with neighbor connectivity
            # If periodic BC is requested but no neighbor exists, that's an error
            if direction not in patch.neighbors or not patch.neighbors[direction]:
                raise ValueError(
                    f"Periodic BC requested for {direction} but no neighbor exists"
                )

        elif bc_type == 1:
            # Zero-gradient (Neumann): copy interior to ghost
            _apply_zero_gradient(patch, direction)

        elif bc_type == 2:
            # Reflective: mirror and potentially flip velocity
            _apply_reflective(patch, direction)

        else:
            raise ValueError(f"Unknown boundary type: {bc_type}")


def _apply_zero_gradient(patch: GridPatch, direction: str) -> None:
    """
    Apply zero-gradient boundary condition.

    Copies the interior cells adjacent to boundary into ghost cells.

    Parameters
    ----------
    patch : GridPatch
        The patch
    direction : str
        Which boundary face
    """
    ng = patch.n_ghost
    ndim = patch.ndim

    # Map direction to axis and side
    direction_map = {
        'left': (0, False), 'right': (0, True),
        'bottom': (1, False), 'top': (1, True),
        'back': (2, False), 'front': (2, True),
    }

    axis, is_upper = direction_map[direction]

    # Build slices for ghost region and interior edge
    if ndim == 1:
        if is_upper:
            ghost_slice = (slice(None), slice(-ng, None))
            interior_slice = (slice(None), slice(-ng-1, -ng))
        else:
            ghost_slice = (slice(None), slice(None, ng))
            interior_slice = (slice(None), slice(ng, ng+1))

    elif ndim == 2:
        if axis == 0:  # x-axis
            if is_upper:
                ghost_slice = (slice(None), slice(-ng, None), slice(None))
                interior_slice = (slice(None), slice(-ng-1, -ng), slice(None))
            else:
                ghost_slice = (slice(None), slice(None, ng), slice(None))
                interior_slice = (slice(None), slice(ng, ng+1), slice(None))
        else:  # y-axis
            if is_upper:
                ghost_slice = (slice(None), slice(None), slice(-ng, None))
                interior_slice = (slice(None), slice(None), slice(-ng-1, -ng))
            else:
                ghost_slice = (slice(None), slice(None), slice(None, ng))
                interior_slice = (slice(None), slice(None), slice(ng, ng+1))

    elif ndim == 3:
        # Build slice for each axis
        # For non-boundary dimensions, use slice(None) to fill ENTIRE ghost slab
        # including edges/corners. Values in edge/corner cells will be filled
        # by subsequent boundary applications in other directions.
        slices_ghost = [slice(None)]  # Field dimension
        slices_interior = [slice(None)]

        for d in range(ndim):
            if d == axis:
                if is_upper:
                    slices_ghost.append(slice(-ng, None))
                    slices_interior.append(slice(-ng-1, -ng))
                else:
                    slices_ghost.append(slice(None, ng))
                    slices_interior.append(slice(ng, ng+1))
            else:
                # Use all cells (including ghost) so edges/corners get filled
                slices_ghost.append(slice(None))
                slices_interior.append(slice(None))

        ghost_slice = tuple(slices_ghost)
        interior_slice = tuple(slices_interior)

    # Copy interior edge to all ghost layers
    interior_values = patch.data[interior_slice]

    # Broadcast to fill all ghost cells
    for i in range(ng):
        if ndim == 1:
            if is_upper:
                patch.data = patch.data.at[:, -ng+i].set(interior_values[:, 0])
            else:
                patch.data = patch.data.at[:, i].set(interior_values[:, 0])
        elif ndim == 2:
            if axis == 0:
                if is_upper:
                    patch.data = patch.data.at[:, -ng+i, :].set(interior_values[:, 0, :])
                else:
                    patch.data = patch.data.at[:, i, :].set(interior_values[:, 0, :])
            else:
                if is_upper:
                    patch.data = patch.data.at[:, :, -ng+i].set(interior_values[:, :, 0])
                else:
                    patch.data = patch.data.at[:, :, i].set(interior_values[:, :, 0])
        elif ndim == 3:
            # Fill each ghost layer for 3D - fill ENTIRE slab including edges/corners
            if axis == 0:  # x-boundary
                if is_upper:
                    patch.data = patch.data.at[:, -ng+i, :, :].set(interior_values[:, 0, :, :])
                else:
                    patch.data = patch.data.at[:, i, :, :].set(interior_values[:, 0, :, :])
            elif axis == 1:  # y-boundary
                if is_upper:
                    patch.data = patch.data.at[:, :, -ng+i, :].set(interior_values[:, :, 0, :])
                else:
                    patch.data = patch.data.at[:, :, i, :].set(interior_values[:, :, 0, :])
            else:  # z-boundary (axis == 2)
                if is_upper:
                    patch.data = patch.data.at[:, :, :, -ng+i].set(interior_values[:, :, :, 0])
                else:
                    patch.data = patch.data.at[:, :, :, i].set(interior_values[:, :, :, 0])


def _apply_reflective(patch: GridPatch, direction: str) -> None:
    """
    Apply reflective boundary condition.

    Mirrors interior cells into ghost cells and flips normal velocity component.

    Parameters
    ----------
    patch : GridPatch
        The patch
    direction : str
        Which boundary face

    Notes
    -----
    For reflective BC:
    - Density, pressure, tangential velocities: mirrored
    - Normal velocity: mirrored and negated
    """
    ng = patch.n_ghost
    ndim = patch.ndim

    # Map direction to axis
    direction_map = {
        'left': (0, False), 'right': (0, True),
        'bottom': (1, False), 'top': (1, True),
        'back': (2, False), 'front': (2, True),
    }

    axis, is_upper = direction_map[direction]

    # Determine which field is the normal momentum
    # Field ordering: [rho, rho*vx, rho*vy, rho*vz, E] (3D)
    #                 [rho, rho*vx, rho*vy, E]          (2D)
    #                 [rho, rho*vx, E]                  (1D)
    mom_field = 1 + axis  # Momentum field for this axis

    # Mirror interior into ghost cells
    if ndim == 1:
        if is_upper:
            # Right boundary
            for i in range(ng):
                ghost_idx = -ng + i
                interior_idx = -ng - 1 - i
                patch.data = patch.data.at[:, ghost_idx].set(patch.data[:, interior_idx])
                # Flip normal velocity
                patch.data = patch.data.at[mom_field, ghost_idx].set(-patch.data[mom_field, interior_idx])
        else:
            # Left boundary
            for i in range(ng):
                ghost_idx = i
                interior_idx = 2*ng - 1 - i
                patch.data = patch.data.at[:, ghost_idx].set(patch.data[:, interior_idx])
                # Flip normal velocity
                patch.data = patch.data.at[mom_field, ghost_idx].set(-patch.data[mom_field, interior_idx])

    elif ndim == 2:
        # 2D reflective boundary
        if axis == 0:  # x-boundary
            if is_upper:
                for i in range(ng):
                    ghost_idx = -ng + i
                    interior_idx = -ng - 1 - i
                    patch.data = patch.data.at[:, ghost_idx, :].set(patch.data[:, interior_idx, :])
                    patch.data = patch.data.at[mom_field, ghost_idx, :].set(-patch.data[mom_field, interior_idx, :])
            else:
                for i in range(ng):
                    ghost_idx = i
                    interior_idx = 2*ng - 1 - i
                    patch.data = patch.data.at[:, ghost_idx, :].set(patch.data[:, interior_idx, :])
                    patch.data = patch.data.at[mom_field, ghost_idx, :].set(-patch.data[mom_field, interior_idx, :])
        else:  # y-boundary (axis == 1)
            if is_upper:
                for i in range(ng):
                    ghost_idx = -ng + i
                    interior_idx = -ng - 1 - i
                    patch.data = patch.data.at[:, :, ghost_idx].set(patch.data[:, :, interior_idx])
                    patch.data = patch.data.at[mom_field, :, ghost_idx].set(-patch.data[mom_field, :, interior_idx])
            else:
                for i in range(ng):
                    ghost_idx = i
                    interior_idx = 2*ng - 1 - i
                    patch.data = patch.data.at[:, :, ghost_idx].set(patch.data[:, :, interior_idx])
                    patch.data = patch.data.at[mom_field, :, ghost_idx].set(-patch.data[mom_field, :, interior_idx])

    elif ndim == 3:
        # 3D reflective boundary - fill ENTIRE slab including edges/corners
        if axis == 0:  # x-boundary
            if is_upper:
                for i in range(ng):
                    ghost_idx = -ng + i
                    interior_idx = -ng - 1 - i
                    patch.data = patch.data.at[:, ghost_idx, :, :].set(patch.data[:, interior_idx, :, :])
                    patch.data = patch.data.at[mom_field, ghost_idx, :, :].set(-patch.data[mom_field, interior_idx, :, :])
            else:
                for i in range(ng):
                    ghost_idx = i
                    interior_idx = 2*ng - 1 - i
                    patch.data = patch.data.at[:, ghost_idx, :, :].set(patch.data[:, interior_idx, :, :])
                    patch.data = patch.data.at[mom_field, ghost_idx, :, :].set(-patch.data[mom_field, interior_idx, :, :])
        elif axis == 1:  # y-boundary
            if is_upper:
                for i in range(ng):
                    ghost_idx = -ng + i
                    interior_idx = -ng - 1 - i
                    patch.data = patch.data.at[:, :, ghost_idx, :].set(patch.data[:, :, interior_idx, :])
                    patch.data = patch.data.at[mom_field, :, ghost_idx, :].set(-patch.data[mom_field, :, interior_idx, :])
            else:
                for i in range(ng):
                    ghost_idx = i
                    interior_idx = 2*ng - 1 - i
                    patch.data = patch.data.at[:, :, ghost_idx, :].set(patch.data[:, :, interior_idx, :])
                    patch.data = patch.data.at[mom_field, :, ghost_idx, :].set(-patch.data[mom_field, :, interior_idx, :])
        else:  # z-boundary (axis == 2)
            if is_upper:
                for i in range(ng):
                    ghost_idx = -ng + i
                    interior_idx = -ng - 1 - i
                    patch.data = patch.data.at[:, :, :, ghost_idx].set(patch.data[:, :, :, interior_idx])
                    patch.data = patch.data.at[mom_field, :, :, ghost_idx].set(-patch.data[mom_field, :, :, interior_idx])
            else:
                for i in range(ng):
                    ghost_idx = i
                    interior_idx = 2*ng - 1 - i
                    patch.data = patch.data.at[:, :, :, ghost_idx].set(patch.data[:, :, :, interior_idx])
                    patch.data = patch.data.at[mom_field, :, :, ghost_idx].set(-patch.data[mom_field, :, :, interior_idx])


def _get_opposite_direction(direction: str) -> str:
    """Get opposite direction for halo exchange."""
    opposites = {
        'left': 'right', 'right': 'left',
        'bottom': 'top', 'top': 'bottom',
        'back': 'front', 'front': 'back',
    }
    return opposites[direction]


def update_ghosts(domain: Domain, boundary_type: int, devices: Optional[List[jax.Device]] = None) -> None:
    """
    Convenience function to update all ghost cells.

    IMPORTANT: Order matters for correctness at domain boundaries!

    Performs:
    1. Halo exchange (fills inter-patch ghost cells from neighbors)
    2. Boundary conditions (fills domain edge ghost cells)

    This order ensures:
    - Interior patches get fresh neighbor data first
    - Domain boundary ghost cells are filled AFTER exchange (won't be overwritten)
    - Corner cells in 2D/3D are properly handled (face exchanges fill corners,
      then domain BC overwrites only the actual boundary corners)

    Parameters
    ----------
    domain : Domain
        The domain
    boundary_type : int
        Boundary condition type for domain edges
    devices : List[jax.Device], optional
        Devices for multi-GPU

    Examples
    --------
    >>> # Single GPU
    >>> update_ghosts(domain, boundary_type=1)
    >>>
    >>> # Multi GPU
    >>> devices = jax.devices()
    >>> update_ghosts(domain, boundary_type=0, devices=devices)
    """
    # Step 1: Exchange halos between patches FIRST
    exchange_halos(domain, devices)

    # Step 2: Apply boundary conditions at domain edges AFTER
    for patch in domain.patches:
        apply_boundary_conditions(patch, boundary_type)
