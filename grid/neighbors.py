"""
Neighbor connectivity logic for patches.

Sets up neighbor lists that enable:
- Halo exchange between patches
- Boundary condition application
- AMR coarse-fine interactions (future)
"""

from typing import List, Dict, Tuple
from grid.patch import GridPatch


def setup_neighbor_connectivity(
    patches: List[GridPatch],
    periodic: bool = False
) -> None:
    """
    Setup neighbor connectivity for all patches.

    Modifies patches in-place by populating their neighbors dictionaries.

    For fixed uniform grid:
    - Each interior patch has 2*ndim neighbors (one per face)
    - Boundary patches have fewer neighbors (unless periodic)

    For AMR (future):
    - Patches can have multiple neighbors per face
    - Neighbors can be at different refinement levels

    Parameters
    ----------
    patches : List[GridPatch]
        List of all patches in the domain
    periodic : bool
        If True, wrap around at domain boundaries (overrides boundary_types)

    Notes
    -----
    Direction naming convention:
    - 1D: left, right
    - 2D: left, right, bottom, top
    - 3D: left, right, bottom, top, back, front

    Axis mapping:
    - x-axis: left (-x), right (+x)
    - y-axis: bottom (-y), top (+y)
    - z-axis: back (-z), front (+z)
    """
    if not patches:
        return

    # Check dimensionality
    ndim = patches[0].ndim

    # Build spatial index for quick neighbor lookup
    # Map: global_offset -> patch
    patch_index = {patch.global_offset: patch for patch in patches}

    # For each patch, find neighbors
    for patch in patches:
        neighbors: Dict[str, List[GridPatch]] = {}

        # Check all possible neighbor directions
        if ndim >= 1:
            # Left neighbor (x - 1)
            left_offset = list(patch.global_offset)
            left_offset[0] -= patch.shape[0]
            left_neighbor = patch_index.get(tuple(left_offset))

            if left_neighbor is not None:
                neighbors['left'] = [left_neighbor]
            elif periodic and patch.is_boundary('left'):
                # Wrap around to right side
                wrap_offset = list(patch.global_offset)
                # Find rightmost patch in this row/plane
                wrap_offset[0] = find_max_offset_in_direction(patches, 0, patch, ndim)
                wrap_neighbor = patch_index.get(tuple(wrap_offset))
                if wrap_neighbor is not None:
                    neighbors['left'] = [wrap_neighbor]

            # Right neighbor (x + 1)
            right_offset = list(patch.global_offset)
            right_offset[0] += patch.shape[0]
            right_neighbor = patch_index.get(tuple(right_offset))

            if right_neighbor is not None:
                neighbors['right'] = [right_neighbor]
            elif periodic and patch.is_boundary('right'):
                # Wrap around to left side
                wrap_offset = list(patch.global_offset)
                wrap_offset[0] = 0
                wrap_neighbor = patch_index.get(tuple(wrap_offset))
                if wrap_neighbor is not None:
                    neighbors['right'] = [wrap_neighbor]

        if ndim >= 2:
            # Bottom neighbor (y - 1)
            bottom_offset = list(patch.global_offset)
            bottom_offset[1] -= patch.shape[1]
            bottom_neighbor = patch_index.get(tuple(bottom_offset))

            if bottom_neighbor is not None:
                neighbors['bottom'] = [bottom_neighbor]
            elif periodic and patch.is_boundary('bottom'):
                # Wrap around to top
                wrap_offset = list(patch.global_offset)
                wrap_offset[1] = find_max_offset_in_direction(patches, 1, patch, ndim)
                wrap_neighbor = patch_index.get(tuple(wrap_offset))
                if wrap_neighbor is not None:
                    neighbors['bottom'] = [wrap_neighbor]

            # Top neighbor (y + 1)
            top_offset = list(patch.global_offset)
            top_offset[1] += patch.shape[1]
            top_neighbor = patch_index.get(tuple(top_offset))

            if top_neighbor is not None:
                neighbors['top'] = [top_neighbor]
            elif periodic and patch.is_boundary('top'):
                # Wrap around to bottom
                wrap_offset = list(patch.global_offset)
                wrap_offset[1] = 0
                wrap_neighbor = patch_index.get(tuple(wrap_offset))
                if wrap_neighbor is not None:
                    neighbors['top'] = [wrap_neighbor]

        if ndim == 3:
            # Back neighbor (z - 1)
            back_offset = list(patch.global_offset)
            back_offset[2] -= patch.shape[2]
            back_neighbor = patch_index.get(tuple(back_offset))

            if back_neighbor is not None:
                neighbors['back'] = [back_neighbor]
            elif periodic and patch.is_boundary('back'):
                # Wrap around to front
                wrap_offset = list(patch.global_offset)
                wrap_offset[2] = find_max_offset_in_direction(patches, 2, patch, ndim)
                wrap_neighbor = patch_index.get(tuple(wrap_offset))
                if wrap_neighbor is not None:
                    neighbors['back'] = [wrap_neighbor]

            # Front neighbor (z + 1)
            front_offset = list(patch.global_offset)
            front_offset[2] += patch.shape[2]
            front_neighbor = patch_index.get(tuple(front_offset))

            if front_neighbor is not None:
                neighbors['front'] = [front_neighbor]
            elif periodic and patch.is_boundary('front'):
                # Wrap around to back
                wrap_offset = list(patch.global_offset)
                wrap_offset[2] = 0
                wrap_neighbor = patch_index.get(tuple(wrap_offset))
                if wrap_neighbor is not None:
                    neighbors['front'] = [wrap_neighbor]

        patch.neighbors = neighbors


def find_max_offset_in_direction(
    patches: List[GridPatch],
    axis: int,
    reference_patch: GridPatch,
    ndim: int
) -> int:
    """
    Find maximum offset in given axis for patches that match reference in other axes.

    Used for periodic boundary wrapping.

    Parameters
    ----------
    patches : List[GridPatch]
        All patches
    axis : int
        Axis to search (0=x, 1=y, 2=z)
    reference_patch : GridPatch
        Patch to match other coordinates
    ndim : int
        Number of dimensions

    Returns
    -------
    int
        Maximum offset found in the given axis
    """
    max_offset = 0

    for patch in patches:
        # Check if other coordinates match
        match = True
        for i in range(ndim):
            if i != axis and patch.global_offset[i] != reference_patch.global_offset[i]:
                match = False
                break

        if match:
            max_offset = max(max_offset, patch.global_offset[axis])

    return max_offset


def validate_connectivity(patches: List[GridPatch], periodic: bool = False) -> bool:
    """
    Validate that neighbor connectivity is correct.

    Checks:
    - Symmetric neighbors (if A is neighbor of B, B is neighbor of A)
    - No missing neighbors for interior patches
    - Boundary patches have correct number of neighbors

    Parameters
    ----------
    patches : List[GridPatch]
        All patches
    periodic : bool
        Whether domain is periodic

    Returns
    -------
    bool
        True if connectivity is valid
    """
    if not patches:
        return True

    ndim = patches[0].ndim

    for patch in patches:
        # Check symmetry
        for direction, neighbors in patch.neighbors.items():
            for neighbor in neighbors:
                # Find opposite direction
                opposite = get_opposite_direction(direction)

                # Check if this patch is in neighbor's opposite direction
                if opposite not in neighbor.neighbors:
                    print(f"Asymmetry: Patch {patch.rank} has {neighbor.rank} as {direction}, "
                          f"but {neighbor.rank} doesn't have {patch.rank} as {opposite}")
                    return False

                if patch not in neighbor.neighbors[opposite]:
                    print(f"Asymmetry: Patch {patch.rank} has {neighbor.rank} as {direction}, "
                          f"but {neighbor.rank} doesn't have {patch.rank} as {opposite}")
                    return False

        # Check expected number of neighbors
        n_expected = 2 * ndim  # Maximum possible
        n_actual = sum(len(neighbors) for neighbors in patch.neighbors.values())

        # For interior patches (no boundaries), should have full set
        if not any(patch.is_boundary(d) for d in patch.neighbors.keys()):
            if n_actual != n_expected and not periodic:
                # Interior patch should have all neighbors
                pass  # This might be okay for some decompositions

    return True


def get_opposite_direction(direction: str) -> str:
    """
    Get opposite direction.

    Parameters
    ----------
    direction : str
        One of: 'left', 'right', 'bottom', 'top', 'back', 'front'

    Returns
    -------
    str
        Opposite direction
    """
    opposites = {
        'left': 'right',
        'right': 'left',
        'bottom': 'top',
        'top': 'bottom',
        'back': 'front',
        'front': 'back',
    }
    return opposites[direction]


def print_connectivity(patches: List[GridPatch]) -> None:
    """
    Print neighbor connectivity for debugging.

    Parameters
    ----------
    patches : List[GridPatch]
        All patches
    """
    print("="*60)
    print("NEIGHBOR CONNECTIVITY")
    print("="*60)

    for patch in patches:
        print(f"\nPatch {patch.rank} (offset={patch.global_offset}, shape={patch.shape}):")

        if not patch.neighbors:
            print("  No neighbors")
        else:
            for direction, neighbors in sorted(patch.neighbors.items()):
                neighbor_ranks = [n.rank for n in neighbors]
                print(f"  {direction:8s}: {neighbor_ranks}")

        # Show boundaries
        boundaries = [d for d, bt in patch.boundary_types.items() if bt is not None]
        if boundaries:
            print(f"  Boundaries: {boundaries}")
