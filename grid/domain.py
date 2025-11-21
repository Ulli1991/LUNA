"""
Domain: Manager for collection of GridPatches.

The Domain class orchestrates multiple patches and provides:
- Global properties (total size, physical extent)
- Iteration over patches
- Global operations (reductions, etc.)

For single-GPU: Domain contains 1 patch
For multi-GPU: Domain contains multiple patches
"""

from typing import List, Tuple, Optional
import jax.numpy as jnp
from grid.patch import GridPatch


class Domain:
    """
    Domain manages a collection of GridPatches.

    Attributes
    ----------
    patches : List[GridPatch]
        All patches in this domain (local to this process)
    global_shape : Tuple[int, ...]
        Global grid dimensions (total across all patches)
    global_extent : Tuple[float, ...]
        Physical size of domain (Lx, Ly, Lz)
    ndim : int
        Spatial dimensionality (1, 2, or 3)
    """

    def __init__(
        self,
        patches: List[GridPatch],
        global_shape: Tuple[int, ...],
        global_extent: Tuple[float, ...],
    ):
        """
        Initialize Domain.

        Parameters
        ----------
        patches : List[GridPatch]
            List of patches owned by this process
        global_shape : Tuple[int, ...]
            Total grid dimensions across all patches
        global_extent : Tuple[float, ...]
            Physical domain size (Lx,) or (Lx, Ly) or (Lx, Ly, Lz)
        """
        if not patches:
            raise ValueError("Domain must contain at least one patch")

        self.patches = patches
        self.global_shape = global_shape
        self.global_extent = global_extent

        # Validate dimensionality consistency
        self.ndim = len(global_shape)
        for patch in patches:
            if patch.ndim != self.ndim:
                raise ValueError(
                    f"Patch dimensionality {patch.ndim} doesn't match "
                    f"domain dimensionality {self.ndim}"
                )

        if len(global_extent) != self.ndim:
            raise ValueError(
                f"global_extent dimensionality {len(global_extent)} doesn't match "
                f"domain dimensionality {self.ndim}"
            )

    @property
    def n_patches(self) -> int:
        """Number of patches in this domain (local to this process)."""
        return len(self.patches)

    @property
    def total_cells(self) -> int:
        """Total number of cells across all patches (local)."""
        return sum(patch.n_cells for patch in self.patches)

    @property
    def total_volume(self) -> float:
        """Total physical volume across all patches (local)."""
        return sum(patch.volume for patch in self.patches)

    @property
    def dx(self) -> float:
        """Global grid spacing in x-direction (assumes uniform)."""
        return self.patches[0].dx

    @property
    def dy(self) -> Optional[float]:
        """Global grid spacing in y-direction (assumes uniform)."""
        return self.patches[0].dy if self.ndim >= 2 else None

    @property
    def dz(self) -> Optional[float]:
        """Global grid spacing in z-direction (assumes uniform)."""
        return self.patches[0].dz if self.ndim == 3 else None

    def get_patch_by_rank(self, rank: int) -> Optional[GridPatch]:
        """
        Get patch by rank.

        Parameters
        ----------
        rank : int
            Rank to search for

        Returns
        -------
        GridPatch or None
            The patch with given rank, or None if not found
        """
        for patch in self.patches:
            if patch.rank == rank:
                return patch
        return None

    def allocate_all(self, n_fields: int, dtype=jnp.float32) -> None:
        """
        Allocate data arrays for all patches.

        Parameters
        ----------
        n_fields : int
            Number of conserved fields
        dtype : dtype
            Data type (default: float32)
        """
        for patch in self.patches:
            patch.allocate_data(n_fields, dtype)

    def validate_decomposition(self) -> bool:
        """
        Validate that patches cover the domain correctly.

        Checks:
        - No overlaps
        - No gaps
        - Covers global domain

        Returns
        -------
        bool
            True if decomposition is valid

        Note
        ----
        This is a simplified check. Full validation would require
        global communication in multi-process setting.
        """
        # For now, just check that patch shapes are reasonable
        for patch in self.patches:
            if any(n <= 0 for n in patch.shape):
                return False

            # Check that global_offset + shape doesn't exceed global_shape
            for i in range(self.ndim):
                if patch.global_offset[i] + patch.shape[i] > self.global_shape[i]:
                    return False

        return True

    def compute_global_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """
        Compute physical bounds of the domain.

        Returns
        -------
        Tuple[Tuple[float, float], ...]
            ((xmin, xmax),) or ((xmin, xmax), (ymin, ymax)) or
            ((xmin, xmax), (ymin, ymax), (zmin, zmax))
        """
        bounds = []
        extents = [self.global_extent[i] for i in range(self.ndim)]

        for extent in extents:
            bounds.append((0.0, extent))

        return tuple(bounds)

    def __iter__(self):
        """Iterate over patches."""
        return iter(self.patches)

    def __len__(self) -> int:
        """Number of patches."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> GridPatch:
        """Get patch by index."""
        return self.patches[idx]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Domain(n_patches={self.n_patches}, "
            f"global_shape={self.global_shape}, "
            f"extent={self.global_extent}, "
            f"total_cells={self.total_cells})"
        )
