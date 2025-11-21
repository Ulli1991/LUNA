"""
GridPatch: The fundamental unit for domain decomposition.

A GridPatch represents a rectangular sub-domain with:
- Interior cells (actual computational domain)
- Ghost/halo cells (for boundary exchange)
- Metadata (size, spacing, neighbors, level)

Design principles:
- Self-contained: knows its own properties
- Agnostic: doesn't know about global structure
- Flexible: supports irregular neighbors (AMR-ready)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import jax.numpy as jnp


@dataclass
class GridPatch:
    """
    A rectangular patch of the computational domain.

    Attributes
    ----------
    rank : int
        GPU/process rank that owns this patch
    shape : Tuple[int, ...]
        Interior cell dimensions (Nx,) or (Nx, Ny) or (Nx, Ny, Nz)
        Does NOT include ghost cells
    dx : float
        Grid spacing in x-direction
    dy : float or None
        Grid spacing in y-direction (None for 1D)
    dz : float or None
        Grid spacing in z-direction (None for 1D/2D)
    global_offset : Tuple[int, ...]
        Position of this patch in global index space
        (ix_start,) or (ix_start, iy_start) or (ix_start, iy_start, iz_start)
    level : int
        Refinement level (0 = base grid, >0 for AMR)
        For fixed grid, always 0
    neighbors : Dict[str, List['GridPatch']]
        Neighbor patches by direction.
        Keys: 'left', 'right', 'bottom', 'top', 'back', 'front'
        Values: List of GridPatch (usually 1 for fixed grid, multiple for AMR)
    boundary_types : Dict[str, Optional[int]]
        Boundary condition type for each face (None if not a boundary)
        0=periodic, 1=zero-gradient, 2=reflective
    data : Optional[jnp.ndarray]
        The actual field data (with ghost cells)
        Shape: (nfields, Nx+2*ng, Ny+2*ng, Nz+2*ng) for 3D
        None until initialized
    n_ghost : int
        Number of ghost cell layers (default: 2 for PLM reconstruction)
    """

    rank: int
    shape: Tuple[int, ...]
    dx: float
    dy: Optional[float] = None
    dz: Optional[float] = None
    global_offset: Tuple[int, ...] = field(default_factory=tuple)
    level: int = 0
    neighbors: Dict[str, List['GridPatch']] = field(default_factory=dict)
    boundary_types: Dict[str, Optional[int]] = field(default_factory=dict)
    data: Optional[jnp.ndarray] = None
    n_ghost: int = 2

    def __post_init__(self):
        """Validate and setup after initialization."""
        # Validate dimensionality consistency
        ndim = len(self.shape)
        if ndim == 1:
            if self.dy is not None or self.dz is not None:
                raise ValueError("1D patch should have dy=None, dz=None")
        elif ndim == 2:
            if self.dy is None:
                raise ValueError("2D patch must have dy specified")
            if self.dz is not None:
                raise ValueError("2D patch should have dz=None")
        elif ndim == 3:
            if self.dy is None or self.dz is None:
                raise ValueError("3D patch must have dy and dz specified")
        else:
            raise ValueError(f"Invalid dimensionality: {ndim}")

        # Validate global_offset
        if len(self.global_offset) != ndim:
            if len(self.global_offset) == 0:
                # Auto-set to zero if not provided
                self.global_offset = tuple([0] * ndim)
            else:
                raise ValueError(
                    f"global_offset dimensionality {len(self.global_offset)} "
                    f"doesn't match shape dimensionality {ndim}"
                )

    @property
    def ndim(self) -> int:
        """Spatial dimensionality (1, 2, or 3)."""
        return len(self.shape)

    @property
    def shape_with_ghosts(self) -> Tuple[int, ...]:
        """Shape including ghost cells."""
        return tuple(n + 2*self.n_ghost for n in self.shape)

    @property
    def volume(self) -> float:
        """Physical volume of the patch (interior only)."""
        vol = self.dx * self.shape[0]
        if self.ndim >= 2:
            vol *= self.dy * self.shape[1]
        if self.ndim == 3:
            vol *= self.dz * self.shape[2]
        return vol

    @property
    def n_cells(self) -> int:
        """Number of interior cells."""
        result = self.shape[0]
        for i in range(1, self.ndim):
            result *= self.shape[i]
        return result

    def is_boundary(self, direction: str) -> bool:
        """
        Check if this patch has a domain boundary in the given direction.

        Parameters
        ----------
        direction : str
            One of: 'left', 'right', 'bottom', 'top', 'back', 'front'

        Returns
        -------
        bool
            True if this face is a domain boundary
        """
        return self.boundary_types.get(direction) is not None

    def get_interior_slice(self) -> Tuple[slice, ...]:
        """
        Get slice to extract interior cells from data array.

        Returns
        -------
        Tuple[slice, ...]
            Slice objects to extract interior (excludes ghost cells)
            Format: (slice(None), ...) for fields dimension, then spatial slices
        """
        ng = self.n_ghost
        spatial_slices = tuple(slice(ng, -ng) for _ in range(self.ndim))
        return (slice(None),) + spatial_slices  # slice(None) for field dimension

    def get_ghost_slice(self, direction: str, send: bool = True) -> Tuple[slice, ...]:
        """
        Get slice to extract ghost cell data for sending or receiving.

        Parameters
        ----------
        direction : str
            One of: 'left', 'right', 'bottom', 'top', 'back', 'front'
        send : bool
            If True, get the interior cells adjacent to boundary (for sending)
            If False, get the ghost cells (for receiving)

        Returns
        -------
        Tuple[slice, ...]
            Slice objects to extract the appropriate region
        """
        ng = self.n_ghost

        # Map direction to (axis, is_upper_boundary)
        direction_map = {
            'left':   (0, False),  # x-axis, lower
            'right':  (0, True),   # x-axis, upper
            'bottom': (1, False),  # y-axis, lower
            'top':    (1, True),   # y-axis, upper
            'back':   (2, False),  # z-axis, lower
            'front':  (2, True),   # z-axis, upper
        }

        if direction not in direction_map:
            raise ValueError(f"Invalid direction: {direction}")

        axis, is_upper = direction_map[direction]

        # Check if this direction makes sense for this dimensionality
        if axis >= self.ndim:
            raise ValueError(f"Direction '{direction}' invalid for {self.ndim}D patch")

        # Build slice tuple
        slices = [slice(None)]  # Field dimension

        for dim in range(self.ndim):
            if dim == axis:
                if send:
                    # Send: interior cells adjacent to boundary
                    if is_upper:
                        slices.append(slice(-2*ng, -ng))  # Last ng interior cells
                    else:
                        slices.append(slice(ng, 2*ng))    # First ng interior cells
                else:
                    # Receive: ghost cells
                    if is_upper:
                        slices.append(slice(-ng, None))   # Upper ghost cells
                    else:
                        slices.append(slice(None, ng))    # Lower ghost cells
            else:
                # For other dimensions, include all cells
                slices.append(slice(None))

        return tuple(slices)

    def allocate_data(self, n_fields: int, dtype=jnp.float32) -> None:
        """
        Allocate memory for field data including ghost cells.

        Parameters
        ----------
        n_fields : int
            Number of conserved fields (3 for 1D, 4 for 2D, 5 for 3D)
        dtype : dtype
            Data type (default: float32)
        """
        shape = (n_fields,) + self.shape_with_ghosts
        self.data = jnp.zeros(shape, dtype=dtype)

    def __repr__(self) -> str:
        """String representation."""
        spacing_str = f"dx={self.dx:.4e}"
        if self.dy is not None:
            spacing_str += f", dy={self.dy:.4e}"
        if self.dz is not None:
            spacing_str += f", dz={self.dz:.4e}"

        return (
            f"GridPatch(rank={self.rank}, shape={self.shape}, "
            f"{spacing_str}, level={self.level}, "
            f"offset={self.global_offset})"
        )
