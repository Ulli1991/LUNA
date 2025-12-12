"""
Grid parameters dataclass.

This module defines a frozen dataclass that bundles all grid-related parameters
and pre-computed constants for efficient parameter passing and memory usage.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GridParams:
    """
    Immutable container for grid parameters and pre-computed constants.

    Attributes
    ----------
    Nx, Ny, Nz : int
        Grid dimensions
    Lx, Ly, Lz : float
        Domain sizes
    dx, dy, dz : float
        Grid spacings
    inv_dx, inv_dy, inv_dz : float
        Inverse grid spacings (for fast multiplication instead of division)
    spatial_dimension : int
        Number of spatial dimensions (1, 2, or 3)
    total_cells : int
        Total number of cells in the grid
    """
    # Grid dimensions
    Nx: int
    Ny: int
    Nz: int

    # Domain sizes
    Lx: float
    Ly: float
    Lz: float

    # Grid spacings
    dx: float
    dy: float
    dz: float

    # Inverse grid spacings (pre-computed for performance)
    inv_dx: float
    inv_dy: float
    inv_dz: float

    # Metadata
    spatial_dimension: int
    total_cells: int

    @classmethod
    def from_config(cls, cfg):
        """
        Create GridParams from a Config object.

        Parameters
        ----------
        cfg : Config
            Configuration object

        Returns
        -------
        GridParams
            Initialized grid parameters
        """
        # Compute grid spacings
        dx = cfg.Lx / cfg.Nx
        dy = cfg.Ly / cfg.Ny
        dz = cfg.Lz / cfg.Nz

        # Compute inverse grid spacings
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_dz = 1.0 / dz

        # Compute total cells
        if cfg.SPATIAL_DIMENSION == 1:
            total_cells = cfg.Nx
        elif cfg.SPATIAL_DIMENSION == 2:
            total_cells = cfg.Nx * cfg.Ny
        else:  # 3D
            total_cells = cfg.Nx * cfg.Ny * cfg.Nz

        return cls(
            Nx=cfg.Nx,
            Ny=cfg.Ny,
            Nz=cfg.Nz,
            Lx=cfg.Lx,
            Ly=cfg.Ly,
            Lz=cfg.Lz,
            dx=dx,
            dy=dy,
            dz=dz,
            inv_dx=inv_dx,
            inv_dy=inv_dy,
            inv_dz=inv_dz,
            spatial_dimension=cfg.SPATIAL_DIMENSION,
            total_cells=total_cells
        )
