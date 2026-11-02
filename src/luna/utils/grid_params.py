"""Frozen dataclass of grid dimensions, spacings and their inverses."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridParams:
    """Grid dimensions, domain sizes, spacings and their inverses, dimension and cell count."""

    Nx: int
    Ny: int
    Nz: int

    Lx: float
    Ly: float
    Lz: float

    dx: float
    dy: float
    dz: float

    inv_dx: float
    inv_dy: float
    inv_dz: float

    spatial_dimension: int
    total_cells: int

    @classmethod
    def from_config(cls, cfg):
        """Build GridParams from a Config."""
        dx = cfg.Lx / cfg.Nx
        dy = cfg.Ly / cfg.Ny
        dz = cfg.Lz / cfg.Nz

        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy
        inv_dz = 1.0 / dz

        if cfg.SPATIAL_DIMENSION == 1:
            total_cells = cfg.Nx
        elif cfg.SPATIAL_DIMENSION == 2:
            total_cells = cfg.Nx * cfg.Ny
        else:
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
            total_cells=total_cells,
        )


def _require_cubic_cells(cfg, who):
    """Raise unless dx == dy (== dz in 3D): `who` uses Lx/Nx for every axis."""
    dx = cfg.Lx / cfg.Nx
    dy = cfg.Ly / cfg.Ny
    dz = cfg.Lz / cfg.Nz if cfg.SPATIAL_DIMENSION == 3 else dx
    if abs(dy - dx) > 1e-12 * dx or abs(dz - dx) > 1e-12 * dx:
        raise ValueError(f"{who} assumes cubic cells: dx={dx:g}, dy={dy:g}, dz={dz:g}")
