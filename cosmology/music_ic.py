"""Reader for MUSIC-style cosmological DM initial conditions (Arepo/Gadget HDF5).

MUSIC (Hahn & Abel 2011) writes Gadget/Arepo HDF5 ICs with a ``Header`` group and
one ``PartType{n}`` group per species; dark matter is ``PartType1``. We read the
DM positions, velocities, and masses plus the cosmology/box header so LUNA can
deposit them onto the mesh and solve gravity with the multigrid Poisson solver.

Conventions (Gadget/Arepo):
  * Header/BoxSize     -- comoving box size in the file's length unit.
  * Header/Time        -- the scale factor a of the IC (= 1/(1+z)).
  * Header/MassTable   -- per-species mass; if MassTable[1] == 0 a per-particle
                          ``PartType1/Masses`` dataset is present instead.
  * PartType1/Coordinates (N,3), Velocities (N,3), ParticleIDs (N).
  * Velocities are the Gadget peculiar velocity u = sqrt(a) dx/dt (relevant to
    the integrator later, not to the gravity solve).

``write_synthetic_dm_ic`` emits a small, format-correct file (a Zel'dovich-
displaced particle grid) as a stand-in for real MUSIC output, so the reader and
the PM pipeline can be exercised before wiring up MUSIC.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DMICs:
    """Dark-matter initial conditions loaded from a MUSIC/Gadget HDF5 file."""
    pos: np.ndarray        # (N, 3) comoving positions, in the file's length unit
    vel: np.ndarray        # (N, 3) Gadget velocities (sqrt(a) dx/dt)
    mass: np.ndarray       # (N,) particle masses
    box_size: float        # comoving box size
    a: float               # scale factor of the IC
    redshift: float
    Omega_m: float
    Omega_L: float
    h: float               # H0 / (100 km/s/Mpc)

    @property
    def n_part(self) -> int:
        return self.pos.shape[0]


def read_music_dm_ic(path: str, part_type: int = 1) -> DMICs:
    """Read DM (``PartType{part_type}``) ICs from a MUSIC/Gadget Arepo-HDF5 file."""
    import h5py

    with h5py.File(path, "r") as f:
        H = dict(f["Header"].attrs)
        box = float(np.asarray(H["BoxSize"]).reshape(-1)[0])
        a = float(H.get("Time", 1.0))
        z = float(H.get("Redshift", 1.0 / a - 1.0))
        Om = float(H.get("Omega0", H.get("OmegaMatter", 0.3)))
        OL = float(H.get("OmegaLambda", 0.7))
        hub = float(H.get("HubbleParam", 0.7))

        g = f[f"PartType{part_type}"]
        pos = np.asarray(g["Coordinates"], dtype=np.float64)
        vel = (np.asarray(g["Velocities"], dtype=np.float64)
               if "Velocities" in g else np.zeros_like(pos))
        n = pos.shape[0]
        if "Masses" in g:
            mass = np.asarray(g["Masses"], dtype=np.float64)
        else:                                          # uniform mass from the table
            mtable = np.asarray(H["MassTable"], dtype=np.float64)
            mass = np.full(n, mtable[part_type], dtype=np.float64)

    return DMICs(pos=pos, vel=vel, mass=mass, box_size=box, a=a, redshift=z,
                 Omega_m=Om, Omega_L=OL, h=hub)


def write_synthetic_dm_ic(path: str, Np: int = 32, box: float = 1.0,
                          a: float = 0.02, Omega_m: float = 0.3,
                          Omega_L: float = 0.7, h: float = 0.7,
                          amp: float = 0.15) -> str:
    """Write a small, format-correct Gadget/Arepo-HDF5 DM IC -- a Zel'dovich-
    displaced ``Np^3`` particle grid (single x-mode), as a stand-in for real
    MUSIC output. Mass is carried in the Header MassTable (no Masses dataset),
    exercising that reader path."""
    import h5py

    q = (np.arange(Np) + 0.5) * (box / Np)
    QX, QY, QZ = np.meshgrid(q, q, q, indexing="ij")
    k = 2.0 * np.pi / box
    psi = amp * np.sin(k * QX) / k                     # Zel'dovich displacement (x)
    X = (QX + psi) % box
    pos = np.stack([X, QY, QZ], axis=-1).reshape(-1, 3).astype(np.float32)
    vel = np.zeros_like(pos)                           # placeholder (unused by gravity)
    n = pos.shape[0]
    m_part = Omega_m * box ** 3 / n                    # mean matter density * V / N

    with h5py.File(path, "w") as f:
        H = f.create_group("Header")
        H.attrs["BoxSize"] = box
        H.attrs["NumPart_ThisFile"] = np.array([0, n, 0, 0, 0, 0], dtype=np.int64)
        H.attrs["NumPart_Total"] = np.array([0, n, 0, 0, 0, 0], dtype=np.uint32)
        H.attrs["MassTable"] = np.array([0, m_part, 0, 0, 0, 0], dtype=np.float64)
        H.attrs["Time"] = a
        H.attrs["Redshift"] = 1.0 / a - 1.0
        H.attrs["Omega0"] = Omega_m
        H.attrs["OmegaLambda"] = Omega_L
        H.attrs["HubbleParam"] = h
        H.attrs["NumFilesPerSnapshot"] = 1
        p1 = f.create_group("PartType1")
        p1.create_dataset("Coordinates", data=pos)
        p1.create_dataset("Velocities", data=vel)
        p1.create_dataset("ParticleIDs", data=np.arange(n, dtype=np.uint64))
    return path


# Arepo/Gadget unit system: length kpc/h, velocity km/s, mass 1e10 Msun/h.
G_AREPO = 43007.1      # G in kpc/h (km/s)^2 / (1e10 Msun/h)


def music_cosmology(ic: DMICs):
    """A cosmology.Cosmology built from the IC header."""
    from cosmology import Cosmology
    return Cosmology(Omega_m=ic.Omega_m, Omega_L=ic.Omega_L, h=ic.h)


def music_run_config(ic: DMICs, Nx: int, base=None):
    """Build a LUNA Config for a cosmological DM run from a loaded MUSIC IC.

    The box size, cosmology (Omega_m, Omega_L, H0), initial scale factor,
    periodic boundary, and the Arepo unit constants are all taken from the IC
    header -- so a cosmo run is configured AT STARTUP straight from the ICs,
    nothing hardcoded at the call site. Units are Gadget/Arepo (length kpc/h,
    velocity km/s, mass 1e10 Msun/h); H0 = 0.1 h km/s/kpc in that system.
    """
    from dataclasses import replace
    from template_config import Config
    cfg = Config() if base is None else base
    return replace(
        cfg,
        SPATIAL_DIMENSION=3, Nx=Nx, Ny=Nx, Nz=Nx,
        Lx=ic.box_size, Ly=ic.box_size, Lz=ic.box_size,
        BOUNDARY=0,                                    # periodic cosmological box
        CosmologicalIntegration=True,
        cosmo_Omega_m=ic.Omega_m, cosmo_Omega_L=ic.Omega_L,
        cosmo_H0=0.1 * ic.h,                           # km/s/kpc (Arepo units)
        a_init=ic.a,
        GRAVITATIONAL_G=G_AREPO,
    )
