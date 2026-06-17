"""Gravitational coupling between the mesh gas and point particles (sinks/stars).

Lives in the gravity package next to the multigrid Poisson solver because this
*is* LUNA gravity physics — it reuses `solve_poisson_fmg` (the same MG/FMG solver
used for gas self-gravity). The particle <-> mesh coupling needs two grid
potentials (each a multigrid Poisson solve):

    phi_gas  = MG( rho_gas )                  gas self-gravity, AND the field
                                              particles feel from the gas
    phi_star = MG( CIC-deposit(particles) )   the field the GAS feels from the
                                              particles  ("Particle-Mesh": the
                                              point masses are smeared to a grid
                                              density first, since the MG solver
                                              acts on a density field, not on
                                              point masses)

Particle <-> particle gravity is deliberately NOT done here — for a coupled run
that is the external N-body code's job (ph4 / PeTar), so LUNA never does the
O(N^2) star-star sum. Gas->particle sampling uses the gas-only phi_gas, so
particles feel no self-force. All grid ops keep a fixed shape and all particle
ops are vmapped over the particle axis (no Python loop), so nothing here forces
an XLA recompile as particles are created/destroyed.
"""

import jax
import jax.numpy as jnp

from gravity.poisson_bc import make_halo_3d
from gravity.poisson_multigrid import _max_levels, solve_poisson_fmg


class ParticleMeshGravity:
    """Gas<->particle gravity built from a config; uses LUNA's MG Poisson."""

    def __init__(self, cfg):
        if cfg.SPATIAL_DIMENSION != 3:
            raise ValueError("ParticleMeshGravity requires SPATIAL_DIMENSION == 3")
        self.cfg = cfg
        self.Nx, self.Ny, self.Nz = cfg.Nx, cfg.Ny, cfg.Nz
        self.dx = cfg.Lx / cfg.Nx
        self.dV = self.dx ** 3
        self.inv_2dx = 0.5 / self.dx
        self.inv_12dx = 1.0 / (12.0 * self.dx)   # 4th-order gradient prefactor
        self.G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))
        self.fourpiG = 4.0 * jnp.pi * self.G

        pb = getattr(cfg, "POISSON_BOUNDARY", None)
        self.boundary = int(cfg.BOUNDARY if pb is None else pb)
        self.halo_fn = make_halo_3d(self.boundary)
        self._poisson_kw = dict(
            n_levels=int(getattr(cfg, "POISSON_N_LEVELS", None) or _max_levels(cfg.Nx)),
            n_vcycles_per_level=getattr(cfg, "POISSON_FMG_VCYCLES_PER_LEVEL", 1),
            n_extra_vcycles=getattr(cfg, "POISSON_FMG_EXTRA_VCYCLES", 0),
            n_pre=getattr(cfg, "POISSON_N_PRE", 2),
            n_post=getattr(cfg, "POISSON_N_POST", 2),
            n_coarsest=getattr(cfg, "POISSON_N_COARSEST", 50),
        )

        # cell-centre coordinate axes (code units)
        xc = (jnp.arange(cfg.Nx) + 0.5) * self.dx
        yc = (jnp.arange(cfg.Ny) + 0.5) * (cfg.Ly / cfg.Ny)
        zc = (jnp.arange(cfg.Nz) + 0.5) * (cfg.Lz / cfg.Nz)
        self.X, self.Y, self.Z = jnp.meshgrid(xc, yc, zc, indexing="ij")

    # -- potential (MG Poisson) + acceleration field ------------------------
    def gas_potential(self, rho):
        """FMG-multigrid solve of ∇²φ = 4πG ρ for a density field ρ."""
        rho_src = rho - jnp.mean(rho) if self.boundary == 0 else rho
        return solve_poisson_fmg(rho_src, self.dx, self.fourpiG,
                                 boundary=self.boundary, **self._poisson_kw)

    def _pad2(self, phi):
        """Two-cell halo for the 4th-order gradient stencil."""
        if self.boundary == 0:            # periodic
            mode = "wrap"
        elif self.boundary in (1, 2):     # Neumann / zero-gradient
            mode = "edge"
        else:
            raise NotImplementedError(
                "4th-order grad_phi supports periodic (0) or Neumann (1/2) "
                f"boundaries; got boundary={self.boundary}. Use order=2 for "
                "isolated (3).")
        return jnp.pad(phi, ((2, 2), (2, 2), (2, 2)), mode=mode)

    def grad_phi(self, phi, order=2):
        """-∇φ on the grid (the gravitational acceleration field).

        order=2 (default): 2nd-order centred difference, 1-cell halo, O(Δx²).
        order=4: 4th-order centred difference, 2-cell halo,
            ∂φ/∂x ≈ [φ(i-2) − 8φ(i-1) + 8φ(i+1) − φ(i+2)] / (12Δx),
        cutting the force error to O(Δx⁴) and reducing grid-scale force
        anisotropy -- the recommended PM force for the cosmological DM particles.
        Periodic / Neumann boundaries only. The stencil is anti-symmetric, so a
        CIC-deposit / CIC-gather pair stays self-force-free.
        """
        if order == 4:
            pp = self._pad2(phi)
            c = self.inv_12dx
            gx = (pp[4:, 2:-2, 2:-2] - 8.0 * pp[3:-1, 2:-2, 2:-2]
                  + 8.0 * pp[1:-3, 2:-2, 2:-2] - pp[:-4, 2:-2, 2:-2]) * c
            gy = (pp[2:-2, 4:, 2:-2] - 8.0 * pp[2:-2, 3:-1, 2:-2]
                  + 8.0 * pp[2:-2, 1:-3, 2:-2] - pp[2:-2, :-4, 2:-2]) * c
            gz = (pp[2:-2, 2:-2, 4:] - 8.0 * pp[2:-2, 2:-2, 3:-1]
                  + 8.0 * pp[2:-2, 2:-2, 1:-3] - pp[2:-2, 2:-2, :-4]) * c
            return gx, gy, gz                 # already -∇φ (force)
        pp = self.halo_fn(phi)
        gx = -(pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) * self.inv_2dx
        gy = -(pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) * self.inv_2dx
        gz = -(pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) * self.inv_2dx
        return gx, gy, gz

    def gas_accel_grid(self, rho):
        """Gravitational acceleration field of a density: -∇(MG(ρ))."""
        return self.grad_phi(self.gas_potential(rho))

    # -- trilinear (CIC) interpolation, vectorized over particles -----------
    def _interp_one(self, field, p):
        """Trilinear interpolation of `field` at a single position `p`."""
        fx = p[0] / self.dx - 0.5
        fy = p[1] / self.dx - 0.5
        fz = p[2] / self.dx - 0.5
        i0 = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, self.Nx - 2)
        j0 = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, self.Ny - 2)
        k0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, self.Nz - 2)
        tx = jnp.clip(fx - i0, 0.0, 1.0)
        ty = jnp.clip(fy - j0, 0.0, 1.0)
        tz = jnp.clip(fz - k0, 0.0, 1.0)
        out = 0.0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = ((tx if di else 1 - tx) * (ty if dj else 1 - ty)
                         * (tz if dk else 1 - tz))
                    out = out + w * field[i0 + di, j0 + dj, k0 + dk]
        return out

    def sample_at_points(self, field, positions):
        """Interpolate a scalar grid `field` at `positions` (M,3) → (M,)."""
        return jax.vmap(lambda p: self._interp_one(field, p))(positions)

    def accel_at_points(self, accel_grid, positions):
        """Sample an acceleration field (gx,gy,gz) at `positions` (M,3) → (M,3)."""
        gx, gy, gz = accel_grid
        ax = self.sample_at_points(gx, positions)
        ay = self.sample_at_points(gy, positions)
        az = self.sample_at_points(gz, positions)
        return jnp.stack([ax, ay, az], axis=1)

    def gas_accel_on_particles(self, rho, positions):
        """Acceleration of the GAS on particles at `positions` (gas-only φ ⇒ no
        self-force). Backs gas->particle (`get_gravity_at_point`)."""
        return self.accel_at_points(self.gas_accel_grid(rho), positions)

    # -- Particle-Mesh: particle mass -> grid density (CIC) -----------------
    def cic_deposit(self, positions, masses, active):
        """Deposit particle masses onto the grid as a density (CIC).

        Vectorized scatter-add over the particle axis — O(N), shape-stable.
        ∑ρ·dV == ∑ active masses (mass-conserving; positions clamped to box).
        This is the prerequisite for the MG solve: the solver acts on a density
        field, so point masses must be smeared to the mesh first.
        """
        w = (masses * active) / self.dV
        fx = positions[:, 0] / self.dx - 0.5
        fy = positions[:, 1] / self.dx - 0.5
        fz = positions[:, 2] / self.dx - 0.5
        i0 = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, self.Nx - 2)
        j0 = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, self.Ny - 2)
        k0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, self.Nz - 2)
        tx = jnp.clip(fx - i0, 0.0, 1.0)
        ty = jnp.clip(fy - j0, 0.0, 1.0)
        tz = jnp.clip(fz - k0, 0.0, 1.0)
        rho = jnp.zeros((self.Nx, self.Ny, self.Nz), dtype=w.dtype)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    wt = ((tx if di else 1 - tx) * (ty if dj else 1 - ty)
                          * (tz if dk else 1 - tz)) * w
                    rho = rho.at[i0 + di, j0 + dj, k0 + dk].add(wt)
        return rho

    def particles_potential(self, positions, masses, active):
        """φ_star: CIC-deposit the particle masses → MG Poisson solve."""
        return self.gas_potential(self.cic_deposit(positions, masses, active))

    # -- gas kick (momentum + energy), same convention as sources/sinks.py --
    def kick_gas(self, U, gx, gy, gz, dt):
        """Apply a gravitational acceleration field (gx,gy,gz) to the gas for dt."""
        rho = U[0]
        nmx = U[1] + dt * rho * gx
        nmy = U[2] + dt * rho * gy
        nmz = U[3] + dt * rho * gz
        mxm = 0.5 * (U[1] + nmx)
        mym = 0.5 * (U[2] + nmy)
        mzm = 0.5 * (U[3] + nmz)
        nE = U[-1] + dt * (mxm * gx + mym * gy + mzm * gz)
        return U.at[1].set(nmx).at[2].set(nmy).at[3].set(nmz).at[-1].set(nE)
