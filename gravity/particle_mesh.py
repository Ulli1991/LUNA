"""Gas <-> particle gravity via two MG Poisson solves: phi_gas = MG(rho_gas)
and phi_star = MG(CIC-deposit(particles)).

Particle-particle gravity is deliberately NOT done here (external N-body's
job); particles sample the gas-only phi_gas, so they feel no self-force.
Fixed grid shapes + vmapped particle ops -> no recompile as particles change.
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
        """-∇φ on the grid: order=2 centred difference (default) or order=4
        (∂φ/∂x ≈ [φ(i-2) − 8φ(i-1) + 8φ(i+1) − φ(i+2)] / (12Δx)). Periodic /
        Neumann boundaries only; anti-symmetric stencil keeps CIC self-force-free."""
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
        # 2nd order on purpose: this force feeds the 2nd-order MUSCL-Hancock
        # gas update, so a higher-order gradient adds cost without accuracy.
        return self.grad_phi(self.gas_potential(rho))

    # -- trilinear (CIC) interpolation, vectorized over particles -----------
    def _interp_one(self, field, p):
        """Trilinear interpolation of `field` at a single position `p`
        (periodic wrap for boundary 0, face-clamped otherwise)."""
        fx = p[0] / self.dx - 0.5
        fy = p[1] / self.dx - 0.5
        fz = p[2] / self.dx - 0.5
        i0 = jnp.floor(fx).astype(jnp.int32)
        j0 = jnp.floor(fy).astype(jnp.int32)
        k0 = jnp.floor(fz).astype(jnp.int32)
        tx = fx - i0
        ty = fy - j0
        tz = fz - k0
        if self.boundary != 0:
            i0 = jnp.clip(i0, 0, self.Nx - 2)
            j0 = jnp.clip(j0, 0, self.Ny - 2)
            k0 = jnp.clip(k0, 0, self.Nz - 2)
            tx = jnp.clip(fx - i0, 0.0, 1.0)
            ty = jnp.clip(fy - j0, 0.0, 1.0)
            tz = jnp.clip(fz - k0, 0.0, 1.0)
        out = 0.0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = ((tx if di else 1 - tx) * (ty if dj else 1 - ty)
                         * (tz if dk else 1 - tz))
                    ix, jy, kz = i0 + di, j0 + dj, k0 + dk
                    if self.boundary == 0:
                        ix = jnp.mod(ix, self.Nx)
                        jy = jnp.mod(jy, self.Ny)
                        kz = jnp.mod(kz, self.Nz)
                    out = out + w * field[ix, jy, kz]
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
    def cic_deposit(self, positions, masses, active, dtype=None):
        """Deposit particle masses onto the grid as a density (CIC scatter-add):
        ∑ρ·dV == ∑ active masses (positions clamped to the box). `dtype` sets
        the grid dtype (float32 halves the scatter cost; default: positions')."""
        w = (masses * active) / self.dV
        fx = positions[:, 0] / self.dx - 0.5
        fy = positions[:, 1] / self.dx - 0.5
        fz = positions[:, 2] / self.dx - 0.5
        i0 = jnp.floor(fx).astype(jnp.int32)
        j0 = jnp.floor(fy).astype(jnp.int32)
        k0 = jnp.floor(fz).astype(jnp.int32)
        tx = fx - i0
        ty = fy - j0
        tz = fz - k0
        if self.boundary != 0:                 # non-periodic: clamp to faces
            i0 = jnp.clip(i0, 0, self.Nx - 2)
            j0 = jnp.clip(j0, 0, self.Ny - 2)
            k0 = jnp.clip(k0, 0, self.Nz - 2)
            tx = jnp.clip(fx - i0, 0.0, 1.0)
            ty = jnp.clip(fy - j0, 0.0, 1.0)
            tz = jnp.clip(fz - k0, 0.0, 1.0)
        if dtype is not None:
            w, tx, ty, tz = (q.astype(dtype) for q in (w, tx, ty, tz))
        rho = jnp.zeros((self.Nx, self.Ny, self.Nz), dtype=w.dtype)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    wt = ((tx if di else 1 - tx) * (ty if dj else 1 - ty)
                          * (tz if dk else 1 - tz)) * w
                    ix, jy, kz = i0 + di, j0 + dj, k0 + dk
                    if self.boundary == 0:     # periodic wrap at the faces
                        ix = jnp.mod(ix, self.Nx)
                        jy = jnp.mod(jy, self.Ny)
                        kz = jnp.mod(kz, self.Nz)
                    rho = rho.at[ix, jy, kz].add(wt)
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
