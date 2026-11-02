"""Gas <-> particle gravity on one device: MG Poisson solves of the gas and of the CIC-deposited particles.

No particle-particle gravity here (the external N-body's job): particles sample the gas-only potential.
"""

import jax
import jax.numpy as jnp

from luna.gravity.poisson_bc import edge_ghosts, make_halo_3d, pad_inplace, wrap_ghosts
from luna.gravity.poisson_multigrid import _max_levels, solve_poisson_fmg
from luna.utils.grid_params import _require_cubic_cells


class ParticleMeshGravity:
    """Gas<->particle gravity built from a config; uses LUNA's MG Poisson."""

    def __init__(self, cfg):
        if cfg.SPATIAL_DIMENSION != 3:
            raise ValueError("ParticleMeshGravity requires SPATIAL_DIMENSION == 3")
        self.cfg = cfg
        self.Nx, self.Ny, self.Nz = cfg.Nx, cfg.Ny, cfg.Nz
        self.dx = cfg.Lx / cfg.Nx
        _require_cubic_cells(cfg, "ParticleMeshGravity")
        self.dV = self.dx ** 3
        self.inv_2dx = 0.5 / self.dx
        self.inv_12dx = 1.0 / (12.0 * self.dx)   # 4th-order gradient prefactor
        self.G = float(cfg.GRAVITATIONAL_G)
        self.fourpiG = 4.0 * jnp.pi * self.G

        pb = cfg.POISSON_BOUNDARY
        self.boundary = int(cfg.BOUNDARY if pb is None else pb)
        self.halo_fn = make_halo_3d(self.boundary)
        self._poisson_kw = dict(
            n_levels=int(cfg.POISSON_N_LEVELS or _max_levels(cfg.Nx)),
            n_vcycles_per_level=cfg.POISSON_FMG_VCYCLES_PER_LEVEL,
            n_extra_vcycles=cfg.POISSON_FMG_EXTRA_VCYCLES,
            n_pre=cfg.POISSON_N_PRE,
            n_post=cfg.POISSON_N_POST,
            n_coarsest=cfg.POISSON_N_COARSEST,
        )

    def gas_potential(self, rho):
        """FMG-multigrid solve of ∇²φ = 4πG ρ for a density field ρ."""
        rho_src = rho - jnp.mean(rho) if self.boundary == 0 else rho
        return solve_poisson_fmg(rho_src, self.dx, self.fourpiG,
                                 boundary=self.boundary, **self._poisson_kw)

    def _pad2(self, phi):
        """Two-cell halo for the 4th-order gradient stencil."""
        if self.boundary == 0:
            return pad_inplace(phi, 2, lambda core, ax: wrap_ghosts(core, ax, 2))
        if self.boundary in (1, 2):
            return pad_inplace(phi, 2, lambda core, ax: edge_ghosts(core, ax, 2))
        from luna.gravity.isolated import extrap_pad
        return extrap_pad(phi, 2)

    def grad_phi(self, phi, order=2):
        """-∇φ on the grid by a centred difference of order 2 or 4; isolated faces extrapolate."""
        if order == 4:
            pp = self._pad2(phi)
            c = self.inv_12dx
            gx = (pp[4:, 2:-2, 2:-2] - 8.0 * pp[3:-1, 2:-2, 2:-2]
                  + 8.0 * pp[1:-3, 2:-2, 2:-2] - pp[:-4, 2:-2, 2:-2]) * c
            gy = (pp[2:-2, 4:, 2:-2] - 8.0 * pp[2:-2, 3:-1, 2:-2]
                  + 8.0 * pp[2:-2, 1:-3, 2:-2] - pp[2:-2, :-4, 2:-2]) * c
            gz = (pp[2:-2, 2:-2, 4:] - 8.0 * pp[2:-2, 2:-2, 3:-1]
                  + 8.0 * pp[2:-2, 2:-2, 1:-3] - pp[2:-2, 2:-2, :-4]) * c
            return gx, gy, gz
        if self.boundary == 3:
            from luna.gravity.isolated import extrap_pad
            pp = extrap_pad(phi, 1)
        else:
            pp = self.halo_fn(phi)
        gx = -(pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) * self.inv_2dx
        gy = -(pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) * self.inv_2dx
        gz = -(pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) * self.inv_2dx
        return gx, gy, gz

    def gas_accel_grid(self, rho):
        """Gravitational acceleration field of a density: -∇(MG(ρ))."""
        # 2nd order on purpose: it feeds the 2nd-order hydro update.
        return self.grad_phi(self.gas_potential(rho))

    def _interp_one(self, field, p):
        """Trilinear interpolation of `field` at one position (periodic wrap or face-clamped)."""
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
        """Acceleration from the gas alone at `positions` (M,3) → (M,3)."""
        return self.accel_at_points(self.gas_accel_grid(rho), positions)

    def cic_deposit(self, positions, masses, active, dtype=None, fixed=None,
                    fixed_bits=17, mass_unit=None):
        """CIC density of the active particle masses (∑ρ·dV = ∑ active masses).

        fixed: jnp.int32/int64 for fixed-point accumulation, 2**fixed_bits units per mean-density cell.
        mass_unit: the mean density, if the caller already knows it.
        """
        if fixed is not None:
            return self._cic_deposit_fixed(positions, masses, active, fixed,
                                           fixed_bits, dtype, mass_unit)
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
        if self.boundary != 0:
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
                    if self.boundary == 0:
                        ix = jnp.mod(ix, self.Nx)
                        jy = jnp.mod(jy, self.Ny)
                        kz = jnp.mod(kz, self.Nz)
                    rho = rho.at[ix, jy, kz].add(wt)
        return rho

    def _cic_deposit_fixed(self, positions, masses, active, int_dtype,
                           bits, dtype=None, mass_unit=None):
        """Fixed-point CIC: bitwise independent of particle order; the rounding tree conserves mass exactly.

        Overflow: (1+delta) must stay below 2**31/S for int32 (16384 at S=2^17).
        """
        w = (masses * active) / self.dV
        if mass_unit is None:
            mass_unit = jnp.sum(w) / float(self.Nx * self.Ny * self.Nz)
        S = float(1 << bits)
        fx = positions[:, 0] / self.dx - 0.5
        fy = positions[:, 1] / self.dx - 0.5
        fz = positions[:, 2] / self.dx - 0.5
        i0 = jnp.floor(fx).astype(jnp.int32)
        j0 = jnp.floor(fy).astype(jnp.int32)
        k0 = jnp.floor(fz).astype(jnp.int32)
        tx, ty, tz = fx - i0, fy - j0, fz - k0
        if self.boundary != 0:
            i0 = jnp.clip(i0, 0, self.Nx - 2)
            j0 = jnp.clip(j0, 0, self.Ny - 2)
            k0 = jnp.clip(k0, 0, self.Nz - 2)
            tx = jnp.clip(fx - i0, 0.0, 1.0)
            ty = jnp.clip(fy - j0, 0.0, 1.0)
            tz = jnp.clip(fz - k0, 0.0, 1.0)
        ftype = jnp.float32 if int_dtype == jnp.int32 else jnp.float64
        tx, ty, tz = (q.astype(ftype) for q in (tx, ty, tz))
        W = jnp.rint((w / mass_unit).astype(ftype) * ftype(S)).astype(int_dtype)
        grid = jnp.zeros((self.Nx, self.Ny, self.Nz), int_dtype)
        for di in (0, 1):
            lo = jnp.rint(W.astype(ftype) * (1.0 - tx)).astype(int_dtype)
            Wx = lo if di == 0 else W - lo
            for dj in (0, 1):
                lo = jnp.rint(Wx.astype(ftype) * (1.0 - ty)).astype(int_dtype)
                Wxy = lo if dj == 0 else Wx - lo
                for dk in (0, 1):
                    lo = jnp.rint(Wxy.astype(ftype) * (1.0 - tz)).astype(int_dtype)
                    upd = lo if dk == 0 else Wxy - lo
                    ix, jy, kz = i0 + di, j0 + dj, k0 + dk
                    if self.boundary == 0:
                        ix = jnp.mod(ix, self.Nx)
                        jy = jnp.mod(jy, self.Ny)
                        kz = jnp.mod(kz, self.Nz)
                    grid = grid.at[ix, jy, kz].add(upd)
        out = jnp.float32 if dtype is None and int_dtype == jnp.int32 else \
            (dtype or jnp.float64)
        return grid.astype(out) * jnp.asarray(mass_unit / S, dtype=out)

    def particles_potential(self, positions, masses, active):
        """φ_star: CIC-deposit the particle masses → MG Poisson solve."""
        return self.gas_potential(self.cic_deposit(positions, masses, active))

    def kick_gas(self, U, gx, gy, gz, dt):
        """Kick the gas momentum and energy by (gx, gy, gz) for dt."""
        rho = U[0]
        nmx = U[1] + dt * rho * gx
        nmy = U[2] + dt * rho * gy
        nmz = U[3] + dt * rho * gz
        mxm = 0.5 * (U[1] + nmx)
        mym = 0.5 * (U[2] + nmy)
        mzm = 0.5 * (U[3] + nmz)
        nE = U[-1] + dt * (mxm * gx + mym * gy + mzm * gz)
        return U.at[1].set(nmx).at[2].set(nmy).at[3].set(nmz).at[-1].set(nE)
