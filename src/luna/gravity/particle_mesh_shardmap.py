"""Sharded star <-> gas bridge for the coupled runner: the PM kick and the SN + wind deposit, each one shard_map.

Gas is sharded P(None, 'x', 'y', 'z'), the star slots replicated P(); every op is batched over all slots.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from luna.gravity.poisson_bc_shardmap import make_scalar_halo_3d_shardmap
from luna.gravity.poisson_multigrid import (
    fmg_solve, gathered_coarse, split_levels, vcycle,
)


class ShardedParticleBridge:
    """Sharded star<->gas gravity + feedback injection (shard_map)."""

    def __init__(self, cfg, mesh):
        self.cfg = cfg
        self.mesh = mesh
        Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz
        dx = cfg.Lx / cfg.Nx
        self.dx = dx
        self.dV = dx ** 3
        self.inv_2dx = 0.5 / dx
        G = float(getattr(cfg, "GRAVITATIONAL_G", 1.0))
        self.fourpiG = 4.0 * jnp.pi * G
        pb = getattr(cfg, "POISSON_BOUNDARY", None)
        bc = int(cfg.BOUNDARY if pb is None else pb)
        self.bc = bc

        self.n_pre = getattr(cfg, "POISSON_N_PRE", 2)
        self.n_post = getattr(cfg, "POISSON_N_POST", 2)
        self.n_coarsest = getattr(cfg, "POISSON_N_COARSEST", 50)
        self.n_vpl = getattr(cfg, "POISSON_FMG_VCYCLES_PER_LEVEL", 1)
        self.n_extra = getattr(cfg, "POISSON_FMG_EXTRA_VCYCLES", 0)
        _nl = getattr(cfg, "POISSON_N_LEVELS", None)

        self.halo_sm = None if bc == 3 else make_scalar_halo_3d_shardmap(mesh)(bc)
        self.mxs = mesh.shape["x"]; self.mys = mesh.shape["y"]; self.mzs = mesh.shape["z"]
        self.nxl = Nx // self.mxs; self.nyl = Ny // self.mys; self.nzl = Nz // self.mzs
        # the shards coarsen to the root grid, which every device solves gathered (the sharded solver's ladder)
        self.n_sh, self.n_levels = split_levels((Nx, Ny, Nz), mesh, int(_nl) if _nl else None)
        self.coarse = self.coarse_top = None
        if bc != 3:
            from luna.gravity.poisson_bc import make_halo_3d
            self.coarse, self.coarse_top = gathered_coarse(
                mesh, make_halo_3d(bc, inplace=False), self.n_levels - self.n_sh, self.n_pre, self.n_post,
                self.n_coarsest, self.n_vpl)

        import numpy as np
        r_inj = 2.0 * dx
        self.r_inj2 = r_inj * r_inj
        hw = int(np.ceil(r_inj / dx)) + 1
        self.Winj = 2 * hw + 1
        self.off = np.arange(-hw, hw + 1)     # host constant: no eager device op between shard_maps

        self._build()

    def _poisson(self, rho_l):
        if self.bc == 3:
            from luna.gravity.isolated import isolated_fmg_local
            return isolated_fmg_local(rho_l, self.dx, self.fourpiG, self.mesh,
                                      (self.Nx, self.Ny, self.Nz), self.n_levels,
                                      self.n_pre, self.n_post, self.n_coarsest,
                                      self.n_vpl, self.n_extra)
        if self.bc == 0:
            rho_l = rho_l - jax.lax.pmean(jnp.mean(rho_l), ("x", "y", "z"))
        rhs = self.fourpiG * rho_l
        phi = fmg_solve(rhs, self.dx, self.halo_sm, self.n_sh,
                        self.n_pre, self.n_post, self.n_coarsest, self.n_vpl,
                        coarse_fn=self.coarse, coarse_top_fn=self.coarse_top)
        for _ in range(self.n_extra):
            phi = vcycle(phi, rhs, self.dx, self.halo_sm,
                         self.n_sh - 1, self.n_pre, self.n_post, self.n_coarsest,
                         coarse_fn=self.coarse)
        return phi

    def _grad(self, phi):
        if self.bc == 3:
            from luna.gravity.isolated import pad_sharded
            pp = pad_sharded(phi, self.mesh, 1, 3)
        else:
            pp = self.halo_sm(phi)
        i = self.inv_2dx
        gx = -(pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) * i
        gy = -(pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) * i
        gz = -(pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) * i
        return gx, gy, gz

    def _cic_local(self, pos, mass, active, x0, y0, z0):
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        nxl, nyl, nzl = self.nxl, self.nyl, self.nzl
        dx, dV = self.dx, self.dV
        w = (mass * active) / dV
        fx = pos[:, 0] / dx - 0.5; fy = pos[:, 1] / dx - 0.5; fz = pos[:, 2] / dx - 0.5
        i0 = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, Nx - 2)
        j0 = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, Ny - 2)
        k0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, Nz - 2)
        tx = jnp.clip(fx - i0, 0.0, 1.0); ty = jnp.clip(fy - j0, 0.0, 1.0); tz = jnp.clip(fz - k0, 0.0, 1.0)
        rho = jnp.zeros((nxl, nyl, nzl), w.dtype)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    li = i0 + di - x0; lj = j0 + dj - y0; lk = k0 + dk - z0
                    m = ((li >= 0) & (li < nxl) & (lj >= 0) & (lj < nyl) & (lk >= 0) & (lk < nzl))
                    wt = ((tx if di else 1 - tx) * (ty if dj else 1 - ty)
                          * (tz if dk else 1 - tz)) * w * m
                    lic = jnp.clip(li, 0, nxl - 1); ljc = jnp.clip(lj, 0, nyl - 1); lkc = jnp.clip(lk, 0, nzl - 1)
                    rho = rho.at[lic, ljc, lkc].add(wt)
        return rho

    def _kick_gas(self, U, gx, gy, gz, dt):
        rho = U[0]
        nmx = U[1] + dt * rho * gx; nmy = U[2] + dt * rho * gy; nmz = U[3] + dt * rho * gz
        mxm = 0.5 * (U[1] + nmx); mym = 0.5 * (U[2] + nmy); mzm = 0.5 * (U[3] + nmz)
        nE = U[-1] + dt * (mxm * gx + mym * gy + mzm * gz)
        return U.at[1].set(nmx).at[2].set(nmy).at[3].set(nmz).at[-1].set(nE)

    def _build(self):
        mesh = self.mesh
        nxl, nyl, nzl = self.nxl, self.nyl, self.nzl
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dV = self.dx, self.dV
        off = self.off; Winj = self.Winj; r_inj2 = self.r_inj2
        AX = ("x", "y", "z")
        vec = P(None, "x", "y", "z")

        @partial(jax.jit)
        @partial(shard_map, mesh=mesh, in_specs=(vec, P(), P(), P(), P()),
                 out_specs=vec, check_rep=False)
        def _grav_on_gas(U, pos, mass, active, dt):
            x0 = jax.lax.axis_index("x") * nxl
            y0 = jax.lax.axis_index("y") * nyl
            z0 = jax.lax.axis_index("z") * nzl
            phi = self._poisson(self._cic_local(pos, mass, active, x0, y0, z0))
            gx, gy, gz = self._grad(phi)
            return self._kick_gas(U, gx, gy, gz, dt)

        self._grav_on_gas = _grav_on_gas

        @partial(jax.jit)
        @partial(shard_map, mesh=mesh,
                 in_specs=(vec, P(), P(), P(), P(), P(), P(), P(), P()),
                 out_specs=vec, check_rep=False)
        def _inject(U, pos, vel, e_sn, m_ej, dm_wind, vterm, fkin, active):
            x0 = jax.lax.axis_index("x") * nxl
            y0 = jax.lax.axis_index("y") * nyl
            z0 = jax.lax.axis_index("z") * nzl
            rho_in = U[0]

            def per_site(pos_i, vel_i, e_i, mej_i, dmw_i, vt_i, fk_i, act_i):
                bi = jnp.round(pos_i / dx - 0.5).astype(jnp.int32)
                sx = bi[0] + off; sy = bi[1] + off; sz = bi[2] + off
                lx = sx - x0; ly = sy - y0; lz = sz - z0
                inb = ((sx >= 0) & (sx < Nx) & (lx >= 0) & (lx < nxl))[:, None, None] \
                    & ((sy >= 0) & (sy < Ny) & (ly >= 0) & (ly < nyl))[None, :, None] \
                    & ((sz >= 0) & (sz < Nz) & (lz >= 0) & (lz < nzl))[None, None, :]
                xs = (sx + 0.5) * dx; ys = (sy + 0.5) * dx; zs = (sz + 0.5) * dx
                rrx = xs[:, None, None] - pos_i[0]; rry = ys[None, :, None] - pos_i[1]; rrz = zs[None, None, :] - pos_i[2]
                r2 = rrx**2 + rry**2 + rrz**2
                msk = (inb & (r2 < r_inj2)).astype(U.dtype) * act_i
                cells = jnp.sum(msk)                       # this shard's part of the count
                lic = jnp.clip(lx, 0, nxl - 1); ljc = jnp.clip(ly, 0, nyl - 1); lkc = jnp.clip(lz, 0, nzl - 1)
                return msk, cells, rrx, rry, rrz, lic, ljc, lkc

            msk, cells, rrx, rry, rrz, lic, ljc, lkc = jax.vmap(per_site)(
                pos, vel, e_sn, m_ej, dm_wind, vterm, fkin, active)
            vol = jnp.maximum(jax.lax.psum(cells, AX), 1e-30) * dV   # global per-site volume

            def gather_rho(lic_i, ljc_i, lkc_i):
                return rho_in[lic_i[:, None, None], ljc_i[None, :, None], lkc_i[None, None, :]]
            rho_b = jax.vmap(gather_rho)(lic, ljc, lkc)              # (Ns,W,W,W)

            r = jnp.sqrt(rrx**2 + rry**2 + rrz**2) + 1e-30
            ex, ey, ez = rrx / r, rry / r, rrz / r
            volb = vol[:, None, None, None]
            # SN ejecta at rest, wind ejecta with the wind velocity
            drho_sn = msk * (m_ej[:, None, None, None] / volb)
            drho_w = msk * (dm_wind[:, None, None, None] / volb)
            # wind velocity: star bulk + radial v_term
            vx = vel[:, 0][:, None, None, None] + vterm[:, None, None, None] * ex
            vy = vel[:, 1][:, None, None, None] + vterm[:, None, None, None] * ey
            vz = vel[:, 2][:, None, None, None] + vterm[:, None, None, None] * ez
            # SN energy: thermal (1-fkin)*E, radial kinetic fkin*E
            e_th = msk * ((1.0 - fkin[:, None, None, None]) * e_sn[:, None, None, None] / volb)
            e_kin = fkin[:, None, None, None] * e_sn[:, None, None, None] / volb
            # as inject_energy: the kick is sized on rho + ejecta
            rho_k = jnp.maximum(rho_b + drho_sn, 1e-30)
            p_sn = jnp.sqrt(2.0 * rho_k * jnp.maximum(e_kin, 0.0)) * msk

            d_rho = drho_sn + drho_w
            d_mx = drho_w * vx + p_sn * ex
            d_my = drho_w * vy + p_sn * ey
            d_mz = drho_w * vz + p_sn * ez
            d_E = (e_th + msk * e_kin
                   + 0.5 * drho_w * (vx**2 + vy**2 + vz**2))

            shp = (msk.shape[0], Winj, Winj, Winj)
            gxi = lic[:, :, None, None] + jnp.zeros(shp, jnp.int32)
            gyi = ljc[:, None, :, None] + jnp.zeros(shp, jnp.int32)
            gzi = lkc[:, None, None, :] + jnp.zeros(shp, jnp.int32)
            z3 = jnp.zeros((nxl, nyl, nzl), U.dtype)
            a_rho = z3.at[gxi, gyi, gzi].add(d_rho)
            a_mx = z3.at[gxi, gyi, gzi].add(d_mx)
            a_my = z3.at[gxi, gyi, gzi].add(d_my)
            a_mz = z3.at[gxi, gyi, gzi].add(d_mz)
            a_E = z3.at[gxi, gyi, gzi].add(d_E)
            U = (U.at[0].add(a_rho).at[1].add(a_mx).at[2].add(a_my)
                 .at[3].add(a_mz).at[4].add(a_E))
            return U

        self._inject = _inject

        sc = P("x", "y", "z")

        @partial(jax.jit)
        @partial(shard_map, mesh=mesh, in_specs=(sc,), out_specs=(sc, sc, sc), check_rep=False)
        def _field(rho):
            return self._grad(self._poisson(rho))

        self._field = _field

        @partial(jax.jit)
        @partial(shard_map, mesh=mesh, in_specs=(sc, sc, sc, P()), out_specs=P(), check_rep=False)
        def _at_points(gx, gy, gz, pos):
            """CIC field values at points: each shard adds the corners it owns, then one psum."""
            x0 = jax.lax.axis_index("x") * nxl
            y0 = jax.lax.axis_index("y") * nyl
            z0 = jax.lax.axis_index("z") * nzl
            f = [pos[:, a] / dx - 0.5 for a in range(3)]
            i0 = [jnp.floor(v).astype(jnp.int32) for v in f]
            t = [v - i for v, i in zip(f, i0)]
            N = (Nx, Ny, Nz)
            if self.bc != 0:
                i0 = [jnp.clip(i, 0, n - 2) for i, n in zip(i0, N)]
                t = [jnp.clip(v - i, 0.0, 1.0) for v, i in zip(f, i0)]
            out = jnp.zeros((pos.shape[0], 3), gx.dtype)
            for di in (0, 1):
                for dj in (0, 1):
                    for dk in (0, 1):
                        w = ((t[0] if di else 1 - t[0]) * (t[1] if dj else 1 - t[1])
                             * (t[2] if dk else 1 - t[2]))
                        g = [i0[0] + di, i0[1] + dj, i0[2] + dk]
                        if self.bc == 0:
                            g = [jnp.mod(i, n) for i, n in zip(g, N)]
                        li, lj, lk = g[0] - x0, g[1] - y0, g[2] - z0
                        own = ((li >= 0) & (li < nxl) & (lj >= 0) & (lj < nyl)
                               & (lk >= 0) & (lk < nzl))
                        li = jnp.clip(li, 0, nxl - 1)
                        lj = jnp.clip(lj, 0, nyl - 1)
                        lk = jnp.clip(lk, 0, nzl - 1)
                        wo = jnp.where(own, w, 0.0)[:, None]
                        out = out + wo * jnp.stack([gx[li, lj, lk], gy[li, lj, lk], gz[li, lj, lk]], axis=1)
            return jax.lax.psum(out, AX)

        self._at_points = _at_points

    def accel_field(self, rho):
        """-grad phi of rho (a sharded (N,N,N) density): three sharded fields."""
        return self._field(rho)

    def accel_at_points(self, field, pos):
        """A sharded acceleration field at positions (M, 3) -> (M, 3), replicated."""
        return self._at_points(field[0], field[1], field[2], pos)

    def gravity_on_gas(self, U, pos, mass, active, dt):
        return self._grav_on_gas(U, pos, mass, active, jnp.asarray(dt, U.dtype))

    def inject(self, U, pos, vel, e_sn, m_ej, dm_wind, vterm, fkin, active):
        return self._inject(U, pos, vel, e_sn, m_ej, dm_wind, vterm, fkin, active)
