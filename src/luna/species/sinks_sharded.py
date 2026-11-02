"""Sinks on a (px, py, pz) device mesh: replicated sink arrays, shard_map hooks.

See docs/TORCH.md ("Sinks on the mesh").
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from luna.gravity.particle_mesh_blocks import AXES, _block_origin, _halo
from luna.setup.state import Particles
from luna.species.sinks import Sinks
from luna.utils.halo_exchange_shardmap import make_halo_shardmap
from luna.utils.shmap import shard_map

S = ("x", "y", "z")


class ShardedSinks(Sinks):
    def __init__(self, mesh, **kw):
        super().__init__(**kw)
        self.mesh = mesh
        self.pdims = tuple(int(mesh.shape[a]) for a in AXES)
        if any(n % p for n, p in zip(self.N3, self.pdims)):
            raise ValueError(f"block decomposition {self.pdims} must divide {self.N3}")
        self.nloc = tuple(n // p for n, p in zip(self.N3, self.pdims))
        if self.creation == "flash" and self.boundary not in (0, 1):
            raise ValueError("flash sink creation on a mesh needs BOUNDARY 0 or 1 "
                             "(its 2-cell potential halo)")
        self._rep = NamedSharding(mesh, P())
        self._halo_u = make_halo_shardmap(mesh, 3)
        PU, Pg, R = P(None, *S), P(*S), P()
        sm = partial(shard_map, mesh=mesh)
        self._divv = jax.jit(sm(self._div_v_block, in_specs=(PU,), out_specs=Pg))
        self._accel = jax.jit(sm(self._accel_block, in_specs=(R, R, R, (Pg, Pg, Pg)),
                                 out_specs=R))
        mom_spec = {k: R for k in ("macc", "sv", "sv2", "scs", "sL", "m_out")}
        self._finish = jax.jit(sm(self._finish_block,
                                  in_specs=(PU, R, R, R, R, Pg, Pg, mom_spec),
                                  out_specs=(PU, R, R, R, R, mom_spec)))
        self._kick_half = jax.jit(self._kick_half_body)
        self._close_half = jax.jit(self._close_half_body)

    def _replicate(self, tree):
        return jax.tree_util.tree_map(
            lambda a: jax.device_put(np.asarray(a), self._rep), tree)

    def assemble(self, host=None, n_total=None):
        return self._replicate(super().assemble(host, n_total))

    def settle_after_restart(self, part):
        return self._replicate(part)

    def _origin(self):
        return _block_origin(self.pdims, self.nloc)

    def _own(self, gidx, axis, org):
        """Clipped local index of global cells `gidx` along `axis`, and the ownership mask."""
        loc = gidx - org[axis]
        own = (loc >= 0) & (loc < self.nloc[axis])
        return jnp.clip(loc, 0, self.nloc[axis] - 1), own

    def _sink_field_on_grid(self, pos, mass, active, origin=None):
        """The single-device kernel on this block's cell coordinates."""
        if origin is None:
            return super()._sink_field_on_grid(pos, mass, active)
        dx, dy, dz = self.dxyz
        nx, ny, nz = self.nloc
        X = ((origin[0] + jnp.arange(nx)) + 0.5) * dx
        Y = ((origin[1] + jnp.arange(ny)) + 0.5) * dy
        Z = ((origin[2] + jnp.arange(nz)) + 0.5) * dz
        X, Y, Z = X[:, None, None], Y[None, :, None], Z[None, None, :]
        return self._field_at(X, Y, Z, self.nloc, pos, mass, active)

    def _accel_block(self, pos, mass, active, g_gas):
        org = self._origin()
        gp = [_halo(q, self.mesh, 1) for q in g_gas]
        dx = self.dx
        Nx, Ny, Nz = self.N3

        def one(p):
            # must match `_interp` exactly (the halo holds the periodic neighbour)
            f = [p[a] / dx - 0.5 for a in range(3)]
            if self.wrap:
                fl = [jnp.floor(x) for x in f]
                i0 = [jnp.mod(fl[a].astype(jnp.int32), n) for a, n in enumerate((Nx, Ny, Nz))]
                t = [f[a] - fl[a] for a in range(3)]
            else:
                i0 = [jnp.clip(jnp.floor(f[a]).astype(jnp.int32), 0, n - 2)
                      for a, n in enumerate((Nx, Ny, Nz))]
                t = [jnp.clip(f[a] - i0[a], 0.0, 1.0) for a in range(3)]
            loc, own = zip(*(self._own(i0[a], a, org) for a in range(3)))
            mine = (own[0] & own[1] & own[2]).astype(p.dtype)
            il = [l + 1 for l in loc]                        # into the halo'd block
            out = []
            for q in gp:
                acc = 0.0
                for di in (0, 1):
                    for dj in (0, 1):
                        for dk in (0, 1):
                            w = ((t[0] if di else 1-t[0]) * (t[1] if dj else 1-t[1])
                                 * (t[2] if dk else 1-t[2]))
                            acc = acc + w * q[il[0]+di, il[1]+dj, il[2]+dk]
                out.append(mine * acc)
            return jnp.stack(out)

        g_at = jax.lax.psum(jax.vmap(one)(pos), S)             # (N, 3), replicated
        N, G, eps2 = self.n_max, self.G, self.eps ** 2
        acc = g_at
        if self.sink_sink:
            rows = []
            for i in range(N):
                dd = jnp.stack([self._image(pos[:, a] - pos[i, a], a) for a in range(3)], 1)
                inv_r3 = active * G * mass * (jnp.sum(dd * dd, axis=1) + eps2) ** -1.5
                inv_r3 = inv_r3.at[i].set(0.0)
                rows.append(g_at[i] + jnp.sum(inv_r3[:, None] * dd, axis=0))
            acc = jnp.stack(rows)
        return acc * active[:, None]

    def _div_v_block(self, U):
        rho = jnp.maximum(U[0], 1e-30)
        v = jnp.stack([U[1] / rho, U[2] / rho, U[3] / rho])
        pv = self._halo_u(v, self.boundary, 3, width=1)
        inv_2dx = self.inv_2dx
        dvx = (pv[0, 2:, 1:-1, 1:-1] - pv[0, :-2, 1:-1, 1:-1]) * inv_2dx
        dvy = (pv[1, 1:-1, 2:, 1:-1] - pv[1, 1:-1, :-2, 1:-1]) * inv_2dx
        dvz = (pv[2, 1:-1, 1:-1, 2:] - pv[2, 1:-1, 1:-1, :-2]) * inv_2dx
        return dvx + dvy + dvz

    def _window(self, i0, j0, k0, org):
        """Local indices and ownership mask of the accretion window around cell (i0, j0, k0)."""
        Nx, Ny, Nz = self.N3
        periodic = self.boundary == 0
        gx, mx = self._block(i0, Nx, periodic)
        gy, my = self._block(j0, Ny, periodic)
        gz, mz = self._block(k0, Nz, periodic)
        lx, ox = self._own(gx, 0, org)
        ly, oy = self._own(gy, 1, org)
        lz, oz = self._own(gz, 2, org)
        own = ox[:, None, None] & oy[None, :, None] & oz[None, None, :]
        edge = (own if periodic else
                own & mx[:, None, None] & my[None, :, None] & mz[None, None, :])
        return (lx, ly, lz), edge

    def _slot_block(self, U, org, P, V, M, A, s, p, v, m, a):
        """`_slot` on this block's part of the window; the sums are partial."""
        dx, dy, dz = self.dxyz
        off = jnp.arange(-self.R_acc, self.R_acc + 1)
        i0 = jnp.floor(p[0] / dx).astype(jnp.int32)
        j0 = jnp.floor(p[1] / dy).astype(jnp.int32)
        k0 = jnp.floor(p[2] / dz).astype(jnp.int32)
        (lx, ly, lz), edge = self._window(i0, j0, k0, org)
        Ub = U[:, lx[:, None, None], ly[None, :, None], lz[None, None, :]]
        RX = (((i0 + off) + 0.5) * dx - p[0])[:, None, None]
        RY = (((j0 + off) + 0.5) * dy - p[1])[None, :, None]
        RZ = (((k0 + off) + 0.5) * dz - p[2])[None, None, :]
        r2 = RX * RX + RY * RY + RZ * RZ
        rho = Ub[0]
        inside = a * (r2 < self.r_acc * self.r_acc) * edge.astype(U.dtype)
        share = self._share(Ub, RX, RY, RZ, inside, p, s, P, V, M, A)
        dm_cell = share * jnp.maximum(rho - self.rho_create, 0.0)
        dM = jnp.sum(dm_cell) * self.dV
        frac = jnp.where(rho > 0, dm_cell / jnp.maximum(rho, 1e-30), 0.0)
        dp = jnp.array([jnp.sum(frac * Ub[1]), jnp.sum(frac * Ub[2]),
                        jnp.sum(frac * Ub[3])]) * self.dV
        safe = jnp.maximum(rho, 1e-30)
        dm = frac * rho * self.dV
        vc = Ub[1:4] / safe
        ke = 0.5 * safe * jnp.sum(vc * vc, axis=0)
        cs = jnp.sqrt(jnp.maximum((self.gamma - 1.0) * (Ub[4] - ke) / safe, 0.0))   # sqrt(P/rho)
        dx_m = jnp.array([jnp.sum(dm_cell * RX), jnp.sum(dm_cell * RY),
                          jnp.sum(dm_cell * RZ)]) * self.dV
        return dict(dM=dM, dp=dp, dm=dm, vc=vc, cs=cs, RX=RX, RY=RY, RZ=RZ,
                    drain=-frac * Ub, idx=(lx, ly, lz), dx_m=dx_m)

    def _finish_block(self, U, pos, vel, mass, active, phi, divv, mom):
        pos, vel, mass, active, mom = self._at_faces(pos, vel, mass, active, mom)
        org = self._origin()
        if self.creation == "flash":
            phip = self._halo_u(phi[None], self.boundary, 3, width=2)[0]
            pos, vel, mass, active, mom, new = self._create_flash(
                U, pos, vel, mass, active, mom, phi, phip, org, self.nloc,
                lambda x: jax.lax.psum(x, S), lambda x: jax.lax.pmax(x, S), self._gather)
            U, pos, vel, mass, active, mom = self._accrete_block(
                U, pos, vel, mass, active, mom, org)
            active = self._drop_empty(new, mass, active)
        else:
            U, pos, vel, mass, active, mom = self._accrete_block(
                U, pos, vel, mass, active, mom, org)
            U, pos, vel, mass, active = self._create_luna_block(
                U, pos, vel, mass, active, phi, divv, org)
        if self.merging:
            pos, vel, mass, active, mom = self._merge(pos, vel, mass, active, mom)
        return U, pos, vel, mass, active, mom

    def _gather(self, x):
        """Every block's `x` (K,), concatenated in block order and replicated."""
        # each block writes its stretch of a zero buffer; the psum keeps it replicated for shard_map
        px, py, pz = self.pdims
        i = (jax.lax.axis_index("x") * py + jax.lax.axis_index("y")) * pz + jax.lax.axis_index("z")
        K = x.shape[0]
        buf = jnp.zeros((px * py * pz * K,), x.dtype)
        return jax.lax.psum(jax.lax.dynamic_update_slice_in_dim(buf, x, i * K, 0), S)

    def _accrete_block(self, U, pos, vel, mass, active, mom, org):
        """`_accrete`: partial sums on this block, totals across the mesh."""
        w = jax.vmap(self._slot_block,
                     in_axes=(None, None, None, None, None, None, 0, 0, 0, 0, 0))(
            U, org, pos, vel, mass, active, jnp.arange(self.n_max), pos, vel, mass, active)
        dM = jax.lax.psum(w["dM"], S)
        dp = jax.lax.psum(w["dp"], S)
        m_new = mass + dM
        pnew = mass[:, None] * vel + dp
        v_new = jnp.where(m_new[:, None] > 0, pnew / m_new[:, None], 0.0)
        dm, vc, cs = w["dm"], w["vc"], w["cs"]
        u = vc - v_new[:, :, None, None, None]
        RX, RY, RZ = w["RX"], w["RY"], w["RZ"]
        sv2 = jnp.stack([jnp.sum(dm * vc[:, c] ** 2, axis=(1, 2, 3)) for c in range(3)], 1)
        scs = jnp.sum(dm * cs, axis=(1, 2, 3))
        sL = jnp.stack([jnp.sum(dm * (RY * u[:, 2] - RZ * u[:, 1]), axis=(1, 2, 3)),
                        jnp.sum(dm * (RZ * u[:, 0] - RX * u[:, 2]), axis=(1, 2, 3)),
                        jnp.sum(dm * (RX * u[:, 1] - RY * u[:, 0]), axis=(1, 2, 3))], 1)
        sv2, scs, sL = (jax.lax.psum(x, S) for x in (sv2, scs, sL))
        mom = {"macc": mom["macc"] + dM, "sv": mom["sv"] + dp,
               "sv2": mom["sv2"] + sv2, "scs": mom["scs"] + scs, "sL": mom["sL"] + sL,
               "m_out": mom["m_out"]}
        lx, ly, lz = w["idx"]
        n, B = lx.shape
        II = jnp.broadcast_to(lx[:, :, None, None], (n, B, B, B)).reshape(-1)
        JJ = jnp.broadcast_to(ly[:, None, :, None], (n, B, B, B)).reshape(-1)
        KK = jnp.broadcast_to(lz[:, None, None, :], (n, B, B, B)).reshape(-1)
        U = U.at[:, II, JJ, KK].add(jnp.moveaxis(w["drain"], 1, 0).reshape(U.shape[0], -1))
        pos = self._to_com(pos, jax.lax.psum(w["dx_m"], S), m_new)
        return U, pos, v_new, m_new, active, mom

    def _create_luna_block(self, U, pos, vel, mass, active, phi, divv, org):
        """`_maybe_create`: the global potential minimum, then its window."""
        Nx, Ny, Nz = self.N3
        nx, ny, nz = self.nloc
        loc = jnp.argmin(phi.reshape(-1))
        li, lj, lk = loc // (ny * nz), (loc // nz) % ny, loc % nz
        gi, gj, gk = li + org[0], lj + org[1], lk + org[2]
        gflat = gi * (Ny * Nz) + gj * Nz + gk
        vmin = jax.lax.pmin(phi.reshape(-1)[loc], S)
        # among ties the smallest flat index, as a single device's argmin
        gflat = jax.lax.pmin(jnp.where(phi.reshape(-1)[loc] == vmin, gflat, Nx * Ny * Nz), S)
        ix, iy, iz = gflat // (Ny * Nz), (gflat // Nz) % Ny, gflat % Nz
        (lx0, ox), (ly0, oy), (lz0, oz) = (self._own(ix, 0, org), self._own(iy, 1, org),
                                           self._own(iz, 2, org))
        owner = (ox & oy & oz).astype(U.dtype)
        rho_pk = jax.lax.psum(owner * U[0, lx0, ly0, lz0], S)
        divv_pk = jax.lax.psum(owner * divv[lx0, ly0, lz0], S)
        dx, dy, dz = self.dxyz
        pk = jnp.array([(ix + 0.5) * dx, (iy + 0.5) * dy, (iz + 0.5) * dz])

        d2 = sum(self._image(pos[:, a] - pk[a], a) ** 2 for a in range(3))
        near = jnp.any(active * (d2 < self.r_sep * self.r_sep))
        free_slot = jnp.argmin(active.astype(jnp.int32))
        has_free = jnp.any(active < 0.5)

        off = jnp.arange(-self.R_acc, self.R_acc + 1)
        (lx, ly, lz), edge = self._window(ix, iy, iz, org)
        sel = (slice(None), lx[:, None, None], ly[None, :, None], lz[None, None, :])
        Ub = U[sel]
        phib = phi[lx[:, None, None], ly[None, :, None], lz[None, None, :]]
        RX = (((ix + off) + 0.5) * dx - pk[0])[:, None, None]
        RY = (((iy + off) + 0.5) * dy - pk[1])[None, :, None]
        RZ = (((iz + off) + 0.5) * dz - pk[2])[None, None, :]
        r2 = RX * RX + RY * RY + RZ * RZ
        rhob = Ub[0]
        rem = (self.create_frac * (r2 < self.r_acc * self.r_acc).astype(rhob.dtype)
               * edge.astype(rhob.dtype))
        M0 = jax.lax.psum(jnp.sum(rem * rhob), S) * self.dV
        p0 = jax.lax.psum(jnp.array([jnp.sum(rem * Ub[1]), jnp.sum(rem * Ub[2]),
                                     jnp.sum(rem * Ub[3])]), S) * self.dV
        v0 = jnp.where(M0 > 0, p0 / M0, 0.0)
        bound = self._bound(rem, Ub, phib, self.dV, lambda x: jax.lax.psum(x, S),
                            lambda x: jax.lax.pmax(x, S))
        do_create = ((rho_pk > self.rho_create) & (divv_pk < 0.0) & bound
                     & (~near) & has_free)
        s = free_slot
        pos = pos.at[s].set(jnp.where(do_create, pk, pos[s]))
        vel = vel.at[s].set(jnp.where(do_create, v0, vel[s]))
        mass = mass.at[s].set(jnp.where(do_create, M0, mass[s]))
        active = active.at[s].set(jnp.where(do_create, 1.0, active[s]))
        keep = jnp.where(do_create, 1.0 - rem, 1.0)
        U = U.at[sel].multiply(keep[None])
        return U, pos, vel, mass, active

    @staticmethod
    def _kick_half_body(p, g, pos, active, h, dt):
        act = active[:, None]
        vel = p + h * g * act
        return vel, pos + dt * vel * act

    @staticmethod
    def _close_half_body(p, acc, active, dt):
        return p + (0.5 * dt) * acc * active[:, None]

    def field_on_gas(self, part):
        return self._sink_field_on_grid, (part.pos, part.mass, part.active)

    def before(self, part, h, dt, stats):
        vel, pos = self._kick_half(part.p, part.g, part.pos, part.active, h, dt)
        return Particles(pos, vel, part.mass, part.active, part.g, extra=part.extra)

    def observe(self, U):
        return self._divv(U)

    def force(self, part, g_gas):
        return self._accel(part.pos, part.mass, part.active, tuple(g_gas))

    def after(self, part, acc, dt, s):
        vel = self._close_half(part.p, acc, part.active, dt)
        return Particles(part.pos, vel, part.mass, part.active, acc, extra=part.extra), None

    def finish(self, U, part, psi, divv):
        mom = part.extra if part.extra is not None else self._replicate(self.zero_moments())
        U, pos, vel, mass, active, mom = self._finish(
            U, part.pos, part.p, part.mass, part.active, psi, divv, mom)
        return U, Particles(pos, vel, mass, active, part.g, extra=mom)
