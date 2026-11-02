"""On-the-fly sink particles on one device: accretion, creation, merging, gas moments.

Gravity and sink rules: see docs/TORCH.md."""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from luna.gravity.poisson_bc import make_halo_3d
from luna.setup.state import Particles


def particles_periodic(cfg):
    """Whether sinks and coupled stars wrap at the box faces (empty: follow the gas)."""
    pb = str(getattr(cfg, "particle_boundary", "") or "").lower()
    if pb not in ("", "periodic", "outflow"):
        raise ValueError(f"particle_boundary={pb!r}: 'periodic', 'outflow' or ''")
    return int(cfg.BOUNDARY) == 0 if pb == "" else pb == "periodic"

ACCRETION_MODES = ("threshold", "federrath")
CREATION_MODES = ("luna", "flash")
CREATE_K = 8        # flash creation: candidate cells examined per step
SLOT_CHUNK = 4      # sink slots per pass of the sink field on the grid
SKIP_EMPTY = True   # a chunk of slots with no live sink adds nothing: skip it (tests turn it off)


def _lowest_k(score, K):
    """The K smallest grid entries by (value, flat index), and their flat indices."""
    # the K rows with the smallest minima hold them all: avoids a whole-grid top_k
    n = score.shape[-1]
    rows = score.reshape(-1, n)
    K1 = min(K, rows.shape[0])
    r = jnp.argsort(jnp.min(rows, axis=1), stable=True)[:K1]
    v = rows[r].reshape(-1)
    f = (r[:, None] * n + jnp.arange(n)[None, :]).reshape(-1)
    o = jnp.lexsort((f, v))[:K]
    return v[o], f[o]


class Sinks:
    name = "sinks"
    needs_gather = False

    def __init__(self, *, N3, box3, G, boundary, n_max, softening, accretion_radius,
                 creation_density, creation_separation, create_frac,
                 sink_sink=True, gamma=5.0 / 3.0, periodic_particles=None,
                 accretion="threshold", creation="luna", merging=False, is_main=True):
        Nx, Ny, Nz = (int(n) for n in N3)
        if accretion not in ACCRETION_MODES:
            raise ValueError(f"sink accretion={accretion!r}: one of {ACCRETION_MODES}")
        if creation not in CREATION_MODES:
            raise ValueError(f"sink creation checks={creation!r}: one of {CREATION_MODES}")
        self.accretion = accretion
        self.creation = creation
        self.merging = bool(merging)
        Lx, Ly, Lz = (float(b) for b in box3)
        self.N3 = (Nx, Ny, Nz)
        self.dx = Lx / Nx
        dx = self.dx
        self.G = float(G)
        self.n_max = int(n_max)
        self.eps = float(softening) or 2.0 * dx
        self.r_acc = float(accretion_radius) or 2.0 * dx
        self.rho_create = float(creation_density) or float("inf")
        self.r_sep = float(creation_separation) or 4.0 * dx
        self.sink_sink = bool(sink_sink)
        self.gamma = float(gamma)
        # cells keep (1 - frac) so the hydro never sees a zero-density cell
        self.create_frac = float(create_frac) or 0.9
        self.halo_fn = make_halo_3d(int(boundary))
        self.boundary = int(boundary)
        self.inv_2dx = 0.5 / dx
        self.dV = dx ** 3
        self.dxyz = (dx, Ly / Ny, Lz / Nz)
        self.box3 = (Lx, Ly, Lz)
        self.wrap = (int(boundary) == 0) if periodic_particles is None else bool(periodic_particles)
        if self.wrap and int(boundary) != 0:
            raise ValueError("periodic particles need a periodic gas box (BOUNDARY=0)")
        self.image_box = self.box3 if self.wrap else None
        # window half-width in cells, capped so the window never wraps onto itself
        # (a cell would be drained twice)
        R = int(np.ceil(self.r_acc / min(self.dxyz))) + 1
        R_max = (min(self.N3) - 1) // 2
        self.R_acc = min(R, R_max)
        if R > R_max and is_main:
            print(f"  ! sink accretion radius {self.r_acc:.3e} needs {R} cells "
                  f"but the grid allows {R_max}; the window is clamped")
        # 1D cell centres only: grid-shaped coordinates are never stored (memory)
        self.xc = (jnp.arange(Nx) + 0.5) * dx
        self.yc = (jnp.arange(Ny) + 0.5) * (Ly / Ny)
        self.zc = (jnp.arange(Nz) + 0.5) * (Lz / Nz)
        self.is_main = is_main
        self._accel = jax.jit(self._sink_accel)
        self._divv = jax.jit(self._div_v)
        self._finish = jax.jit(self._accrete_and_create)
        if is_main:
            print(f"✓ Sinks (on-the-fly, PM+direct N-body): N_max={self.n_max}, "
                  f"ρ_create={self.rho_create:.3e}, r_acc={self.r_acc:.3e}, "
                  f"ε={self.eps:.3e}, BC={int(boundary)}")

    def assemble(self, host=None, n_total=None):
        N = self.n_max
        return Particles(jnp.zeros((N, 3)), jnp.zeros((N, 3)), jnp.zeros(N),
                         jnp.zeros(N),
                         extra=self.zero_moments())

    def zero_moments(self):
        """Zeroed running sums over each sink's accreted gas, keyed as below.

        macc dm, sv dm v, sv2 dm v^2, scs dm c_s, sL dm (r - r_sink) x (v - v_sink), m_out left the box."""
        N = self.n_max
        return {"macc": jnp.zeros(N), "sv": jnp.zeros((N, 3)),
                "sv2": jnp.zeros((N, 3)), "scs": jnp.zeros(N),
                "sL": jnp.zeros((N, 3)), "m_out": jnp.zeros(N)}

    def _image(self, d, axis):
        """Nearest periodic image of a separation along `axis` (identity unless wrapping)."""
        if not self.wrap:
            return d
        L = self.box3[axis]
        return d - L * jnp.floor(d / L + 0.5)

    def _at_faces(self, pos, vel, mass, active, mom):
        """Wrap live sinks into the box, or (outflow) free the slot of one that left into `m_out`."""
        L = jnp.asarray(self.box3, pos.dtype)
        live = active[:, None] > 0
        if self.wrap:
            return jnp.where(live, jnp.mod(pos, L), pos), vel, mass, active, mom
        out = (active > 0) & jnp.any((pos < 0.0) | (pos >= L), axis=1)
        keep = 1.0 - out.astype(mass.dtype)
        mom = {k: (v + out * mass if k == "m_out" else
                   v * (keep[:, None] if v.ndim == 2 else keep))
               for k, v in mom.items()}
        return pos, vel * keep[:, None], mass * keep, active * keep, mom

    def baseline(self, stats):
        pass

    def _sink_field_on_grid(self, pos, mass, active):
        """The sinks' softened point-mass field on the grid."""
        X = self.xc[:, None, None]; Y = self.yc[None, :, None]; Z = self.zc[None, None, :]
        return self._field_at(X, Y, Z, self.N3, pos, mass, active)

    def _field_at(self, X, Y, Z, shape, pos, mass, active):
        """The sinks' field at cells X, Y, Z (broadcastable to `shape`)."""
        # fori_loop over slot chunks: unrolled, XLA keeps one grid of scratch per slot
        G, eps2 = self.G, self.eps ** 2
        C = min(SLOT_CHUNK, self.n_max)
        pad = -self.n_max % C
        if pad:
            pos = jnp.concatenate([pos, jnp.zeros((pad, 3), pos.dtype)])
            mass = jnp.concatenate([mass, jnp.zeros(pad, mass.dtype)])
            active = jnp.concatenate([active, jnp.zeros(pad, active.dtype)])

        def chunk(c, g):
            p = jax.lax.dynamic_slice_in_dim(pos, c * C, C)
            m = jax.lax.dynamic_slice_in_dim(mass, c * C, C)
            a = jax.lax.dynamic_slice_in_dim(active, c * C, C)

            def add(g):
                gx, gy, gz = g
                for i in range(C):
                    ddx = self._image(X - p[i, 0], 0)
                    ddy = self._image(Y - p[i, 1], 1)
                    ddz = self._image(Z - p[i, 2], 2)
                    # rsqrt cubed: an FP64 pow is much slower
                    r = jax.lax.rsqrt(ddx*ddx + ddy*ddy + ddz*ddz + eps2)
                    inv_r3 = a[i] * G * m[i] * r * r * r
                    gx = gx - ddx * inv_r3; gy = gy - ddy * inv_r3; gz = gz - ddz * inv_r3
                return gx, gy, gz
            if not SKIP_EMPTY:
                return add(g)
            # an empty slot adds exactly zero, so skipping its chunk changes no bit
            return jax.lax.cond(jnp.any(a != 0.0), add, lambda g: g, g)

        # zeros shaped like the coordinates: under shard_map the carry must be per-device
        z = jnp.broadcast_to(0.0 * (X + Y + Z), shape)
        return jax.lax.fori_loop(0, (self.n_max + pad) // C, chunk, (z, z, z))

    def _interp(self, field, p):
        """Trilinear (CIC) interpolation of a grid field at a position p."""
        dx = self.dx
        Nx, Ny, Nz = self.N3
        fx = p[0] / dx - 0.5; fy = p[1] / dx - 0.5; fz = p[2] / dx - 0.5
        if self.wrap:
            fl = [jnp.floor(f) for f in (fx, fy, fz)]
            i0, j0, k0 = (fl[a].astype(jnp.int32) for a in range(3))
            tx, ty, tz = fx - fl[0], fy - fl[1], fz - fl[2]

            def at(i, j, k):
                return field[jnp.mod(i, Nx), jnp.mod(j, Ny), jnp.mod(k, Nz)]
        else:
            i0 = jnp.clip(jnp.floor(fx).astype(jnp.int32), 0, Nx - 2)
            j0 = jnp.clip(jnp.floor(fy).astype(jnp.int32), 0, Ny - 2)
            k0 = jnp.clip(jnp.floor(fz).astype(jnp.int32), 0, Nz - 2)
            tx = jnp.clip(fx - i0, 0.0, 1.0); ty = jnp.clip(fy - j0, 0.0, 1.0)
            tz = jnp.clip(fz - k0, 0.0, 1.0)

            def at(i, j, k):
                return field[i, j, k]
        out = 0.0
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    w = ((tx if di else 1-tx) * (ty if dj else 1-ty)
                         * (tz if dk else 1-tz))
                    out = out + w * at(i0+di, j0+dj, k0+dk)
        return out

    def _sink_accel(self, pos, mass, active, g_gas):
        """Sink accelerations: interpolated gas field plus direct softened sink-sink terms."""
        # sink_sink is off under an external N-body code (PeTar), which owns that force
        N, G, eps2 = self.n_max, self.G, self.eps ** 2
        gx_gas, gy_gas, gz_gas = g_gas
        ax = jnp.zeros(N); ay = jnp.zeros(N); az = jnp.zeros(N)
        for i in range(N):
            gxi = self._interp(gx_gas, pos[i]); gyi = self._interp(gy_gas, pos[i])
            gzi = self._interp(gz_gas, pos[i])
            ddx = self._image(pos[:, 0] - pos[i, 0], 0)
            ddy = self._image(pos[:, 1] - pos[i, 1], 1)
            ddz = self._image(pos[:, 2] - pos[i, 2], 2)
            if self.sink_sink:
                inv_r3 = active * G * mass * (ddx*ddx + ddy*ddy + ddz*ddz + eps2) ** -1.5
                inv_r3 = inv_r3.at[i].set(0.0)
                gxi = gxi + jnp.sum(inv_r3 * ddx)
                gyi = gyi + jnp.sum(inv_r3 * ddy)
                gzi = gzi + jnp.sum(inv_r3 * ddz)
            ax = ax.at[i].set(active[i] * gxi); ay = ay.at[i].set(active[i] * gyi)
            az = az.at[i].set(active[i] * gzi)
        return jnp.stack([ax, ay, az], axis=1)

    def _div_v(self, U):
        """div v on the grid: the converging-flow indicator for sink formation."""
        halo_fn, inv_2dx = self.halo_fn, self.inv_2dx
        rho = jnp.maximum(U[0], 1e-30)
        vx = U[1] / rho; vy = U[2] / rho; vz = U[3] / rho
        px = halo_fn(vx); py = halo_fn(vy); pz = halo_fn(vz)
        dvx = (px[2:, 1:-1, 1:-1] - px[:-2, 1:-1, 1:-1]) * inv_2dx
        dvy = (py[1:-1, 2:, 1:-1] - py[1:-1, :-2, 1:-1]) * inv_2dx
        dvz = (pz[1:-1, 1:-1, 2:] - pz[1:-1, 1:-1, :-2]) * inv_2dx
        return dvx + dvy + dvz

    def _block(self, i0, n, periodic):
        """Wrapped indices of the (2R+1) window around cell `i0`, and the in-box mask."""
        off = jnp.arange(-self.R_acc, self.R_acc + 1)
        g = i0 + off
        return jnp.mod(g, n), (True if periodic else ((g >= 0) & (g < n)))

    def _claims(self, RX, RY, RZ, p, P, A):
        """Number of active sinks whose r_acc sphere reaches each cell of the window."""
        periodic = self.boundary == 0
        r2acc = self.r_acc * self.r_acc

        def one(q, aq):
            d = [R + (p[c] - q[c]) for c, R in enumerate((RX, RY, RZ))]
            if periodic:
                d = [x - L * jnp.floor(x / L + 0.5) for x, L in zip(d, self.box3)]
            return aq * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < r2acc)
        return jnp.sum(jax.vmap(one)(P, A), axis=0)

    def _federrath_take(self, Ub, RX, RY, RZ, p, s, P, V, M, A):
        """1 where slot `s` takes the cell's excess under Federrath+2010 sec. 2.3, else 0."""
        periodic = self.boundary == 0
        G, eps2 = self.G, self.eps ** 2
        r2acc = self.r_acc * self.r_acc
        r2in = max(0.2 * self.r_acc, self.dx) ** 2
        safe = jnp.maximum(Ub[0], 1e-30)
        vc = Ub[1:4] / safe
        ke = 0.5 * safe * jnp.sum(vc * vc, axis=0)
        cs = jnp.sqrt(jnp.maximum((self.gamma - 1.0) * (Ub[4] - ke) / safe, 0.0))
        phi_edge = jax.lax.rsqrt(r2acc + eps2)

        def one(q, vq, mq, aq):
            d = [R + (p[c] - q[c]) for c, R in enumerate((RX, RY, RZ))]
            if periodic:
                d = [x - L * jnp.floor(x / L + 0.5) for x, L in zip(d, self.box3)]
            r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            u = [vc[c] - vq[c] for c in range(3)]
            e = (0.5 * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
                 - G * mq * (jax.lax.rsqrt(r2 + eps2) - phi_edge)
                 - (2.0 * np.pi / 3.0) * G * self.rho_create * (r2acc - r2))
            infall = (d[0] * u[0] + d[1] * u[1] + d[2] * u[2]) < 1e-5 * cs * jnp.sqrt(r2)
            ok = (aq > 0) & (r2 < r2acc) & ((r2 < r2in) | ((e < 0.0) & infall))
            return ok, jnp.where(ok, e, jnp.inf)
        ok, e = jax.vmap(one)(P, V, M, A)
        return (ok[s] & (jnp.argmin(e, axis=0) == s)).astype(Ub.dtype)

    def _share(self, Ub, RX, RY, RZ, inside, p, s, P, V, M, A):
        """This slot's share of each cell's excess (even split, or all of it under federrath)."""
        if self.accretion == "federrath":
            return inside * self._federrath_take(Ub, RX, RY, RZ, p, s, P, V, M, A)
        return inside / jnp.maximum(self._claims(RX, RY, RZ, p, P, A), 1.0)

    def _slot(self, U, P, V, M, A, s, p, v, m, a):
        """One slot's accretion on its window; returns the drain instead of writing U."""
        Nx, Ny, Nz = self.N3
        dx, dy, dz = self.dxyz
        periodic = self.boundary == 0
        off = jnp.arange(-self.R_acc, self.R_acc + 1)

        i0 = jnp.floor(p[0] / dx).astype(jnp.int32)
        j0 = jnp.floor(p[1] / dy).astype(jnp.int32)
        k0 = jnp.floor(p[2] / dz).astype(jnp.int32)
        ix, mx = self._block(i0, Nx, periodic)
        iy, my = self._block(j0, Ny, periodic)
        iz, mz = self._block(k0, Nz, periodic)
        Ub = U[:, ix[:, None, None], iy[None, :, None], iz[None, None, :]]

        # unwrapped offsets: the nearest image
        RX = (((i0 + off) + 0.5) * dx - p[0])[:, None, None]
        RY = (((j0 + off) + 0.5) * dy - p[1])[None, :, None]
        RZ = (((k0 + off) + 0.5) * dz - p[2])[None, None, :]
        r2 = RX * RX + RY * RY + RZ * RZ
        edge = (1.0 if periodic else
                (mx[:, None, None] & my[None, :, None] & mz[None, None, :])
                .astype(U.dtype))

        rho = Ub[0]
        inside = a * (r2 < self.r_acc * self.r_acc) * edge
        share = self._share(Ub, RX, RY, RZ, inside, p, s, P, V, M, A)
        dm_cell = share * jnp.maximum(rho - self.rho_create, 0.0)
        dM = jnp.sum(dm_cell) * self.dV
        frac = jnp.where(rho > 0, dm_cell / jnp.maximum(rho, 1e-30), 0.0)
        dp = jnp.array([jnp.sum(frac * Ub[1]), jnp.sum(frac * Ub[2]),
                        jnp.sum(frac * Ub[3])]) * self.dV

        m_new = m + dM
        pnew = m * v + dp
        vnew = jnp.where(m_new > 0, pnew / m_new, 0.0)

        safe = jnp.maximum(rho, 1e-30)
        dm = frac * rho * self.dV
        vc = Ub[1:4] / safe
        ke = 0.5 * safe * jnp.sum(vc * vc, axis=0)
        # isothermal sqrt(P/rho), as Torch seeds its stars' dispersion
        cs = jnp.sqrt(jnp.maximum((self.gamma - 1.0) * (Ub[4] - ke) / safe, 0.0))
        # relative to the sink's new velocity, which is what a star inherits
        u = vc - vnew[:, None, None, None]
        moments = (
            jnp.array([jnp.sum(dm * vc[0] ** 2), jnp.sum(dm * vc[1] ** 2),
                       jnp.sum(dm * vc[2] ** 2)]),
            jnp.sum(dm * cs),
            jnp.array([jnp.sum(dm * (RY * u[2] - RZ * u[1])),
                       jnp.sum(dm * (RZ * u[0] - RX * u[2])),
                       jnp.sum(dm * (RX * u[1] - RY * u[0]))]),
        )
        drain = -frac * Ub
        dx_m = jnp.array([jnp.sum(dm_cell * RX), jnp.sum(dm_cell * RY),
                          jnp.sum(dm_cell * RZ)]) * self.dV
        return dM, dp, m_new, vnew, drain, (ix, iy, iz), moments, dx_m

    def _accrete(self, U, pos, vel, mass, active, mom=None):
        """Accrete the gas above rho_sink inside r_acc for every slot; accumulate `mom` if given."""
        dM, dp, m_new, v_new, drain, idx, moments, dx_m = jax.vmap(
            self._slot, in_axes=(None, None, None, None, None, 0, 0, 0, 0, 0))(
                U, pos, vel, mass, active, jnp.arange(self.n_max), pos, vel, mass, active)

        ix, iy, iz = idx
        n, B = ix.shape
        II = jnp.broadcast_to(ix[:, :, None, None], (n, B, B, B)).reshape(-1)
        JJ = jnp.broadcast_to(iy[:, None, :, None], (n, B, B, B)).reshape(-1)
        KK = jnp.broadcast_to(iz[:, None, None, :], (n, B, B, B)).reshape(-1)
        U = U.at[:, II, JJ, KK].add(jnp.moveaxis(drain, 1, 0).reshape(U.shape[0], -1))
        pos = self._to_com(pos, dx_m, m_new)

        if mom is not None:
            sv2, scs, sL = moments
            mom = {
                "macc": mom["macc"] + dM,
                "sv": mom["sv"] + dp,
                "sv2": mom["sv2"] + sv2,
                "scs": mom["scs"] + scs,
                "sL": mom["sL"] + sL,
                "m_out": mom["m_out"],
            }
        return (U, pos, v_new, m_new, active) if mom is None else (
            U, pos, v_new, m_new, active, mom)

    @staticmethod
    def _bound(w, Ub, phib, dV, psum=lambda x: x, pmax=lambda x: x):
        """Whether the gas weighted by `w` is bound, E_th + E_kin <= E_grav; psum/pmax combine blocks."""
        safe = jnp.maximum(Ub[0], 1e-30)
        vb = Ub[1:4] / safe
        eint = Ub[4] - 0.5 * safe * jnp.sum(vb * vb, axis=0)
        M = jnp.maximum(psum(jnp.sum(w * Ub[0])), 1e-30)
        vcm = psum(jnp.stack([jnp.sum(w * Ub[1 + a]) for a in range(3)])) / M
        ph_max = pmax(jnp.max(jnp.where(w > 0, phib, -jnp.inf)))
        E_th = psum(jnp.sum(w * eint)) * dV
        du = vb - vcm[:, None, None, None]
        E_kin = 0.5 * psum(jnp.sum(w * Ub[0] * jnp.sum(du * du, axis=0))) * dV
        E_grav = psum(jnp.sum(w * (ph_max - phib) * Ub[0])) * dV
        return E_th + E_kin <= E_grav

    def _maybe_create(self, U, pos, vel, mass, active, phi, divv):
        """Make a sink at the global potential minimum if it passes the creation checks."""
        dV, r_acc, r_sep = self.dV, self.r_acc, self.r_sep
        Nx, Ny, Nz = self.N3
        dx, dy, dz = self.dxyz
        periodic = self.boundary == 0
        off = jnp.arange(-self.R_acc, self.R_acc + 1)
        rho = U[0]

        # candidate: the potential minimum (SEREN, Bate, Bonnell & Price 1995)
        idx = jnp.argmin(phi.reshape(-1))
        ix = idx // (Ny * Nz)
        iy = (idx // Nz) % Ny
        iz = idx % Nz
        rho_pk = rho[ix, iy, iz]
        divv_pk = divv[ix, iy, iz]
        pk = jnp.array([self.xc[ix], self.yc[iy], self.zc[iz]])

        d2 = sum(self._image(pos[:, a] - pk[a], a) ** 2 for a in range(3))
        near = jnp.any(active * (d2 < r_sep * r_sep))
        free_slot = jnp.argmin(active.astype(jnp.int32))
        has_free = jnp.any(active < 0.5)

        jx, mx = self._block(ix, Nx, periodic)
        jy, my = self._block(iy, Ny, periodic)
        jz, mz = self._block(iz, Nz, periodic)
        sel = (slice(None), jx[:, None, None], jy[None, :, None],
               jz[None, None, :])
        Ub = U[sel]
        phib = phi[jx[:, None, None], jy[None, :, None], jz[None, None, :]]

        RX = (((ix + off) + 0.5) * dx - pk[0])[:, None, None]
        RY = (((iy + off) + 0.5) * dy - pk[1])[None, :, None]
        RZ = (((iz + off) + 0.5) * dz - pk[2])[None, None, :]
        r2 = RX * RX + RY * RY + RZ * RZ
        edge = (1.0 if periodic else
                (mx[:, None, None] & my[None, :, None] & mz[None, None, :])
                .astype(U.dtype))

        rhob = Ub[0]
        rem = self.create_frac * (r2 < r_acc * r_acc).astype(rhob.dtype) * edge
        M0 = jnp.sum(rem * rhob) * dV
        p0 = jnp.array([jnp.sum(rem * Ub[1]), jnp.sum(rem * Ub[2]),
                        jnp.sum(rem * Ub[3])]) * dV
        v0 = jnp.where(M0 > 0, p0 / M0, 0.0)
        bound = self._bound(rem, Ub, phib, dV)

        do_create = (rho_pk > self.rho_create) & (divv_pk < 0.0) & bound \
            & (~near) & has_free
        s = free_slot
        pos = pos.at[s].set(jnp.where(do_create, pk, pos[s]))
        vel = vel.at[s].set(jnp.where(do_create, v0, vel[s]))
        mass = mass.at[s].set(jnp.where(do_create, M0, mass[s]))
        active = active.at[s].set(jnp.where(do_create, 1.0, active[s]))
        keep = jnp.where(do_create, 1.0 - rem, 1.0)
        U = U.at[sel].multiply(keep[None])
        return U, pos, vel, mass, active

    def _accrete_and_create(self, U, pos, vel, mass, active, phi, divv, mom):
        pos, vel, mass, active, mom = self._at_faces(pos, vel, mass, active, mom)
        if self.creation == "flash":
            # FLASH's order: massless new sinks first, then every sink accretes
            phip = jnp.pad(phi, 2, mode="wrap" if self.boundary == 0 else "edge")
            ident = lambda x: x                                        # noqa: E731
            pos, vel, mass, active, mom, new = self._create_flash(
                U, pos, vel, mass, active, mom, phi, phip, (0, 0, 0), self.N3,
                ident, ident, ident)
            U, pos, vel, mass, active, mom = self._accrete(U, pos, vel, mass, active, mom)
            active = self._drop_empty(new, mass, active)
        else:
            U, pos, vel, mass, active, mom = self._accrete(
                U, pos, vel, mass, active, mom)
            U, pos, vel, mass, active = self._maybe_create(
                U, pos, vel, mass, active, phi, divv)
        if self.merging:
            pos, vel, mass, active, mom = self._merge(pos, vel, mass, active, mom)
        return U, pos, vel, mass, active, mom

    def _to_com(self, pos, dx_m, m_new):
        """federrath: move each sink to the centre of mass with its accreted gas; else unchanged."""
        if self.accretion != "federrath":
            return pos
        inv = jnp.where(m_new > 0, 1.0 / jnp.where(m_new > 0, m_new, 1.0), 0.0)
        return pos + dx_m * inv[:, None]

    def _cell_window(self, c, R, org, nloc):
        """Local indices of the (2R+1)^3 cells around global cell `c`, and the owned in-box mask."""
        periodic = self.boundary == 0
        idx, own = [], []
        for a, n in enumerate(self.N3):
            g = c[a] + jnp.arange(-R, R + 1)
            loc = jnp.mod(g, n) - org[a]
            ok = (loc >= 0) & (loc < nloc[a])
            if not periodic:
                ok = ok & (g >= 0) & (g < n)
            idx.append(jnp.clip(loc, 0, nloc[a] - 1))
            own.append(ok)
        return idx, own[0][:, None, None] & own[1][None, :, None] & own[2][None, None, :]

    def _near_sinks(self, pos, active, org, nloc):
        """Mask of this block's cells within 2 r_acc of an active sink."""
        R = int(np.ceil(2.0 * self.r_acc / min(self.dxyz))) + 1
        R = min(R, (min(self.N3) - 1) // 2)
        lim2 = (2.0 * self.r_acc) ** 2
        nx, ny, nz = nloc

        def one(p, a):
            c = [jnp.floor(p[i] / self.dxyz[i]).astype(jnp.int32) for i in range(3)]
            (ix, iy, iz), own = self._cell_window(c, R, org, nloc)
            d2 = 0.0
            for i, sh in enumerate(((-1, 1, 1), (1, -1, 1), (1, 1, -1))):
                centre = (jnp.mod(c[i] + jnp.arange(-R, R + 1), self.N3[i]) + 0.5) * self.dxyz[i]
                d2 = d2 + self._image(centre - p[i], i).reshape(sh) ** 2
            hit = own & (a > 0) & (d2 <= lim2)
            flat = (ix[:, None, None] * ny + iy[None, :, None]) * nz + iz[None, None, :]
            return jnp.where(hit, flat, nx * ny * nz).reshape(-1)
        idx = jax.vmap(one)(pos, active).reshape(-1)
        near = jnp.zeros(nx * ny * nz, bool).at[idx].set(True, mode="drop")
        return near.reshape(nloc)

    def _create_flash(self, U, pos, vel, mass, active, mom, phi, phip, org, nloc,
                      psum, pmax, gather):
        """FLASH-Torch sink creation (Federrath+2010 sec. 2.2) on the CREATE_K deepest candidates."""
        Ny, Nz = self.N3[1], self.N3[2]
        dx = self.dxyz[0]
        ny, nz = nloc[1], nloc[2]
        rho = U[0]

        def box_min(p, w):
            for axis, h in enumerate(w):
                n = p.shape[axis] - 4
                p = functools.reduce(jnp.minimum, [
                    jax.lax.slice_in_dim(p, 2 + o, 2 + o + n, axis=axis) for o in range(-h, h + 1)])
            return p
        # V is exactly the union of these three boxes
        vmin = jnp.minimum(jnp.minimum(box_min(phip, (2, 2, 1)), box_min(phip, (2, 1, 2))),
                           box_min(phip, (1, 2, 2)))
        cand = (rho > self.rho_create) & (vmin - phi >= -1e-5 * jnp.abs(phi))
        # excluded before the ranking, so a sink on the deepest minimum cannot hide the rest
        cand = cand & ~self._near_sinks(pos, active, org, nloc)
        score = jnp.where(cand, phi, jnp.inf)
        K = min(CREATE_K, score.size)
        val, li = _lowest_k(score, K)
        gflat = (((li // (ny * nz) + org[0]) * Ny + ((li // nz) % ny + org[1])) * Nz
                 + (li % nz + org[2]))
        val, gflat = gather(val), gather(gflat)
        order = jnp.lexsort((gflat, val))[:K]
        val, gflat = val[order], gflat[order]
        cells = jnp.stack([gflat // (Ny * Nz), (gflat // Nz) % Ny, gflat % Nz], 1)
        pk = (cells + 0.5) * jnp.asarray(self.dxyz)

        o = jnp.arange(-2, 3)
        OX, OY, OZ = o[:, None, None], o[None, :, None], o[None, None, :]
        O2 = OX * OX + OY * OY + OZ * OZ
        ring = jnp.maximum(jnp.maximum(jnp.abs(OX), jnp.abs(OY)), jnp.abs(OZ)) == 1
        inv_o = jnp.where(O2 > 0, 1.0 / jnp.sqrt(jnp.maximum(O2, 1).astype(U.dtype)), 0.0)
        dV, g1 = self.dV, self.gamma - 1.0

        def checks(c, v):
            (ix, iy, iz), own = self._cell_window(c, 2, org, nloc)
            Ub = U[:, ix[:, None, None], iy[None, :, None], iz[None, None, :]]
            phb = phi[ix[:, None, None], iy[None, :, None], iz[None, None, :]]
            safe = jnp.maximum(Ub[0], 1e-30)
            vb = Ub[1:4] / safe
            eint = Ub[4] - 0.5 * safe * jnp.sum(vb * vb, axis=0)
            mid = own[2, 2, 2].astype(U.dtype)
            rho_c = jnp.maximum(psum(mid * Ub[0, 2, 2, 2]), 1e-30)
            v_c = psum(mid * vb[:, 2, 2, 2])
            cs_c = jnp.sqrt(jnp.maximum(g1 * psum(mid * eint[2, 2, 2]) / rho_c, 0.0))
            vrad = (OX * (vb[0] - v_c[0]) + OY * (vb[1] - v_c[1]) + OZ * (vb[2] - v_c[2])) * inv_o
            away = psum(jnp.sum((own & ring & (vrad > 1e-5 * cs_c)).astype(U.dtype)))
            w = (own & (O2 <= 9)).astype(U.dtype)
            M = jnp.maximum(psum(jnp.sum(w * Ub[0])), 1e-30)
            vcm = psum(jnp.stack([jnp.sum(w * Ub[1 + a]) for a in range(3)])) / M
            ph_max = pmax(jnp.max(jnp.where(w > 0, phb, -jnp.inf)))
            E_th = psum(jnp.sum(w * eint)) * dV
            du = vb - vcm[:, None, None, None]
            E_kin = 0.5 * psum(jnp.sum(w * Ub[0] * jnp.sum(du * du, axis=0))) * dV
            E_grav = psum(jnp.sum(w * (ph_max - phb) * Ub[0])) * dV
            return (jnp.isfinite(v) & (away == 0.0) & (2.0 * E_th <= E_grav)
                    & (E_th + E_kin <= E_grav))
        ok = jax.vmap(checks)(cells, val)

        d2 = sum(self._image(pk[:, None, a] - pos[None, :, a], a) ** 2 for a in range(3))
        ok = ok & ~jnp.any((active[None, :] > 0) & (d2 <= (2.0 * self.r_acc) ** 2), axis=1)
        # a deeper new candidate within 1.1 sqrt(3) dx takes this one's place
        dc2 = sum(self._image(pk[:, None, a] - pk[None, :, a], a) ** 2 for a in range(3))
        k = jnp.arange(K)
        shadow = ok[None, :] & (k[None, :] < k[:, None]) & (dc2 <= (1.1 * np.sqrt(3.0) * dx) ** 2)
        ok = ok & ~jnp.any(shadow, axis=1)

        free = jnp.argsort(active > 0.5, stable=True)
        n_free = jnp.sum(active < 0.5)
        rank = jnp.cumsum(ok) - 1
        slot = jnp.where(ok & (rank < n_free), free[jnp.clip(rank, 0, self.n_max - 1)], self.n_max)
        pos = pos.at[slot].set(pk, mode="drop")
        vel = vel.at[slot].set(0.0, mode="drop")
        mass = mass.at[slot].set(0.0, mode="drop")
        active = active.at[slot].set(1.0, mode="drop")
        new = jnp.zeros(self.n_max, U.dtype).at[slot].set(1.0, mode="drop")
        keep = 1.0 - new
        mom = {kk: (v if kk == "m_out" else v * keep.reshape((-1,) + (1,) * (v.ndim - 1)))
               for kk, v in mom.items()}
        return pos, vel, mass, active, mom, new

    def _drop_empty(self, new, mass, active):
        """A sink born this step that accreted nothing is removed (FLASH)."""
        return active * (1.0 - new * (mass <= 0.0))

    def _merge(self, pos, vel, mass, active, mom):
        """Merge close, approaching, bound pairs, the lighter into the heavier (Federrath+2010 sec. 3.5)."""
        n = self.n_max
        d = [self._image(pos[None, :, a] - pos[:, None, a], a) for a in range(3)]  # x_j - x_i
        dv = [vel[None, :, a] - vel[:, None, a] for a in range(3)]
        r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        idx = jnp.arange(n)
        mi, mj = mass[:, None], mass[None, :]
        msum = mi + mj
        mu = jnp.where(msum > 0, mi * mj / jnp.where(msum > 0, msum, 1.0), 0.0)
        E = (0.5 * mu * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2])
             - self.G * mi * mj * jax.lax.rsqrt(r2 + self.eps ** 2))
        pair = ((active[:, None] > 0) & (active[None, :] > 0) & (idx[:, None] != idx[None, :])
                & (r2 < self.r_acc * self.r_acc)
                & (d[0] * dv[0] + d[1] * dv[1] + d[2] * dv[2] < 0.0) & (E < 0.0))
        can = pair & ((mi > mj) | ((mi == mj) & (idx[:, None] < idx[None, :])))  # i takes j
        has = jnp.any(can, axis=0)
        into = jnp.argmin(jnp.where(can, E, jnp.inf), axis=0)
        goes = has & ~has[into]
        W = ((into[None, :] == idx[:, None]) & goes[None, :]).astype(mass.dtype)   # W[i, j]
        Wm = W * mass[None, :]
        M = mass + jnp.sum(Wm, axis=1)
        keep = 1.0 - goes.astype(mass.dtype)
        inv = jnp.where(M > 0, 1.0 / jnp.where(M > 0, M, 1.0), 0.0)[:, None]
        P = mass[:, None] * vel + jnp.stack([jnp.sum(Wm * vel[None, :, a], axis=1)
                                             for a in range(3)], 1)
        shift = jnp.stack([jnp.sum(Wm * d[a], axis=1) for a in range(3)], 1)
        vel = jnp.where(M[:, None] > 0, P * inv, vel) * keep[:, None]
        pos = pos + shift * inv

        def add(v):
            got = jnp.sum(W.reshape(W.shape + (1,) * (v.ndim - 1)) * v[None], axis=1)
            return (v + got) * keep.reshape((-1,) + (1,) * (v.ndim - 1))
        mom = {k: (v if k == "m_out" else add(v)) for k, v in mom.items()}
        return pos, vel, M * keep, active * keep, mom

    def deposit(self, part):
        return None

    def field_on_gas(self, part):
        """The sinks' field as a program `(fn, args)` the gas kick evaluates in place."""
        return self._sink_field_on_grid, (part.pos, part.mass, part.active)

    def before(self, part, h, dt, stats):
        """Half-kick on the carried acceleration, then a full drift."""
        act = part.active[:, None]
        vel = part.p + h * part.g * act
        pos = part.pos + dt * vel * act
        return Particles(pos, vel, part.mass, part.active, part.g,
                         extra=part.extra)

    def observe(self, U):
        """div v before the kick, for the creation test."""
        return self._divv(U)

    def force(self, part, g_gas):
        return self._accel(part.pos, part.mass, part.active, g_gas)

    def after(self, part, acc, dt, s):
        vel = part.p + (0.5 * dt) * acc * part.active[:, None]
        return Particles(part.pos, vel, part.mass, part.active, acc,
                         extra=part.extra), None

    def finish(self, U, part, psi, divv):
        mom = part.extra if part.extra is not None else self.zero_moments()
        U, pos, vel, mass, active, mom = self._finish(
            U, part.pos, part.p, part.mass, part.active, psi, divv, mom)
        return U, Particles(pos, vel, mass, active, part.g, extra=mom)

    def dt_limit(self, stats):
        return None

    def check(self, stats, step):
        pass

    def report(self, part):
        """One-line sink summary."""
        # host sums only: rank 0 alone calls this, so a jnp reduction would hang
        m, act = np.asarray(part.mass), np.asarray(part.active)
        out = (part.extra or {}).get("m_out")
        gone = "" if out is None else f", left the box {np.asarray(out).sum():.3e}"
        return f"   🟣 sinks: {int(act.sum())} active, total M_sink={m.sum():.3e}{gone}"

    def snapshot(self, cfg, part, t, idx):
        """Write the sinks to one small npz per snapshot."""
        import os
        pos, vel, mass, act = (np.asarray(a) for a in (part.pos, part.p, part.mass, part.active))
        np.savez(os.path.join(cfg.output_dir, f"sinks_{idx:03d}.npz"),
                 pos=pos, vel=vel, mass=mass, active=act, t=np.float64(t))

    def checkpoint_arrays(self, part):
        return {}

    def from_checkpoint(self, part, arrays):
        return part


@functools.lru_cache(maxsize=None)
def _moment_kernel(n, shape, lengths, gamma, r_acc, periodic, mesh=None, spec=None,
                   cs2_max=None):
    """Jitted (n, 14) gas-moment sums over each position's (2R+1)^3 window (psum'd on a mesh)."""
    nx, ny, nz = shape
    d = np.array(lengths, float) / np.array(shape, float)
    R = int(np.ceil(r_acc / d.min())) + 1
    R = min(R, (min(shape) - 1) // 2)
    off = jnp.arange(-R, R + 1)

    def sums(U, positions, active, org, nloc):
        def one(p, on):
            c0 = jnp.floor(p / jnp.asarray(d)).astype(jnp.int32)
            idx, own, rel = [], [], []
            for a, n_ax in enumerate(shape):
                g = c0[a] + off
                inside_ax = True if periodic else (g >= 0) & (g < n_ax)
                gw = jnp.mod(g, n_ax)
                loc = gw - org[a]
                mine = (loc >= 0) & (loc < nloc[a]) & inside_ax
                idx.append(jnp.clip(loc, 0, nloc[a] - 1))
                own.append(mine)
                rel.append((g + 0.5) * d[a] - p[a])        # unwrapped: nearest image
            Ub = U[:, idx[0][:, None, None], idx[1][None, :, None], idx[2][None, None, :]]
            RX, RY, RZ = rel[0][:, None, None], rel[1][None, :, None], rel[2][None, None, :]
            w_own = (own[0][:, None, None] & own[1][None, :, None] & own[2][None, None, :])
            rho = jnp.maximum(Ub[0], 1e-30)
            vx, vy, vz = Ub[1] / rho, Ub[2] / rho, Ub[3] / rho
            ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz)
            cs = jnp.sqrt(jnp.maximum((gamma - 1.0) * (Ub[4] - ke), 1e-30) / rho)   # isothermal
            w_own = w_own & (RX * RX + RY * RY + RZ * RZ < r_acc * r_acc)
            if cs2_max is not None:
                w_own = w_own & (cs * cs <= cs2_max)
            w = jnp.where(w_own, rho, 0.0) * on
            S = lambda q: jnp.sum(w * q)                                            # noqa: E731
            return jnp.stack([
                jnp.sum(w), S(cs), S(vx), S(vy), S(vz), S(vx * vx), S(vy * vy), S(vz * vz),
                S(RX), S(RY), S(RZ),
                S(RY * vz - RZ * vy), S(RZ * vx - RX * vz), S(RX * vy - RY * vx)])
        return jax.vmap(one)(positions, active)

    if mesh is None:
        return jax.jit(lambda U, P, A: sums(U, P, A, (0, 0, 0), shape))

    from jax.sharding import PartitionSpec as Ps
    from luna.utils.shmap import shard_map
    axes = (tuple(spec)[1:] + (None,) * 3)[:3]
    counts = [int(mesh.shape[ax]) if ax is not None else 1 for ax in axes]
    nloc = tuple(n_ax // c for n_ax, c in zip(shape, counts))
    names = tuple(ax for ax in axes if ax is not None)

    def body(U, P, A):
        org = tuple(jax.lax.axis_index(ax) * nloc[a] if ax is not None else 0
                    for a, ax in enumerate(axes))
        return jax.lax.psum(sums(U, P, A, org, nloc), names)

    return jax.jit(shard_map(body, mesh=mesh, in_specs=(Ps(None, *axes), Ps(), Ps()),
                             out_specs=Ps()))


def _moments_from_sums(S):
    """(n, 14) window sums -> mean_cs, mean_velocity, var_velocity, angular_momentum."""
    M = np.maximum(S[:, 0], 1e-30)[:, None]
    mv = S[:, 2:5] / M
    var = S[:, 5:8] / M - mv * mv
    rm = S[:, 8:11]
    L = (S[:, 11:14] - np.cross(rm, mv)) / M
    return S[:, 1] / M[:, 0], mv, var, L


def sink_gas_moments(cfg, U, positions, gamma=None, r_acc=None, cs2_max=None):
    """Mass-weighted moments of the gas within ``r_acc`` of each position (code units).

    Returns dict mean_cs, mean_velocity, var_velocity, angular_momentum, mass; ``cs2_max`` caps P/rho."""
    positions = np.atleast_2d(np.asarray(positions))
    n = positions.shape[0]
    res = dict(mean_cs=np.zeros(n), mean_velocity=np.zeros((n, 3)),
               var_velocity=np.zeros((n, 3)), angular_momentum=np.zeros((n, 3)),
               mass=np.zeros(n))
    if n == 0:
        return res
    g = float(cfg.gamma if gamma is None else gamma)
    dx = cfg.Lx / cfg.Nx
    if r_acc is None:
        r_acc = float(getattr(cfg, "sink_accretion_radius", 0.0)) or 2.0 * dx

    # pad to a power of two: one compile per size class
    n_pad = 1 << (n - 1).bit_length()
    pos = np.zeros((n_pad, 3))
    pos[:n] = positions
    on = np.zeros(n_pad)
    on[:n] = 1.0

    sh = getattr(U, "sharding", None)
    mesh = getattr(sh, "mesh", None)
    sharded = mesh is not None and mesh.size > 1
    kernel = _moment_kernel(n_pad, (int(cfg.Nx), int(cfg.Ny), int(cfg.Nz)),
                            (float(cfg.Lx), float(cfg.Ly), float(cfg.Lz)),
                            g, float(r_acc), int(cfg.BOUNDARY) == 0,
                            mesh if sharded else None,
                            tuple(sh.spec) if sharded else None,
                            None if cs2_max is None else float(cs2_max))
    S = np.asarray(jax.device_get(kernel(U, jnp.asarray(pos), jnp.asarray(on))))[:n]
    cs, mv, var, am = _moments_from_sums(S)
    res.update(mean_cs=cs, mean_velocity=mv, var_velocity=var, angular_momentum=am,
               mass=S[:, 0] * (cfg.Lx / cfg.Nx) * (cfg.Ly / cfg.Ny) * (cfg.Lz / cfg.Nz))
    return res


def accreted_gas_moments(carry):
    """Per-active-sink moments of the accreted gas from carry = (pos, vel, mass, active, mom)."""
    _pos, _vel, _mass, active, mom = carry
    active = np.asarray(active)
    sel = np.where(active > 0.5)[0]
    n = len(sel)
    res = dict(mean_cs=np.zeros(n), mean_velocity=np.zeros((n, 3)),
               var_velocity=np.zeros((n, 3)), angular_momentum=np.zeros((n, 3)))
    if n == 0:
        return res
    M = np.maximum(np.asarray(mom["macc"])[sel], 1e-30)
    sv = np.asarray(mom["sv"])[sel]
    mean_v = sv / M[:, None]
    res["mean_cs"] = np.asarray(mom["scs"])[sel] / M
    res["mean_velocity"] = mean_v
    res["var_velocity"] = np.maximum(np.asarray(mom["sv2"])[sel] / M[:, None] - mean_v**2, 0.0)
    res["angular_momentum"] = np.asarray(mom["sL"])[sel] / M[:, None]
    return res
