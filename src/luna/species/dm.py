"""Collisionless dark matter on the particle mesh: CIC deposit, KDK leapfrog.

Exchange and conservation rules: see docs/ARCHITECTURE.md.
"""
import numpy as np

from luna.setup.state import Particles


def _fixed_ceiling(prec):
    """Largest 1+delta a fixed-point deposit can hold before integer overflow."""
    import jax.numpy as jnp
    bits = 63 if prec.fixed_int is jnp.int64 else 31
    return float(2 ** bits - 1) / float(2 ** prec.fixed_bits)


def _host_ids(host, n_total):
    """The host rows' particle IDs: the IC's, else the row index."""
    import jax
    dt = np.int64 if jax.config.jax_enable_x64 else np.int32
    if host.ids is not None:
        return np.asarray(host.ids, dt)
    if n_total is not None:
        raise ValueError("[dm] a chunked read must carry particle IDs")
    return np.arange(np.asarray(host.pos).shape[0], dtype=dt)


def dtau_accel(eta, eps_soft, gmax):
    """dtau ~ eta sqrt(eps / |g|_max): the N-body accuracy criterion."""
    return eta * np.sqrt(eps_soft / (gmax + 1e-30))


class DarkMatter:
    name = "dm"
    needs_gather = True

    def __init__(self, backend, *, box, dx, eta, eps_soft, prec, capacity_factor,
                 face_capacity_factor, resort_every, is_main=True):
        self.backend = backend
        self.box = float(box)
        self.dx = float(dx)
        self.eta = float(eta)
        self.eps_soft = float(eps_soft)
        self.prec = prec
        self.capacity_factor = float(capacity_factor)
        self.fcf = float(face_capacity_factor) or None
        self.resort_every = int(resort_every)
        # the 1-cell ghost tolerates +-0.5 dx of drift (CIC corner at pos/dx - 0.5)
        self.drift_margin = 0.45 * self.dx
        self.is_main = is_main
        self.cum_drift = 0.0
        self.n_exchanges = 0
        self.exchanged = False
        self.n_over = 0
        self.n_active0 = None
        self.mass0 = None
        self._deposit = None

    def assemble(self, host, n_total=None):
        """Device buffers from the host `Particles` (canonical momenta).

        `n_total`: the global count on a chunked read (each rank holds its block's rows).
        """
        mass_host = np.asarray(host.mass)
        pos, p, mass, act, ids = self.backend.assemble_particles(
            host.pos, host.p, mass_host, _host_ids(host, n_total), n_total,
            self.capacity_factor)
        if n_total is not None:
            pos, p, mass, act, ids, n_over = self.backend.settle(pos, p, mass, act, ids,
                                                                 host.pos)
            if int(n_over) != 0:
                raise RuntimeError(f"[dm] IC settle: {int(n_over)} particles overflowed "
                                   "the counted face buffers")
        m_total = (float(n_total) * float(np.mean(mass_host)) if n_total is not None
                   else float(np.sum(mass_host)))
        # fixed-point unit: the global mean density (constant, since mass is conserved)
        self._deposit = self.backend.make_deposit(m_total / self.box ** 3)
        if self.is_main:
            extra = (f" (capacity {self.backend.capacity} per block)"
                     if getattr(self.backend, "sharded", False) else "")
            print(f"[dm] {n_total or pos.shape[0]} particles, {self.prec}{extra}", flush=True)
        return Particles(pos, p, mass, act, ids=ids)

    def settle_after_restart(self, part):
        """Checkpointed buffers may hold particles just outside their block."""
        pos, p, mass, act, ids, n_over = self.backend.exchange(
            part.pos, part.p, part.mass, part.active, part.ids, self.fcf)
        if int(n_over) != 0:
            raise RuntimeError(f"[dm] restart settle: {int(n_over)} movers dropped")
        self.cum_drift = 0.0
        return Particles(pos, p, mass, act, part.g, ids=ids)

    def baseline(self, stats):
        """The count and mass every later exchange must reproduce."""
        self.n_active0 = int(stats["n_active"])
        self.mass0 = float(stats["mass"])

    def deposit(self, part):
        return self._deposit(part.pos, part.mass, part.active)

    def field_on_gas(self, part):
        return None

    def before(self, part, h, dt, stats):
        """Half-kick, drift, and an exchange when the drift could leave the ghost margin."""
        step_drift = (np.sqrt(stats["p2_max"]) + 0.5 * np.sqrt(stats["g2_max"]) * dt) * dt
        self.exchanged = (self.cum_drift + step_drift) > self.drift_margin
        pos, p_half = self.backend.kick_drift(part.pos, part.p, part.g, dt)
        mass, act, ids, n_over = part.mass, part.active, part.ids, 0
        if self.exchanged:
            pos, p_half, mass, act, ids, n_over = self.backend.exchange(
                pos, p_half, mass, act, ids, self.fcf)
            self.cum_drift = 0.0
            self.n_exchanges += 1
        else:
            self.cum_drift += step_drift
        self.n_over = n_over
        return Particles(pos, p_half, mass, act, part.g, ids=ids)

    def observe(self, U):
        return None

    def force(self, part, g_grid):
        return self.backend.gather(g_grid, part.pos)

    def after(self, part, graw, dt, s):
        """Closing half-kick on the new force; returns the particles and max |g|^2."""
        p, g, g2 = self.backend.close_kick(graw, part.p, part.active, dt, s)
        pos, mass, act, ids = part.pos, part.mass, part.active, part.ids
        if self.resort_every and self.exchanged and self.n_exchanges % self.resort_every == 0:
            pos, p, mass, act, g, ids = self.backend.resort(pos, p, mass, act, g, ids)
        return Particles(pos, p, mass, act, g, ids=ids), g2

    def finish(self, U, part, psi, obs):
        return U, part

    def dt_limit(self, stats):
        return dtau_accel(self.eta, self.eps_soft, np.sqrt(stats["g2_max"]))

    def check(self, stats, step):
        """Raise on lost particles at an exchange or a fixed-point deposit overflow."""
        if self.exchanged and (int(self.n_over) != 0
                               or int(stats["n_active"]) != self.n_active0
                               or abs(stats["mass"] - self.mass0) > 1e-8 * self.mass0):
            raise RuntimeError(
                f"[dm] exchange conservation violated at step {step}: overflow="
                f"{int(self.n_over)}, n_active {int(stats['n_active'])} vs {self.n_active0}, "
                f"mass {stats['mass']:.15e} vs {self.mass0:.15e}")
        if self.prec.fixed_deposit and stats["delta_min"] < -1.0 - 1e-9:
            raise RuntimeError(
                f"fixed-point deposit: delta_min = {stats['delta_min']:.3f} < -1, i.e. a "
                "NEGATIVE density cell: an integer wrap in the scatter, which the "
                "1+delta_max ceiling check cannot catch. Use gravity_precision='double' (int64).")
        if self.prec.fixed_deposit and 1.0 + stats["delta_max"] > 0.5 * _fixed_ceiling(self.prec):
            raise RuntimeError(
                f"fixed-point deposit: 1+delta_max = {1 + stats['delta_max']:.0f} is within "
                f"2x of the integer ceiling {_fixed_ceiling(self.prec):.0f} "
                f"({self.prec.fixed_int.__name__} at S=2^{self.prec.fixed_bits}). Use "
                "gravity_precision='double' (int64).")

    def snapshot(self, cfg, part, a, ic, n_total):
        """Write a Gadget particle snapshot with ParticleIDs (each process its own rows)."""
        import jax
        from luna.file_io.write_gadget import write_gadget_dm_snapshot
        pos = self.backend.local_rows(part.pos)
        p = self.backend.local_rows(part.p)
        act = self.backend.local_rows(part.active) > 0
        ids = None if part.ids is None else self.backend.local_rows(part.ids)[act]
        if jax.process_count() == 1:
            write_gadget_dm_snapshot(cfg, pos[act], p[act] / a ** 1.5, a, ic, ids=ids)
        else:
            write_gadget_dm_snapshot(cfg, pos[act], p[act] / a ** 1.5, a, ic,
                                     rank=jax.process_index(), nproc=jax.process_count(),
                                     n_total=int(n_total), ids=ids)

    def checkpoint_arrays(self, part):
        arrays = {"dm_pos": part.pos, "dm_p": part.p, "dm_mass": part.mass,
                  "dm_active": part.active}
        if part.ids is not None:
            arrays["dm_ids"] = part.ids
        return arrays

    def from_checkpoint(self, part, arrays):
        """Particles from checkpoint arrays (IDs optional)."""
        ids = arrays.get("dm_ids")
        if ids is None and part.ids is not None and self.is_main:
            print("[dm] checkpoint carries no particle IDs: snapshots will have none",
                  flush=True)
        return Particles(arrays["dm_pos"], arrays["dm_p"], arrays["dm_mass"],
                         arrays["dm_active"], part.g, ids=ids)
