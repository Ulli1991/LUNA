"""The gravity stage: one Poisson solve per step for the gas and every particle species, kick-drift-kick.

psi is solved with a unit coefficient and scaled at kick time by `clock.kick_scale(a)`, carried as `s_kick`.
"""
import jax

from luna.setup.run_report import require_finite


class Gravity:
    name = "gravity"

    def __init__(self, backend, clock, species, gas, *, order_gas=2, order_particles=4,
                 is_main=True, traced=False, energy="kick"):
        """traced: the stage runs inside a jit/scan, so stats stay device arrays and nothing syncs the host.

        energy: 'kick' (the work of the cell-centred kick) or 'flux' (the hydro step's face mass fluxes times the
        face potential difference, time-centred: E_kin + E_th + W conserved to the solve's tolerance).
        """
        if energy not in ("kick", "flux"):
            raise ValueError(f"gravity_energy={energy!r}: 'kick' or 'flux'")
        # what 'flux' can run is luna.setup.gravity_energy's (resolve_gravity_energy); the stage takes its answer
        self.energy = energy
        self.pre_solve = []           # flux: ops (state, h, dt) -> state that change the gas mass, run before the solve
        self.traced = bool(traced)
        self.backend = backend
        self.clock = clock
        self.species = list(species)
        self.gas = bool(gas)
        self.order_gas = order_gas
        self.order_particles = order_particles
        self.is_main = is_main
        self.events = [False, False]  # a species changed the grid mass at once: last step, the one before
        self.history_valid = None     # whether the last solve's seed was extrapolated

    def assemble(self, state, init):
        """Species buffers from the initial conditions' host arrays."""
        for sp in self.species:
            host = init.dm if sp.name == "dm" else None
            state = state.with_species(sp.name, sp.assemble(host, init.n_dm))
        return state

    def settle_after_restart(self, state):
        for sp in self.species:
            if hasattr(sp, "settle_after_restart"):
                state = state.with_species(sp.name, sp.settle_after_restart(state.species[sp.name]))
        return state

    def init(self, state):
        """The opening solve, which the first half-kick needs; later steps reuse the previous step's solve."""
        state = self.refresh(state)
        for sp in self.species:
            sp.baseline(state.stats)
        return state

    def refresh(self, state):
        """A fresh solve and forces for the gas as it is now; flux mode needs one after U changed outside the step."""
        src, stats = self._source(state)
        psi = self.backend.solve(src, None)
        if self.energy == "flux":
            # the step's energy source uses this psi: bring it to the warm solve's tolerance, as every later one
            psi = self.backend.solve(src, psi)
        return self._forces(state, psi, 0.0, stats, obs={}, events=False)

    def before(self, state, h, dt):
        U = state.U
        if self.energy == "flux":
            U, work = self.backend.kick_gas_work(U, state.g_gas, state.s_kick, h)
            state = state.update(kick_work=work)
        elif self.gas:
            U = self.backend.kick_gas(U, state.g_gas, state.s_kick, h,
                                      extra=self._extra_fields(state))
        species = dict(state.species)
        for sp in self.species:
            species[sp.name] = sp.before(species[sp.name], h, dt, state.stats)
        return state.update(U=U, species=species)

    def after(self, state, h, dt):
        U_pre = None
        if self.energy == "flux" and self.pre_solve:
            # the closing solve must see every non-hydro mass change of the step (booked in _forces)
            U_pre = state.U
            for op in self.pre_solve:
                state = op(state, h, dt)
        src, stats = self._source(state)
        # ROCm: an unsynced burst of eager dispatches can wedge the queue; one mid-step sync bounds it
        if not self.traced:
            jax.block_until_ready(src)
        psi0, s0 = state.psi, state.s_kick
        psi, state = self._solve(state, src, dt)
        obs = {sp.name: sp.observe(state.U) for sp in self.species}
        return self._forces(state, psi, dt, stats, obs, psi0=psi0, s0=s0, U_pre=U_pre)

    def _solve(self, state, src, dt):
        """The warm solve from the carried psi, extrapolated in time when the history allows it."""
        o = self.backend.opts
        if not (o.extrapolate and not self.traced and state.psi is not None):
            return self.backend.solve(src, state.psi), state
        q = dt / state.dt_prev if state.dt_prev > 0 else 0.0
        # psi_n - psi_n-1 spans an event that finished the step before last
        valid = (state.psi_prev is not None and 1.0 / o.dt_ratio_max <= q <= o.dt_ratio_max
                 and not self.events[1])
        self.history_valid = valid
        psi = self.backend.solve(src, state.psi, prev=state.psi_prev if valid else None,
                                 q=q if valid else 0.0)
        self.events = [False, self.events[0]]
        return psi, state.update(psi_prev=state.psi, dt_prev=float(dt))

    def _source(self, state):
        U = state.U if self.gas else None            # sliced inside the program
        rho_dm = None
        for sp in self.species:
            d = sp.deposit(state.species[sp.name])
            if d is not None:
                rho_dm = d if rho_dm is None else rho_dm + d
        dm = next((state.species[sp.name] for sp in self.species if sp.needs_gather), None)
        p, mass, active = (dm.p, dm.mass, dm.active) if dm is not None else (None, None, None)
        return self.backend.source(U, rho_dm, p, mass, active)

    def _extra_fields(self, state):
        """The species' field programs; the kick evaluates them inside itself."""
        progs = (sp.field_on_gas(state.species[sp.name]) for sp in self.species)
        return tuple(p for p in progs if p is not None) or None

    def _forces(self, state, psi, dt, stats, obs, events=True, psi0=None, s0=None, U_pre=None):
        """Gas and species closing half-kicks from a fresh psi, then the per-step events and the stats.

        psi0, s0: the potential and kick scale the step opened with (gravity_energy='flux'); U_pre: the gas before
        the pre-solve ops, whose mass change books W_booked = sum drho Phibar dV and energy change E_booked.
        """
        s = self.clock.kick_scale(state.a)
        U = state.U
        g_gas = self.backend.grad(psi, self.order_gas) if self.gas else None
        species = dict(state.species)
        # under a trace dt is not a Python number; a zero-length kick is a no-op anyway
        if self.energy == "flux" and psi0 is not None:
            if state.mass_flux is None or state.kick_work is None:
                raise RuntimeError("gravity_energy='flux': the step carries no face mass fluxes or opening "
                                   "kick work; the hydro must run Hydro.step_flux (luna.setup.evolve.step)")
            U, moved = self.backend.close_flux(U, g_gas, s, 0.5 * dt, psi0, psi, s0, state.mass_flux,
                                               state.kick_work, dt)
            stats = {**stats, "flux_moved": moved}
            if U_pre is not None:
                b = self.backend
                drho = b.component(state.U, 0) - b.component(U_pre, 0)
                stats["W_booked"] = b.booked_work(psi0, psi, s0, s, drho)
                stats["E_booked"] = b.grid_sum(b.component(state.U, -1) - b.component(U_pre, -1)) * b.dx ** 3
        elif self.gas and (self.traced or dt > 0.0):
            U = self.backend.kick_gas(U, g_gas, s, 0.5 * dt, extra=self._extra_fields(state))
        if self.energy == "flux":
            # the same stats keys every step: a traced loop carries them (autodiff_revolve's scan)
            zero = jax.numpy.zeros((), U.dtype)
            stats = {"flux_moved": zero, "W_booked": zero, "E_booked": zero, **stats}
        g_part = None
        if any(sp.needs_gather for sp in self.species):
            g_part = self.backend.grad(psi, self.order_particles)
        g2 = []
        for sp in self.species:
            part = species[sp.name]
            graw = sp.force(part, g_part if sp.needs_gather else g_gas)
            part, g2_sp = sp.after(part, graw, dt, s)
            if g2_sp is not None:
                g2.append(g2_sp)
            species[sp.name] = part
        if events:
            for sp in self.species:
                before = species[sp.name]
                U, species[sp.name] = sp.finish(U, before, psi, obs.get(sp.name))
                if self._slots_changed(before, species[sp.name]):
                    self.events[0] = True
        if self.traced:
            host = {k: v for k, v in stats.items() if v is not None}
        else:
            host = {k: float(v) for k, v in stats.items() if v is not None}
            host["g2_max"] = max((float(v) for v in g2), default=0.0)
            info = getattr(self.backend, "last_info", None)
            if info is not None:
                host["mg_cycles"], host["mg_residual"] = float(info[0]), float(info[1])
            if self.history_valid is not None:
                host["mg_extrapolated"] = float(self.history_valid)
        return state.update(U=U, psi=psi, g_gas=g_gas, s_kick=s, species=species, stats=host)

    def _slots_changed(self, old, new):
        """True when an extrapolated seed needs to know: a species gained or lost live slots."""
        o = self.backend.opts
        if not o.extrapolate or self.traced or getattr(old, "active", None) is None:
            return False
        return float(jax.numpy.sum(old.active)) != float(jax.numpy.sum(new.active))

    def dt_limit(self, state):
        lim = [sp.dt_limit(state.stats) for sp in self.species]
        lim = [v for v in lim if v is not None]
        return min(lim) if lim else float("inf")

    def check(self, state):
        if self.energy == "flux" and not self.traced and self.backend.opts.warm_tol > 0:
            res = state.stats.get("mg_residual")
            if res is not None and res > self.backend.opts.warm_tol:
                raise RuntimeError(
                    f"gravity_energy='flux': the warm solve stopped at {state.stats.get('mg_cycles'):.0f} cycles "
                    f"with |r|/|f| = {res:.3e} > gravity_warm_tol = {self.backend.opts.warm_tol:.1e} at step "
                    f"{state.step}; the energy is conserved only to the solve's residual. Raise "
                    "gravity_warm_max_cycles.")
        if self.gas:
            require_finite(state.stats["rho_gas_max"], what="gas density", step=state.step,
                           t=state.t, hint="the gravity solve saw a non-finite gas density")
        for sp in self.species:
            sp.check(state.stats, state.step)

    def deposit_of(self, state, name):
        """A species' density on the grid for the snapshot writer (a collective: every rank must call it)."""
        sp = next(s for s in self.species if s.name == name)
        return sp.deposit(state.species[name])

    def report(self, state):
        parts = [f"delta_max={state.stats['delta_max']:.3f}"] if "delta_max" in state.stats else []
        for sp in self.species:
            if hasattr(sp, "report"):
                parts.append(sp.report(state.species[sp.name]))
        return "  ".join(parts)
