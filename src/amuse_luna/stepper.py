"""Resident LUNA state with an I/O-free `evolve_to`, for the AMUSE worker."""
import jax

from luna.setup.evolve import step as _evolve_step
from luna.setup.evolve import timestep as _timestep
from luna.setup.step_timing import StepTiming


class LunaState:
    """A LUNA run held open between calls: `state` plus the assembled physics."""

    def __init__(self, state, physics, cfg, mesh=None, sharding=None):
        self._s = state
        self.physics = physics
        self.cfg = cfg
        self.mesh = mesh
        self._sharding = sharding if sharding is not None else getattr(state.U, "sharding", None)
        self._stale = False      # U was set from outside since the last solve
        self._faces = None       # the face report (audit C8), read after every step: refused or logged
        if getattr(physics.hydro, "step_report", None) is not None:
            from luna.setup.face_report import FaceLog
            self._faces = FaceLog(cfg, physics.hydro.grid, jax.process_index() == 0)
        every = int(getattr(cfg, "timing_every", 0) or 0)
        self._timing = (StepTiming(every, getattr(cfg, "output_dir", "."), jax.process_index() == 0,
                                   fresh=not getattr(cfg, "restart", False)) if every > 0 else None)

    @property
    def U(self):
        return self._s.U

    @U.setter
    def U(self, value):
        # Put U back in the run's own sharding; a GSPMD-chosen layout is
        # misread by the next shard_map program.
        sh = self._sharding
        if sh is not None and getattr(value, "sharding", None) != sh:
            value = jax.device_put(value, sh)
        self._s.U = value
        self._stale = True

    @property
    def t(self):
        return float(self._s.t)

    @property
    def step(self):
        return int(self._s.step)

    @property
    def sources(self):
        return self._s.sources

    @property
    def species(self):
        return self._s.species

    @property
    def state(self):
        """The underlying `luna.setup.state.State`, for callers that need it."""
        return self._s

    def _refresh_gravity(self):
        """gravity_energy='flux' after U was set from outside: Phi^n must be the potential of this rho^n."""
        grav = self.physics.gravity
        if self._stale and grav is not None and grav.energy == "flux":
            self._s = grav.refresh(self._s)
        self._stale = False

    def compute_dt(self):
        """The CFL-limited dt for the current state, unclamped."""
        dt, _ = _timestep(self._s, self.physics, float("inf"))
        return float(dt)

    def step_once(self, dt):
        """Advance by exactly `dt`."""
        self._refresh_gravity()
        self._s = self._step(dt, self._s.t + dt)

    def _step(self, dt, t_new, timer=None):
        """evolve.step, then its face report read (one sync per step) and refused or logged."""
        reports = [] if self._faces is not None else None
        s = _evolve_step(self._s, dt, t_new, self.physics, timer, reports)
        if reports:
            self._faces.push(reports[0], s.step, float(t_new))
            self._faces.flush()
        return s

    def evolve_to(self, t_target, max_steps=None):
        """Advance to `t_target` on the CFL dt; returns the number of substeps."""
        self._refresh_gravity()
        n = 0
        while self._s.t < t_target:
            timer = self._timing.timer(self._s.step + 1) if self._timing is not None else None
            if timer is None:
                dt, t_new = _timestep(self._s, self.physics, t_target)
                self._s = self._step(dt, t_new)
            else:
                with timer:
                    dt, t_new = _timestep(self._s, self.physics, t_target, timer)
                    self._s = self._step(dt, t_new, timer)
                self._timing.record(timer, self._s, dt)
            n += 1
            if max_steps is not None and n >= max_steps and self._s.t < t_target:
                raise RuntimeError(
                    f"evolve_to: exceeded max_steps={max_steps} at "
                    f"t={self._s.t:.6e} (target {t_target:.6e})")
        return n


def build(cfg, *, is_main=False, U0=None):
    """Assemble a run from `cfg` as a `LunaState`; `U0` replaces the initial U."""
    from luna.ic_generation.ic import resolve
    from luna.setup.assemble import assemble
    from luna.setup.evolve import initial_state, release_initial
    from luna.setup.ics import read_ics
    from luna.setup.mesh import build_mesh, decomposition
    from luna.setup.run_kind import run_kind

    cfg = resolve(cfg)
    kind = run_kind(cfg)
    layout = decomposition(cfg, kind)
    mesh, _ = build_mesh(cfg, layout, kind)
    init = read_ics(cfg, mesh, kind, layout, is_main)
    physics = assemble(cfg, mesh, kind, layout, init, is_main)
    st = initial_state(init, physics)
    release_initial(init, physics)
    # the run's layout, not that of U0 (usually one unsharded array)
    sharding = getattr(st.U, "sharding", None)
    if U0 is not None:
        st.U = U0 if sharding is None else jax.device_put(U0, sharding)
    if physics.gravity is not None:
        # Opening solve (left out of initial_state); after the U0 override so
        # the potential belongs to the gas the caller set.
        st = physics.gravity.init(st)
    return LunaState(st, physics, cfg, mesh, sharding=sharding)
