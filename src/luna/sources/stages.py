"""Operator-split sources as stages with `before`/`after` halves (and optional `dt_limit`).

See docs/ARCHITECTURE.md ("The loop"); carried state lives in `state.sources[name]`.
"""


class CellLocal:
    """Cell-local stage applying the same update over h before and after the hydro."""

    def __init__(self, name, apply):
        self.name = name
        self._apply = apply

    def before(self, state, h, dt):
        return state.update(U=self._apply(state.U, h))

    after = before


class ComovingEnergy:
    """The (3 gamma - 5) internal-energy term: factor (a_mid / a_0)^p before, (a_1 / a_mid)^p after the hydro."""
    name = "comoving_energy"

    def __init__(self, clock, exponent, scale):
        self._clock, self._p, self._scale = clock, float(exponent), scale

    def before(self, state, h, dt):
        a_mid = float(self._clock.scale(state.t + h))
        return state.update(U=self._scale(state.U, (a_mid / float(state.a)) ** self._p))

    def after(self, state, h, dt):
        a_mid = float(self._clock.scale(state.t - h))
        return state.update(U=self._scale(state.U, (float(state.a) / a_mid) ** self._p))


class Cooling(CellLocal):
    """Explicit sub-cycled cooling; keeps the solver for the substep diagnostics."""

    def __init__(self, solver, cfg):
        from luna.sources.cooling import apply_cooling
        super().__init__("cooling", lambda U, h: apply_cooling(U, h, cfg, solver))
        self.solver = solver
        self.cfg = cfg

    def substeps_needed(self, U, dt, mesh):
        from luna.sources.cooling import cooling_substeps_needed
        return int(cooling_substeps_needed(U, dt, self.cfg, self.solver, mesh=mesh))


class Wind(CellLocal):
    """CC85 injection, its dt bounded by the CFL dt at the sound speed it gives a cold cell.

    The hydro rule (cell_dt) at rest, dt = CFL / sum_d (c / dx_d), at the sound speed of the
    first half-injection into the emptiest cell, c^2 = gamma (gamma-1) e_dot dt / (2 rho_min),
    solved for dt. Fixed at setup from rho_min; it only matters at the cold start.
    """

    def __init__(self, cc85, *, gamma, dxs, CFL, rho_min):
        super().__init__("cc85", cc85["apply"])
        self.apply_eta = cc85["apply_eta"]
        coef = float(CFL) / sum(1.0 / float(d) for d in dxs) * (
            2.0 * float(rho_min) / (float(gamma) * (float(gamma) - 1.0)
                                    * float(cc85["e_dot"]))) ** 0.5
        self.dt_max = coef ** (2.0 / 3.0)

    def dt_limit(self, state):
        return self.dt_max


def injection_rho_min(U, mask, mesh=None):
    """Return the lowest density inside the injection sphere, reduced on the host."""
    # Host NumPy, not a device reduction: a mesh reduction here stalls the first CFL call on ROCm.
    import jax
    import numpy as np
    rho_by_dev = {s.device: np.asarray(s.data)[0] for s in U.addressable_shards}
    lo = np.inf
    for s in mask.addressable_shards:
        m = np.asarray(s.data) > 0
        rho = rho_by_dev[s.device]
        if m.any():
            lo = min(lo, float(rho[m].min()))
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        lo = float(np.min(multihost_utils.process_allgather(np.asarray(lo))))
    return lo


class Forcing:
    """OU forcing: one draw per step in `after`, impulse h F(a_old) + h F(a_new) (trapezoid)."""
    name = "forcing"

    def __init__(self, ou):
        self.ou = ou
        self._kick, self._evolve = ou["kick"], ou["evolve"]
        self.carry0 = (ou["a"], ou["key"])

    def before(self, state, h, dt):
        a, _ = state.sources[self.name]
        return state.update(U=self._kick(state.U, a, h))

    def after(self, state, h, dt):
        a, key = state.sources[self.name]
        a, key = self._evolve(a, key, dt)
        return state.update(U=self._kick(state.U, a, h)).with_source(self.name, (a, key))


class ForcingTheta:
    """Forcing with differentiable theta = (sigma_1..sigma_M, zeta[, T_ac]): carries the unit state b, kicks with a(b; theta)."""
    name = "forcing"

    def __init__(self, ou, theta, tac=False):
        import jax
        self._kick, drv = ou["kick"], ou["driver"]
        if tac:
            sig, zeta, T = theta[:-2], theta[-2], theta[-1]

            def evolve(b, key, dt):
                new_key, sub = jax.random.split(key)          # the split of ou["unit_evolve"]
                return drv.unit_evolve(b, dt, sub, T), new_key
            self._evolve = evolve
        else:
            sig, zeta = theta[:-1], theta[-1]
            self._evolve = ou["unit_evolve"]
        self._amp = lambda b: drv.amplitudes(b, sig, zeta)
        self.carry0 = (ou["b"], ou["key"])

    def before(self, state, h, dt):
        b, _ = state.sources[self.name]
        return state.update(U=self._kick(state.U, self._amp(b), h))

    def after(self, state, h, dt):
        b, key = state.sources[self.name]
        b, key = self._evolve(b, key, dt)
        return state.update(U=self._kick(state.U, self._amp(b), h)).with_source(self.name, (b, key))


class Radiation:
    """Radiation field stage; `after` runs the half-step in reverse order (docs/RADIATION.md)."""
    name = "radiation"

    def __init__(self, rad):
        self._before, self._after = rad["before"], rad["after"]
        self.carry0 = rad["R0"]
        self.dt_max = rad["dt_max"]
        self._dt_ion = rad.get("dt_ion")
        self._finite = rad.get("finite")

    def before(self, state, h, dt):
        U, R = self._before(state.U, state.sources[self.name], h)
        return state.update(U=U).with_source(self.name, R)

    def after(self, state, h, dt):
        U, R = self._after(state.U, state.sources[self.name], h)
        return state.update(U=U).with_source(self.name, R)

    def dt_limit(self, state):
        # Quiet is not correct: a NaN field keeps stepping (ionize_implicit read it as fully ionised)
        if self._finite is not None and not bool(self._finite(state.sources[self.name])):
            raise RuntimeError(
                f"non-finite radiation field at step {state.step} (t={state.t:.6e}): the "
                "radiation carry went NaN/Inf. Last good snapshot is the newest in the output dir.")
        if self._dt_ion is None:
            return self.dt_max
        return min(self.dt_max, float(self._dt_ion(state.U, state.sources[self.name])))


class Corrector:
    """Learned 3D flux correction, applied once after the hydro with the full dt (innermost stage)."""
    name = "corrector"

    def __init__(self, apply, params):
        self._apply, self.params = apply, params

    def with_params(self, params):
        return Corrector(self._apply, params)

    def before(self, state, h, dt):
        return state

    def after(self, state, h, dt):
        if self.params is None:
            return state
        return state.update(U=self._apply(self.params, state.U, dt))


class Orbits:
    """Prescribed-orbit point masses: half at the start position, hydro, advance, half at the end."""
    name = "particles"

    def __init__(self, pm):
        self._half, self._advance = pm["half"], pm["advance"]
        self.carry0 = pm["pos"]

    def before(self, state, h, dt):
        return state.update(U=self._half(state.U, state.sources[self.name], h))

    def after(self, state, h, dt):
        pos = self._advance(state.sources[self.name], dt)
        return state.update(U=self._half(state.U, pos, h)).with_source(self.name, pos)
