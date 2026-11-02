"""What a run carries from one step to the next.

One `State` for every kind of run: the gas grid (or None), the clock's time
and scale factor, the potential of the last gravity solve with the force it
gives the gas, one `Particles` per species, and each source's own carry. The
loop passes it whole; every stage returns a new one with `dataclasses.replace`.
"""
from dataclasses import dataclass, field, fields, replace
from typing import Any

import jax


@dataclass
class Particles:
    """One species' arrays. `p` is the canonical momentum (super-comoving
    runs) or the velocity (proper time); `g` is the force at the particles
    from the last solve, already multiplied by the kick scale, carried across
    the step for the opening half-kick. `active` marks the live slots of a
    fixed-capacity buffer (all ones when the buffer is exact). `ids` are the
    persistent particle IDs, moved with every exchange and resort."""
    pos: Any
    p: Any
    mass: Any
    active: Any = None
    g: Any = None
    extra: Any = None             # per-species extras, a dict of (N, ...) arrays;
                                  # None for species that carry none. Sinks use
                                  # it for the accreted-gas moments a sink->star
                                  # prescription needs (species/sinks.py).
    ids: Any = None               # (N,) integer IDs; None for species without them


@dataclass
class State:
    U: Any = None                 # gas grid, sharded; None for dark matter only
    t: float = 0.0                # the clock's time (tau for comoving)
    a: float = 1.0                # clock.scale(t), kept so no stage interpolates
    step: int = 0
    psi: Any = None               # potential of the last solve, unit coefficient
    psi_prev: Any = None          # the solve before it, kept only to extrapolate the next seed
    dt_prev: float = 0.0          # the step that took psi_prev to psi
    g_gas: Any = None             # -grad psi on the grid, unscaled
    s_kick: float = 1.0           # kick scale of the last solve, clock.kick_scale(a)
    mass_flux: Any = None         # gravity_energy='flux': this step's face mass fluxes (Fx, Fy, Fz)
    kick_work: Any = None         # gravity_energy='flux': the energy the opening half-kick added
    species: dict = field(default_factory=dict)   # name -> Particles
    sources: dict = field(default_factory=dict)   # name -> that source's carry
    stats: dict = field(default_factory=dict)     # host floats of the last solve

    def update(self, **kw):
        return replace(self, **kw)

    def with_species(self, name, part):
        return replace(self, species={**self.species, name: part})

    def with_source(self, name, carry):
        return replace(self, sources={**self.sources, name: carry})


def _pytree(cls):
    """Every field is data, so a State can be the carry of a lax.scan."""
    names = [f.name for f in fields(cls)]
    jax.tree_util.register_dataclass(cls, data_fields=names, meta_fields=[])
    return cls


_pytree(Particles)
_pytree(State)
