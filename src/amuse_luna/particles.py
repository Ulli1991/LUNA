"""Fixed-capacity, tag-addressed host particle store for the Torch/AMUSE coupling.

Constant array shapes plus an `active` mask, so jitted kernels never recompile; see docs/TORCH.md."""

import jax.numpy as jnp
import numpy as np


# What Torch parks on our particles, with its unit name (resolved in
# interface.py, the only module that imports amuse); None = a bare code.
PARKED_ATTRS = {
    "oldmass":       "mass",
    "creation_time": "time",
    "radius":        "length",
    "stype":         None,
    "nion":          "rate",
    "eion":          "photon_energy",
    "sigh":          "area",
    "npep":          "rate",
    "epep":          "photon_energy",
    "sigd":          "area",
    "wind_mass":     "mass_rate",
    "wind_vel":      "speed",
    "rel_mass":      "mass",
    "rel_age":       "time",
    "corem":         "mass",
    "co_corem":      "mass",
}


def default_star_schema(dtype=jnp.float64):
    """Every attribute Torch exchanges with the hydro code for a star."""
    return {
        "position": ((3,), dtype),
        "velocity": ((3,), dtype),
        "acceleration": ((3,), dtype),   # from the last bridge kick
        "mass": ((), dtype),
        "oldmass": ((), dtype),
        "creation_time": ((), dtype),
        "radius": ((), dtype),
        "stype": ((), jnp.int32),        # stellar type code

        # -- radiation: photon rate, mean energy, cross section (H, then PE/dust)
        "nion": ((), dtype),
        "eion": ((), dtype),
        "sigh": ((), dtype),
        "npep": ((), dtype),
        "epep": ((), dtype),
        "sigd": ((), dtype),

        # -- winds
        "wind_mass": ((), dtype),
        "wind_vel": ((), dtype),
        "wind_bgdy": ((), dtype),        # density at the wind's start (FLASH BGDY; 0 = not yet)

        # -- SeBa state, parked here between steps
        "rel_mass": ((), dtype),         # relative (ZAMS) mass
        "rel_age": ((), dtype),
        "corem": ((), dtype),
        "co_corem": ((), dtype),
    }


def default_sink_schema(dtype=jnp.float64):
    """Star attributes plus the accreted-gas moments Torch reads from sinks."""
    s = default_star_schema(dtype)
    s.update({
        "mean_cs": ((), dtype),
        "mean_velocity": ((3,), dtype),
        "var_velocity": ((3,), dtype),     # per axis
        "angular_momentum": ((3,), dtype),
    })
    return s


def empty_arrays(capacity, schema):
    """Allocate the static host arrays for a store of the given capacity."""
    arrays = {}
    for name, (cshape, dt) in schema.items():
        arrays[name] = np.zeros((capacity,) + tuple(cshape), dtype=np.dtype(dt))
    arrays["active"] = np.zeros((capacity,), dtype=bool)
    return arrays


def _host(a, dtype=None):
    """A writable numpy copy of `a` (a jax array hands numpy a read-only view)."""
    return np.array(a, dtype=dtype, copy=True)


class ParticleCapacityError(RuntimeError):
    """Raised when an add would exceed the pre-allocated capacity `N_max`."""


class ParticleStore:
    """Fixed-capacity, tag-addressed particle container; add/remove only flip `active`."""

    def __init__(self, capacity, schema, name="particles"):
        self.capacity = int(capacity)
        self.schema = dict(schema)
        self.name = name
        self.arrays = empty_arrays(self.capacity, self.schema)
        # Free slots as a stack; pop the lowest index first for determinism.
        self._free = list(range(self.capacity - 1, -1, -1))
        self._tag_to_slot = {}
        self._slot_to_tag = {}
        self._next_tag = 0

    # -- introspection ------------------------------------------------------
    @property
    def n_active(self):
        return len(self._tag_to_slot)

    def active_tags(self):
        """Tags of all live particles, in ascending tag order."""
        return np.array(sorted(self._tag_to_slot), dtype=np.int64)

    def slots_for(self, tags):
        """Slot indices for `tags`, via a cached sorted-tag searchsorted."""
        tags = np.asarray(np.atleast_1d(tags), dtype=np.int64).ravel()
        if getattr(self, "_lookup", None) is None:
            t = np.fromiter(self._tag_to_slot.keys(), dtype=np.int64, count=len(self._tag_to_slot))
            o = np.argsort(t)
            self._lookup = (t[o], np.fromiter(self._tag_to_slot.values(), dtype=np.int64,
                                              count=len(self._tag_to_slot))[o])
        keys, slots = self._lookup
        i = np.searchsorted(keys, tags)
        bad = (i >= len(keys)) | (keys[np.minimum(i, len(keys) - 1)] != tags) if len(keys) else np.ones(len(tags), bool)
        if bad.any():
            raise KeyError(int(tags[np.argmax(bad)]))
        return slots[i]

    def _validate_attr(self, name):
        if name not in self.schema:
            raise KeyError(f"{self.name!r} store has no attribute {name!r}; "
                           f"known: {sorted(self.schema)}")

    # -- dynamic add/remove (host-side; outside jit) ------------------------
    def add(self, n=None, tags=None, **attr_values):
        """Create `n` particles (attributes as name=array, rest zero); return their tags."""
        if n is None:
            if tags is not None:
                n = len(np.atleast_1d(tags))
            elif attr_values:
                n = len(np.atleast_1d(next(iter(attr_values.values()))))
            else:
                raise ValueError("add() needs either n= or at least one attribute array")
        n = int(n)
        if n <= 0:
            return np.empty(0, dtype=np.int64)
        if n > len(self._free):
            raise ParticleCapacityError(
                f"{self.name!r}: add({n}) exceeds remaining capacity "
                f"{len(self._free)} (N_max={self.capacity}). Increase the cap.")

        slots = np.array([self._free.pop() for _ in range(n)], dtype=np.int64)
        if tags is None:
            tags = np.array([self._new_tag() for _ in range(n)], dtype=np.int64)
        else:
            tags = np.asarray(np.atleast_1d(tags), dtype=np.int64)
            used = [int(t) for t in tags if int(t) in self._tag_to_slot]
            if used:
                raise ValueError(f"{self.name!r}: tags already in use: {used}")
            self._next_tag = max(self._next_tag, int(tags.max()) + 1)
        for t, s in zip(tags, slots):
            self._tag_to_slot[int(t)] = int(s)
            self._slot_to_tag[int(s)] = int(t)
        self._lookup = None

        self.arrays["active"][slots] = True
        for name, vals in attr_values.items():
            self._validate_attr(name)
            self.arrays[name][slots] = np.asarray(vals)
        return tags

    def remove(self, tags):
        """Remove particles by tag: clear the mask, zero the mass, free slots."""
        tags = np.atleast_1d(tags)
        if tags.size == 0:
            return
        slots = self.slots_for(tags)
        self.arrays["active"][slots] = False
        self.arrays["mass"][slots] = 0.0                          # no phantom gravity
        for t, s in zip(tags, slots):
            del self._tag_to_slot[int(t)]
            del self._slot_to_tag[int(s)]
            self._free.append(int(s))
        self._lookup = None

    def _new_tag(self):
        t = self._next_tag
        self._next_tag += 1
        return t

    # -- checkpoint ---------------------------------------------------------
    def dump(self):
        """Everything a restart needs: the arrays as numpy, the tag map."""
        return {"arrays": {k: np.asarray(v) for k, v in self.arrays.items()},
                "tag_to_slot": {int(t): int(s) for t, s in self._tag_to_slot.items()},
                "next_tag": int(self._next_tag)}

    def load(self, d):
        """The inverse of `dump`, onto a store of the same capacity and schema."""
        for k, v in d["arrays"].items():
            if k not in self.arrays:
                raise KeyError(f"{self.name!r}: checkpoint carries unknown attribute {k!r}")
            self.arrays[k] = _host(v, dtype=self.arrays[k].dtype)
        self._tag_to_slot = {int(t): int(s) for t, s in d["tag_to_slot"].items()}
        self._slot_to_tag = {s: t for t, s in self._tag_to_slot.items()}
        taken = set(self._slot_to_tag)
        self._free = [s for s in range(self.capacity - 1, -1, -1) if s not in taken]
        self._next_tag = int(d["next_tag"])
        self._lookup = None

    # -- by-tag get/set -----------------------------------------------------
    def get(self, name, tags=None):
        """Gather attribute `name` for `tags` (all live particles if None)."""
        self._validate_attr(name)
        if tags is None:
            tags = self.active_tags()
        slots = self.slots_for(tags)
        return self.arrays[name][slots]

    def set(self, name, tags, values):
        """Scatter `values` into `name` for `tags`, cast to its dtype (AMUSE sends float64)."""
        self._validate_attr(name)
        slots = self.slots_for(np.atleast_1d(tags))
        target = self.arrays[name]
        target[slots] = np.asarray(values).astype(target.dtype)

    # -- device-array exchange with the jitted physics ----------------------
    def update_arrays(self, new_arrays):
        """Take a jitted op's new values; it may change values, never the slot layout."""
        if set(new_arrays) != set(self.arrays):
            raise ValueError("update_arrays: key set changed — shape/layout must "
                             "be preserved across the jitted physics op")
        self.arrays = {k: _host(v, dtype=self.arrays[k].dtype) for k, v in new_arrays.items()}
