"""LUNA block-structured AMR (dimension-generic, JAX).

See DESIGN.md for the architecture and locked decisions. The package layers:
`geometry` (Morton ids, prolong/restrict), `blocks`/`halo`/`state` (topology,
ghost fill, AMRState pytree), `hydro` (per-block MUSCL-Hancock + seam fluxes),
`reflux`, `driver` (multi-level advance, Berger-Oliger subcycle, regrid),
`shard` (multi-device/multi-process distribution), `poisson` (level-by-level
gravity), `particles` (DM on the finest covering block), `cosmo` (super-
comoving), and `hybrid` (dense-L0 experiment). Consumers import the modules
directly (see `setup/runner_amr.py`); only the config and geometry core are
re-exported here.
"""

from amr.config import AMRConfig
from amr import geometry

__all__ = ["AMRConfig", "geometry"]
