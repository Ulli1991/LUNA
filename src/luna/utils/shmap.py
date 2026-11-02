"""``shard_map`` with jax's replication/varying-axes check disabled where it rejects LUNA's scans.

The check is a typing rule: the multigrid smoother's scan carries start replicated and become varying.
"""
import inspect

import jax
from jax.experimental.shard_map import shard_map as _shard_map

_params = inspect.signature(_shard_map).parameters
_CHECK_KW = ("check_vma" if "check_vma" in _params
             else "check_rep" if "check_rep" in _params else None)


def _needs_bypass():
    """Return True for jax 0.6-0.9, the only versions that reject LUNA's scan carries."""
    # disabling the check on jax >= 0.10 stalls multi-GPU runs on ROCm: leave it on there
    try:
        major, minor = (int(v) for v in jax.__version__.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (0, 6) and (major, minor) < (0, 10)


_BYPASS = _CHECK_KW is not None and _needs_bypass()


def shard_map(f, *args, **kwargs):
    """jax shard_map, with the varying-axes check off where needed."""
    if _BYPASS:
        kwargs.setdefault(_CHECK_KW, False)
    return _shard_map(f, *args, **kwargs)
