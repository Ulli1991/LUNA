"""Cosmology subsystem: FLRW background + super-comoving integration helpers.

Re-exports the public API from ``background.py`` so that ``import cosmology`` and
``from cosmology import Cosmology, EdS, ...`` keep working unchanged. Future
cosmological-volume modules (Friedmann tables, cosmological ICs, the PM coupling)
live alongside ``background.py`` in this package.
"""
from .background import Cosmology, EdS, a_to_redshift, redshift_to_a

__all__ = ["Cosmology", "EdS", "redshift_to_a", "a_to_redshift"]
