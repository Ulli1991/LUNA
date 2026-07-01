# Read-only reference copy of upstream JAX-AMR (https://github.com/JA4S/JAX-AMR).
#
# This package is NOT a LUNA runtime dependency. It is vendored solely for:
#   1. MIT-license attribution (see LICENSE), and
#   2. parity diffing: importing upstream `amr.py` functions in tests to
#      validate LUNA's dimension-generic re-implementation in `amr/` against
#      the original 2D behaviour.
#
# We deliberately do NOT re-export `amraux` here: upstream `amraux.py` does
# `from tkinter import N`, which crashes on a headless cluster node. Importing
# `third_party.jaxamr.amr` works without it (amr.py only needs simple_solver).
