"""Operator-split physics source terms.

Each module exposes a setup_* factory that returns either None (disabled)
or a per-step apply function the runner calls after the hydro update.
"""
