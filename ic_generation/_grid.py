"""Leaf helper shared by the IC generators (kept import-cycle-free: ic.py
imports every IC module, so the modules must not import back from ic.py)."""


def local_range(n, sl):
    """(start, end) cell range for an axis: the full [0, n) or the slice sl."""
    return (0, n) if sl is None else sl
