"""Leaf helper for the IC generators; must not import ic.py (ic.py imports every IC module)."""


def local_range(n, sl):
    """(start, end) cell range for an axis: the full [0, n) or the slice sl."""
    return (0, n) if sl is None else sl
