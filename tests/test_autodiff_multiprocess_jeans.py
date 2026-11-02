"""The jeans IC inversion's derivatives on 8 processes equal one device (see tests/autodiff_multiprocess.py)."""
from tests.autodiff_multiprocess import check_across_processes


def test_derivatives_across_processes_equal_one_device(tmp_path):
    check_across_processes("jeans", tmp_path)
