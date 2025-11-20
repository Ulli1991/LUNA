"""
Memory tracking utilities for JAX hydro simulations.
Supports GPU (CUDA/ROCm) and CPU memory tracking.
"""

import os
import jax


def get_memory_usage():
    """
    Get current memory usage in GB.
    Returns GPU memory if available, otherwise system memory.

    Returns
    -------
    current_gb : float or None
        Current memory usage in GB
    peak_gb : float or None
        Peak memory usage in GB (if available)
    """
    try:
        # Try to get GPU memory usage
        devices = jax.devices()
        device = devices[0]

        # For GPU backends (CUDA/ROCm), query device memory
        if device.platform in ['gpu', 'cuda', 'rocm']:
            # Get memory stats from JAX
            memory_stats = device.memory_stats()
            if memory_stats:
                bytes_in_use = memory_stats.get('bytes_in_use', 0)
                peak_bytes = memory_stats.get('peak_bytes_in_use', 0)
                gb_in_use = bytes_in_use / (1024**3)
                gb_peak = peak_bytes / (1024**3)
                return gb_in_use, gb_peak

        # Fallback: try system memory
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            gb_in_use = mem_info.rss / (1024**3)
            return gb_in_use, gb_in_use
        except ImportError:
            return None, None

    except Exception as e:
        print(f"Warning: Could not query memory usage: {e}")
        return None, None


def print_memory_usage(label=""):
    """
    Print current memory usage with optional label.

    Parameters
    ----------
    label : str
        Optional prefix label for the output
    """
    current_gb, peak_gb = get_memory_usage()
    if current_gb is not None:
        if peak_gb is not None and peak_gb != current_gb:
            print(f"{label}Memory usage: {current_gb:.2f} GB (peak: {peak_gb:.2f} GB)")
        else:
            print(f"{label}Memory usage: {current_gb:.2f} GB")
    else:
        print(f"{label}Memory tracking not available")
