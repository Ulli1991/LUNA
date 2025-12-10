"""
Memory tracking utilities for JAX hydro simulations.
Supports GPU (CUDA/ROCm) and CPU memory tracking.
"""

import os
import jax


def get_memory_usage():
    """
    Get current memory usage in GB across all devices.
    Returns GPU memory if available, otherwise system memory.

    Returns
    -------
    per_device_gb : list of float or None
        Memory usage per device in GB
    total_gb : float or None
        Total memory usage across all devices in GB
    """
    try:
        # Try to get GPU memory usage (only local/addressable devices)
        devices = jax.local_devices()

        # For GPU backends (CUDA/ROCm), query device memory
        if devices and devices[0].platform in ['gpu', 'cuda', 'rocm']:
            per_device = []
            for device in devices:
                memory_stats = device.memory_stats()
                if memory_stats:
                    bytes_in_use = memory_stats.get('bytes_in_use', 0)
                    gb_in_use = bytes_in_use / (1024**3)
                    per_device.append(gb_in_use)
                else:
                    per_device.append(0.0)

            total_gb = sum(per_device)
            return per_device, total_gb

        # Fallback: try system memory
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            gb_in_use = mem_info.rss / (1024**3)
            return [gb_in_use], gb_in_use
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
    per_device, total = get_memory_usage()
    if per_device is not None and total is not None:
        if len(per_device) > 1:
            per_device_str = ", ".join([f"{mem:.2f}" for mem in per_device])
            print(f"{label}Memory usage: {total:.2f} GB total across {len(per_device)} GPUs [{per_device_str} GB]")
        else:
            print(f"{label}Memory usage: {total:.2f} GB")
    else:
        print(f"{label}Memory tracking not available")
