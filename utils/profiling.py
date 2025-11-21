"""
Profiling utilities for measuring performance of simulation components.

Usage:
    from utils.profiling import ProfileTimer

    with ProfileTimer("flux_computation"):
        result = compute_flux(...)
"""

import time
import jax
from collections import defaultdict
from contextlib import contextmanager


class ProfileTimer:
    """
    Context manager for timing code sections with GPU synchronization.

    Automatically handles JAX's asynchronous execution by calling block_until_ready().
    """

    # Class-level storage for all timings
    _timings = defaultdict(list)
    _enabled = False

    @classmethod
    def enable(cls):
        """Enable profiling."""
        cls._enabled = True
        cls._timings.clear()

    @classmethod
    def disable(cls):
        """Disable profiling."""
        cls._enabled = False

    @classmethod
    def is_enabled(cls):
        """Check if profiling is enabled."""
        return cls._enabled

    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            Name of the code section being timed
        """
        self.name = name
        self.start_time = None

    def __enter__(self):
        if not self._enabled:
            return self

        # Force GPU synchronization before timing
        jax.block_until_ready(jax.numpy.array(0.0))
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled or self.start_time is None:
            return

        # Force GPU synchronization after timing
        jax.block_until_ready(jax.numpy.array(0.0))
        elapsed = (time.time() - self.start_time) * 1000  # Convert to ms
        self._timings[self.name].append(elapsed)

    @classmethod
    def report(cls, min_samples=10):
        """
        Print a summary report of all timings.

        Parameters
        ----------
        min_samples : int
            Minimum number of samples before including in report
        """
        if not cls._timings:
            print("No profiling data collected.")
            return

        print("="*80)
        print("PERFORMANCE PROFILE")
        print("="*80)
        print()

        # Calculate statistics for each section
        stats = {}
        total_time = 0.0

        for name, times in cls._timings.items():
            if len(times) < min_samples:
                continue

            # Skip first few samples (JIT compilation warmup)
            times_filtered = times[3:] if len(times) > 3 else times

            if not times_filtered:
                continue

            avg_time = sum(times_filtered) / len(times_filtered)
            min_time = min(times_filtered)
            max_time = max(times_filtered)
            total_time += avg_time

            stats[name] = {
                'avg': avg_time,
                'min': min_time,
                'max': max_time,
                'count': len(times_filtered)
            }

        # Sort by average time (descending)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['avg'], reverse=True)

        # Print header
        print(f"{'Section':<30} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12} {'% Total':<10} {'Samples':<10}")
        print("-"*80)

        # Print each section
        for name, stat in sorted_stats:
            pct = (stat['avg'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<30} {stat['avg']:>10.2f}  {stat['min']:>10.2f}  {stat['max']:>10.2f}  {pct:>8.1f}%  {stat['count']:>8}")

        print("-"*80)
        print(f"{'TOTAL':<30} {total_time:>10.2f}  {'':>10}  {'':>10}  {'100.0%':>10}  {'':>8}")
        print()

        # Identify bottlenecks
        print("BOTTLENECK ANALYSIS")
        print("-"*80)
        if sorted_stats:
            top3 = sorted_stats[:3]
            for i, (name, stat) in enumerate(top3, 1):
                pct = (stat['avg'] / total_time * 100) if total_time > 0 else 0
                print(f"{i}. {name}: {pct:.1f}% of total time")
        print()

    @classmethod
    def get_stats(cls):
        """Return raw timing statistics."""
        return dict(cls._timings)


def profile_function(func, name=None):
    """
    Decorator to profile a function.

    Parameters
    ----------
    func : callable
        Function to profile
    name : str, optional
        Name for profiling (defaults to function name)

    Returns
    -------
    wrapped : callable
        Wrapped function with profiling
    """
    if name is None:
        name = func.__name__

    def wrapper(*args, **kwargs):
        with ProfileTimer(name):
            result = func(*args, **kwargs)
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()
            return result

    return wrapper


@contextmanager
def profile_section(name):
    """
    Context manager for profiling a code section.

    Usage:
        with profile_section("my_computation"):
            result = expensive_operation()
    """
    with ProfileTimer(name):
        yield
