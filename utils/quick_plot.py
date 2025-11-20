import os
import time

import jax
import jax.numpy as jnp

from config import Config
from ic_generation.ic import initial_conditions
from time_integration.cfl import compute_dt
from file_io.write_snapshot import write_snapshot

cfg = Config()
print("Generating density plots from all HDF5 snapshots...")
from utils.plot_density2d import plot_density2d
plot_density2d(cfg.output_dir, cfg.base_name)
