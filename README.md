# jax-hydro

A JAX-based hydrodynamics simulation code for solving the Euler equations in 1D, 2D, and 3D.

## Setup

1. **Configuration**: Copy the template configuration file to create your local config:
   ```bash
   cp template_config.py config.py
   ```
   Then edit `config.py` to set your simulation parameters. This file is gitignored and won't be tracked.

2. **Dependencies**: Ensure you have JAX, NumPy, h5py, and matplotlib installed:
   ```bash
   pip install jax jaxlib numpy h5py matplotlib
   ```

## Usage

Run a simulation by executing:
```bash
python main.py
```

Configure your simulation in `config.py` by setting parameters like:
- `test_problem`: Which initial condition to use (e.g., "Sod-Shock", "Kelvin-Helmholtz")
- `SPATIAL_DIMENSION`: 1, 2, or 3
- `TIME_SOLVER`: Integrator choice (0=RK2, 1=RK4, 2=MUSCL-Hancock)
- Grid parameters, time limits, output settings, etc.
