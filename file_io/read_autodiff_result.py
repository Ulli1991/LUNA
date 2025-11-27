import h5py
import jax.numpy as jnp

def read_autodiff_result(filename):
    """
    Read autodiff results from HDF5 file.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'U0': Initial conservative variables
        - 'U_final': Final conservative variables
        - 'grads': Gradients dJ/dU0
        - 'observable_name': Name of observable
        - 'observable_value': Value of observable
        - 'x': x-coordinates
        - 'y': y-coordinates (if 2D/3D)
        - 'z': z-coordinates (if 3D)
        - 'metadata': Dictionary of simulation parameters
    """

    with h5py.File(filename, "r") as f:
        # Read initial condition
        U0 = jnp.array(f["initial/U"])

        # Read final state
        U_final = jnp.array(f["final/U"])

        # Read gradients
        grads = jnp.array(f["gradients/dJ_dU0"])

        # Read observable info
        observable_name = f["observable"].attrs["name"]
        observable_value = f["observable"].attrs["value"]

        # Check if observation point exists
        obs_attrs = dict(f["observable"].attrs)

        # Read coordinates
        x = jnp.array(f["coordinates/x"])

        spatial_dim = f["metadata"].attrs["spatial_dimension"]

        if spatial_dim >= 2:
            y = jnp.array(f["coordinates/y"])
        else:
            y = None

        if spatial_dim == 3:
            z = jnp.array(f["coordinates/z"])
        else:
            z = None

        # Read metadata
        metadata = dict(f["metadata"].attrs)

        # Read gradient statistics
        grad_stats = dict(f["gradients"].attrs)

    data = {
        "U0": U0,
        "U_final": U_final,
        "grads": grads,
        "observable_name": observable_name,
        "observable_value": observable_value,
        "observable_attrs": obs_attrs,
        "x": x,
        "y": y,
        "z": z,
        "metadata": metadata,
        "gradient_stats": grad_stats,
    }

    return data
