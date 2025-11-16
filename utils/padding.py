import jax.numpy as jnp

def periodic_pad(u):
    return jnp.pad(u, ((0,0),(1,1),(1,1)), mode="wrap")

