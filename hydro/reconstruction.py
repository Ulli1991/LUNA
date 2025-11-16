import jax.numpy as jnp

def plm_reconstruct(U_pad, dUx, dUy):
    Qc = U_pad[:,1:-1,1:-1]

    UL = Qc[:, :-1, :] + 0.5*dUx[:, :-1, :]
    UR = Qc[:, 1:,  :] - 0.5*dUx[:, 1:, :]

    UB = Qc[:, :, :-1] + 0.5*dUy[:, :, :-1]
    UT = Qc[:, :, 1: ] - 0.5*dUy[:, :, 1: ]

    return UL, UR, UB, UT

