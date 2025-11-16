import matplotlib.pyplot as plt
import os

def save_density(rho, t, Lx, Ly, snapshot_idx, output_dir):
    """
    Save a density snapshot as a PNG file.

    Parameters
    ----------
    rho : array_like
        2D density array (Nx, Ny)
    t : float
        Current simulation time
    Lx, Ly : float
        Domain lengths in x and y directions
    output_dir : str
        Directory to save snapshots
    """
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(rho.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='magma', vmin=0.75, vmax=2.25)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Density at t={t:.3f}')
    plt.colorbar(im, ax=ax, label='rho')

    filename = os.path.join(output_dir, f'density_{snapshot_idx:03d}.png')
    plt.savefig(filename, dpi=300)
    plt.close(fig)

