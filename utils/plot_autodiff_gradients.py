"""
Plotting functions for visualizing autodiff gradients.

Creates publication-quality plots showing:
1. Gradients vs spatial position
2. Initial condition with gradient overlay
3. Comparison of forward and adjoint sensitivities
"""
import matplotlib.pyplot as plt
import numpy as np
from file_io.read_autodiff_result import read_autodiff_result

def plot_gradients_1d(filename, output_prefix="autodiff_gradients"):
    """
    Create comprehensive gradient visualization for 1D problems.

    Generates multiple plots:
    1. All gradients on separate subplots
    2. Initial condition overlay with gradients
    3. Normalized gradients (for comparison)

    Parameters
    ----------
    filename : str
        Path to HDF5 file with autodiff results
    output_prefix : str
        Prefix for output filenames (default: 'autodiff_gradients')
    """

    # Load data
    data = read_autodiff_result(filename)

    x = data["x"]
    U0 = data["U0"]
    U_final = data["U_final"]
    grads = data["grads"]
    obs_name = data["observable_name"]
    obs_value = data["observable_value"]
    metadata = data["metadata"]

    # Extract variables
    rho0 = U0[0]
    momx0 = U0[1]
    E0 = U0[-1]

    # Extract gradients
    dJ_drho = grads[0]
    dJ_dmomx = grads[1]
    dJ_dE = grads[-1]

    # Get observation point if applicable
    obs_attrs = data["observable_attrs"]
    obs_x = obs_attrs.get("observation_position_x", None)

    # ========================================
    # PLOT 1: All gradients (stacked subplots)
    # ========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Gradient w.r.t. density
    axes[0].plot(x, dJ_drho, 'b-', linewidth=1.5)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    if obs_x is not None:
        axes[0].axvline(obs_x, color='r', linestyle='--', alpha=0.5, label=f'Obs. point')
    axes[0].set_ylabel(r'$\partial J / \partial \rho_0$', fontsize=12)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Gradient w.r.t. momentum
    axes[1].plot(x, dJ_dmomx, 'g-', linewidth=1.5)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    if obs_x is not None:
        axes[1].axvline(obs_x, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel(r'$\partial J / \partial (\rho u)_0$', fontsize=12)
    axes[1].grid(alpha=0.3)

    # Gradient w.r.t. energy
    axes[2].plot(x, dJ_dE, 'r-', linewidth=1.5)
    axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
    if obs_x is not None:
        axes[2].axvline(obs_x, color='r', linestyle='--', alpha=0.5)
    axes[2].set_ylabel(r'$\partial J / \partial E_0$', fontsize=12)
    axes[2].set_xlabel('Position x', fontsize=12)
    axes[2].grid(alpha=0.3)

    # Title
    title = f"Gradients: {obs_name}\n"
    title += f"Test: {metadata['test_problem']}, J = {obs_value:.6e}"
    fig.suptitle(title, fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_stacked.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_stacked.png")
    plt.close()

    # ========================================
    # PLOT 2: Initial condition + gradients overlay
    # ========================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Initial condition
    ax_top = axes[0]
    vx0 = momx0 / rho0

    # Compute pressure for IC
    ke0 = 0.5 * momx0**2 / rho0
    internal0 = E0 - ke0
    pressure0 = (metadata['gamma'] - 1.0) * internal0

    ax_top.plot(x, rho0, 'b-', label=r'$\rho_0$', linewidth=1.5)
    ax_top.set_ylabel('Density', fontsize=12, color='b')
    ax_top.tick_params(axis='y', labelcolor='b')
    ax_top.grid(alpha=0.3)

    ax_twin = ax_top.twinx()
    ax_twin.plot(x, pressure0, 'r-', label=r'$p_0$', linewidth=1.5)
    ax_twin.set_ylabel('Pressure', fontsize=12, color='r')
    ax_twin.tick_params(axis='y', labelcolor='r')

    if obs_x is not None:
        ax_top.axvline(obs_x, color='k', linestyle='--', alpha=0.5, linewidth=2)

    ax_top.set_title('Initial Condition', fontsize=12, fontweight='bold')

    # Bottom: Gradients (normalized to [0,1] for comparison)
    ax_bottom = axes[1]

    # Normalize gradients for visualization
    def normalize(arr):
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max - arr_min > 1e-12:
            return (arr - arr_min) / (arr_max - arr_min)
        return arr * 0

    ax_bottom.plot(x, normalize(dJ_drho), 'b-', label=r'$\partial J/\partial \rho_0$ (norm.)', linewidth=1.5, alpha=0.7)
    ax_bottom.plot(x, normalize(dJ_dmomx), 'g-', label=r'$\partial J/\partial (\rho u)_0$ (norm.)', linewidth=1.5, alpha=0.7)
    ax_bottom.plot(x, normalize(dJ_dE), 'r-', label=r'$\partial J/\partial E_0$ (norm.)', linewidth=1.5, alpha=0.7)

    if obs_x is not None:
        ax_bottom.axvline(obs_x, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Obs. point')

    ax_bottom.set_xlabel('Position x', fontsize=12)
    ax_bottom.set_ylabel('Normalized Gradient', fontsize=12)
    ax_bottom.set_title('Gradient Sensitivity (Normalized)', fontsize=12, fontweight='bold')
    ax_bottom.legend(fontsize=10)
    ax_bottom.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_overlay.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_overlay.png")
    plt.close()

    # ========================================
    # PLOT 3: Domain of influence
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute total gradient magnitude
    grad_magnitude = np.sqrt(dJ_drho**2 + dJ_dmomx**2 + dJ_dE**2)

    # Threshold for "significant" gradients (1% of max)
    threshold = 0.01 * np.max(grad_magnitude)
    influence_region = grad_magnitude > threshold

    ax.fill_between(x, 0, grad_magnitude, alpha=0.5, color='purple', label='Gradient magnitude')
    ax.axhline(threshold, color='k', linestyle='--', alpha=0.5, label=f'Threshold ({0.01*100:.0f}% of max)')

    if obs_x is not None:
        ax.axvline(obs_x, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Observation point')

    # Estimate domain of influence boundaries
    influence_indices = np.where(influence_region)[0]
    if len(influence_indices) > 0:
        left_bound = x[influence_indices[0]]
        right_bound = x[influence_indices[-1]]
        ax.axvline(left_bound, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Domain of influence')
        ax.axvline(right_bound, color='orange', linestyle=':', linewidth=2, alpha=0.7)

        # Annotate domain size
        domain_size = right_bound - left_bound
        ax.text(0.02, 0.98, f'Influence domain: [{left_bound:.3f}, {right_bound:.3f}]\nSize: {domain_size:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

    ax.set_xlabel('Position x', fontsize=12)
    ax.set_ylabel(r'$\|\nabla_U J\|$', fontsize=12)
    ax.set_title(f'Domain of Influence Analysis\nObservable: {obs_name}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_domain.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_domain.png")
    plt.close()

    # ========================================
    # PLOT 4: Forward and Final state comparison
    # ========================================
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Extract final state
    rho_final = U_final[0]
    momx_final = U_final[1]
    E_final = U_final[-1]
    vx_final = momx_final / rho_final
    ke_final = 0.5 * momx_final**2 / rho_final
    internal_final = E_final - ke_final
    pressure_final = (metadata['gamma'] - 1.0) * internal_final

    # Top: Initial vs Final density
    axes[0].plot(x, rho0, 'b--', label=r'$\rho_0$ (initial)', linewidth=1.5, alpha=0.7)
    axes[0].plot(x, rho_final, 'b-', label=r'$\rho_{final}$', linewidth=1.5)
    if obs_x is not None:
        axes[0].axvline(obs_x, color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title('Evolution: Initial → Final', fontsize=12, fontweight='bold')

    # Bottom: Gradient magnitude
    axes[1].fill_between(x, 0, grad_magnitude, alpha=0.5, color='purple')
    if obs_x is not None:
        axes[1].axvline(obs_x, color='r', linestyle='--', alpha=0.5, linewidth=2, label='Observation')
    axes[1].set_xlabel('Position x', fontsize=12)
    axes[1].set_ylabel(r'$\|\nabla_U J\|$', fontsize=12)
    axes[1].set_title('Sensitivity (Gradient Magnitude)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_evolution.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_prefix}_evolution.png")
    plt.close()

    print(f"\nAll plots saved with prefix: {output_prefix}")
    print(f"  - {output_prefix}_stacked.png: Individual gradients")
    print(f"  - {output_prefix}_overlay.png: IC + normalized gradients")
    print(f"  - {output_prefix}_domain.png: Domain of influence")
    print(f"  - {output_prefix}_evolution.png: Initial → Final + gradients")

def plot_gradients_summary(filename):
    """
    Quick summary plot showing the most important gradient information.

    Parameters
    ----------
    filename : str
        Path to HDF5 file with autodiff results
    """
    data = read_autodiff_result(filename)

    x = data["x"]
    grads = data["grads"]
    obs_name = data["observable_name"]
    metadata = data["metadata"]

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(np.sum(grads**2, axis=0))

    plt.figure(figsize=(10, 5))
    plt.fill_between(x, 0, grad_magnitude, alpha=0.6, color='purple')
    plt.xlabel('Position x', fontsize=12)
    plt.ylabel(r'Gradient Magnitude $\|\nabla_U J\|$', fontsize=12)
    plt.title(f'Sensitivity to Initial Conditions\nTest: {metadata["test_problem"]}, Observable: {obs_name}',
              fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: gradient_summary.png")
    plt.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_autodiff_gradients.py <autodiff_result.hdf5>")
        sys.exit(1)

    filename = sys.argv[1]
    plot_gradients_1d(filename)
    plot_gradients_summary(filename)
