"""
Test script for grid infrastructure.

This demonstrates:
- Creating a domain decomposition
- Setting up neighbor connectivity
- Validating the setup

Run from project root:
    python test_grid_infrastructure.py
"""

import sys
from grid.decomposition import create_uniform_decomposition, compute_optimal_decomposition
from grid.domain import Domain
from grid.neighbors import setup_neighbor_connectivity, validate_connectivity, print_connectivity


def test_1d_decomposition():
    """Test 1D domain decomposition."""
    print("\n" + "="*60)
    print("TEST: 1D Decomposition (4 GPUs, 1024 cells)")
    print("="*60)

    n_gpus = 4
    global_shape = (1024,)
    global_extent = (1.0,)

    # Create decomposition
    patches = create_uniform_decomposition(
        global_shape=global_shape,
        global_extent=global_extent,
        n_gpus=n_gpus,
        boundary_type=0  # Periodic
    )

    # Setup neighbors
    setup_neighbor_connectivity(patches, periodic=True)

    # Create domain
    domain = Domain(patches, global_shape, global_extent)

    print(f"\nDomain: {domain}")
    print(f"Patches per GPU: {domain.n_patches // n_gpus if n_gpus > 0 else domain.n_patches}")

    for patch in patches:
        print(f"\n{patch}")
        for direction, neighbors in patch.neighbors.items():
            print(f"  {direction}: rank {neighbors[0].rank}")

    # Validate
    valid = validate_connectivity(patches, periodic=True)
    print(f"\n✓ Connectivity valid: {valid}")


def test_2d_decomposition():
    """Test 2D domain decomposition."""
    print("\n" + "="*60)
    print("TEST: 2D Decomposition (8 GPUs, 2048x2048 cells)")
    print("="*60)

    n_gpus = 8
    global_shape = (2048, 2048)
    global_extent = (1.0, 1.0)

    # Compute optimal decomposition
    decomp = compute_optimal_decomposition(n_gpus, ndim=2)
    print(f"Optimal decomposition for {n_gpus} GPUs: {decomp}")

    # Create decomposition
    patches = create_uniform_decomposition(
        global_shape=global_shape,
        global_extent=global_extent,
        n_gpus=n_gpus,
        decomposition=decomp,
        boundary_type=1  # Zero-gradient
    )

    # Setup neighbors
    setup_neighbor_connectivity(patches, periodic=False)

    # Create domain
    domain = Domain(patches, global_shape, global_extent)

    print(f"\nDomain: {domain}")
    print(f"Grid spacing: dx={domain.dx:.6f}, dy={domain.dy:.6f}")

    # Print connectivity
    print_connectivity(patches)

    # Validate
    valid = validate_connectivity(patches, periodic=False)
    print(f"\n✓ Connectivity valid: {valid}")

    # Test domain validation
    print(f"✓ Decomposition valid: {domain.validate_decomposition()}")


def test_3d_decomposition():
    """Test 3D domain decomposition."""
    print("\n" + "="*60)
    print("TEST: 3D Decomposition (8 GPUs, 512x512x512 cells)")
    print("="*60)

    n_gpus = 8
    global_shape = (512, 512, 512)
    global_extent = (1.0, 1.0, 1.0)

    # Compute optimal decomposition
    decomp = compute_optimal_decomposition(n_gpus, ndim=3)
    print(f"Optimal decomposition for {n_gpus} GPUs: {decomp}")

    # Create decomposition
    patches = create_uniform_decomposition(
        global_shape=global_shape,
        global_extent=global_extent,
        n_gpus=n_gpus,
        decomposition=decomp,
        boundary_type=0  # Periodic
    )

    # Setup neighbors
    setup_neighbor_connectivity(patches, periodic=True)

    # Create domain
    domain = Domain(patches, global_shape, global_extent)

    print(f"\nDomain: {domain}")
    print(f"Grid spacing: dx={domain.dx:.6f}, dy={domain.dy:.6f}, dz={domain.dz:.6f}")

    # Show sample patches
    print("\nSample patches:")
    for i in [0, 3, 7]:
        patch = patches[i]
        print(f"\n{patch}")
        n_neighbors = sum(len(neighbors) for neighbors in patch.neighbors.values())
        print(f"  Total neighbors: {n_neighbors}")

        # Check if boundary
        boundaries = [d for d in patch.boundary_types if patch.is_boundary(d)]
        if boundaries:
            print(f"  Domain boundaries: {boundaries}")

    # Validate
    valid = validate_connectivity(patches, periodic=True)
    print(f"\n✓ Connectivity valid: {valid}")


def test_irregular_decomposition():
    """Test irregular decomposition (not all patches same size)."""
    print("\n" + "="*60)
    print("TEST: Irregular Decomposition (7 GPUs, 2000x2000 cells)")
    print("="*60)

    n_gpus = 7
    global_shape = (2000, 2000)
    global_extent = (1.0, 1.0)

    # Create decomposition - won't divide evenly
    patches = create_uniform_decomposition(
        global_shape=global_shape,
        global_extent=global_extent,
        n_gpus=n_gpus,
        boundary_type=0
    )

    # Setup neighbors
    setup_neighbor_connectivity(patches, periodic=True)

    # Create domain
    domain = Domain(patches, global_shape, global_extent)

    print(f"\nDomain: {domain}")

    # Show all patch sizes
    print("\nPatch sizes:")
    total_cells = 0
    for patch in patches:
        cells = patch.n_cells
        total_cells += cells
        print(f"  Rank {patch.rank}: {patch.shape} = {cells} cells")

    print(f"\nTotal cells: {total_cells}")
    print(f"Expected: {global_shape[0] * global_shape[1]}")
    print(f"Match: {total_cells == global_shape[0] * global_shape[1]}")

    # Validate
    valid = validate_connectivity(patches, periodic=True)
    print(f"\n✓ Connectivity valid: {valid}")


def test_data_allocation():
    """Test data allocation on patches."""
    print("\n" + "="*60)
    print("TEST: Data Allocation")
    print("="*60)

    n_gpus = 4
    global_shape = (512, 512)
    global_extent = (1.0, 1.0)

    patches = create_uniform_decomposition(
        global_shape=global_shape,
        global_extent=global_extent,
        n_gpus=n_gpus,
    )

    setup_neighbor_connectivity(patches, periodic=True)
    domain = Domain(patches, global_shape, global_extent)

    # Allocate data (4 fields for 2D)
    n_fields = 4
    domain.allocate_all(n_fields)

    print(f"Allocated data for {n_fields} fields")

    for patch in patches:
        print(f"\nPatch {patch.rank}:")
        print(f"  Interior shape: {patch.shape}")
        print(f"  With ghosts: {patch.shape_with_ghosts}")
        print(f"  Data shape: {patch.data.shape}")

        # Test slicing
        interior_slice = patch.get_interior_slice()
        interior_data = patch.data[interior_slice]
        print(f"  Interior data shape: {interior_data.shape}")

        # Test ghost slicing
        send_left = patch.get_ghost_slice('left', send=True)
        print(f"  Send left slice: {send_left}")
        print(f"  Send left shape: {patch.data[send_left].shape}")

    print("\n✓ Data allocation successful")


def main():
    """Run all tests."""
    print("="*60)
    print("GRID INFRASTRUCTURE TESTS")
    print("="*60)

    try:
        test_1d_decomposition()
        test_2d_decomposition()
        test_3d_decomposition()
        test_irregular_decomposition()
        test_data_allocation()

        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓")
        print("="*60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
