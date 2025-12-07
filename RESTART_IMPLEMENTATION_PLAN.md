# Restart/Checkpoint System Implementation Plan

## Overview
Implement a restart/checkpoint system that allows simulations to be stopped and resumed from saved state. This is critical for long-running simulations on systems with time limits.

---

## 1. Configuration Extension

### New Parameters in `config.py`

Add to the `Config` dataclass:

```python
# Restart/Checkpoint settings
RestartOn: bool = False  # Enable restart functionality
TimeBetRestart: float = 7200.0  # Time in seconds between checkpoints (default: 2 hours)
RestartFromFile: str = ""  # Path to checkpoint file to restart from (empty = fresh start)
CheckpointDir: str = "checkpoints"  # Directory for checkpoint files (separate from output_dir)
KeepLastNCheckpoints: int = 2  # Keep only the N most recent checkpoints (0 = keep all)
```

### Rationale:
- **`RestartOn`**: Master switch - when False, no checkpoint overhead
- **`TimeBetRestart`**: Wall-clock time (not simulation time) between checkpoints
- **`RestartFromFile`**: Explicit path to resume from a specific checkpoint
- **`CheckpointDir`**: Keep checkpoints separate from regular output snapshots
- **`KeepLastNCheckpoints`**: Disk space management - automatically clean old checkpoints

---

## 2. Checkpoint File Structure

### File Format: HDF5 (`.chk.hdf5`)

**Filename Pattern:** `checkpoint_<timestamp>.chk.hdf5`
- Example: `checkpoint_20250312_143022.chk.hdf5`
- Timestamp format: `YYYYMMDD_HHMMSS`
- Extension `.chk.hdf5` distinguishes from regular snapshots

### HDF5 Structure:

```
checkpoint_<timestamp>.chk.hdf5
├── simulation_state (group)
│   ├── U (dataset)                    # Conservative state array: (Nvar, Nx[, Ny[, Nz]])
│   ├── t (scalar dataset)             # Current simulation time (float)
│   ├── step (scalar dataset)          # Current step number (int)
│   ├── snapshot_idx (scalar dataset)  # Next snapshot index (int)
│   └── wallclock_elapsed (scalar)     # Total wallclock time elapsed (seconds)
│
├── snapshot_schedule (group)
│   └── snapshot_times (dataset)       # Array of snapshot times
│
├── metadata (group)
│   ├── checkpoint_wallclock (scalar)  # Wallclock time when checkpoint was written
│   ├── checkpoint_timestamp (string)  # Human-readable timestamp
│   └── jax_version (string)           # JAX version for compatibility checking
│
└── config (group)  # All config parameters as attributes or datasets
    ├── SPATIAL_DIMENSION (attr)
    ├── Nx, Ny, Nz (attrs)
    ├── Lx, Ly, Lz (attrs)
    ├── gamma, CFL (attrs)
    ├── ORDER, LIMITER, BOUNDARY (attrs)
    ├── TIME_SOLVER (attr)
    ├── t_start, t_end, time_bet_snaps (attrs)
    ├── use_smooth_limiters, smooth_limiter_sharpness (attrs)
    └── ... (all other config params)
```

### Why This Structure?

1. **`simulation_state` group**: All dynamic state variables together
2. **`snapshot_schedule` group**: Pre-computed snapshot times (avoids recomputation)
3. **`metadata` group**: Provenance and compatibility information
4. **`config` group**: Complete configuration for validation and reproducibility

---

## 3. Implementation Components

### 3.1 New Module: `file_io/checkpoint.py`

Functions needed:

```python
def write_checkpoint(
    checkpoint_path: str,
    U: jnp.ndarray,
    t: float,
    step: int,
    snapshot_idx: int,
    snapshot_times: jnp.ndarray,
    wallclock_elapsed: float,
    config: Config
) -> None:
    """
    Write complete simulation state to checkpoint file.

    - Creates checkpoint_dir if it doesn't exist
    - Writes all state variables and metadata
    - Uses atomic write (temp file + rename) to prevent corruption
    """
    pass

def read_checkpoint(
    checkpoint_path: str
) -> dict:
    """
    Read checkpoint file and return simulation state.

    Returns:
        dict with keys:
            'U': Conservative state array
            't': Current time
            'step': Step number
            'snapshot_idx': Next snapshot index
            'snapshot_times': Pre-computed snapshot times
            'wallclock_elapsed': Elapsed wallclock time
            'config': Dictionary of config parameters
    """
    pass

def validate_checkpoint(
    checkpoint_data: dict,
    current_config: Config
) -> tuple[bool, list[str]]:
    """
    Validate checkpoint compatibility with current configuration.

    Checks:
        - Grid dimensions match (Nx, Ny, Nz)
        - Spatial dimension matches
        - Physics parameters match (gamma)
        - State array shape is correct

    Returns:
        (is_valid, list_of_warnings_or_errors)
    """
    pass

def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int
) -> None:
    """
    Remove old checkpoint files, keeping only the N most recent.

    - Sorts by modification time
    - Keeps the keep_last_n newest files
    - Deletes older files
    """
    pass
```

### 3.2 Modify `main.py` - Initialization Section

**Location:** Lines 34-155 (before main loop)

**Changes:**

```python
# After config loading (around line 50):

# Initialize wallclock timer
wallclock_start = time.time()
wallclock_elapsed = 0.0
last_checkpoint_time = wallclock_start

# Check for restart
if cfg.RestartOn and cfg.RestartFromFile:
    print(f"Restarting from checkpoint: {cfg.RestartFromFile}")

    # Read checkpoint
    checkpoint_data = read_checkpoint(cfg.RestartFromFile)

    # Validate checkpoint
    is_valid, messages = validate_checkpoint(checkpoint_data, cfg)
    if not is_valid:
        print("ERROR: Checkpoint validation failed:")
        for msg in messages:
            print(f"  - {msg}")
        sys.exit(1)

    # Extract state
    U = checkpoint_data['U']
    t = checkpoint_data['t']
    step = checkpoint_data['step']
    snapshot_idx = checkpoint_data['snapshot_idx']
    snapshot_times = checkpoint_data['snapshot_times']
    wallclock_elapsed = checkpoint_data['wallclock_elapsed']

    print(f"  Resuming from t={t:.4e}, step={step}")
    print(f"  Next snapshot at t={snapshot_times[snapshot_idx]:.4e}")

else:
    # Fresh start - existing initialization code
    U = initialize_state(cfg)
    t = cfg.t_start
    step = 0
    snapshot_times = compute_snapshot_times(cfg)
    snapshot_idx = 0
    wallclock_elapsed = 0.0
```

### 3.3 Modify `main.py` - Main Loop

**Location:** Lines 156-201 (main time-stepping loop)

**Changes:**

Add checkpoint logic inside the loop (after snapshot saving):

```python
# Inside main loop, after snapshot saving (around line 201):

# Checkpoint if RestartOn is enabled
if cfg.RestartOn:
    current_wallclock = time.time()
    time_since_last_checkpoint = current_wallclock - last_checkpoint_time

    if time_since_last_checkpoint >= cfg.TimeBetRestart:
        # Update total elapsed time
        wallclock_elapsed += (current_wallclock - wallclock_start)

        # Create checkpoint
        checkpoint_name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.chk.hdf5"
        checkpoint_path = os.path.join(cfg.CheckpointDir, checkpoint_name)

        print(f"\nWriting checkpoint at t={t:.4e}, step={step}")
        print(f"  Wallclock elapsed: {wallclock_elapsed/3600:.2f} hours")

        write_checkpoint(
            checkpoint_path=checkpoint_path,
            U=U,
            t=t,
            step=step,
            snapshot_idx=snapshot_idx,
            snapshot_times=snapshot_times,
            wallclock_elapsed=wallclock_elapsed,
            config=cfg
        )

        print(f"  Checkpoint saved: {checkpoint_path}")

        # Clean up old checkpoints if requested
        if cfg.KeepLastNCheckpoints > 0:
            cleanup_old_checkpoints(cfg.CheckpointDir, cfg.KeepLastNCheckpoints)

        # Reset checkpoint timer
        last_checkpoint_time = current_wallclock
```

---

## 4. Implementation Strategy & Order

### Phase 1: Core Checkpoint I/O (No Restart Yet)
1. Create `file_io/checkpoint.py` module
2. Implement `write_checkpoint()` function
3. Implement `read_checkpoint()` function
4. Add config parameters to `config.py`
5. Test checkpoint writing during simulation (but don't restart yet)

**Test:** Run a simulation with `RestartOn=True`, verify checkpoint files are created and contain correct data

---

### Phase 2: Checkpoint Management
1. Implement `cleanup_old_checkpoints()` function
2. Test that old checkpoints are removed correctly

**Test:** Run simulation that creates 5+ checkpoints with `KeepLastNCheckpoints=2`, verify only 2 newest remain

---

### Phase 3: Restart Functionality
1. Implement `validate_checkpoint()` function
2. Modify `main.py` initialization to detect and load restart file
3. Test restart from checkpoint

**Test Cases:**
- Restart from checkpoint and verify simulation continues correctly
- Verify snapshot_idx is restored (don't re-save old snapshots)
- Run for 10 steps, checkpoint, restart, verify t and step match
- Compare restarted simulation output to non-restarted (should be identical)

---

### Phase 4: Robustness & Edge Cases
1. Add atomic write (temp file + rename) to prevent corruption
2. Handle missing directories (create automatically)
3. Add error handling for corrupted checkpoint files
4. Test validation catches incompatible checkpoints (wrong grid size, etc.)

**Test Cases:**
- Restart with mismatched grid dimensions (should fail gracefully)
- Corrupted checkpoint file (should fail gracefully)
- Missing checkpoint directory (should be created)

---

### Phase 5: Multi-GPU Support (If Needed)
1. Extend checkpoint system for `main_multi.py`
2. Checkpoint global `U` state (not local subdomains)
3. Test on multi-GPU setup

**Consideration:** For `pmap` version, need to gather global state before checkpointing

---

## 5. Testing Strategy

### Unit Tests
Create `tests/test_checkpoint.py`:
- Test checkpoint write/read roundtrip
- Test validation catches errors
- Test cleanup removes correct files
- Test atomic write prevents corruption

### Integration Tests
1. **Continuity Test:**
   - Run sim for N steps → checkpoint → restart → run N more steps
   - Compare to running 2N steps without checkpoint
   - Results should be **bit-identical**

2. **Snapshot Schedule Test:**
   - Checkpoint between snapshots
   - Verify restarted sim doesn't re-save old snapshots
   - Verify snapshot numbering continues correctly

3. **Wallclock Timer Test:**
   - Run for 1 hour → checkpoint → restart → run 1 more hour
   - Verify wallclock_elapsed ≈ 2 hours

4. **Cleanup Test:**
   - Create multiple checkpoints
   - Verify old ones are deleted

---

## 6. User Workflow Examples

### Fresh Start (No Restart)
```python
# config.py or user config
RestartOn = False  # Default - no checkpoint overhead
```

### Enable Checkpointing (But Start Fresh)
```python
# config.py
RestartOn = True
TimeBetRestart = 7200.0  # Checkpoint every 2 hours
CheckpointDir = "checkpoints"
KeepLastNCheckpoints = 2  # Keep only 2 most recent
RestartFromFile = ""  # Empty = fresh start
```

### Restart from Checkpoint
```python
# config.py
RestartOn = True
RestartFromFile = "checkpoints/checkpoint_20250312_143022.chk.hdf5"
# All other restart parameters should remain the same
```

**Command Line:**
```bash
# Fresh start with checkpointing
python main.py

# Restart from checkpoint (modify config first)
python main.py  # with RestartFromFile set
```

---

## 7. Key Design Decisions & Rationale

### Decision 1: Separate Checkpoint Directory
**Why:** Checkpoints are operational files (frequently created/deleted), while snapshots are scientific output (kept permanently). Separating them prevents accidental deletion and cluttering output.

### Decision 2: Keep Conservative Variables `U`, Not Primitive
**Why:** `U` is the actual simulation state. Primitive variables (rho, vx, vy, vz, internal_energy) are derived and can introduce roundoff errors during conversion.

### Decision 3: Store `snapshot_idx`
**Why:** Prevents re-saving snapshots that were already written before checkpoint. Without this, restarting would overwrite existing snapshot files.

### Decision 4: Wallclock Time, Not Simulation Time
**Why:** The purpose is to checkpoint before hitting system time limits (e.g., SLURM job walltime). Simulation time `t` can advance slowly if timesteps are small.

### Decision 5: Timestamp in Filename, Not Step Number
**Why:** Multiple restart chains could have overlapping step numbers. Timestamps are unique and provide chronological ordering.

### Decision 6: Atomic Write (Temp File + Rename)
**Why:** If simulation crashes during checkpoint write, we don't want a corrupted checkpoint. Writing to `.tmp` then renaming ensures atomicity.

---

## 8. Future Extensions (Not Part of Initial Implementation)

### Extension 1: Auto-Restart on SLURM
- Script wrapper that automatically submits next job with `RestartFromFile` set
- Useful for long simulations needing multiple job submissions

### Extension 2: Checkpoint on Signal
- Catch SIGTERM (SLURM sends this before killing job)
- Write emergency checkpoint before exit
- Requires signal handlers in Python

### Extension 3: Compression
- HDF5 supports GZIP compression
- Trade-off: smaller files vs. slower I/O
- Add config option `CheckpointCompression: bool = False`

### Extension 4: Parallel I/O for Multi-GPU
- Use MPI-IO or parallel HDF5 for large multi-GPU runs
- Each rank writes its local subdomain
- More complex but scales better

---

## 9. Potential Issues & Mitigations

### Issue 1: Checkpoint I/O Time
**Problem:** Writing large arrays to disk can be slow (seconds to minutes)
**Mitigation:**
- JAX arrays are lazy; explicitly copy to CPU with `np.array(U)`
- Use asynchronous I/O if possible (write in background thread)
- Acceptable overhead if `TimeBetRestart` is large (hours)

### Issue 2: JAX JIT Recompilation After Restart
**Problem:** JIT cache is lost after restart; first few steps will be slow
**Mitigation:**
- This is expected JAX behavior
- JIT cache warms up quickly (few steps)
- Not a correctness issue, just performance

### Issue 3: Floating-Point Reproducibility
**Problem:** Restarted simulation might diverge from non-restarted due to roundoff
**Mitigation:**
- Store full float64 precision in checkpoint (even if sim uses float32)
- Test shows JAX is remarkably reproducible if RNG state is not involved
- For RNG (if added later), must checkpoint random state too

### Issue 4: Config Drift
**Problem:** User modifies config between runs, checkpoint becomes invalid
**Mitigation:**
- `validate_checkpoint()` checks critical parameters
- Fail early with clear error message
- Consider storing config hash for exact validation

---

## 10. Success Criteria

The implementation is successful when:

1. ✅ Simulations can run with `RestartOn=True` and automatically checkpoint
2. ✅ Checkpoints are written at specified wallclock intervals
3. ✅ Simulations can restart from checkpoint and continue seamlessly
4. ✅ Restarted simulations produce **identical results** to non-restarted runs
5. ✅ Old checkpoints are cleaned up automatically
6. ✅ Invalid checkpoints are detected and rejected with clear error messages
7. ✅ System works for 1D, 2D, and 3D simulations
8. ✅ Documentation explains user workflow clearly

---

## 11. Estimated Complexity

- **New code:** ~400-600 lines
  - `checkpoint.py`: ~300 lines (write, read, validate, cleanup)
  - `main.py` modifications: ~80 lines
  - Config additions: ~10 lines
  - Tests: ~200 lines

- **Modified files:**
  - `config.py` (add parameters)
  - `main.py` (initialization + loop)
  - New file: `file_io/checkpoint.py`
  - New file: `tests/test_checkpoint.py` (optional but recommended)

- **Testing time:** 2-4 hours
  - Unit tests: 30 min
  - Integration tests: 1-2 hours
  - Multi-dimensional tests (1D/2D/3D): 1 hour

---

## 12. Summary

This plan provides a complete, robust restart/checkpoint system that:
- Allows simulations to survive time limits
- Minimizes overhead when not needed (RestartOn=False)
- Stores complete state for exact restarts
- Manages disk space automatically
- Validates compatibility on restart
- Maintains separate checkpoint and output files
- Works with existing HDF5 infrastructure

The phased implementation strategy ensures each component can be tested independently before integration, reducing risk of bugs.

---

## Next Steps

1. **Review this plan** - discuss any modifications or concerns
2. **Implement Phase 1** - core checkpoint write/read
3. **Test Phase 1** - verify checkpoint files contain correct data
4. **Proceed through phases** - iteratively adding functionality
5. **Final integration testing** - ensure full workflow works end-to-end

**Estimated time to complete:** 1-2 days of focused development + testing
