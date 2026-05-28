"""
Ornstein-Uhlenbeck turbulent driving (Federrath/Schmidt 2010 style).

Design choices for sharded JAX:
- A small set of low-k driving modes is enumerated once at setup.
- Each mode carries a complex 3-vector amplitude evolved in time as an
  Ornstein-Uhlenbeck process.
- The real-space force F(x) is reconstructed by direct plane-wave
  summation: F(x) = 2 Re sum_k a_k exp(i k . x). No FFT, so no global
  communication when U is sharded across devices.

The driver returns a force field with the same shape as the velocity field
and integrates as a momentum source:
    (rho v)_new = (rho v)_old + rho F dt
    E_new       = E_old + (rho v) . F dt + 0.5 rho |F|^2 dt^2

References
----------
Federrath, Roman-Duval, Klessen, Schmidt, Mac Low (2010) A&A 512, A81.
Schmidt, Federrath, Klessen (2008) PRL 101, 194505.
"""
from dataclasses import dataclass, field
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class OUDriverConfig:
    """Configuration for the OU turbulent driver.

    All wavenumbers are in units of 2 pi / L (so k=1 is the first nonzero
    Fourier mode along a box edge of length L).
    """
    k_min: float = 1.0
    k_max: float = 3.0
    solenoidal_weight: float = 0.5     # zeta in Federrath et al. (1.0 sol, 0.0 comp)
    autocorrelation_time: float = 1.0  # T_ac in code time units
    rms_acceleration: float = 1.0      # target RMS of |F| in code units
    seed: int = 42                     # base PRNG seed (per-mode keys derived)
    spectrum: str = "parabolic"        # "parabolic" or "flat"

    # Box lengths captured at build time (not part of the steady state).
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0


def _enumerate_modes(cfg: OUDriverConfig):
    """Enumerate integer wavevectors with k_min <= |n| <= k_max.

    We keep only a half-space (lexicographic n_z>0, or n_z=0 and n_y>0, or
    n_z=n_y=0 and n_x>0) so the reconstructed real-space field is real:
        F(x) = 2 Re sum_{k in H} a_k exp(i k . x)

    Returns
    -------
    n_vec : (M, 3) integer array of mode triples
    k_vec : (M, 3) physical wavenumbers (2 pi n / L)
    k_mag : (M,) magnitudes |k|
    weight : (M,) spectral weight (parabolic or flat in |n|)
    """
    k_min_n = cfg.k_min
    k_max_n = cfg.k_max
    n_ceil = int(np.ceil(k_max_n))

    ns = []
    for nz in range(-n_ceil, n_ceil + 1):
        for ny in range(-n_ceil, n_ceil + 1):
            for nx in range(-n_ceil, n_ceil + 1):
                if (nx, ny, nz) == (0, 0, 0):
                    continue
                # Half-space selection so conjugate pairs are not double counted.
                if nz < 0:
                    continue
                if nz == 0 and ny < 0:
                    continue
                if nz == 0 and ny == 0 and nx <= 0:
                    continue
                rad = np.sqrt(nx * nx + ny * ny + nz * nz)
                if rad < k_min_n - 1e-9 or rad > k_max_n + 1e-9:
                    continue
                ns.append((nx, ny, nz, rad))

    if not ns:
        raise ValueError(
            f"No driving modes found for k_min={cfg.k_min}, k_max={cfg.k_max}"
        )

    n_arr = np.array([(a, b, c) for (a, b, c, _) in ns], dtype=np.int32)
    rad = np.array([r for (_, _, _, r) in ns], dtype=np.float64)

    if cfg.spectrum == "flat":
        w = np.ones_like(rad)
    elif cfg.spectrum == "parabolic":
        # Parabolic profile in |n|, peaked at the midpoint of the band.
        k_peak = 0.5 * (cfg.k_min + cfg.k_max)
        half = 0.5 * (cfg.k_max - cfg.k_min)
        if half == 0.0:
            w = np.ones_like(rad)
        else:
            w = 1.0 - ((rad - k_peak) / half) ** 2
            w = np.maximum(w, 0.0)
    else:
        raise ValueError(f"Unknown spectrum '{cfg.spectrum}'")

    # Physical wavevector (2 pi / L for each axis).
    two_pi = 2.0 * np.pi
    k_vec = np.column_stack(
        [two_pi * n_arr[:, 0] / cfg.Lx,
         two_pi * n_arr[:, 1] / cfg.Ly,
         two_pi * n_arr[:, 2] / cfg.Lz]
    )
    k_mag = np.linalg.norm(k_vec, axis=1)

    return n_arr, k_vec, k_mag, w


def _projection_tensors(k_vec: np.ndarray, zeta: float) -> np.ndarray:
    """Per-mode projection tensor P_ij = zeta (delta - khat khat) + (1-zeta) khat khat.

    Returns (M, 3, 3).
    """
    eps = 1e-30
    k_mag = np.linalg.norm(k_vec, axis=1)
    khat = k_vec / np.maximum(k_mag, eps)[:, None]
    M = k_vec.shape[0]
    delta = np.eye(3)[None, :, :]
    kk = khat[:, :, None] * khat[:, None, :]
    return zeta * (delta - kk) + (1.0 - zeta) * kk


@dataclass
class OUDriver:
    """OU turbulent driver. Build once at setup, then evolve the state."""

    cfg: OUDriverConfig
    k_vec: jnp.ndarray = field(init=False)        # (M, 3) physical wavenumbers
    k_mag: jnp.ndarray = field(init=False)        # (M,)
    P: jnp.ndarray = field(init=False)            # (M, 3, 3) projection tensor
    weight: jnp.ndarray = field(init=False)       # (M,) normalized to <|F|^2>=1
    n_modes: int = field(init=False)

    def __post_init__(self):
        _, k_vec_np, k_mag_np, w_np = _enumerate_modes(self.cfg)
        P_np = _projection_tensors(k_vec_np, self.cfg.solenoidal_weight)

        # Normalize weights so that the steady-state <|F(x)|^2>_x of the driver
        # equals cfg.rms_acceleration^2. For independent zero-mean complex
        # modes a_k of variance s_k^2 = weight_k each, summed over the half-
        # space with the factor of 2 from Re, the spatial mean square is
        # 2 * sum_k tr(P_k) * s_k^2. We pick s_k^2 = weight_k * C with C set
        # so the total equals rms_acceleration^2.
        tr_P = np.einsum("mii->m", P_np)
        norm_sum = 2.0 * float(np.sum(tr_P * w_np))
        if norm_sum <= 0.0:
            raise ValueError("Mode projection sum is non-positive; check zeta")
        scale = (self.cfg.rms_acceleration ** 2) / norm_sum
        sigma2 = w_np * scale          # per-mode amplitude variance
        sigma = np.sqrt(sigma2)

        self.k_vec = jnp.asarray(k_vec_np, dtype=jnp.float32)
        self.k_mag = jnp.asarray(k_mag_np, dtype=jnp.float32)
        self.P = jnp.asarray(P_np, dtype=jnp.float32)
        self.weight = jnp.asarray(sigma, dtype=jnp.float32)   # std dev per mode
        self.n_modes = int(k_vec_np.shape[0])

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def init_state(self, key: jax.Array) -> jax.Array:
        """Draw steady-state amplitudes a ~ CN(0, P_k * sigma_k^2)."""
        k1, k2 = jax.random.split(key)
        real = jax.random.normal(k1, (self.n_modes, 3), dtype=jnp.float32)
        imag = jax.random.normal(k2, (self.n_modes, 3), dtype=jnp.float32)
        z = (real + 1j * imag) / jnp.sqrt(2.0)
        # Project onto sol/comp and scale by sigma_k.
        a = jnp.einsum("mij,mj->mi", self.P, z) * self.weight[:, None]
        return a

    def evolve(self, a: jax.Array, dt: float, key: jax.Array) -> jax.Array:
        """One OU step: a_new = a*exp(-dt/T) + sigma*sqrt(1-exp(-2dt/T)) * eta."""
        T_ac = self.cfg.autocorrelation_time
        decay = jnp.exp(-dt / T_ac)
        diffuse = jnp.sqrt(jnp.maximum(1.0 - decay * decay, 0.0))

        k1, k2 = jax.random.split(key)
        eta_r = jax.random.normal(k1, (self.n_modes, 3), dtype=jnp.float32)
        eta_i = jax.random.normal(k2, (self.n_modes, 3), dtype=jnp.float32)
        eta = (eta_r + 1j * eta_i) / jnp.sqrt(2.0)
        eta = jnp.einsum("mij,mj->mi", self.P, eta) * self.weight[:, None]

        return decay * a + diffuse * eta

    # ------------------------------------------------------------------
    # Real-space force field
    # ------------------------------------------------------------------
    def force_field(self, a: jax.Array, X: jax.Array, Y: jax.Array, Z: jax.Array
                    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Evaluate F(x) = 2 Re sum_k a_k exp(i k . x).

        X, Y, Z : real-space coordinates with the same shape (typically the
                  cell centers of the *local* shard).

        Returns Fx, Fy, Fz with that same shape.

        Memory-friendly: scan over modes so the per-rank peak is just three
        spatial arrays + the scalar phase, never (M, *spatial). At 1024^3 /
        8 GPUs with ~30 modes the old materialised-phase version asked for
        ~32 GB per rank, which is the OOM we kept hitting.
        """
        ar = jnp.real(a)  # (M, 3)
        ai = jnp.imag(a)

        def step(carry, mk_ak):
            Fx, Fy, Fz = carry
            k_vec, a_r, a_i = mk_ak
            phase = k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z
            c = jnp.cos(phase)
            s = jnp.sin(phase)
            Fx = Fx + 2.0 * (a_r[0] * c - a_i[0] * s)
            Fy = Fy + 2.0 * (a_r[1] * c - a_i[1] * s)
            Fz = Fz + 2.0 * (a_r[2] * c - a_i[2] * s)
            return (Fx, Fy, Fz), None

        zero = jnp.zeros_like(X)
        (Fx, Fy, Fz), _ = jax.lax.scan(
            step, (zero, zero, zero),
            (self.k_vec, ar, ai)
        )
        return Fx, Fy, Fz

    # ------------------------------------------------------------------
    # Application to a 3D conservative state
    # ------------------------------------------------------------------
    def apply(self, U: jax.Array, a: jax.Array, dt: float,
              X: jax.Array, Y: jax.Array, Z: jax.Array) -> jax.Array:
        """Apply F as a momentum source to U=[rho, rho vx, rho vy, rho vz, E].

        Increments momentum by rho F dt and total energy by (rho v) . F dt
        + 0.5 rho |F|^2 dt^2 (exact for explicit Euler velocity update).
        """
        rho = U[0]
        mx, my, mz = U[1], U[2], U[3]
        E = U[4]

        Fx, Fy, Fz = self.force_field(a, X, Y, Z)
        dmx = rho * Fx * dt
        dmy = rho * Fy * dt
        dmz = rho * Fz * dt

        # E_new - E_old = (mx*Fx + my*Fy + mz*Fz)*dt + 0.5 rho |F|^2 dt^2
        dE = ((mx * Fx + my * Fy + mz * Fz) * dt
              + 0.5 * rho * (Fx * Fx + Fy * Fy + Fz * Fz) * dt * dt)

        U_new = U.at[1].add(dmx)
        U_new = U_new.at[2].add(dmy)
        U_new = U_new.at[3].add(dmz)
        U_new = U_new.at[4].add(dE)
        return U_new


# ----------------------------------------------------------------------
# Convenience: build local-shard coordinate arrays
# ----------------------------------------------------------------------
def local_cell_centers(cfg, x_slice=None, y_slice=None, z_slice=None):
    """Return cell-center coordinate arrays (X, Y, Z) for a local shard,
    matching the convention used by the IC generators (cell-centered)."""
    Nx, Ny, Nz = cfg.Nx, cfg.Ny, cfg.Nz
    Lx, Ly, Lz = cfg.Lx, cfg.Ly, cfg.Lz
    dx, dy, dz = Lx / Nx, Ly / Ny, Lz / Nz

    x_full = jnp.arange(Nx, dtype=jnp.float32) * dx + 0.5 * dx
    y_full = jnp.arange(Ny, dtype=jnp.float32) * dy + 0.5 * dy
    z_full = jnp.arange(Nz, dtype=jnp.float32) * dz + 0.5 * dz

    if x_slice is not None:
        x_full = x_full[x_slice[0]:x_slice[1]]
    if y_slice is not None:
        y_full = y_full[y_slice[0]:y_slice[1]]
    if z_slice is not None:
        z_full = z_full[z_slice[0]:z_slice[1]]

    X, Y, Z = jnp.meshgrid(x_full, y_full, z_full, indexing="ij")
    return X, Y, Z
