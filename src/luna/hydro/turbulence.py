"""Ornstein-Uhlenbeck turbulent driving (Federrath+2010) by direct plane-wave summation.

Few low-k modes summed in real space, so no FFT and no global communication when sharded.
exp(ik.x) factorises per axis, so the sum is 1D tables in x, 2D tables in (y, z) and one contraction.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class OUDriverConfig:
    """OU driver settings; wavenumbers in units of 2 pi / L."""

    k_min: float = 1.0
    k_max: float = 3.0
    solenoidal_weight: float = 0.5  # zeta in Federrath et al. (1.0 sol, 0.0 comp)
    autocorrelation_time: float = 1.0  # T_ac in code time units
    rms_acceleration: float = 1.0  # target RMS of |F| in code units
    seed: int = 42
    spectrum: str = "parabolic"  # "parabolic" or "flat"

    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0


def _enumerate_modes(cfg: OUDriverConfig):
    """Half-space integer modes k_min <= |n| <= k_max: (n_vec (M,3), k_vec (M,3), k_mag (M,), weight (M,))."""
    k_min_n = cfg.k_min
    k_max_n = cfg.k_max
    n_ceil = int(np.ceil(k_max_n))

    ns = []
    for nz in range(-n_ceil, n_ceil + 1):
        for ny in range(-n_ceil, n_ceil + 1):
            for nx in range(-n_ceil, n_ceil + 1):
                if (nx, ny, nz) == (0, 0, 0):
                    continue
                # Half-space only: the conjugate mode is implied by 2 Re.
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
        raise ValueError(f"No driving modes found for k_min={cfg.k_min}, k_max={cfg.k_max}")

    n_arr = np.array([(a, b, c) for (a, b, c, _) in ns], dtype=np.int32)
    rad = np.array([r for (_, _, _, r) in ns], dtype=np.float64)

    if cfg.spectrum == "flat":
        w = np.ones_like(rad)
    elif cfg.spectrum == "parabolic":
        k_peak = 0.5 * (cfg.k_min + cfg.k_max)
        half = 0.5 * (cfg.k_max - cfg.k_min)
        if half == 0.0:
            w = np.ones_like(rad)
        else:
            w = 1.0 - ((rad - k_peak) / half) ** 2
            w = np.maximum(w, 0.0)
    else:
        raise ValueError(f"Unknown spectrum '{cfg.spectrum}'")

    two_pi = 2.0 * np.pi
    k_vec = np.column_stack(
        [
            two_pi * n_arr[:, 0] / cfg.Lx,
            two_pi * n_arr[:, 1] / cfg.Ly,
            two_pi * n_arr[:, 2] / cfg.Lz,
        ]
    )
    k_mag = np.linalg.norm(k_vec, axis=1)

    return n_arr, k_vec, k_mag, w


def _projection_tensors(k_vec: np.ndarray, zeta: float) -> np.ndarray:
    """Per-mode projection P_ij = zeta (delta - khat khat) + (1-zeta) khat khat, shape (M, 3, 3)."""
    eps = 1e-30
    k_mag = np.linalg.norm(k_vec, axis=1)
    khat = k_vec / np.maximum(k_mag, eps)[:, None]
    delta = np.eye(3)[None, :, :]
    kk = khat[:, :, None] * khat[:, None, :]
    return zeta * (delta - kk) + (1.0 - zeta) * kk


@dataclass
class OUDriver:
    """OU turbulent driver: build once at setup, then evolve its mode amplitudes."""

    cfg: OUDriverConfig
    k_vec: jnp.ndarray = field(init=False)  # (M, 3) physical wavenumbers
    k_mag: jnp.ndarray = field(init=False)  # (M,)
    P: jnp.ndarray = field(init=False)  # (M, 3, 3) projection tensor
    weight: jnp.ndarray = field(init=False)  # (M,) per-mode amplitude std dev
    n_modes: int = field(init=False)
    kx: jnp.ndarray = field(init=False)  # (M,) x-wavenumber of each mode
    pair_of_mode: jnp.ndarray = field(init=False)  # (M, P) one-hot: mode -> its (ky, kz) pair
    k_pair: jnp.ndarray = field(init=False)  # (P, 2) the distinct (ky, kz)
    # FP64 only when the force must be reproducible across layouts (differentiated runs).
    dtype: object = jnp.float32

    def __post_init__(self):
        n_np, k_vec_np, k_mag_np, w_np = _enumerate_modes(self.cfg)
        P_np = _projection_tensors(k_vec_np, self.cfg.solenoidal_weight)

        # <|F|^2> = 2 sum_k sigma_k^2 tr(P_k^2) (tr(P^2), not tr(P)); scale it to rms^2.
        tr_PP = np.einsum("mij,mij->m", P_np, P_np)
        norm_sum = 2.0 * float(np.sum(tr_PP * w_np))
        if norm_sum <= 0.0:
            raise ValueError("Mode projection sum is non-positive; check zeta")
        scale = (self.cfg.rms_acceleration**2) / norm_sum
        sigma2 = w_np * scale
        sigma = np.sqrt(sigma2)

        khat = k_vec_np / np.linalg.norm(k_vec_np, axis=1)[:, None]
        self.kk = jnp.asarray(khat[:, :, None] * khat[:, None, :], dtype=self.dtype)
        self.k_vec = jnp.asarray(k_vec_np, dtype=self.dtype)
        self.k_mag = jnp.asarray(k_mag_np, dtype=self.dtype)
        self.P = jnp.asarray(P_np, dtype=self.dtype)
        self.weight = jnp.asarray(sigma, dtype=self.dtype)
        self.n_modes = int(k_vec_np.shape[0])

        pairs, pair_idx = np.unique(n_np[:, 1:], axis=0, return_inverse=True)
        self.kx = jnp.asarray(k_vec_np[:, 0], dtype=self.dtype)
        self.pair_of_mode = jnp.asarray(np.eye(len(pairs))[pair_idx.ravel()], dtype=self.dtype)
        self.k_pair = jnp.asarray(2.0 * np.pi * pairs / np.array([self.cfg.Ly, self.cfg.Lz]),
                                  dtype=self.dtype)

    def init_state(self, key: jax.Array) -> jax.Array:
        """Draw steady-state amplitudes a ~ CN(0, P_k * sigma_k^2)."""
        k1, k2 = jax.random.split(key)
        real = jax.random.normal(k1, (self.n_modes, 3), dtype=jnp.float32)
        imag = jax.random.normal(k2, (self.n_modes, 3), dtype=jnp.float32)
        z = (real + 1j * imag) / jnp.sqrt(2.0)
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

    def unit_state(self, key: jax.Array) -> jax.Array:
        """init_state's draw before projection and weighting: a = weight P(zeta) b."""
        k1, k2 = jax.random.split(key)
        real = jax.random.normal(k1, (self.n_modes, 3), dtype=jnp.float32)
        imag = jax.random.normal(k2, (self.n_modes, 3), dtype=jnp.float32)
        return ((real + 1j * imag) / jnp.sqrt(2.0)).astype(jnp.result_type(self.dtype, 1j))

    def unit_evolve(self, b: jax.Array, dt: float, key: jax.Array, T_ac=None) -> jax.Array:
        """evolve() on the unit state: the same draws, no projection or weight; T_ac may be traced."""
        T_ac = self.cfg.autocorrelation_time if T_ac is None else T_ac
        decay = jnp.exp(-dt / T_ac)
        diffuse = jnp.sqrt(jnp.maximum(1.0 - decay * decay, 0.0))
        k1, k2 = jax.random.split(key)
        eta_r = jax.random.normal(k1, (self.n_modes, 3), dtype=jnp.float32)
        eta_i = jax.random.normal(k2, (self.n_modes, 3), dtype=jnp.float32)
        eta = ((eta_r + 1j * eta_i) / jnp.sqrt(2.0)).astype(b.dtype)   # weak decay/diffuse would keep complex64
        return decay * b + diffuse * eta

    def amplitudes(self, b: jax.Array, sigma: jax.Array, zeta) -> jax.Array:
        """Mode amplitudes a = sigma_k P_k(zeta) b_k from the unit state, differentiable in sigma and zeta."""
        P = zeta * (jnp.eye(3, dtype=self.dtype) - self.kk) + (1.0 - zeta) * self.kk
        return jnp.einsum("mij,mj->mi", P, b) * sigma[:, None]

    def force_field(self, a: jax.Array, x: jax.Array, y: jax.Array, z: jax.Array,
                    constrain=None) -> jax.Array:
        """F(x) = 2 Re sum_k a_k exp(ik.x) on the grid x (x) y (x) z of 1D coordinates, shape (3, Nx, Ny, Nz)."""
        hi = jax.lax.Precision.HIGHEST
        pin = constrain or (lambda arr, kind: arr)
        ar, ai = jnp.real(a).astype(self.dtype), jnp.imag(a).astype(self.dtype)
        px = self.kx[:, None] * x[None, :]
        cx, sx = jnp.cos(px), jnp.sin(px)
        S = self.pair_of_mode
        # G_p(x) = sum over the modes of pair p of a_k exp(i kx x), per component c.
        Gr = (jnp.einsum("mp,mc,mx->cpx", S, ar, cx, precision=hi)
              - jnp.einsum("mp,mc,mx->cpx", S, ai, sx, precision=hi))
        Gi = (jnp.einsum("mp,mc,mx->cpx", S, ar, sx, precision=hi)
              + jnp.einsum("mp,mc,mx->cpx", S, ai, cx, precision=hi))
        pyz = (self.k_pair[:, 0, None, None] * y[None, :, None]
               + self.k_pair[:, 1, None, None] * z[None, None, :])
        Er, Ei = pin(jnp.cos(pyz), "yz"), pin(jnp.sin(pyz), "yz")
        Gr, Gi = pin(Gr, "x"), pin(Gi, "x")
        F = 2.0 * (jnp.einsum("cpx,pyz->cxyz", Gr, Er, precision=hi)
                   - jnp.einsum("cpx,pyz->cxyz", Gi, Ei, precision=hi))
        return pin(F, "xyz")

    def apply(self, U: jax.Array, a: jax.Array, dt: float, x: jax.Array, y: jax.Array,
              z: jax.Array, constrain=None) -> jax.Array:
        """Add rho F dt to the 3D momentum and the consistent (rho v).F dt + rho|F|^2 dt^2/2 to E."""
        rho = U[0]
        mx, my, mz = U[1], U[2], U[3]

        F = self.force_field(a, x, y, z, constrain).reshape((3,) + rho.shape)
        Fx, Fy, Fz = F[0], F[1], F[2]
        dmx = rho * Fx * dt
        dmy = rho * Fy * dt
        dmz = rho * Fz * dt

        dE = (mx * Fx + my * Fy + mz * Fz) * dt + 0.5 * rho * (
            Fx * Fx + Fy * Fy + Fz * Fz
        ) * dt * dt

        # two .at adds, not four: bitwise equal on CPU and GPU, 1.5x faster (MI300A 512^3,
        # docs/history/CODE_NOTES.md); .at keeps U's dtype (a stack/concatenate of a weakly typed FP64 state
        # with the FP32 force is FP32)
        return U.at[1:4].add(jnp.stack([dmx, dmy, dmz])).at[4].add(dE)
