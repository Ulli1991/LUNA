"""Taylor-Green vortex: LUNA's compressible run against the stored spectral incompressible reference.

The reference is spectral/taylor_green/tg_spectral_N<N>.npz in luna-reference-solutions: tools/tg_spectral_reference.py, packed by
tools/tg_reference_pack.py. A missing reference is an error, never a skip."""
import numpy as np

from luna.reference_data import reference_path

GAMMA = 1.4
L = 2.0 * np.pi


def reference(n_spec=128):
    """The spectral run's velocity coefficients at t = 1, 2 (luna-reference-solutions, tools/tg_reference_pack.py) as a dict."""
    path = reference_path(f"spectral/taylor_green/tg_spectral_N{n_spec}.npz")
    with np.load(path) as z:
        c = np.zeros((len(z["field_t"]), *z["shape"]), np.complex64)
        c[tuple(z["coord"].T)] = z["value"]
        return dict(field_t=z["field_t"], field_c=c, kc=z["kc"], N=z["N"], provenance=str(z["provenance"]))


def ref_cells(ref, t, n):
    """Reference velocity (3, n, n, n) averaged over LUNA's cells (centres at (i + 1/2) h) at time t."""
    i = int(np.argmin(np.abs(ref["field_t"] - t)))
    assert abs(ref["field_t"][i] - t) < 1e-6, f"no stored field at t={t}"
    c, kc = ref["field_c"][i].astype(np.complex128), int(ref["kc"])
    m = max(n, 1 << int(np.ceil(np.log2(2 * kc + 2))))
    k = np.r_[0:kc + 1, -kc:0].astype(float)
    h = L / n
    f = np.exp(0.5j * k * h) * np.sinc(0.5 * k * h / np.pi)       # half-cell shift and cell average
    fz = f[:kc + 1]
    c = c * f[None, :, None, None] * f[None, None, :, None] * fz[None, None, None, :]
    full = np.zeros((3, m, m, m // 2 + 1), np.complex128)
    idx = np.r_[0:kc + 1, m - kc:m]
    full[:, idx[:, None], idx[None, :], :kc + 1] = c
    u = np.fft.irfftn(full * float(m) ** 3, s=(m, m, m), axes=(1, 2, 3))
    s = m // n
    return u[:, ::s, ::s, ::s]


def luna_ic(n, mach):
    """Conserved TG state (5, n, n, n): rho = 1, |u| <= 1, p = 1/(gamma M^2) + the incompressible pressure."""
    h = L / n
    x = (np.arange(n) + 0.5) * h
    s1, s2 = np.sinc(0.5 * h / np.pi), np.sinc(h / np.pi)          # cell averages of k = 1 and k = 2 modes
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = s1 ** 3 * np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -s1 ** 3 * np.cos(X) * np.sin(Y) * np.cos(Z)
    p = 1.0 / (GAMMA * mach ** 2) + (s2 * np.cos(2 * X) + s2 * np.cos(2 * Y)) * (s2 * np.cos(2 * Z) + 2) / 16
    rho = np.ones_like(X)
    return np.stack([rho, u, v, np.zeros_like(X), p / (GAMMA - 1) + 0.5 * (u * u + v * v)])


def luna_run(n, mach, times, order=1, mesh_shape=None):
    """LUNA states at `times` (minmod, HLLC, FP64, default dt) from the TG state."""
    from tests._helpers import accuracy_cfg, evolve
    cfg = accuracy_cfg(test_problem="Sod-Shock-3D-AMR", SPATIAL_DIMENSION=3, Nx=n, Ny=n, Nz=n,
                       Lx=L, Ly=L, Lz=L, BOUNDARY=0, gamma=GAMMA, ORDER=order)
    states, _ = evolve(cfg, luna_ic(n, mach), list(times), mesh_shape)
    return states


def _modes(u):
    """(energy per Fourier mode, |k|) of a periodic velocity field (3, n, n, n)."""
    n = u.shape[1]
    uh = np.fft.rfftn(u, axes=(1, 2, 3)) / float(n) ** 3
    k = np.fft.fftfreq(n, 1.0 / n)
    kz = np.fft.rfftfreq(n, 1.0 / n)
    kk = np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + kz[None, None, :] ** 2)
    w = np.where(kz == 0, 1.0, 2.0)[None, None, :]
    return 0.5 * np.sum(np.abs(uh) ** 2, axis=0) * w * L ** 3, kk


def spectrum(u):
    """Shell-binned (rint |k|) kinetic-energy spectrum of a periodic velocity field."""
    e, kk = _modes(u)
    return np.bincount(np.rint(kk).astype(int).ravel(), e.ravel())[: u.shape[1] // 2 + 1]


def enstrophy(u):
    """0.5 * integral |curl u|^2 of a periodic velocity field, spectrally."""
    e, kk = _modes(u)
    return float(np.sum(kk ** 2 * e))


def metrics(U, U0, ref, t):
    """KE loss fraction, rel. velocity L2 error, enstrophy ratio, density rms and E(k) ratio at time t."""
    n = U.shape[1]
    ke = lambda W: 0.5 * float(np.sum((W[1] ** 2 + W[2] ** 2 + W[3] ** 2) / W[0]))   # noqa: E731
    u = U[1:4] / U[0]
    ur = ref_cells(ref, t, n)
    sl, sr = spectrum(u), spectrum(ur)
    return dict(ke_loss=1.0 - ke(U) / ke(U0),
                l2=float(np.sqrt(np.sum((u - ur) ** 2) / np.sum(ur ** 2))),
                z_ratio=enstrophy(u) / enstrophy(ur),
                drho=float(np.std(U[0])), ek_ratio=sl / np.maximum(sr, 1e-300), e_ref=sr)


def k_eff(ek_ratio, e_ref, frac=0.5, floor=1e-10):
    """First shell k >= 2 holding reference energy (> floor of the total) where LUNA's E(k) is below frac of it."""
    k = np.arange(ek_ratio.size)
    live = (k >= 2) & (e_ref[: k.size] > floor * e_ref.sum())
    low = np.nonzero(live & (ek_ratio < frac))[0]
    return float(low[0]) if low.size else float(k[live].max())
