"""Kelvin-Helmholtz linear theory of a periodic double tanh shear layer: the compressible eigenproblem
(Fourier collocation in y), its fastest mode, and periodic cell averages. Checked against Michalke 1964
in tools/kh_stability.py; the reference of tests/test_hydro_accuracy_kh.py. The hydro step never calls it."""
import math

import numpy as np

K_MICHALKE = 0.4446        # most unstable k*d of u = tanh(y/d) (Michalke 1964)
SIGMA_MICHALKE = 0.1897    # its growth rate in units of U/d, u in [-U, U]


def profile(y, Ly, d, dU):
    """The periodic double tanh layer (streams at +-dU/2)."""
    return 0.5 * dU * (np.tanh((y - 0.25 * Ly) / d) - np.tanh((y - 0.75 * Ly) / d) - 1.0)


def geometry(dU=1.0, cs=2.0, gamma=5.0 / 3.0):
    """Box and base-state constants: d from Lx = 1 at Michalke's k."""
    Lx = 1.0
    d = K_MICHALKE * Lx / (2.0 * math.pi)
    return dict(Lx=Lx, Ly=2.0 * Lx, d=d, dU=dU, cs=cs, gamma=gamma, rho=1.0,
                p=cs * cs / gamma, k=2.0 * math.pi / Lx)


def linear_theory(k, Ly, d, dU, cs, gamma, ny, rho=1.0):
    """Largest growth rate of the linearised compressible Euler equations
    about (rho, u(y), p) for perturbations ~ exp(i k x + s t); Fourier
    collocation in y with ny points. Returns max Re(s)."""
    p = rho * cs * cs / gamma
    y = np.arange(ny) * Ly / ny
    kap = 2.0 * math.pi * np.fft.fftfreq(ny, Ly / ny)
    kap[ny // 2] = 0.0
    F = np.fft.fft(np.eye(ny), axis=0)
    D = np.real(np.fft.ifft(1j * kap[:, None] * F, axis=0))
    u = profile(y, Ly, d, dU)
    du = D @ u
    I = np.eye(ny)
    ik = 1j * k
    A = np.block([
        [-ik * np.diag(u), -np.diag(du), -ik * I / rho],
        [0 * I, -ik * np.diag(u), -D / rho],
        [-gamma * p * ik * I, -gamma * p * D, -ik * np.diag(u)]])
    return float(np.max(np.linalg.eigvals(A).real))


def linear_mode(g, ny=1024):
    """(s, q, y): the fastest eigenvalue of linear_theory's matrix and its (u', v', p') on ny
    Fourier points y, scaled to max |v'| = 1."""
    k, Ly, d, dU, cs, gamma, rho = g["k"], g["Ly"], g["d"], g["dU"], g["cs"], g["gamma"], g["rho"]
    p = rho * cs * cs / gamma
    y = np.arange(ny) * Ly / ny
    kap = 2.0 * math.pi * np.fft.fftfreq(ny, Ly / ny)
    kap[ny // 2] = 0.0
    F = np.fft.fft(np.eye(ny), axis=0)
    D = np.real(np.fft.ifft(1j * kap[:, None] * F, axis=0))
    u = profile(y, Ly, d, dU)
    I = np.eye(ny)
    ik = 1j * k
    A = np.block([
        [-ik * np.diag(u), -np.diag(D @ u), -ik * I / rho],
        [0 * I, -ik * np.diag(u), -D / rho],
        [-gamma * p * ik * I, -gamma * p * D, -ik * np.diag(u)]])
    w, V = np.linalg.eig(A)
    j = int(np.argmax(w.real))
    q = V[:, j].reshape(3, ny)
    return w[j], q / np.abs(q[1]).max(), y


def cell_average_y(f, Ly, n):
    """Averages over n periodic cells [j Ly/n, (j+1) Ly/n] of f given on len(f) Fourier points."""
    m = len(f)
    kap = 2.0 * math.pi * np.fft.fftfreq(m, Ly / m)
    h = Ly / n
    fh = np.fft.fft(f) * np.sinc(kap * h / (2.0 * math.pi)) * np.exp(0.5j * kap * h)
    return np.fft.ifft(fh)[:: m // n]
