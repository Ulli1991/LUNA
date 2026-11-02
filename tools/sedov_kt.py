"""Sedov blast in rho = rho0 r^-omega, planar/cylindrical/spherical (j = 1, 2, 3), after Kamm & Timmes 2007
(LA-UR-07-2849): standard, singular and vacuum families. The removable singularities omega2, omega3 are refused.
Their eqs. 58-60 for the singular alpha are misprinted; alpha here is eq. 54 integrated, which matches their Table 6."""
import numpy as np
from scipy.integrate import quad

EPS = 1e-4                                   # KT's tolerance for the singular / removable cases


class Sedov:
    """The self-similar solution for (j, omega, gamma)."""

    def __init__(self, j, omega, gamma):
        self.j, self.w, self.g = j, float(omega), float(gamma)
        j, w, g = self.j, self.w, self.g
        self.V2 = 4.0 / ((j + 2 - w) * (g + 1))                               # eq. 5
        self.Vstar = 2.0 / (j * (g - 1) + 2)                                  # eq. 6
        self.w2 = (2 * (g - 1) + j) / g
        self.w3 = j * (2 - g)
        if abs(self.V2 - self.Vstar) <= EPS:
            self.kind = "singular"
        elif self.V2 < self.Vstar - EPS:
            self.kind = "standard"
        else:
            self.kind = "vacuum"
        if self.kind != "singular" and (abs(w - self.w2) <= EPS or abs(w - self.w3) <= EPS):
            raise NotImplementedError(f"omega = {w} is at a removable singularity (omega2 {self.w2}, omega3 {self.w3})")
        q = j + 2 - w
        self.a0 = 2.0 / q                                                     # eqs. 19-24
        self.a2 = -(g - 1) / (g * (self.w2 - w))
        self.a1 = q * g / (2 + j * (g - 1)) * (2 * (j * (2 - g) - w) / (g * q * q) - self.a2)
        self.a3 = (j - w) / (g * (self.w2 - w))
        self.a4 = q * (j - w) / (self.w3 - w) * self.a1
        self.a5 = (w * (1 + g) - 2 * j) / (self.w3 - w)
        self.a = q * (g + 1) / 4.0                                            # eqs. 25-29
        self.b = (g + 1) / (g - 1)
        self.c = g * q / 2.0
        den = q * (g + 1) - 2 * (2 + j * (g - 1))                           # zero exactly in the singular case
        self.d = q * (g + 1) / den if self.kind != "singular" else np.inf
        self.e = (2 + j * (g - 1)) / 2.0
        self.Vmin = 2.0 / (q * g) if self.kind == "standard" else 2.0 / q    # eqs. 61, 62
        self.alpha = self._alpha()

    def funcs(self, dV):
        """(lambda, f, g, h, dlambda/dV) at V = Vmin + dV (eqs. 30-38); the factor vanishing at Vmin from dV exactly."""
        dV = np.asarray(dV, float)
        V = self.Vmin + dV
        x1 = self.a * V
        x2 = self.b * self.c * dV if self.kind == "standard" else self.b * (self.c * V - 1)
        x3 = self.d * (1 - self.e * V)
        x4 = -self.b * self.c / self.g * dV if self.kind == "vacuum" else self.b * (1 - self.c * V / self.g)
        lam = x1 ** -self.a0 * x2 ** -self.a2 * x3 ** -self.a1
        dlam = -lam * (self.a0 / x1 * self.a + self.a2 / x2 * self.b * self.c - self.a1 / x3 * self.d * self.e)
        f = x1 * lam
        w = self.w
        gg = x1 ** (self.a0 * w) * x2 ** (self.a3 + self.a2 * w) * x3 ** (self.a4 + self.a1 * w) * x4 ** self.a5
        h = x1 ** (self.a0 * self.j) * x3 ** (self.a4 + self.a1 * (w - 2)) * x4 ** (1 + self.a5)
        return lam, f, gg, h, dlam

    def _quad(self, fn):
        """int_Vmin^V2 fn(dV) dV with dV = (V2 - Vmin) u^6: removes KT's power-law endpoint singularities."""
        span = self.V2 - self.Vmin
        return quad(lambda u: fn(span * u ** 6) * span * 6 * u ** 5, 0.0, 1.0, limit=500,
                    epsabs=0, epsrel=1e-13)[0]

    def _alpha(self):
        """Eq. 54: alpha = 8 C0 / ((gamma^2 - 1) q^2) int_0^1 (g f^2 + h) lambda^(j-1) dlambda."""
        j, g, q = self.j, self.g, self.j + 2 - self.w
        C0 = {1: 2.0, 2: 2 * np.pi, 3: 4 * np.pi}[j]
        pre = 8 * C0 / ((g * g - 1) * q * q)
        if self.kind == "singular":
            return pre / j                                                    # f = lam, g = lam^(j-2), h = lam^j
        def integrand(dV):
            lam, f, gg, h, dlam = self.funcs(dV)
            return (gg * f * f + h) * lam ** (j - 1) * dlam
        return pre * self._quad(integrand)

    def J12(self):
        """KT eqs. 55, 56 (Tables 4 and 8); eq. 56 is misprinted with lambda^(j+1), eq. 54 needs lambda^(j-1)."""
        j, g, q = self.j, self.g, self.j + 2 - self.w
        def i1(dV):
            lam, f, gg, h, dlam = self.funcs(dV)
            V = self.Vmin + dV
            return (g + 1) / (g - 1) * lam ** (j + 1) * gg * V * V * dlam
        def i2(dV):
            lam, f, gg, h, dlam = self.funcs(dV)
            return 8 / ((g + 1) * q * q) * lam ** (j - 1) * h * dlam
        return self._quad(i1), self._quad(i2)

    def profile(self, n=400_000):
        """Dense (lambda, f, g, h), increasing in lambda, from the inner end (origin or vacuum edge) to the shock."""
        if self.kind == "singular":
            lam = np.linspace(0.0, 1.0, n)
            return lam, lam, lam ** (self.j - 2), lam ** self.j
        lam, f, gg, h, _ = self.funcs((self.V2 - self.Vmin) * np.geomspace(1e-300, 1.0, n))
        o = np.argsort(lam)
        return lam[o], f[o], gg[o], h[o]

    def r_shock(self, t, E, rho0):
        """Eq. 12."""
        return (E * t * t / (self.alpha * rho0)) ** (1.0 / (self.j + 2 - self.w))

    def solution(self, r, t, E, rho0, p_amb=0.0):
        """(rho, v, p) at radii r (eqs. 69-76); ambient rho0 r^-omega, at rest, p_amb outside the shock."""
        r = np.asarray(r, float)
        j, w, g = self.j, self.w, self.g
        r2 = self.r_shock(t, E, rho0)
        us = 2.0 / (j + 2 - w) * r2 / t                                       # eq. 13
        rho_pre = rho0 * r2 ** -w
        u2, rho2, p2 = 2 * us / (g + 1), (g + 1) / (g - 1) * rho_pre, 2 * rho_pre * us * us / (g + 1)
        lam, f, gg, h = self.profile()
        x = r / r2
        inside = x < 1.0
        rho = np.where(inside, rho2 * np.interp(x, lam, gg, left=0.0), rho0 * np.where(r > 0, r, 1.0) ** -w)
        v = np.where(inside, u2 * np.interp(x, lam, f, left=0.0), 0.0)
        p = np.where(inside, p2 * np.interp(x, lam, h, left=0.0), p_amb)
        return rho, v, p, r2

    def r_vacuum(self, t, E, rho0):
        """The vacuum boundary (0 unless vacuum)."""
        return 0.0 if self.kind != "vacuum" else self.r_shock(t, E, rho0) * self.profile()[0][0]


if __name__ == "__main__":
    # KT07 Tables 4-9 (gamma 1.4, rho0 1, t 1)
    for j, w, E, alpha, J1, J2, r2, extra in [
            (1, 0.0, 0.0673185, 0.538548, 0.197928, 0.175834, 0.5, None),
            (2, 0.0, 0.311357, 0.984041, 0.0654053, 0.0495650, 0.75, None),
            (3, 0.0, 0.851072, 0.851060, 0.0296269, 0.0211647, 1.0, None),
            (2, 5 / 3, 2.45749, 4.80856, None, None, 0.75, None),
            (3, 7 / 3, 4.90875, 4.90875, None, None, 1.0, None),
            (2, 1.7, 2.67315, 5.18062, 0.856238, 0.158561, 0.75, 0.154090),
            (3, 2.4, 5.45670, 5.45670, 0.454265, 0.0828391, 1.0, 0.272644)]:
        s = Sedov(j, w, 1.4)
        line = f"j {j} omega {w:.4f} {s.kind:9s} alpha {s.alpha:.6f} (KT {alpha}, {s.alpha / alpha - 1:+.1e})"
        line += f" r2 {s.r_shock(1.0, E, 1.0):.6f} (KT {r2})"
        if J1 is not None and s.kind != "singular":
            a, b = s.J12()
            line += f" J1 {a:.6g} ({a / J1 - 1:+.1e}) J2 {b:.6g} ({b / J2 - 1:+.1e})"
        if extra:
            line += f" r_vac {s.r_vacuum(1.0, E, 1.0):.6f} (KT {extra})"
        print(line)
    # Sedov 1959 via KT Tables 1-3 (lambda, f, g, h at V)
    for j, rows in [(1, [(0.9420, 0.5400, 0.9157, 0.6662, 0.7917), (0.4912, 0.4800, 0.4244, 0.0641, 0.4037)]),
                    (2, [(0.9476, 0.4000, 0.9097, 0.5161, 0.7122), (0.6390, 0.3600, 0.5521, 0.0362, 0.3867)]),
                    (3, [(0.9080, 0.3060, 0.8335, 0.2275, 0.5238), (0.6788, 0.2870, 0.5844, 0.0174, 0.3732)])]:
        s = Sedov(j, 0.0, 1.4)
        for lam, V, f, g, h in rows:
            L, F, G, H, _ = s.funcs(V - s.Vmin)
            print(f"  j {j} V {V}: lambda {L:.4f} ({lam}) f {F:.4f} ({f}) g {G:.4f} ({g}) h {H:.4f} ({h})")
