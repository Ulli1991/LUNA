"""Exact Riemann solution for the Sod shock tube (Toro 2009, ch. 4).

Standard two-state Riemann problem: Newton iteration on the star-region
pressure p* using the two-shock/two-rarefaction flux functions, then
self-similar sampling in xi = (x - x0)/t.

Self-check (Toro Test 1, gamma=1.4, WL=(1,0,1), WR=(0.125,0,0.1)):
p* = 0.30313, u* = 0.92745 (Toro Table 4.2).
"""
import numpy as np


def _fK(p, rhoK, pK, gamma):
    """Toro's f_K(p) and its derivative for one side."""
    aK = np.sqrt(gamma * pK / rhoK)
    if p > pK:                                   # shock
        AK = 2.0 / ((gamma + 1.0) * rhoK)
        BK = (gamma - 1.0) / (gamma + 1.0) * pK
        f = (p - pK) * np.sqrt(AK / (p + BK))
        df = np.sqrt(AK / (p + BK)) * (1.0 - 0.5 * (p - pK) / (p + BK))
    else:                                        # rarefaction
        f = (2.0 * aK / (gamma - 1.0)) * ((p / pK) ** ((gamma - 1.0)
                                                       / (2.0 * gamma)) - 1.0)
        df = (1.0 / (rhoK * aK)) * (p / pK) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


def riemann_star(rhoL, uL, pL, rhoR, uR, pR, gamma):
    """Star-region (p*, u*) by Newton iteration (Toro sec. 4.3.2)."""
    p = max(0.5 * (pL + pR), 1e-12)              # initial guess
    for _ in range(60):
        fL, dfL = _fK(p, rhoL, pL, gamma)
        fR, dfR = _fK(p, rhoR, pR, gamma)
        dp = (fL + fR + (uR - uL)) / (dfL + dfR)
        p_new = max(p - dp, 1e-12)
        if abs(p_new - p) < 1e-14 * p:
            p = p_new
            break
        p = p_new
    fL, _ = _fK(p, rhoL, pL, gamma)
    fR, _ = _fK(p, rhoR, pR, gamma)
    u = 0.5 * (uL + uR) + 0.5 * (fR - fL)
    return p, u


def sod_exact(x, t, x0=0.5, gamma=1.4,
              WL=(1.0, 0.0, 1.0), WR=(0.125, 0.0, 0.1)):
    """Exact solution (rho, u, p) at positions x, time t.
    WL/WR = (rho, u, p) left/right primitive states."""
    rhoL, uL, pL = WL
    rhoR, uR, pR = WR
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    ps, us = riemann_star(rhoL, uL, pL, rhoR, uR, pR, gamma)

    x = np.asarray(x, dtype=np.float64)
    xi = (x - x0) / max(t, 1e-300)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    g1 = (gamma - 1.0) / (gamma + 1.0)
    g2 = (gamma - 1.0) / (2.0 * gamma)

    left = xi <= us
    # --- left side of the contact ---
    if ps > pL:                                  # left shock
        rhoLs = rhoL * ((ps / pL + g1) / (g1 * ps / pL + 1.0))
        SL = uL - aL * np.sqrt((gamma + 1.0) / (2.0 * gamma) * ps / pL + g2)
        in_l = left & (xi < SL)
        star_l = left & (xi >= SL)
        rho[in_l], u[in_l], p[in_l] = rhoL, uL, pL
        rho[star_l], u[star_l], p[star_l] = rhoLs, us, ps
    else:                                        # left rarefaction
        rhoLs = rhoL * (ps / pL) ** (1.0 / gamma)
        aLs = aL * (ps / pL) ** g2
        SH, ST = uL - aL, us - aLs
        head = left & (xi < SH)
        fan = left & (xi >= SH) & (xi < ST)
        star_l = left & (xi >= ST)
        rho[head], u[head], p[head] = rhoL, uL, pL
        ufan = 2.0 / (gamma + 1.0) * (aL + 0.5 * (gamma - 1.0) * uL + xi[fan])
        afan = aL - 0.5 * (gamma - 1.0) * (ufan - uL)
        rho[fan] = rhoL * (afan / aL) ** (2.0 / (gamma - 1.0))
        u[fan] = ufan
        p[fan] = pL * (afan / aL) ** (2.0 * gamma / (gamma - 1.0))
        rho[star_l], u[star_l], p[star_l] = rhoLs, us, ps

    right = ~left
    # --- right side of the contact ---
    if ps > pR:                                  # right shock
        rhoRs = rhoR * ((ps / pR + g1) / (g1 * ps / pR + 1.0))
        SR = uR + aR * np.sqrt((gamma + 1.0) / (2.0 * gamma) * ps / pR + g2)
        star_r = right & (xi <= SR)
        out_r = right & (xi > SR)
        rho[star_r], u[star_r], p[star_r] = rhoRs, us, ps
        rho[out_r], u[out_r], p[out_r] = rhoR, uR, pR
    else:                                        # right rarefaction
        rhoRs = rhoR * (ps / pR) ** (1.0 / gamma)
        aRs = aR * (ps / pR) ** g2
        SH, ST = uR + aR, us + aRs
        star_r = right & (xi <= ST)
        fan = right & (xi > ST) & (xi <= SH)
        tail = right & (xi > SH)
        rho[star_r], u[star_r], p[star_r] = rhoRs, us, ps
        ufan = 2.0 / (gamma + 1.0) * (-aR + 0.5 * (gamma - 1.0) * uR + xi[fan])
        afan = aR + 0.5 * (gamma - 1.0) * (ufan - uR)
        rho[fan] = rhoR * (afan / aR) ** (2.0 / (gamma - 1.0))
        u[fan] = ufan
        p[fan] = pR * (afan / aR) ** (2.0 * gamma / (gamma - 1.0))
        rho[tail], u[tail], p[tail] = rhoR, uR, pR

    return {"rho": rho, "u": u, "p": p, "p_star": ps, "u_star": us}


if __name__ == "__main__":
    ps, us = riemann_star(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 1.4)
    print(f"Sod (gamma=1.4): p*={ps:.5f} (ref 0.30313)  u*={us:.5f} (ref 0.92745)")
    ok = abs(ps - 0.30313) < 2e-5 and abs(us - 0.92745) < 2e-5
    print("SELF-CHECK:", "PASS" if ok else "FAIL")
