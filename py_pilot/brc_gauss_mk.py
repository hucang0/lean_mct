#!/usr/bin/env python3
"""Bond-resummed stationary closure (BRC) on the MK model, Gaussian-cage version (gate G2).

Cage density  rho~(u) = G_{sigma^2}(u)  (isotropic Gaussian, variance sigma^2 per component), so that the
plateau is f_s(k) = exp(-k^2 sigma^2) and the relative coordinate of a bond, r = u - v + c, is Gaussian with
variance s = 2 sigma^2 around the relative cage-centre vector c.

Kernel (SM Sec. S5 A, restored-pair stationary memory):
    M(k,inf) = (rho/k^2) * int d^3c  |A(k,c)|^2 / Z(c),
    A(k,c)   = int du dv rho~(u) rho~(v) delta(|u-v+c|-1) (khat.shat) e^{ik.u}
             = e^{-k^2 sigma^2/4} e^{-ik.c/2}  int dOmega_s (khat.shat) G_s(c-shat) e^{ik.shat/2}      (Gaussian)
    Z(c)     = int du dv rho~(u) rho~(v) Theta(|u-v+c|-1) = 1 - int_{|r|<1} G_s(r-c) d^3r.
MSD level (k->0):  f/(1-f) -> 1/(k^2 sigma^2),  M -> m0/k^2,
    m0 = (rho/3) int d^3c |J(c)|^2 / Z(c),   J(c) = int dOmega_s shat G_s(shat-c) = J(c) chat,
closure  sigma^2 m0(sigma^2) = 1.   The same with Z -> 1 is the diagonal (single-collision) theory at the same
level, m0_diag = (rho/3) int d^3c |J|^2 = (rho/(6 pi^2)) int dq q^4 f~(q)^2 e^{-q^2 s}  (Parseval).

Both equations are linear in rho, so  phi(sigma^2) = (pi/6) / (sigma^2 m0(sigma^2)/rho)  and the caged
solution exists iff phi >= phi_c = min_{sigma^2} phi(sigma^2).

Run:  python3 brc_gauss.py   -> results/brc_gauss.json, logs to stdout.
"""
from __future__ import annotations
import json, math, time, sys, os
import numpy as np
from scipy.special import erfc
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

PI = math.pi


def mayer_ft(q):
    q = np.asarray(q, dtype=float)
    out = np.empty_like(q)
    small = q < 1e-6
    out[~small] = -4.0 * PI / q[~small] ** 3 * (np.sin(q[~small]) - q[~small] * np.cos(q[~small]))
    out[small] = -4.0 * PI / 3.0
    return out


# ----------------------------------------------------------------------------- Gaussian-cage building blocks
def Z_gauss(c, s):
    """Z(c) = int_{|r|>1} G_s(r-c) d^3r  (probability that two independent Gaussian cages at distance c do not
    overlap). erfc form, accurate also where Z is exponentially small (c inside the unit ball, small s)."""
    c = np.asarray(c, dtype=float)
    sq = math.sqrt(2.0 * s)
    a, b = (1.0 - c) / sq, (1.0 + c) / sq
    term = (s * (np.exp(-a * a) - np.exp(-b * b))
            + c * math.sqrt(PI * s / 2.0) * (erfc(a) + erfc(b)))
    Z = term / (c * math.sqrt(2.0 * PI * s))
    # c -> 0 limit: Z(0) = int_{r>1} G_s(r) = erfc(1/sqrt(2s)) + sqrt(2/(pi s)) e^{-1/(2s)}
    tiny = c < 1e-6
    if np.any(tiny):
        Z = np.where(tiny, erfc(1.0 / sq) + math.sqrt(2.0 / (PI * s)) * math.exp(-1.0 / (2.0 * s)), Z)
    return Z


def J_gauss(c, s):
    """J(c) = int dOmega_s (shat.chat) G_s(shat - c)  (radial component of the mean 'contact vector')."""
    c = np.asarray(c, dtype=float)
    a = c / s
    pref = (2.0 * PI * s) ** -1.5 * 2.0 * PI
    with np.errstate(divide='ignore', invalid='ignore'):
        J = pref * ((1.0 / a - 1.0 / a ** 2) * np.exp(-(1.0 - c) ** 2 / (2.0 * s))
                    + (1.0 / a + 1.0 / a ** 2) * np.exp(-(1.0 + c) ** 2 / (2.0 * s)))
    # small-a series: 2cosh(a)/a - 2sinh(a)/a^2 = (2/3) a + a^3/15 + ...
    small = a < 1e-3
    if np.any(small):
        Js = (2.0 * PI * s) ** -1.5 * 2.0 * PI * np.exp(-(1.0 + c ** 2) / (2.0 * s)) * (2.0 * a / 3.0 + a ** 3 / 15.0)
        J = np.where(small, Js, J)
    return J


def c_grid(s, n=4000):
    """c grid covering the shell |c-1| <~ 8 sqrt(s) plus the inside of the ball."""
    cmax = 1.0 + 8.0 * math.sqrt(s) + 0.5
    return np.linspace(1e-4, cmax, n)


def m0_over_rho(s, n=4000, zfloor=0.0):
    """(1/3) int 4 pi c^2 dc J^2/Z  and  (1/3) int 4 pi c^2 dc J^2   (per unit rho), Simpson on a fine grid."""
    c = c_grid(s, n)
    J = J_gauss(c, s); Z = Z_gauss(c, s)
    if zfloor > 0: Z = np.maximum(Z, zfloor)
    w = 4.0 * PI * c ** 2
    from scipy.integrate import simpson
    with np.errstate(divide='ignore', invalid='ignore'):
        integ = np.where(Z > 0, J ** 2 / Z, 0.0)
    brc = simpson(w * integ, x=c) / 3.0
    diag = simpson(w * J ** 2, x=c) / 3.0
    return brc, diag


def m0_diag_qspace(s):
    """Parseval form of the diagonal MSD-level kernel: (1/(6 pi^2)) int_0^inf q^4 f~(q)^2 e^{-q^2 s} dq (per rho)."""
    f = lambda q: q ** 4 * mayer_ft(np.array([q]))[0] ** 2 * math.exp(-q * q * s)
    qmax = math.sqrt(60.0 / s) + 5.0
    val, err = quad(f, 0.0, qmax, limit=2000)
    return val / (6.0 * PI ** 2)


def phi_of_sigma2(sig2, variant='brc', n=4000):
    s = 2.0 * sig2
    brc, diag = m0_over_rho(s, n)
    m = brc if variant == 'brc' else diag
    return (PI / 6.0) / (sig2 * m)


def critical(variant, n=4000, bracket=(0.002, 0.5)):
    """phi_c = min over sigma^2 of phi(sigma^2); returns (phi_c, sigma2_c, MSD plateau 6 sigma^2)."""
    res = minimize_scalar(lambda ls: phi_of_sigma2(math.exp(ls), variant, n),
                          bounds=(math.log(bracket[0]), math.log(bracket[1])), method='bounded',
                          options=dict(xatol=1e-10))
    sig2 = math.exp(res.x)
    return res.fun, sig2


def sigma2_glass(phi, variant, n=4000, sig2_c=None):
    """Caged branch (smaller sigma^2) of sigma^2 m0(sigma^2) = 1 at density phi, or None in the liquid."""
    from scipy.optimize import brentq
    phic, s2c = critical(variant, n) if sig2_c is None else (None, sig2_c)
    g = lambda ls: phi_of_sigma2(math.exp(ls), variant, n) - phi
    lo = math.log(1e-3)
    if g(math.log(s2c)) > 0: return None
    return math.exp(brentq(g, lo, math.log(s2c), xtol=1e-12))


# ----------------------------------------------------------------------------- k-resolved Gaussian kernel (check)
def kernel_gauss_k(k, sig2, rho, nc=600, nthc=24, nth=96, nph=64, zfloor=1e-300):
    """M_G(k,inf) = (rho/k^2) e^{-k^2 sig2/2} int d^3c (1/Z) |int dOmega_s (khat.shat) G_s(c-shat) e^{ik.shat/2}|^2
    by direct quadrature (2D angular inner, (c, theta_c) outer). Used to check the Legendre machinery of
    brc_kres.py and to run the single-k closure variant."""
    s = 2.0 * sig2
    c = c_grid(s, nc)
    xc, wc = np.polynomial.legendre.leggauss(nthc)          # cos(theta_c)
    xs, ws = np.polynomial.legendre.leggauss(nth)           # cos(theta_s)
    ph = (np.arange(nph) + 0.5) * 2 * PI / nph              # phi_s (periodic trapezoid)
    sin_s = np.sqrt(1 - xs ** 2)
    # unit vectors shat (nth, nph, 3) with k along z
    shat = np.stack([sin_s[:, None] * np.cos(ph)[None, :], sin_s[:, None] * np.sin(ph)[None, :],
                     np.broadcast_to(xs[:, None], (nth, nph))], axis=-1)
    phase = np.exp(0.5j * k * xs)[:, None]                  # e^{ik.shat/2}
    wgt = (ws[:, None] * (2 * PI / nph)) * xs[:, None] * phase   # dOmega (khat.shat) e^{ik.shat/2}
    pref = (2.0 * PI * s) ** -1.5
    Zc = np.maximum(Z_gauss(c, s), zfloor)
    out = 0.0
    for ic, cc in enumerate(c):
        # chat = (sin th_c, 0, cos th_c); c - shat squared = c^2 + 1 - 2c (chat.shat)
        chat_dot_s = np.sqrt(1 - xc ** 2)[:, None, None] * shat[None, :, :, 0] + xc[:, None, None] * shat[None, :, :, 2]
        G = pref * np.exp(-(cc * cc + 1.0 - 2.0 * cc * chat_dot_s) / (2.0 * s))   # (nthc, nth, nph)
        inner = np.einsum('abc,bc->a', G, wgt)
        out += cc * cc * (c[1] - c[0]) * np.sum(wc * np.abs(inner) ** 2) * 2 * PI / Zc[ic]
    return rho / k ** 2 * math.exp(-k * k * sig2 / 2.0) * out


def main():
    t0 = time.time()
    out = {}
    # grid-convergence of the 1D integrals
    for n in (1000, 2000, 4000, 8000):
        pc_b, s2_b = critical('brc', n); pc_d, s2_d = critical('diag', n)
        print(f"n={n:5d}  BRC-G: phi_c={pc_b:.6f} sigma2_c={s2_b:.6f}   DIAG-G: phi_c={pc_d:.6f} sigma2_c={s2_d:.6f}")
    pc_b, s2_b = critical('brc', 8000); pc_d, s2_d = critical('diag', 8000)
    # Parseval check of the diagonal MSD-level kernel
    for s2 in (0.03, 0.06, 0.1):
        s = 2 * s2
        brc, diag = m0_over_rho(s, 8000)
        print(f"sigma2={s2}: m0_diag/rho real-space={diag:.8f}  q-space={m0_diag_qspace(s):.8f}  m0_brc/rho={brc:.8f}  ratio brc/diag={brc/diag:.4f}")
    out['phi_c_BRC_gauss_MSD'] = pc_b; out['sigma2_c_BRC'] = s2_b; out['MSD_plateau_c_BRC'] = 6 * s2_b
    out['phi_c_DIAG_gauss_MSD'] = pc_d; out['sigma2_c_DIAG'] = s2_d; out['MSD_plateau_c_DIAG'] = 6 * s2_d
    print(f"\nBRC-G (MSD level):  phi_c = {pc_b:.5f}, sigma^2_c = {s2_b:.5f} (MSD plateau 6 sigma^2 = {6*s2_b:.4f})")
    print(f"DIAG-G (MSD level): phi_c = {pc_d:.5f}, sigma^2_c = {s2_d:.5f} (MSD plateau = {6*s2_d:.4f})")
    print(f"reference: replica-Gaussian 1.65 (MK App. A) = Szamel (9)-(11); exact MC 1.82; diagonal k-resolved N=2 2.797")
    # phi(sigma^2) curves
    sig2s = np.logspace(math.log10(0.004), math.log10(0.6), 60)
    out['curve_sigma2'] = sig2s.tolist()
    out['curve_phi_brc'] = [phi_of_sigma2(x, 'brc', 4000) for x in sig2s]
    out['curve_phi_diag'] = [phi_of_sigma2(x, 'diag', 4000) for x in sig2s]
    # caged branch sigma^2(phi) and f_s(k) = exp(-k^2 sigma^2) at k = 3,5,7,10 for a few densities
    tab = {}
    for phi in (1.65, 1.7, 1.8, 1.9, 2.0, 2.2, 2.8):
        row = {}
        for var, s2c in (('brc', s2_b), ('diag', s2_d)):
            s2 = sigma2_glass(phi, var, 4000, s2c)
            row[var] = None if s2 is None else dict(sigma2=s2, MSD=6 * s2,
                                                      fs={k: math.exp(-k * k * s2) for k in (3, 5, 7, 10)})
        tab[str(phi)] = row
        def fmt(r): return 'liquid' if r is None else f"sigma2={r['sigma2']:.4f} MSD={r['MSD']:.3f} f(3,5,7,10)=" + ','.join(f"{r['fs'][k]:.3f}" for k in (3, 5, 7, 10))
        print(f"phi={phi}: BRC-G {fmt(row['brc'])} | DIAG-G {fmt(row['diag'])}")
    out['caged_branch'] = tab
    print("simulation phi=2.0: f_s(3,5,7,10) = 0.85, 0.65, 0.45, 0.23, MSD plateau 0.117")

    # single-k closure variant with the k-resolved Gaussian kernel: f/(1-f) = M_G(k) at k = k*
    print("\nk-resolved Gaussian kernel: M_G(k; sigma^2) and the ratio to f/(1-f), at phi = 2.0 on the MSD branch")
    s2 = tab['2.0']['brc']['sigma2']; rho = 6 * 2.0 / PI
    rowk = {}
    for k in (0.2, 1.0, 3.0, 5.0, 7.0, 10.0):
        Mk = kernel_gauss_k(k, s2, rho)
        f = math.exp(-k * k * s2)
        rowk[str(k)] = dict(M=Mk, f_over_1mf=f / (1 - f), ratio=Mk * (1 - f) / f)
        print(f"  k={k:5.1f}  M_G={Mk:.5f}  f/(1-f)={f/(1-f):.5f}  ratio={Mk*(1-f)/f:.4f}")
    out['kres_gauss_check_phi2'] = rowk
    # small-k consistency: k^2 M_G(k->0) should equal m0 = rho * brc
    brc, diag = m0_over_rho(2 * s2, 8000)
    print(f"  k^2 M_G(k=0.2) = {0.04*rowk['0.2']['M']:.6f}  vs  m0 = {rho*brc:.6f}  (= 1/sigma^2 = {1/s2:.6f} on the closure)")
    # single-k closure: find sigma^2(phi) from f/(1-f) = M_G(k*) at k* = 7 and the critical density
    def phi_single_k(sig2, kstar):
        f = math.exp(-kstar ** 2 * sig2)
        return (PI / 6.0) * (f / (1 - f)) / kernel_gauss_k(kstar, sig2, 1.0, nc=300, nthc=16, nth=64, nph=48)
    for kstar in (3.0, 7.0):
        res = minimize_scalar(lambda ls: phi_single_k(math.exp(ls), kstar), bounds=(math.log(0.01), math.log(0.3)),
                              method='bounded', options=dict(xatol=1e-4))
        print(f"single-k closure at k*={kstar}: phi_c = {res.fun:.4f}, sigma^2_c = {math.exp(res.x):.4f}")
        out[f'phi_c_BRC_gauss_singlek_{kstar}'] = res.fun; out[f'sigma2_c_singlek_{kstar}'] = math.exp(res.x)
    out['seconds'] = time.time() - t0
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'brc_gauss.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f"done in {time.time()-t0:.1f} s")


if __name__ == '__main__':
    main()
