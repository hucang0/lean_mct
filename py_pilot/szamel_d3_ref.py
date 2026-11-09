#!/usr/bin/env python3
"""
Independent implementation of Szamel PRL 119, 155502 (2017) Eqs. (9)-(11) / (II.1)
for hard spheres in the Mari-Kurchan model (g(r) = Theta(r-1), all neighbours, density rho),
d = 3.  T = 1, sigma = 1.

Derivation (my own, from the paper):
  Eq (II.1):  M = (n T/2d) \int ds e^{-s^2/2TM}/(2 pi T M)^{d/2}  |F_s|^2 / Z_s
      Z_s = \int_{r>1} d^d r e^{-M r^2/2 + s.r},   F_s = \int d^d r delta(r-1) e^{-M r^2/2 + s.r} rhat
  Put s = M c (c = "cage-centre" of the neighbour in the relative coordinate):
      e^{-M r^2/2 + M c.r} = e^{M c^2/2} e^{-M|r-c|^2/2}
      => Z_s = e^{Mc^2/2} (2pi/M)^{d/2} q(c),   q(c) = \int_{|r|>1} d^d r (M/2pi)^{d/2} e^{-M|r-c|^2/2}
         F_s = e^{Mc^2/2} (2pi/M)^{d/2} grad_c q(c)
      and ds e^{-s^2/2M}/(2 pi M)^{d/2} = dc (M/2pi)^{d/2} e^{-Mc^2/2}
  =>  M = (rho/2d) \int d^d c |grad q(c)|^2 / q(c)                                     (*)
  with A = 1/(2M) (Szamel II.6) the per-component single-particle cage variance,
  MSD plateau = 2 d A.
  (*) is also exactly  1 = -(rho A/d) d/dA \int d^d c q_A ln q_A  (the m->1 replica eq.),
  using the heat-equation identity d/dA f_{2A} = Laplacian f_{2A} and one integration by parts.

d = 3 closed forms (c = |c|, x = M c):
  q(c)  = [e^{-M(1-c)^2/2} - e^{-M(1+c)^2/2}]/(c sqrt(2 pi M))
          + (1/2)[erfc((1-c) sqrt(M/2)) + erfc((1+c) sqrt(M/2))]
  q'(c) = M sqrt(2M/pi) e^{-M(1+c^2)/2} [cosh x/x - sinh x/x^2]
"""
import numpy as np
from scipy import integrate, optimize, special, stats
import sys

SQRT2PI = np.sqrt(2 * np.pi)


def q3(c, M):
    """q(c) in d = 3, numerically safe for all c >= 0, M > 0."""
    c = np.asarray(c, dtype=float)
    a = np.exp(-M * (1 - c) ** 2 / 2) * (-np.expm1(-2 * M * c))  # = e^{-M(1-c)^2/2} - e^{-M(1+c)^2/2}
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(c > 0, a / (c * np.sqrt(2 * np.pi * M)), np.sqrt(2 * M / np.pi) * np.exp(-M / 2))
    t2 = 0.5 * (special.erfc((1 - c) * np.sqrt(M / 2)) + special.erfc((1 + c) * np.sqrt(M / 2)))
    return t1 + t2


def dq3(c, M):
    """dq/dc in d = 3."""
    c = np.asarray(c, dtype=float)
    x = M * c
    # e^{-M(1+c^2)/2} [cosh x / x - sinh x / x^2]  written without overflow:
    # = e^{-M(1-c)^2/2} * (1/2) [ (1+e^{-2x})/x - (1-e^{-2x})/x^2 ]
    with np.errstate(divide="ignore", invalid="ignore"):
        em = np.exp(-2 * x)
        bracket = 0.5 * ((1 + em) / x - (1 - em) / x ** 2)
        small = x / 3.0 - x ** 3 / 30.0  # series of (x cosh x - sinh x)/x^2, times e^{-x}~ handled below
        # for small x use the direct series of [cosh x/x - sinh x/x^2] = x/3 + x^3/30 + ... times e^{-M(1+c^2)/2}
        direct_small = np.exp(-M * (1 + c ** 2) / 2) * (x / 3.0 + x ** 3 / 30.0 + x ** 5 / 840.0)
        val = np.where(x < 1e-2, direct_small, np.exp(-M * (1 - c) ** 2 / 2) * bracket)
    return M * np.sqrt(2 * M / np.pi) * val


def check_closed_forms(M=11.0):
    cs = np.array([0.0, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.5])
    q_ref = stats.ncx2.sf(M, 3, M * cs ** 2)  # P(M|c+g|^2 > M), M|c+g|^2 ~ ncx2(3, M c^2)
    q_me = q3(cs, M)
    h = 1e-5
    dq_num = (q3(cs + h, M) - q3(np.maximum(cs - h, 0), M)) / (cs + h - np.maximum(cs - h, 0))
    dq_me = dq3(cs, M)
    print("check q(c) vs scipy ncx2.sf and q'(c) vs finite difference, M =", M)
    for c, a, b, e, f in zip(cs, q_me, q_ref, dq_me, dq_num):
        print(f"  c={c:4.2f}  q={a:.12e} ref={b:.12e} rel={abs(a-b)/max(b,1e-300):.1e}   q'={e:.10e} fd={f:.10e}")


def integral_d3(M):
    """I(M) = \int d^3c |grad q|^2/q = 4 pi \int_0^inf c^2 q'(c)^2/q(c) dc."""
    f = lambda c: c ** 2 * dq3(c, M) ** 2 / q3(c, M)
    w = 1.0 / np.sqrt(M)
    pts = [0.0, max(1 - 8 * w, 0.0), 1 - 3 * w, 1.0, 1 + 3 * w, 1 + 8 * w, 1 + 20 * w]
    pts = sorted(set(max(p, 0.0) for p in pts))
    tot = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        if b > a:
            val, err = integrate.quad(f, a, b, limit=400, epsabs=0, epsrel=1e-12)
            tot += val
    val, err = integrate.quad(f, pts[-1], np.inf, limit=400, epsabs=1e-16, epsrel=1e-12)
    tot += val
    return 4 * np.pi * tot


def rho_of_M(M, d=3):
    """Density at which a plateau with memory M exists: rho = 2 d M / I(M)."""
    return 2 * d * M / integral_d3(M)


def main():
    check_closed_forms()
    print()
    # scan
    Ms = np.exp(np.linspace(np.log(2.0), np.log(200.0), 60))
    rhos = np.array([rho_of_M(M) for M in Ms])
    print("scan  M  ->  rho(M), phi(M) = pi rho/6, A = 1/(2M), MSD = 6A")
    for M, r in zip(Ms, rhos):
        print(f"  M={M:9.4f}  rho={r:10.6f}  phi={np.pi*r/6:9.6f}  A={1/(2*M):.5f}  MSD={3/M:.5f}")
    i0 = int(np.argmin(rhos))
    res = optimize.minimize_scalar(rho_of_M, bracket=(Ms[max(i0 - 1, 0)], Ms[i0], Ms[min(i0 + 1, len(Ms) - 1)]),
                                   tol=1e-10)
    Mc, rhoc = res.x, res.fun
    phic = np.pi * rhoc / 6
    print()
    print("FOLD (d=3):")
    print(f"  M_c   = {Mc:.6f}")
    print(f"  rho_c = {rhoc:.6f}   phi_c = pi rho_c/6 = {phic:.6f}   phihat = 2^3 phi/3 = {8*phic/3:.5f}")
    print(f"  A_c   = 1/(2M_c) = {1/(2*Mc):.6f} (per-component cage variance)   MSD plateau = 6 A_c = {3/Mc:.6f}")
    return Mc, rhoc, phic


if __name__ == "__main__":
    main()
