#!/usr/bin/env python3
"""Gaussian (MSD-level) cage closures for 3D hard spheres with liquid structure (PY / Verlet-Weis).

Conventions as in ../restored_pair/brc_gauss.py and SM Sec. S5 B: cage = isotropic Gaussian, per-component
variance A (= sigma^2 of the SM, = K of the replica formula), f_s(k) = exp(-k^2 A), MSD plateau = 6A; relative
coordinate r = u - v + c has variance s = 2A about the relative cage-centre vector c.  gamma_s = 3D Gaussian of
variance s per component.  Z(c) = (gamma_s * Theta(.-1))(c), J = dZ/dc (closed forms of SM Eq. (ZJ)).
Every closure has the form   Lambda(A; phi) = rho * A * K(A; phi) = 1,   rho = 6 phi/pi,
with K = (1/3) int d^3c [integrand].  phi_c = lowest phi at which max_A Lambda >= 1 (fold); glass branch =
smaller-A root.  Variants (integrand of K):
  a      : y(c) J(c)^2 / Z(c)                    cavity-weighted centre measure rho y Z, bare contact force
  a0     : y(c) J(c)^2                           same, 1/Z removed
  amix   : y(1)^2 J(c)^2 / q(c)                  exact-marginal pair measure (gamma_s * g), bare contact force
  MZ     : q'(c)^2 / q(c),  q = gamma_s * g      potential-of-mean-force = m->1 replica (Mangeat-Zamponi 2016 Eq. 30)
  MZ0    : q'(c)^2           = (1/6pi^2) int k^4 h^2 e^{-k^2 s}
  b      : Jc(c)^2 / Z(c),  Jc = d/dc (gamma_s * c_PY)   (MCT vertex as effective Mayer function) -- divergent, see notes
  b0     : Jc(c)^2           = (1/6pi^2) int k^4 c^2 e^{-k^2 s}
  GMCTV  : (1/6pi^2) int k^4 c^2 S e^{-k^2 s}    Gaussian/Vineyard reduction of tagged MCT (Ikeda-Miyazaki Eq. 6)
"""
from __future__ import annotations
import math
import numpy as np
from scipy.fft import dst, dct
from scipy.special import erfc
from scipy.optimize import minimize_scalar, brentq

PI = math.pi


def Z_gauss(c, s):
    c = np.asarray(c, float)
    sq = math.sqrt(2.0 * s)
    a, b = (1.0 - c) / sq, (1.0 + c) / sq
    # s(e^{-a^2}-e^{-b^2}) = s e^{-a^2}(1-e^{-4c/(2s)...}) written stably
    ea = np.exp(-a * a) * (-np.expm1(-(b * b - a * a)))
    Z = s * ea / (c * math.sqrt(2.0 * PI * s)) + 0.5 * (erfc(a) + erfc(b))
    return Z


def J_gauss(c, s):
    c = np.asarray(c, float)
    x = c / s
    pref = 2.0 * PI * (2.0 * PI * s) ** -1.5
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        em = np.exp(-2 * x)
        # e^{-(1-c)^2/2s} [ (1/x - 1/x^2) + (1/x + 1/x^2) e^{-2x} ]  (x = c/s)
        br = (1 / x - 1 / x ** 2) + (1 / x + 1 / x ** 2) * em
        J = pref * np.exp(-(1 - c) ** 2 / (2 * s)) * br
        small = x < 1e-2
        Js = pref * np.exp(-(1 + c ** 2) / (2 * s)) * (2 * x / 3 + x ** 3 / 15)
    return np.where(small, Js, J)


class Grid:
    def __init__(self, N=2 ** 17, dr=5e-4):
        self.N, self.dr = N, dr
        self.r = np.arange(1, N) * dr
        self.dk = PI / (N * dr)
        self.k = np.arange(1, N) * self.dk

    def r_from_k(self, Fk):
        return dst(self.k * Fk, type=1) / 2 * self.dk / (2 * PI ** 2 * self.r)

    def r_from_k_with_deriv(self, Fk):
        """F(r) and F'(r) from F^(k):  F = P/r, P = (1/2pi^2) sum k F^ sin(kr) dk, P' via DCT-I."""
        P = dst(self.k * Fk, type=1) / 2 * self.dk / (2 * PI ** 2)
        x = np.zeros(self.N + 1); x[1:self.N] = self.k ** 2 * Fk
        Pp = (dct(x, type=1) / 2)[1:self.N] * self.dk / (2 * PI ** 2)
        F = P / self.r
        Fp = Pp / self.r - P / self.r ** 2
        return F, Fp


def kspace_K(G, Fk2, s):
    """(1/6 pi^2) int dk k^4 Fk2(k) e^{-k^2 s}  (trapezoid on the DST k grid; integrand ->0 at both ends)."""
    return np.sum(G.k ** 4 * Fk2 * np.exp(-G.k ** 2 * s)) * G.dk / (6 * PI ** 2)


class Closures:
    """Holds the structure at one phi on grid G and evaluates K_variant(A)."""
    def __init__(self, G, rho, y_r, hk, ck=None, Sk=None, y1=None, cmax=30.0):
        self.G, self.rho = G, rho
        self.y_r, self.hk, self.ck, self.Sk = y_r, hk, ck, Sk
        self.y1 = y1
        self.ic = G.r <= cmax
        self.c = G.r[self.ic]
        self.w = 4 * PI * self.c ** 2 * G.dr / 3.0      # (1/3) int d^3c, trapezoid (endpoints negligible)

    def _ZJ(self, s):
        return Z_gauss(self.c, s), J_gauss(self.c, s)

    def K(self, variant, A, zfloor=0.0):
        s = 2.0 * A
        G = self.G
        if variant in ('a', 'a0', 'amix', 'b'):
            Z, J = self._ZJ(s)
        if variant == 'a':
            y = self.y_r[self.ic]
            with np.errstate(divide='ignore', invalid='ignore'):
                f = np.where(Z > 1e-280, y * J ** 2 / Z, 0.0)
            return np.sum(self.w * f)
        if variant == 'a0':
            return np.sum(self.w * self.y_r[self.ic] * J ** 2)
        if variant in ('MZ', 'amix'):
            q, qp = G.r_from_k_with_deriv(self.hk * np.exp(-G.k ** 2 * s / 2))
            q = 1.0 + q[self.ic]; qp = qp[self.ic]
            m = q > 1e-11
            if variant == 'MZ':
                return np.sum(self.w[m] * qp[m] ** 2 / q[m])
            return self.y1 ** 2 * np.sum(self.w[m] * J[m] ** 2 / q[m])
        if variant == 'MZ0':
            return kspace_K(G, self.hk ** 2, s)
        if variant == 'b0':
            return kspace_K(G, self.ck ** 2, s)
        if variant == 'GMCTV':
            return kspace_K(G, self.ck ** 2 * self.Sk, s)
        if variant == 'b':
            _, Jc = G.r_from_k_with_deriv(self.ck * np.exp(-G.k ** 2 * s / 2))
            Jc = Jc[self.ic]
            Zf = np.maximum(Z, zfloor)
            with np.errstate(divide='ignore', invalid='ignore'):
                f = np.where(Zf > 1e-300, Jc ** 2 / Zf, np.inf)
            return np.sum(self.w * f)
        raise ValueError(variant)

    def Lam(self, variant, A, **kw):
        return self.rho * A * self.K(variant, A, **kw)

    def maxLam(self, variant, lo=1e-4, hi=0.3, n=36, **kw):
        As = np.logspace(math.log10(lo), math.log10(hi), n)
        L = np.array([self.Lam(variant, a, **kw) for a in As])
        i = int(np.argmax(L))
        i0, i1 = max(i - 1, 0), min(i + 1, n - 1)
        res = minimize_scalar(lambda la: -self.Lam(variant, math.exp(la), **kw),
                              bounds=(math.log(As[i0]), math.log(As[i1])), method='bounded',
                              options=dict(xatol=1e-7))
        return -res.fun, math.exp(res.x), (As, L)

    def glass_A(self, variant, Amax, lo=2e-5, **kw):
        g = lambda la: self.Lam(variant, math.exp(la), **kw) - 1.0
        if g(math.log(Amax)) < 0:
            return None
        if g(math.log(lo)) > 0:
            return None
        return math.exp(brentq(g, math.log(lo), math.log(Amax), xtol=1e-10))
