#!/usr/bin/env python3
"""Percus-Yevick (and Verlet-Weis) structure of 3D hard spheres, diameter 1, packing fraction phi.

PY (Wertheim-Thiele): for r<1  c(r) = -[a + b r + (phi a/2) r^3],
    a = (1+2phi)^2/(1-phi)^4,  b = -6 phi (1+phi/2)^2/(1-phi)^4,   c(r) = 0 for r>1.
PY closure: y(r) = 1 + gamma(r), gamma = h - c.  Inside the core h=-1 so y = -c (polynomial);
outside c=0 so g = y = 1 + gamma.  In Fourier space gamma^ = rho c^^2/(1 - rho c^), which is known in closed
form, so the OZ equation is inverted ONCE by a 3D radial Fourier transform (DST-I), no iteration needed.

Radial 3D transform conventions:  F^(k) = (4 pi/k) int_0^inf r F(r) sin(kr) dr,
                                   F(r)  = (1/(2 pi^2 r)) int_0^inf k F^(k) sin(kr) dk.
Grid: r_i = i dr, k_j = j pi/(N dr), i,j = 1..N-1  (DST-I).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.fft import dst

PI = math.pi


def py_coeffs(phi):
    a = (1 + 2 * phi) ** 2 / (1 - phi) ** 4
    b = -6 * phi * (1 + phi / 2) ** 2 / (1 - phi) ** 4
    return a, b, phi * a / 2


def c_py_r(r, phi):
    a, b, d = py_coeffs(phi)
    r = np.asarray(r, float)
    return np.where(r < 1, -(a + b * r + d * r ** 3), 0.0)


def y_py_inside(r, phi):
    a, b, d = py_coeffs(phi)
    return a + b * r + d * r ** 3


def c_py_k(k, phi):
    """Analytic c^(k) = (4pi/k) int_0^1 r c(r) sin(kr) dr; recursion for k>=2, Gauss-Legendre for k<2."""
    a, b, d = py_coeffs(phi)
    k = np.asarray(k, float)
    out = np.empty_like(k)
    big = k >= 2.0
    kb = k[big]
    # S_n = int_0^1 r^n sin(kr) dr,  C_n = int_0^1 r^n cos(kr) dr
    s, c = np.sin(kb), np.cos(kb)
    C0 = s / kb
    S0 = (1 - c) / kb
    S1 = -c / kb + C0 / kb
    C1 = s / kb - S0 / kb
    S2 = -c / kb + 2 * C1 / kb
    C2 = s / kb - 2 * S1 / kb
    S3 = -c / kb + 3 * C2 / kb
    C3 = s / kb - 3 * S2 / kb
    S4 = -c / kb + 4 * C3 / kb
    out[big] = -(4 * PI / kb) * (a * S1 + b * S2 + d * S4)
    sm = ~big
    if np.any(sm):
        x, w = np.polynomial.legendre.leggauss(64)
        r = 0.5 * (x + 1); w = 0.5 * w
        ks = k[sm]
        with np.errstate(invalid='ignore', divide='ignore'):
            sinc = np.where(ks[:, None] * r[None, :] > 0, np.sin(ks[:, None] * r[None, :]) / (ks[:, None] * r[None, :] + 1e-300), 1.0)
        sinc[ks == 0, :] = 1.0
        cr = -(a + b * r + d * r ** 3)
        out[sm] = 4 * PI * np.sum(w * r ** 2 * cr * sinc, axis=1)
    return out


class Grid:
    def __init__(self, N=2 ** 17, dr=5e-4):
        self.N, self.dr = N, dr
        self.r = np.arange(1, N) * dr
        self.dk = PI / (N * dr)
        self.k = np.arange(1, N) * self.dk

    def r_from_k(self, Fk):
        """F(r_i) = (1/(2 pi^2 r_i)) sum_j k_j F^(k_j) sin(k_j r_i) dk."""
        return dst(self.k * Fk, type=1) / 2 * self.dk / (2 * PI ** 2 * self.r)

    def k_from_r(self, Fr):
        """F^(k_j) = (4 pi/k_j) sum_i r_i F(r_i) sin(k_j r_i) dr."""
        return dst(self.r * Fr, type=1) / 2 * self.dr * 4 * PI / self.k


def i_contact(grid):
    i1 = int(round(1.0 / grid.dr)) - 1     # index with r == 1.0
    assert abs(grid.r[i1] - 1.0) < 1e-12
    return i1


def py_structure(phi, grid: Grid):
    """Returns dict with r, k, c^(k), S(k), h^(k), and on the r grid: c(r), y(r), g(r), h(r)."""
    rho = 6 * phi / PI
    ck = c_py_k(grid.k, phi)
    Sk = 1.0 / (1.0 - rho * ck)
    hk = ck * Sk
    gamk = rho * ck ** 2 * Sk          # gamma^ = h^ - c^
    gam = grid.r_from_k(gamk)
    r = grid.r
    y = np.where(r < 1, y_py_inside(r, phi), 1 + gam)
    g = np.where(r < 1, 0.0, 1 + gam)
    i1 = i_contact(grid)
    # node r=1: assign mid value of the jump to h, g (second-order trapezoid treatment of the discontinuity)
    g_node = g.copy(); g_node[i1] = 0.5 * (1 + gam[i1])
    return dict(phi=phi, rho=rho, r=r, k=grid.k, ck=ck, Sk=Sk, hk=hk, c_r=c_py_r(r, phi), y=y, g=g,
                g_node=g_node, gam=gam, g1=1 + gam[i1], y_in_poly=lambda x: y_py_inside(x, phi))


# ------------------------------------------------------------------ Verlet-Weis (as in Mangeat-Zamponi Eqs. (57)-(59))
def g_cs_contact(phi):
    return (1 - phi / 2) / (1 - phi) ** 3


def g_py_contact(phi):
    return (1 + phi / 2) / (1 - phi) ** 2


def vw_structure(phi, grid: Grid, interior='continue'):
    """Verlet-Weis g(r) for r>=1: g_PY(xi r; phi*) + (dg1/r) cos(alpha(r-1)) exp(-alpha(r-1)),
    phi* = phi - phi^2/16, xi = (phi/phi*)^(1/3) (>1, VW reduced diameter), dg1 = g_CS(1;phi) - g_PY(xi;phi*),
    alpha = 24 dg1/(phi* g_PY(1;phi*)).  g = 0 inside the core.
    The VW form does not define y(r) inside the core; it is needed ONLY by version (a). Two (non-unique) choices:
      interior='continue': y(r<1) = y_PY(xi r; phi*) + dg(r)   (analytic continuation of the VW formula)
      interior='scale'   : y(r<1) = y_PY(r; phi) * g_CS(1)/g_PY(1)
      interior='shift'   : y(r<1) = y_PY(xi r; phi*) + dg1"""
    phis = phi - phi ** 2 / 16
    xi = (phi / phis) ** (1 / 3)
    st = py_structure(phis, grid)
    r = grid.r
    from scipy.interpolate import CubicSpline
    # y_PY(r; phi*) is smooth (C^1) across r=1, so a spline of y on the whole grid is fine
    sel = r < 40
    ys = CubicSpline(r[sel], st['y'][sel])
    gpy_xi = float(ys(xi))           # xi > 1 so this is g_PY(xi; phi*)
    dg1 = g_cs_contact(phi) - gpy_xi
    alpha = 24 * dg1 / (phis * st['g1'])
    def dg(x):
        return dg1 / x * np.cos(alpha * (x - 1)) * np.exp(-alpha * (x - 1))
    xr = np.minimum(xi * r, r[sel][-1])
    yv_out = ys(xr) + dg(r)
    if interior == 'continue':      # pathological: dg grows like e^{alpha(1-r)} inside the core (alpha ~ 20)
        y = np.where(r < 1, ys(np.minimum(xi * r, r[sel][-1])) + dg(r), yv_out)
    elif interior == 'shift':       # y(r<1) = y_PY(xi r; phi*) + dg1  (continuous at r = 1)
        y = np.where(r < 1, ys(xi * r) + dg1, yv_out)
    else:
        a = py_structure(phi, grid)
        y = np.where(r < 1, a['y'] * g_cs_contact(phi) / g_py_contact(phi), yv_out)
    g = np.where(r < 1, 0.0, yv_out)
    i1 = i_contact(grid)
    g_node = g.copy(); g_node[i1] = 0.5 * yv_out[i1]
    h_node = g_node - 1
    hk = grid.k_from_r(h_node)
    rho = 6 * phi / PI
    Sk = 1 + rho * hk
    return dict(phi=phi, rho=rho, r=r, k=grid.k, hk=hk, Sk=Sk, y=y, g=g, g_node=g_node, g1=yv_out[i1],
                phis=phis, xi=xi, dg1=dg1, alpha=alpha)
