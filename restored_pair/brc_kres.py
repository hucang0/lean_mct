#!/usr/bin/env python3
"""Bond-resummed stationary closure (BRC) on the MK model, wavevector-resolved version (gate G3).

Unknown: plateau f(k) on the Franosch grid k = h (j + 1/2), j = 0..M-1, h = qmax/M.
Cage:    rho~^(k) = sqrt(f(k))  (phase assumption), K^(q) = f(q) = Fourier transform of the relative-displacement
         density K = rho~ * rho~.
Kernel:  M(k) = (rho/k^2) int d^3c |A(k,c)|^2 / Z(c),
         A^(k,q) := int d^3c e^{iq.c} A(k,c) = -i (khat.q) f~(q) rho~^(|k-q|) rho~^(q)        (exact Fourier form)
         Z(c) = 1 - int_{|r|<1} K(r-c) d^3r = 1 - IFT[ v^(q) f(q) ](c),  v^(q) = -f~(q) = FT of the unit ball.
         With k along z, A^(k,q) is axially symmetric; Legendre expansion in x = khat.qhat,
         A^(k,q,x) = sum_l A^_l(q) P_l(x),  A_l(c) = (-i)^l (1/2pi^2) int q^2 dq j_l(qc) A^_l(q),
         int dOmega_c |A|^2 = 4 pi sum_l |A_l(c)|^2/(2l+1).
         Z -> 1 gives Parseval: (rho/k^2) int d^3q/(2pi)^3 (khat.q)^2 f~^2 f(|k-q|) f(q) = diagonal N=2 kernel (B).
Closure: f/(1-f) = M[f]  (exact plateau equation of the Cichocki-Hess memory equation with the BRC kernel).

Iteration f <- M[f]/(1+M[f]) from a Gaussian start (BRC-G sigma^2 at that density when it exists), optional damping.

python3 brc_kres.py check            : Gaussian-input checks (Parseval, Legendre vs direct quadrature, Z vs analytic)
python3 brc_kres.py scan             : fixed points at phi = 1.5, 1.7, 1.9, 2.2, 2.8 (+ 2.0 for the plateau comparison)
python3 brc_kres.py bisect lo hi     : bisection for phi_c
python3 brc_kres.py conv             : numerical-parameter convergence of M(k) at fixed f
"""
from __future__ import annotations
import sys, os, json, math, time
import numpy as np
from scipy.special import spherical_jn, eval_legendre
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
from brc_gauss import Z_gauss, J_gauss, sigma2_glass, critical, kernel_gauss_k   # noqa: E402

PI = math.pi


def mayer_ft(q):
    q = np.asarray(q, dtype=float)
    out = np.full_like(q, -4.0 * PI / 3.0)
    m = q > 1e-6
    out[m] = -4.0 * PI / q[m] ** 3 * (np.sin(q[m]) - q[m] * np.cos(q[m]))
    return out


class BRCKernel:
    """Precomputes everything that does not depend on f; evaluates M[f](k) on the grid."""

    def __init__(self, M=100, qmax=40.0, Nq=800, Nx=96, L=48, Nc=400, cmax=4.0, zfloor=1e-10, cache=True):
        self.M, self.qmax, self.h = M, qmax, qmax / M
        self.k = self.h * (np.arange(M) + 0.5)
        self.Nq, self.Nx, self.L, self.Nc, self.cmax, self.zfloor = Nq, Nx, L, Nc, cmax, zfloor
        # fine q grid (trapezoid), Legendre nodes, c grid (midpoints)
        self.q = np.linspace(0.0, qmax, Nq + 1)
        self.wq = np.full(Nq + 1, self.q[1] - self.q[0]); self.wq[0] *= 0.5; self.wq[-1] *= 0.5
        self.x, self.wx = np.polynomial.legendre.leggauss(Nx)
        self.c = (np.arange(Nc) + 0.5) * (cmax / Nc); self.dc = cmax / Nc
        self.ft_q = mayer_ft(self.q)
        ls = np.arange(L + 1)
        # Legendre projector: Ahat_l(q) = (2l+1)/2 sum_m w_m P_l(x_m) Ahat(q, x_m)
        self.Pl = np.stack([eval_legendre(l, self.x) for l in ls]) * (0.5 * (2 * ls + 1))[:, None] * self.wx[None, :]
        # Hankel matrices T_l[c, q] = (1/2pi^2) w_q q^2 j_l(q c)
        tag = f"hankel_M{M}_q{qmax:g}_Nq{Nq}_L{L}_Nc{Nc}_c{cmax:g}.npy"
        path = os.path.join(HERE, 'cache', tag)          # optional cache of the Hankel matrices (~0.13 GB at the defaults)
        if cache and os.path.exists(path):
            self.T = np.load(path)
        else:
            t0 = time.time()
            z = np.outer(self.c, self.q)
            self.T = np.empty((L + 1, Nc, Nq + 1))
            for l in ls:
                self.T[l] = spherical_jn(l, z) * (self.wq * self.q ** 2)[None, :] / (2 * PI ** 2)
            print(f"[hankel matrices built in {time.time()-t0:.1f} s, {self.T.nbytes/1e9:.2f} GB]", flush=True)
            if cache:
                os.makedirs(os.path.dirname(path), exist_ok=True); np.save(path, self.T)
        self.inv2l1 = 1.0 / (2 * ls + 1)
        # bipolar discretised diagonal vertex of SM Sec. S2 D (for the baseline and the Parseval check)
        kh = np.arange(M) + 0.5
        K_, Q_, P_ = kh[:, None, None], kh[None, :, None], kh[None, None, :]
        mask = (P_ >= np.abs(K_ - Q_) + 0.5 - 1e-9) & (P_ <= K_ + Q_ - 0.5 + 1e-9)
        W = self.h ** 5 / (16 * PI ** 2) * (Q_ * P_ / K_ ** 3) * (K_ ** 2 + Q_ ** 2 - P_ ** 2) ** 2 * mayer_ft(self.k)[None, :, None] ** 2
        self.W_over_k2 = np.where(mask, W, 0.0) / self.k[:, None, None] ** 2     # rho = 1

    # ---- interpolants of f and sqrt(f) on [0, qmax] with even extension; zero beyond qmax
    def _splines(self, f):
        f = np.clip(np.asarray(f, dtype=float), 0.0, 1.0)
        kk = np.concatenate([-self.k[::-1], self.k]); ff = np.concatenate([f[::-1], f])
        sf = CubicSpline(kk, ff); sr = CubicSpline(kk, np.sqrt(ff))
        def F(p):
            p = np.abs(p); out = np.where(p <= self.qmax, sf(np.minimum(p, self.qmax)), 0.0); return np.clip(out, 0.0, 1.0)
        def R(p):
            p = np.abs(p); out = np.where(p <= self.qmax, sr(np.minimum(p, self.qmax)), 0.0); return np.clip(out, 0.0, 1.0)
        return F, R

    def Zc(self, f):
        """Z(c) on the c grid from f on the fine q grid (l = 0 Hankel transform of v^ f)."""
        F, _ = self._splines(f)
        vf = -self.ft_q * F(self.q)
        return 1.0 - self.T[0] @ vf

    def evaluate(self, f, rho, return_parts=False):
        """M_brc(k), M_diag(k) (both with the same quadrature), and diagnostics."""
        F, R = self._splines(f)
        rq = R(self.q)                                   # rho~^(q) on the fine grid
        Z = self.Zc(f)
        Zg = np.maximum(Z, self.zfloor)
        Mb = np.empty(self.M); Md = np.empty(self.M)
        Sc_all = None
        for j, k in enumerate(self.k):
            p = np.sqrt(np.maximum(k * k + self.q[:, None] ** 2 - 2 * k * self.q[:, None] * self.x[None, :], 0.0))
            Ahat = (self.q * self.ft_q * rq)[:, None] * self.x[None, :] * R(p)      # (Nq+1, Nx), -i dropped
            Al_q = self.Pl @ Ahat.T                                                   # (L+1, Nq+1)
            Al_c = np.einsum('lcq,lq->lc', self.T, Al_q)                             # (L+1, Nc)
            S = 4 * PI * np.einsum('lc,l->c', Al_c ** 2, self.inv2l1)                # int dOmega_c |A|^2
            wc = self.dc * self.c ** 2
            Mb[j] = rho / k ** 2 * np.sum(wc * S / Zg)
            Md[j] = rho / k ** 2 * np.sum(wc * S)
        if return_parts:
            return Mb, Md, Z
        return Mb, Md

    def diag_bipolar(self, f, rho):
        """Discretised diagonal kernel (SM Sec. S2 D)  (rho/k^2) sum_{q,p} W~_s(k;q,p) f(q) f(p)  (D0 = 1)."""
        return rho * np.einsum('kqp,q,p->k', self.W_over_k2, f, f)

    # ---- fixed-point iteration
    def iterate(self, f0, rho, maxit=3000, tol=1e-9, damp=1.0, liquid_floor=1e-4, log_every=50, label=''):
        f = np.array(f0, dtype=float)
        hist = []
        t0 = time.time()
        for it in range(1, maxit + 1):
            Mb, Md = self.evaluate(f, rho)
            fnew = Mb / (1.0 + Mb)
            d = float(np.max(np.abs(fnew - f)))
            f = damp * fnew + (1 - damp) * f
            fk7 = float(np.interp(7.0, self.k, f))
            hist.append((it, d, fk7))
            if log_every and (it % log_every == 0 or it <= 3):
                print(f"  {label} it={it:5d}  dmax={d:.3e}  f(k=3,5,7,10)={np.interp(3,self.k,f):.4f},{np.interp(5,self.k,f):.4f},{fk7:.4f},{np.interp(10,self.k,f):.4f}  [{time.time()-t0:.0f}s]", flush=True)
            if fk7 < liquid_floor and np.interp(3.0, self.k, f) < 0.05:
                return dict(status='liquid', f=f, iterations=it, dmax=d, hist=hist, seconds=time.time() - t0)
            if d < tol:
                return dict(status='glass', f=f, iterations=it, dmax=d, hist=hist, seconds=time.time() - t0)
        return dict(status='unresolved', f=f, iterations=maxit, dmax=d, hist=hist, seconds=time.time() - t0)

    # ---- admissible cage: rho~ >= 0 in real space, Z from K = rho~ * rho~ in real space (0 <= Z <= 1 exactly)
    def _ugrid(self):
        if not hasattr(self, 'u'):
            self.Nu = 1200; self.umax = 6.0
            self.u = (np.arange(self.Nu) + 0.5) * (self.umax / self.Nu); self.du = self.umax / self.Nu
            # sine-transform matrices between the fine q grid and the u grid
            self.S_qu = np.sin(np.outer(self.u, self.q))                      # (Nu, Nq+1)
        return self.u

    def admissible(self, f):
        """Project f onto the admissible class: rho~ = IFT sqrt(f) clipped to >= 0 and renormalised.
        Returns dict(rq = rho~^(q) on the fine grid, f_adm on the k grid, rho_u, clipped_mass, Z(c))."""
        u = self._ugrid()
        _, R = self._splines(f)
        rq0 = R(self.q)
        rho_u = (self.S_qu @ (self.wq * self.q * rq0)) / (2 * PI ** 2 * u)      # rho~(u)
        neg = np.clip(-rho_u, 0, None)
        clipped_mass = float(np.sum(4 * PI * u ** 2 * neg) * self.du)
        rho_u = np.clip(rho_u, 0.0, None)
        norm = float(np.sum(4 * PI * u ** 2 * rho_u) * self.du)
        rho_u /= norm
        # back to q: rho~^(q) = (4 pi / q) int u sin(qu) rho~(u) du   (q = 0 -> 1 by normalisation)
        with np.errstate(divide='ignore', invalid='ignore'):
            rq = (4 * PI / self.q) * (self.S_qu.T @ (u * rho_u)) * self.du
        rq[0] = 1.0
        rq = np.clip(rq, 0.0, 1.0)
        f_adm = np.interp(self.k, self.q, rq) ** 2
        # K(w) = (2 pi / w) int u rho(u) [C(u+w) - C(|u-w|)] du,  C(t) = int_0^t t' rho(t') dt'
        C = np.concatenate([[0.0], np.cumsum(u * rho_u) * self.du])
        tgrid = np.concatenate([[0.0], u + 0.5 * self.du])                       # C defined at cell edges
        Cf = lambda t: np.interp(t, tgrid, C)
        w = u
        Kw = np.empty_like(w)
        for i, ww in enumerate(w):
            Kw[i] = (2 * PI / ww) * np.sum(u * rho_u * (Cf(u + ww) - Cf(np.abs(u - ww)))) * self.du
        # I(c) = int_{|r|<1} K(r-c) d^3r = (2 pi / c) int_0^1 r [D(r+c) - D(|r-c|)] dr,  D(t) = int_0^t w K(w) dw
        D = np.concatenate([[0.0], np.cumsum(w * Kw) * self.du])
        Df = lambda t: np.interp(t, tgrid, D)
        r = (np.arange(400) + 0.5) / 400.0; dr = 1.0 / 400
        Z = np.empty_like(self.c)
        for i, cc in enumerate(self.c):
            Z[i] = 1.0 - (2 * PI / cc) * np.sum(r * (Df(r + cc) - Df(np.abs(r - cc)))) * dr
        Z = np.clip(Z, 0.0, 1.0)
        return dict(rq=rq, f_adm=f_adm, rho_u=rho_u, clipped_mass=clipped_mass, Z=Z, K=Kw)

    def evaluate_adm(self, f, rho):
        """BRC and diagonal kernels evaluated for the admissible projection of f."""
        adm = self.admissible(f)
        rq = adm['rq']
        kk = np.concatenate([-self.q[:0:-1], self.q]); rr = np.concatenate([rq[:0:-1], rq])
        sr = CubicSpline(kk, rr)
        R = lambda p: np.clip(np.where(np.abs(p) <= self.qmax, sr(np.minimum(np.abs(p), self.qmax)), 0.0), 0.0, 1.0)
        Zg = np.maximum(adm['Z'], self.zfloor)
        Mb = np.empty(self.M); Md = np.empty(self.M)
        wc = self.dc * self.c ** 2
        for j, k in enumerate(self.k):
            p = np.sqrt(np.maximum(k * k + self.q[:, None] ** 2 - 2 * k * self.q[:, None] * self.x[None, :], 0.0))
            Ahat = (self.q * self.ft_q * rq)[:, None] * self.x[None, :] * R(p)
            Al_q = self.Pl @ Ahat.T
            Al_c = np.einsum('lcq,lq->lc', self.T, Al_q)
            S = 4 * PI * np.einsum('lc,l->c', Al_c ** 2, self.inv2l1)
            Mb[j] = rho / k ** 2 * np.sum(wc * S / Zg)
            Md[j] = rho / k ** 2 * np.sum(wc * S)
        return Mb, Md, adm

    def iterate_adm(self, f0, rho, maxit=2000, tol=1e-8, damp=1.0, kc=None, liquid_floor=1e-4, log_every=25, label=''):
        """Fixed-point iteration with the admissible projection; kc = window: for k > kc the new f is continued
        by the Gaussian through f(kc)."""
        f = np.array(f0, dtype=float); hist = []; t0 = time.time()
        for it in range(1, maxit + 1):
            Mb, Md, adm = self.evaluate_adm(f, rho)
            fnew = Mb / (1.0 + Mb)
            if kc is not None:
                jc = int(np.argmin(np.abs(self.k - kc)))
                s2t = -math.log(max(fnew[jc], 1e-300)) / self.k[jc] ** 2
                fnew[jc + 1:] = np.exp(-self.k[jc + 1:] ** 2 * s2t)
            d = float(np.max(np.abs(fnew - f)))
            f = damp * fnew + (1 - damp) * f
            fk = [float(np.interp(kk, self.k, f)) for kk in (3, 5, 7, 10)]
            hist.append((it, d, fk[2], adm['clipped_mass'], float(adm['Z'].min())))
            if log_every and (it % log_every == 0 or it <= 5):
                print(f"  {label} it={it:5d} dmax={d:.3e} f(3,5,7,10)={fk[0]:.4f},{fk[1]:.4f},{fk[2]:.4f},{fk[3]:.4f} "
                      f"clipped={adm['clipped_mass']:.2e} Zmin={adm['Z'].min():.2e} [{time.time()-t0:.0f}s]", flush=True)
            if fk[2] < liquid_floor and fk[0] < 0.05:
                return dict(status='liquid', f=f, iterations=it, dmax=d, hist=hist, seconds=time.time() - t0, adm=adm)
            if d < tol:
                return dict(status='glass', f=f, iterations=it, dmax=d, hist=hist, seconds=time.time() - t0, adm=adm)
            if f.min() > 0.999:
                return dict(status='runaway', f=f, iterations=it, dmax=d, hist=hist, seconds=time.time() - t0, adm=adm)
        return dict(status='unresolved', f=f, iterations=maxit, dmax=d, hist=hist, seconds=time.time() - t0, adm=adm)

    def rho_tilde(self, f, u):
        """Real-space cage density rho~(u) = (1/2pi^2 u) int q sin(qu) sqrt(f) dq (positivity diagnostic)."""
        _, R = self._splines(f)
        rq = R(self.q)
        u = np.asarray(u, dtype=float)
        return np.array([np.sum(self.wq * self.q * np.sin(self.q * uu) * rq) / (2 * PI ** 2 * uu) for uu in u])


def gaussian_f(k, sig2):
    return np.exp(-k * k * sig2)


def start_f(K, phi, s2c):
    s2 = sigma2_glass(phi, 'brc', 4000, s2c)
    if s2 is None:
        s2 = 0.02
        note = 'BRC-G liquid at this phi; start from sigma^2 = 0.02'
    else:
        note = f'start from BRC-G caged branch sigma^2 = {s2:.5f}'
    return gaussian_f(K.k, s2), s2, note


def cmd_check():
    K = BRCKernel()
    phi = 2.0; rho = 6 * phi / PI
    s2 = 0.0199
    f = gaussian_f(K.k, s2)
    Mb, Md, Z = K.evaluate(f, rho, return_parts=True)
    Mbip = K.diag_bipolar(f, rho)
    print("Gaussian input sigma^2=%.4f, phi=%.2f" % (s2, phi))
    print("  Z(c) Fourier vs analytic (max abs diff): %.2e" % np.max(np.abs(Z - Z_gauss(K.c, 2 * s2))))
    print("  k     M_brc(Legendre)   M_G(direct 2D quad)   M_diag(Parseval)   M_diag(bipolar grid)   f/(1-f)")
    for kk in (0.2, 1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0):
        j = int(np.argmin(np.abs(K.k - kk)))
        Mg = kernel_gauss_k(K.k[j], s2, rho)
        ff = f[j]
        print(f"  {K.k[j]:5.1f}  {Mb[j]:14.6f}   {Mg:14.6f}        {Md[j]:12.6f}      {Mbip[j]:12.6f}      {ff/(1-ff):10.5f}")
    # a non-Gaussian input: simulation-like heavier tail f = exp(-k^2 s2/(1+ a k^2 s2))
    print("Parseval check with a non-Gaussian f:")
    f2 = np.exp(-K.k ** 2 * 0.03 / (1 + 0.4 * K.k ** 2 * 0.03))
    Mb2, Md2 = K.evaluate(f2, rho); Mbip2 = K.diag_bipolar(f2, rho)
    for kk in (1.0, 3.0, 7.0, 15.0):
        j = int(np.argmin(np.abs(K.k - kk)))
        print(f"  k={K.k[j]:5.1f}  M_diag(Parseval)={Md2[j]:.6f}  bipolar={Mbip2[j]:.6f}  M_brc={Mb2[j]:.6f}  ratio brc/diag={Mb2[j]/Md2[j]:.4f}")
    # diagonal N=2 baseline on this grid (reproduce 2.7966): iterate the bipolar sum
    def diag_fixed_point(phi):
        rho = 6 * phi / PI; f = np.ones(K.M)
        for it in range(200000):
            Mk = K.diag_bipolar(f, rho); fn = Mk / (1 + Mk)
            d = np.max(np.abs(fn - f)); f = fn
            if f.max() < 1e-7: return 'liquid', it
            if d < 1e-12: return ('glass' if f.max() > 1e-3 else 'liquid'), it
        return 'unresolved', it
    lo, hi = 2.6, 3.0
    for _ in range(14):
        mid = 0.5 * (lo + hi); st, _it = diag_fixed_point(mid)
        if st == 'glass': hi = mid
        else: lo = mid
    print(f"diagonal N=2 (bipolar, M=100, qmax=40) phi_c in [{lo:.5f}, {hi:.5f}]  (SM Table S1: 2.7966)")


def cmd_conv():
    """Numerical-parameter convergence of M_brc(k) for a fixed non-Gaussian f."""
    base = dict(Nq=800, Nx=96, L=48, Nc=400, cmax=4.0, zfloor=1e-10)
    phi = 2.0; rho = 6 * phi / PI
    ref = None
    for name, kw in (('base', {}), ('Nq1600', dict(Nq=1600)), ('Nx160', dict(Nx=160)), ('L72', dict(L=72)),
                     ('Nc800', dict(Nc=800)), ('cmax6', dict(cmax=6.0)), ('zfloor1e-6', dict(zfloor=1e-6)), ('zfloor1e-14', dict(zfloor=1e-14))):
        p = dict(base); p.update(kw)
        K = BRCKernel(cache=False, **p)
        f = np.exp(-K.k ** 2 * 0.03 / (1 + 0.4 * K.k ** 2 * 0.03))
        Mb, Md = K.evaluate(f, rho)
        if ref is None: ref = Mb
        idx = [int(np.argmin(np.abs(K.k - kk))) for kk in (1, 3, 7, 10, 15)]
        print(f"{name:12s} M_brc(k=1,3,7,10,15) = " + ' '.join(f"{Mb[j]:.6f}" for j in idx) + f"   max rel diff vs base = {np.max(np.abs(Mb-ref)/ref):.2e}")


def run_density(K, phi, s2c, maxit=2000, damp=1.0, kc=None, tag=None, start_sigma2=None):
    rho = 6 * phi / PI
    if start_sigma2 is None:
        f0, s2, note = start_f(K, phi, s2c)
    else:
        s2 = start_sigma2; f0 = gaussian_f(K.k, s2); note = f'start from sigma^2 = {s2}'
    print(f"phi={phi} (kc={kc}, damp={damp}): {note}", flush=True)
    r = K.iterate_adm(f0, rho, maxit=maxit, damp=damp, kc=kc, label=f"phi={phi}")
    f = r['f']; adm = r['adm']
    out = dict(phi=phi, kc=kc, damp=damp, status=r['status'], iterations=r['iterations'], dmax=r['dmax'], seconds=r['seconds'],
               start_sigma2=s2, k=K.k.tolist(), f=f.tolist(), clipped_mass=adm['clipped_mass'], Zmin=float(adm['Z'].min()),
               f_at=dict(k3=float(np.interp(3, K.k, f)), k5=float(np.interp(5, K.k, f)), k7=float(np.interp(7, K.k, f)), k10=float(np.interp(10, K.k, f))),
               hist=r['hist'])
    if r['status'] == 'glass':
        s2eff = float(-np.log(f[0]) / K.k[0] ** 2)
        out['sigma2_eff_smallk'] = s2eff; out['MSD_plateau'] = 6 * s2eff
        out['rho_tilde_u'] = K.u.tolist(); out['rho_tilde'] = adm['rho_u'].tolist()
        Mb, Md, _ = K.evaluate_adm(f, rho)
        out['M_brc'] = Mb.tolist(); out['M_diag_same_f'] = Md.tolist()
        fG = np.exp(-K.k ** 2 * s2eff)
        out['f_gauss_same_MSD_at'] = dict(k3=float(np.interp(3, K.k, fG)), k7=float(np.interp(7, K.k, fG)), k10=float(np.interp(10, K.k, fG)))
        hist = r['hist']; mono = all(hist[i][2] >= hist[i + 1][2] - 1e-12 for i in range(len(hist) - 1))
        out['monotone_f7'] = bool(mono)
        print(f"  -> GLASS: f(3,5,7,10) = {out['f_at']['k3']:.4f}, {out['f_at']['k5']:.4f}, {out['f_at']['k7']:.4f}, {out['f_at']['k10']:.4f}; "
              f"MSD plateau (small-k) = {6*s2eff:.4f}; Gaussian with same MSD: f(3,7,10) = {out['f_gauss_same_MSD_at']['k3']:.3f},{out['f_gauss_same_MSD_at']['k7']:.3f},{out['f_gauss_same_MSD_at']['k10']:.3f}; "
              f"clipped mass {adm['clipped_mass']:.2e}; iterations {r['iterations']}; monotone in f(7): {mono}", flush=True)
    else:
        print(f"  -> {r['status'].upper()} after {r['iterations']} iterations ({r['seconds']:.0f} s); f(7)={out['f_at']['k7']:.4f}", flush=True)
    if tag:
        json.dump(out, open(os.path.join(HERE, 'results', f'kres_{tag}.json'), 'w'))
    return out


def cmd_scan(phis, kc=None, damp=1.0, grid=None, start=None):
    K = BRCKernel(**(grid or {}))
    pc, s2c = critical('brc', 4000)
    res = {}
    for phi in phis:
        res[str(phi)] = run_density(K, phi, s2c, kc=kc, damp=damp, tag=f"phi{phi}_kc{kc}_M{K.M}_q{K.qmax:g}", start_sigma2=start)
    json.dump(res, open(os.path.join(HERE, 'results', f'kres_scan_kc{kc}_M{K.M}_q{K.qmax:g}.json'), 'w'))


def cmd_bisect(lo, hi, nmax=8, maxit=1500, kc=None, damp=1.0, grid=None):
    K = BRCKernel(**(grid or {}))
    pc, s2c = critical('brc', 4000)
    log = []
    for i in range(nmax):
        mid = 0.5 * (lo + hi)
        r = run_density(K, mid, s2c, maxit=maxit, kc=kc, damp=damp, tag=f"bisect_{mid:.5f}_kc{kc}_M{K.M}_q{K.qmax:g}")
        log.append(dict(phi=mid, status=r['status'], iterations=r['iterations'], f7=r['f_at']['k7']))
        if r['status'] == 'glass': hi = mid
        elif r['status'] == 'liquid': lo = mid
        else:
            print(f"  {r['status']} at phi={mid:.5f} (f7={r['f_at']['k7']:.4f}); treating as " + ('glass' if r['f_at']['k7'] > 0.05 else 'liquid'))
            if r['f_at']['k7'] > 0.05: hi = mid
            else: lo = mid
        print(f"bisection: phi_c in [{lo:.5f}, {hi:.5f}]", flush=True)
        json.dump(dict(lo=lo, hi=hi, kc=kc, M=K.M, qmax=K.qmax, log=log), open(os.path.join(HERE, 'results', f'kres_bisect_kc{kc}_M{K.M}_q{K.qmax:g}.json'), 'w'), indent=1)
    return lo, hi


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['check', 'conv', 'scan', 'bisect', 'trace'])
    ap.add_argument('args', nargs='*', type=float)
    ap.add_argument('--kc', type=float, default=None)
    ap.add_argument('--damp', type=float, default=1.0)
    ap.add_argument('--nmax', type=int, default=8)
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--qmax', type=float, default=40.0)
    ap.add_argument('--start', type=float, default=None, help='start sigma^2 (default: BRC-G branch)')
    a = ap.parse_args()
    GRID = dict(M=a.M, qmax=a.qmax, Nq=int(20 * a.qmax))
    if a.cmd == 'check': cmd_check()
    elif a.cmd == 'conv': cmd_conv()
    elif a.cmd == 'scan': cmd_scan(a.args or [2.0, 2.8, 2.2, 1.9, 1.7, 1.5], kc=a.kc, damp=a.damp, grid=GRID, start=a.start)
    elif a.cmd == 'bisect': cmd_bisect(a.args[0], a.args[1], a.nmax, kc=a.kc, damp=a.damp, grid=GRID)
    elif a.cmd == 'trace':
        K = BRCKernel(); phi = a.args[0] if a.args else 2.0; rho = 6 * phi / PI
        f = gaussian_f(K.k, a.args[1] if len(a.args) > 1 else 0.0199)
        idx = [int(np.argmin(np.abs(K.k - kk))) for kk in (0.2, 1, 3, 5, 7, 10, 15, 20, 25, 30, 39.8)]
        print("k:      " + ' '.join(f"{K.k[j]:8.1f}" for j in idx))
        for it in range(1, 9):
            Mb, Md, adm = K.evaluate_adm(f, rho)
            print(f"it{it} f:  " + ' '.join(f"{f[j]:8.2e}" for j in idx) + f"  clipped={adm['clipped_mass']:.2e} Zmin={adm['Z'].min():.2e} rho(0)={adm['rho_u'][0]:.2f}")
            print(f"  ratio: " + ' '.join(f"{Mb[j]*(1-f[j])/max(f[j],1e-300):8.2e}" for j in idx))
            fnew = Mb / (1 + Mb)
            if a.kc is not None:
                jc = int(np.argmin(np.abs(K.k - a.kc))); s2t = -math.log(fnew[jc]) / K.k[jc] ** 2
                fnew[jc + 1:] = np.exp(-K.k[jc + 1:] ** 2 * s2t)
            f = a.damp * fnew + (1 - a.damp) * f
