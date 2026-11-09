#!/usr/bin/env python3
"""Variational non-Gaussian (homogeneous) cage, m -> 1, MK random-shift hard spheres, d = 3.

Functional (entropy of a state, product cages + pairwise non-overlap; = m->1 limit of the replicated
second-virial free entropy with a factorised molecule; Bethe form on the MK tree with marginals ~ cage):
    s[rho~] = -int rho~ ln rho~ + (rho/2) int d^3R  Z(R) ln Z(R),   Z = 1 - B*K,  K = rho~*rho~,  B = unit ball.
Euler-Lagrange (isotropic, positivity automatic):
    rho~(u) ∝ exp[ rho (L * B * rho~)(u) ],   L = -ln Z            ("full": bond hard core resummed)
Matched ablation (single collision, i.e. Z ln Z expanded to (1-Z)^2/2):
    L = 1 - Z                                                        ("single")
Gaussian restriction (infinite space, SM Eq. S27): full -> sigma^2 (rho/3) int |J|^2/Z = 1 (phi_c = 1.7762, = Szamel = replica m->1);
                      single -> same with Z -> 1 (phi_c = 3.2369, diagonal MSD level).
Note: this numerical Euler-Lagrange solver operates on a finite radial ball u <= rmax = 4.0 (default),
giving the static non-Gaussian folds phi_c = 1.7887 (full) and phi_c = 3.3339 (single, l = 1) in SM Tables S15-S16.
Observables: f_s(k) = |rho^~(k)|^2 (two independent draws from the cage), MSD plateau = 2 <|u|^2>.
All convolutions are done in real space with the bipolar formula, so every function stays >= 0 (no clipping).
"""
import json, math, sys, time, argparse, os
import numpy as np
sys.dont_write_bytecode = True
PI = math.pi


class Grid:
    def __init__(self, dr=0.004, rmax=4.0):
        self.dr = dr
        self.r = (np.arange(int(round(rmax / dr))) + 0.5) * dr
        self.N = len(self.r)
        self.w = 4 * PI * self.r ** 2 * dr          # radial measure for int d^3u
        r = self.r
        # bipolar convolution:  (f*g)(y) = (2 pi / y) int du u f(u) [G(u+y) - G(|u-y|)],  G(v) = int_0^v v' g(v') dv'
        self.U = r[None, :]; self.Y = r[:, None]
        self.hi = self.U + self.Y; self.lo = np.abs(self.U - self.Y)
        # unit-ball helper for (B*K)
        # Z(R) = int d^3y K(y) [ |R - y| >= 1 ] = 2 pi int dy y^2 K(y) (1 + clip((y^2+R^2-1)/(2yR), -1, 1))
        Y, R = r[None, :], r[:, None]
        mu = np.clip((Y ** 2 + R ** 2 - 1.0) / (2 * Y * R), -1.0, 1.0)
        self.Zw = 2 * PI * Y ** 2 * (1.0 + mu) * dr      # (R, y)
        self.Bw = 2 * PI * Y ** 2 * (1.0 - mu) * dr      # (B*g)(R) = int_{|R-y|<1} g

    def conv(self, f, g):
        r, dr = self.r, self.dr
        vg = r * g
        Gc = np.concatenate([[0.0], np.cumsum(vg) * dr])          # G at edges 0, dr, 2dr, ...
        edges = np.arange(self.N + 1) * dr
        Ghi = np.interp(self.hi, edges, Gc, right=Gc[-1]); Glo = np.interp(self.lo, edges, Gc)
        return (2 * PI / r) * ((Ghi - Glo) @ (r * f * dr))

    def ft(self, f, k):
        k = np.atleast_1d(k)
        return np.array([np.sum(self.w * f * (np.sinc(kk * self.r / PI))) for kk in k])

    def norm(self, f):
        return f / np.sum(self.w * f)


def gauss(G, s2):
    return np.exp(-G.r ** 2 / (2 * s2)) / (2 * PI * s2) ** 1.5


def step_parts(G, rt):
    K = G.conv(rt, rt)
    Z = np.clip(G.Zw @ K, 1e-300, 1.0)
    Brt = G.Bw @ rt
    return K, Z, Brt


def update(G, rt, rho, variant):
    K, Z, Brt = step_parts(G, rt)
    if variant == 'full':
        L = -np.log(Z)
    elif variant == 'single':
        L = 1.0 - Z
    else:                       # 'nK': K collisions, L = sum_{m=1}^K (1-Z)^m / m
        x = 1.0 - Z; L = sum(x ** m / m for m in range(1, int(variant[1:]) + 1))
    W = rho * G.conv(L, Brt)
    lw = W - W.max()
    new = G.norm(np.exp(lw))
    return new, Z


def solve(phi, variant='full', s2_start=0.02, G=None, damp=0.5, maxit=4000, tol=1e-10, verbose=False):
    G = G or Grid()
    rho = 6 * phi / PI
    rt = G.norm(gauss(G, s2_start))
    lr = np.log(np.maximum(rt, 1e-300))
    t0 = time.time(); msd_prev = None
    for it in range(1, maxit + 1):
        new, Z = update(G, rt, rho, variant)
        ln_new = np.log(np.maximum(new, 1e-300))
        lr = (1 - damp) * lr + damp * ln_new
        rt = G.norm(np.exp(lr - lr.max()))
        u2 = float(np.sum(G.w * rt * G.r ** 2))
        msd = 2 * u2
        if verbose and (it % 50 == 0 or it < 4):
            print(f"  {variant} phi={phi} it={it} MSD={msd:.5f}", flush=True)
        if msd > 1.5:   # cage dissolved (liquid)
            return dict(status='liquid', phi=phi, variant=variant, iterations=it)
        if msd_prev is not None and abs(msd - msd_prev) < tol * max(msd, 1e-6) and it > 20:
            break
        msd_prev = msd
    status = 'glass' if it < maxit else 'unresolved'
    k = np.array([3.0, 5.0, 7.0, 10.0])
    fk = G.ft(rt, k) ** 2
    kk = np.linspace(0.2, 30, 150)
    fkk = G.ft(rt, kk) ** 2
    u4 = float(np.sum(G.w * rt * G.r ** 4))
    alpha2 = 3 * (2 * u4 + 2 * (u2 ** 2) / 3 * 0 + 0) / 1.0  # placeholder, replaced below
    # non-Gaussian parameter of the displacement d = u - u', u,u' iid:  <d^2> = 2<u^2>, <d^4> = 2<u^4> + (10/3)<u^2>^2
    d2 = 2 * u2; d4 = 2 * u4 + (10.0 / 3.0) * u2 ** 2
    alpha2 = 3 * d4 / (5 * d2 ** 2) - 1
    out = dict(status=status, phi=phi, variant=variant, iterations=it, seconds=time.time() - t0,
               msd=msd, f=dict(zip(['k3', 'k5', 'k7', 'k10'], fk.tolist())),
               fG_sameMSD=dict(zip(['k3', 'k5', 'k7', 'k10'], np.exp(-k ** 2 * msd / 6).tolist())),
               alpha2=alpha2, Zmin=float(Z.min()), k=kk.tolist(), fk=fkk.tolist(),
               r=G.r[::5].tolist(), rho_tilde=rt[::5].tolist(), tail_mass_r_gt_1=float(np.sum((G.w * rt)[G.r > 1.0])))
    return out


def gaussian_check(G, variant):
    """Stationarity of s[rho~] restricted to Gaussians: phi(sigma^2) from the virial-like identity, min over sigma^2."""
    best = None
    for s2 in np.geomspace(0.01, 0.2, 60):
        rt = G.norm(gauss(G, s2))
        K, Z, Brt = step_parts(G, rt)
        # sigma^2 (rho/3) int |grad Z|^2 / Z = 1   (full)   or  with Z -> 1 (single)
        dZ = np.gradient(Z, G.r)
        I = np.sum(G.w * dZ ** 2 / (Z if variant == 'full' else 1.0))
        rho = 3.0 / (s2 * I); phi = PI * rho / 6
        if best is None or phi < best[0]:
            best = (phi, s2)
    return best


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('args', nargs='*', type=float)
    ap.add_argument('--variant', default='full'); ap.add_argument('--dr', type=float, default=0.004)
    ap.add_argument('--rmax', type=float, default=4.0); ap.add_argument('--damp', type=float, default=0.5)
    ap.add_argument('--s2', type=float, default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    G = Grid(a.dr, a.rmax)
    if a.cmd == 'gcheck':
        for v in ('full', 'single'):
            print(v, 'Gaussian-restricted fold (phi_c, sigma2_c) =', gaussian_check(G, v))
    elif a.cmd == 'run':
        for phi in a.args:
            s2 = a.s2 if a.s2 else 0.02
            o = solve(phi, a.variant, s2, G, a.damp, verbose=False)
            if o['status'] != 'liquid':
                print(f"{a.variant} phi={phi}: {o['status']} it={o['iterations']} MSD={o['msd']:.4f} f(3,5,7,10)=" +
                      " ".join(f"{o['f'][q]:.3f}" for q in ('k3', 'k5', 'k7', 'k10')) +
                      f"  fG(7)={o['fG_sameMSD']['k7']:.3f} alpha2={o['alpha2']:.3f} tail>1={o['tail_mass_r_gt_1']:.1e} [{o['seconds']:.0f}s]", flush=True)
            else:
                print(f"{a.variant} phi={phi}: liquid (it={o['iterations']})", flush=True)
            json.dump(o, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f"vng_{a.variant}_phi{phi:.3f}_dr{a.dr}.json"), 'w'))
