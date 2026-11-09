#!/usr/bin/env python3
"""Sort-free permutation-symmetric GPU solver for the long-time GMCT hierarchy
(correlator closure, arXiv:2008.10550 Eq. (12)-(14)), Percus-Yevick hard spheres.

Same storage as gmct_sym_gpu.py (colex rank of sorted tuples), but all ranks of inserted/removed
tuples are computed from prefix sums of binomials instead of sorting:

  R sorted of length m, prefix sums  P0[s] = sum_{j<s} C(r_j + j,     j+1)
                                     P1[s] = sum_{j<s} C(r_j + j + 1, j+2)
                                     P2[s] = sum_{j<s} C(r_j + j + 2, j+3)
                                     Pm[s] = sum_{1<=j<s} C(r_j + j - 1, j)
  insert k (s_k = #{r <= k}):   rank(R u k)      = P0[s_k] + C(k + s_k, s_k + 1) + P1[m] - P1[s_k]
  insert q <= p:                rank(R u {q,p})  = P0[s_q] + C(q+s_q, s_q+1) + P1[s_p] - P1[s_q]
                                                   + P2[m] - P2[s_p] + C(p+s_p+1, s_p+2)
  remove position j:            rank(R \\ r_j)    = P0[j] + Pm[m] - Pm[j+1]
Optional fp32 GEMM for the (q,p) contraction (--gemm32); storage and accumulation stay fp64.
"""
from __future__ import annotations
import argparse, json, time
from math import comb
import numpy as np
import torch
from gmct_dense_gpu import build_vertex


class SymHierarchy2:
    def __init__(self, M, N, device, dtype, chunk_elems=400_000_000, gemm32=False):
        self.M, self.N, self.dev, self.dt = M, N, device, dtype
        self.L = N - 1
        self.chunk_elems = chunk_elems
        self.gemm32 = gemm32
        A = M + N + 4
        tab = np.zeros((A, N + 4), dtype=np.int64)
        for a in range(A):
            for j in range(N + 4):
                tab[a, j] = comb(a, j)
        self.binom = torch.tensor(tab, device=device)
        self.counts = {m: comb(M + m - 1, m) for m in range(0, N + 1)}
        self.cb = {j: self.binom[j:j + M + 1, j + 1].contiguous() for j in range(N + 1)}
        self.ar = torch.arange(M, device=device, dtype=torch.int64)
        self.qq, self.pp = torch.meshgrid(self.ar, self.ar, indexing='ij')   # (M, M)
        self.qmin = torch.minimum(self.qq, self.pp).reshape(-1)              # (M^2,)
        self.pmax = torch.maximum(self.qq, self.pp).reshape(-1)

    # ---------------------------------------------------------------- combinatorics
    def unrank(self, r, m):
        out = torch.empty((r.shape[0], m), dtype=torch.int64, device=self.dev)
        rem = r.clone()
        for j in range(m - 1, -1, -1):
            cb = self.cb[j]
            a = torch.searchsorted(cb, rem, right=True) - 1
            out[:, j] = a
            rem = rem - cb[a]
        return out

    def all_tuples(self, m, start, stop):
        return self.unrank(torch.arange(start, stop, device=self.dev, dtype=torch.int64), m)

    def prefix(self, R):
        """Prefix sums P0, P1, P2, Pm of shape (B, m+1) and s[v] = #{r <= v} of shape (B, M)."""
        B, m = R.shape
        j = torch.arange(m, device=self.dev)
        def cum(shift, order_add):
            if m == 0:
                return torch.zeros((B, 1), dtype=torch.int64, device=self.dev)
            vals = self.binom[R + j + shift, j + order_add]                 # (B, m)
            return torch.cat([torch.zeros((B, 1), dtype=torch.int64, device=self.dev), torch.cumsum(vals, 1)], 1)
        P0 = cum(0, 1); P1 = cum(1, 2); P2 = cum(2, 3)
        if m >= 1:
            vals = self.binom[R + j - 1, j]                                  # C(r_j + j - 1, j); j=0 term dropped below
            vals[:, 0] = 0
            Pm = torch.cat([torch.zeros((B, 1), dtype=torch.int64, device=self.dev), torch.cumsum(vals, 1)], 1)
        else:
            Pm = torch.zeros((B, 1), dtype=torch.int64, device=self.dev)
        if m == 0:
            s = torch.zeros((B, self.M), dtype=torch.int64, device=self.dev)
        else:
            s = torch.searchsorted(R, self.ar.view(1, -1).expand(B, -1), right=True)   # (B, M)
        return P0, P1, P2, Pm, s

    def rank_insert1(self, P0, P1, s, m):
        """ranks of R u {k} for all k: (B, M)."""
        k = self.ar.view(1, -1)
        return P0.gather(1, s) + self.binom[k + s, s + 1] + P1[:, m:m + 1] - P1.gather(1, s)

    def rank_insert2(self, P0, P1, P2, s, m):
        """ranks of R u {q,p} for all (q,p): (B, M*M)."""
        v = self.ar.view(1, -1)
        alpha = P0.gather(1, s) - P1.gather(1, s) + self.binom[v + s, s + 1]              # as the smaller element
        beta = P1.gather(1, s) - P2.gather(1, s) + self.binom[v + s + 1, s + 2]           # as the larger element
        return alpha[:, self.qmin] + beta[:, self.pmax] + P2[:, m:m + 1]

    def rank_remove(self, P0, Pm, j, m):
        return P0[:, j] + Pm[:, m] - Pm[:, j + 1]

    # ---------------------------------------------------------------- statics per phi
    def set_phi(self, Wt, Om):
        self.Wt, self.Om = Wt, Om
        WtT = Wt.reshape(self.M, self.M * self.M).T.contiguous()
        self.WtT = WtT.float() if self.gemm32 else WtT
        self.invOmn4 = {}
        for m in range(1, self.L + 1):
            cnt = self.counts[m]
            inv = torch.empty(cnt, device=self.dev, dtype=self.dt)
            step = max(1, self.chunk_elems // (m * 4))
            for s0 in range(0, cnt, step):
                t = self.all_tuples(m, s0, min(cnt, s0 + step))
                inv[s0:s0 + t.shape[0]] = 1.0 / Om[t].sum(dim=1) ** 2
            self.invOmn4[m] = inv

    # ---------------------------------------------------------------- kernels
    def contract(self, fnext, idx):
        F = fnext[idx]                                             # (B, M^2)
        if self.gemm32:
            return (F.float() @ self.WtT).to(self.dt)
        return F @ self.WtT

    def level_memory(self, n, f):
        M = self.M
        acc = torch.zeros(self.counts[n], device=self.dev, dtype=self.dt)
        cntR = self.counts[n - 1]
        step = max(1, self.chunk_elems // (M * M))
        for s0 in range(0, cntR, step):
            R = self.all_tuples(n - 1, s0, min(cntR, s0 + step)); m = n - 1
            P0, P1, P2, Pm, s = self.prefix(R)
            H = self.contract(f[n + 1], self.rank_insert2(P0, P1, P2, s, m))      # (B, M)
            self.scatter(acc, R, P0, P1, s, m, H)
        return acc * self.invOmn4[n]

    def scatter(self, acc, R, P0, P1, s, m, H):
        idx = self.rank_insert1(P0, P1, s, m)                                      # (B, M)
        cnt = (s - torch.searchsorted(R, self.ar.view(1, -1).expand(R.shape[0], -1), right=False)) if m > 0 \
            else torch.zeros_like(idx)                                             # count of k in R
        vals = (cnt.to(self.dt) + 1.0) * self.Om.view(1, -1) * H
        acc.index_add_(0, idx.reshape(-1), vals.reshape(-1))

    def top_memory(self, f):
        M, N, L = self.M, self.N, self.L
        f1, fL = f[1], f[L]
        # Hs over (L-2)-tuples
        cntRp = self.counts[L - 2]
        Hs = torch.empty((cntRp, M), device=self.dev, dtype=self.dt)
        step = max(1, self.chunk_elems // (M * M))
        for s0 in range(0, cntRp, step):
            Rp = self.all_tuples(L - 2, s0, min(cntRp, s0 + step)); m = L - 2
            P0, P1, P2, Pm, s = self.prefix(Rp)
            Hs[s0:s0 + Rp.shape[0]] = self.contract(fL, self.rank_insert2(P0, P1, P2, s, m))
        U = torch.einsum('kqp,q->kp', self.Wt, f1)
        UT = U.T.contiguous()
        acc = torch.zeros(self.counts[L], device=self.dev, dtype=self.dt)
        cntR = self.counts[L - 1]
        step = max(1, self.chunk_elems // (M * 8))
        for s0 in range(0, cntR, step):
            R = self.all_tuples(L - 1, s0, min(cntR, s0 + step)); m = L - 1; B = R.shape[0]
            P0, P1, P2, Pm, s = self.prefix(R)
            F1 = fL[self.rank_insert1(P0, P1, s, m)]                                # (B, M) indexed by p
            H = 2.0 * (F1 @ UT)
            for j in range(m):
                H = H + f1[R[:, j]].unsqueeze(1) * Hs[self.rank_remove(P0, Pm, j, m)]
            H = H / N
            self.scatter(acc, R, P0, P1, s, m, H)
        return acc * self.invOmn4[L]

    def iterate(self, f=None, maxit=100000, tol=1e-11, liquid_floor=1e-7, log_every=0):
        if f is None:
            f = [None] + [torch.ones(self.counts[n], device=self.dev, dtype=self.dt) for n in range(1, self.L + 1)]
        ratios = []; dprev = None
        for it in range(1, maxit + 1):
            fnew = [None] * (self.L + 1)
            for n in range(1, self.L):
                Mn = self.level_memory(n, f); fnew[n] = Mn / (1.0 + Mn)
            Mn = self.top_memory(f); fnew[self.L] = Mn / (1.0 + Mn)
            d = float(torch.max(torch.abs(fnew[1] - f[1])))
            if dprev is not None and dprev > 0: ratios.append(d / dprev)
            dprev = d; f = fnew; fmax = float(f[1].max())
            E = float(np.exp(np.mean(np.log(ratios[-20:])))) if len(ratios) >= 5 else float('nan')
            if log_every and it % log_every == 0:
                print(f"    it={it:6d} f1max={fmax:.6e} dmax={d:.3e} E~{E:.5f}", flush=True)
            if fmax < liquid_floor:
                return dict(f=f, status='liquid', iterations=it, dmax=d, f1max=fmax, E=E)
            if d < tol:
                return dict(f=f, status='glass' if fmax > 1e-3 else 'liquid', iterations=it, dmax=d, f1max=fmax, E=E)
        return dict(f=f, status='unresolved', iterations=maxit, dmax=d, f1max=fmax, E=E)


def solve_sym(H, M, qmax, phi, warm=None, **kw):
    Wt, Om, k, S, c = build_vertex(M, qmax, phi, H.dev, H.dt)
    H.set_phi(Wt, Om)
    t0 = time.time(); res = H.iterate(f=warm, **kw); res['seconds'] = time.time() - t0; res['phi'] = phi
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100); ap.add_argument('--qmax', type=float, default=40.0)
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--lo', type=float, required=True); ap.add_argument('--hi', type=float, required=True)
    ap.add_argument('--tol', type=float, default=1e-4); ap.add_argument('--dtype', default='float64')
    ap.add_argument('--maxit', type=int, default=100000); ap.add_argument('--log_every', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=400_000_000); ap.add_argument('--gemm32', action='store_true')
    ap.add_argument('--out', default='runs_sym.jsonl'); ap.add_argument('--warm', action='store_true')
    ap.add_argument('--conv_tol', type=float, default=1e-11)
    ap.add_argument('--mode', choices=['bisect', 'extrap'], default='bisect',
                    help='extrap: glass-side points hi, hi-d, hi-2d ... with (1-E)^2 linear extrapolation')
    ap.add_argument('--npts', type=int, default=3)
    a = ap.parse_args()
    dev = torch.device('cuda'); dt = getattr(torch, a.dtype)
    H = SymHierarchy2(a.M, a.N, dev, dt, chunk_elems=a.chunk, gemm32=a.gemm32)
    print(f"sym2 solver M={a.M} N={a.N} stored levels 1..{a.N-1}, top count={H.counts[a.N-1]:,} gemm32={a.gemm32}", flush=True)
    log = open(a.out, 'a')
    def record(phi, r, tag=''):
        print(f"  phi={phi:.7f} {r['status']:10s} it={r['iterations']:6d} f1max={r['f1max']:.4e} E={r['E']:.6f} t={r['seconds']:.1f}s {tag}", flush=True)
        log.write(json.dumps(dict(M=a.M, qmax=a.qmax, N=a.N, closure='corr', solver='sym2', phi=phi, status=r['status'],
                                  iterations=r['iterations'], seconds=r['seconds'], f1max=r['f1max'], E=r['E'])) + "\n"); log.flush()
    lo, hi = a.lo, a.hi; warm = None; glass_pts = []
    if a.mode == 'bisect':
        while hi - lo > a.tol:
            mid = 0.5 * (lo + hi)
            r = solve_sym(H, a.M, a.qmax, mid, warm=warm, maxit=a.maxit, tol=a.conv_tol, log_every=a.log_every)
            record(mid, r)
            if r['status'] == 'glass':
                hi = mid; glass_pts.append((mid, r['E']))
                if a.warm: warm = r['f']
            elif r['status'] == 'liquid':
                lo = mid
            else:
                print("  unresolved (maxit reached): bracket NOT updated; stopping", flush=True); break
    else:
        # extrapolation mode: shrink from hi towards lo on the glass side
        phi = hi; d = (hi - lo) / 4
        while len(glass_pts) < a.npts:
            r = solve_sym(H, a.M, a.qmax, phi, warm=warm, maxit=a.maxit, tol=a.conv_tol, log_every=a.log_every)
            record(phi, r, tag='[extrap scan]')
            if r['status'] == 'glass':
                glass_pts.append((phi, r['E'])); hi = min(hi, phi)
                if a.warm: warm = r['f']
                phi = phi - d
                if phi <= lo: d /= 2; phi = hi - d
            elif r['status'] == 'liquid':
                lo = max(lo, phi); d /= 2; phi = hi - d
            else:
                print("  unresolved (maxit reached): point discarded, continuing with larger maxit", flush=True)
                a.maxit *= 2; d /= 2; phi = hi - d
    est = None
    if len(glass_pts) >= 2:
        pts = sorted(glass_pts)[:3]
        x = np.array([p for p, E in pts]); y = np.array([(1 - E) ** 2 for p, E in pts])
        A = np.vstack([x, np.ones_like(x)]).T; coef = np.linalg.lstsq(A, y, rcond=None)[0]
        est = -coef[1] / coef[0]
        print(f"  (1-E)^2 extrapolation from {len(pts)} glass points {pts}: phi_c ~ {est:.6f}", flush=True)
    log.write(json.dumps(dict(M=a.M, qmax=a.qmax, N=a.N, closure='corr', solver='sym2', result=True, lo=lo, hi=hi,
                              extrap=est, glass_pts=glass_pts)) + "\n"); log.close()
    print(f"RESULT sym2 M={a.M} N={a.N}: bracket ({lo:.7f}, {hi:.7f})" + (f"  extrap {est:.6f}" if est else ""), flush=True)
