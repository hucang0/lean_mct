#!/usr/bin/env python3
"""Dense GPU solver for the long-time (plateau) GMCT hierarchy of Percus-Yevick hard spheres.

Two mean-field closures:
  corr : arXiv:2008.10550 Eq. (14)  f_N = (1/N) sum_i f_1(k_i) f_{N-1}(K\\k_i);  levels 1..N-1 dynamical
  mem  : arXiv:1909.00428 Eq. (8)   m_N = [1/(N-1)][1/Om_N^2] sum_i Om_{N-1}^2 m_{N-1} f_1;  levels 1..N dynamical

Grid: M equidistant wavenumbers k = h (j + 1/2), j = 0..M-1, h = qmax/M (qmax = 40 reproduces the
Luo-Janssen grid at M = 100).  Vertex V(k;q,p) as in arXiv:2008.10550 Eq. (11) (n = 1 form, with
Omega_1^2(k) divided out; the level-n prefactor Omega_1^2(k_i)/Omega_n^4 is applied separately).

Correlators are stored as dense symmetric tensors f_n of shape (M,)*n.  Memory limits:
M=100: N_dyn <= 4 (1e8 entries) in fp64, N_dyn = 5 (1e10) only in fp32 on a 96 GB GPU.
"""
from __future__ import annotations
import argparse, json, math, time, sys
import numpy as np
import torch
from numpy.polynomial.legendre import leggauss


# ----------------------------------------------------------------------------- statics
def py_ck(k, phi):
    l1 = (1 + 2 * phi) ** 2 / (1 - phi) ** 4
    l2 = -(1 + phi / 2) ** 2 / (1 - phi) ** 4
    xg, wg = leggauss(400)
    r = 0.5 * (xg + 1.0); w = 0.5 * wg
    cr = -l1 - 6 * phi * l2 * r - 0.5 * phi * l1 * r ** 3
    return np.array([4 * np.pi / kv * np.sum(w * r * cr * np.sin(kv * r)) for kv in k])


def build_vertex(M, qmax, phi, device, dtype):
    h = qmax / M
    khat = np.arange(M) + 0.5
    k = h * khat
    rho = 6 * phi / np.pi
    c = py_ck(k, phi)
    S = 1.0 / (1.0 - rho * c)
    Om = k ** 2 / S                                   # D0 = 1
    K = khat[:, None, None]; Q = khat[None, :, None]; P = khat[None, None, :]
    mask = (P >= np.abs(K - Q) + 0.5 - 1e-9) & (P <= K + Q - 0.5 + 1e-9)
    brack = (Q ** 2 - P ** 2 + K ** 2) * c[None, :, None] + (P ** 2 - Q ** 2 + K ** 2) * c[None, None, :]
    pref = rho * h ** 5 / (32 * np.pi ** 2)
    Wt = pref * (Q * P / K ** 3) * brack ** 2 * S[None, :, None] * S[None, None, :]   # W~(k;q,p), no 1/Om
    Wt = np.where(mask, Wt, 0.0)
    return (torch.tensor(Wt, device=device, dtype=dtype), torch.tensor(Om, device=device, dtype=dtype),
            k, S, c)


# ----------------------------------------------------------------------------- hierarchy
def omega_n(Om, n):
    """Omega_n^2(K) = sum_i Om[k_i] as a dense (M,)*n tensor."""
    M = Om.shape[0]
    out = torch.zeros((M,) * n, device=Om.device, dtype=Om.dtype)
    for i in range(n):
        shape = [1] * n; shape[i] = M
        out = out + Om.reshape(shape)
    return out


def contract_qp(Wt, g):
    """H[k, R] = sum_{q,p} W~(k;q,p) g[q,p,R]  for g of shape (M,M,*R)."""
    M = Wt.shape[0]
    gR = g.reshape(M * M, -1)
    return (Wt.reshape(M, M * M) @ gR).reshape((M,) + g.shape[2:])


def assemble_level(Om, Omn, H, n):
    """M_n[K] = sum_i Om[k_i] * H[k_i, K\\i] / Omn[K]^2,  H of shape (M,)*n with axis 0 = active leg."""
    M = Om.shape[0]
    acc = torch.zeros_like(Omn)
    for i in range(n):
        shape = [1] * n; shape[i] = M
        acc += Om.reshape(shape) * torch.movedim(H, 0, i)
    return acc / Omn ** 2


def top_level_corr(Wt, f1, fNm1, N):
    """H_{N-1}[k, R] for the correlator closure, R an (N-2)-tuple:
       f_N(q,p,R) = (1/N)[ f1(q) f_{N-1}(p,R) + f1(p) f_{N-1}(q,R) + sum_j f1(r_j) f_{N-1}(q,p,R\\r_j) ]."""
    M = Wt.shape[0]
    U = torch.einsum('kqp,q->kp', Wt, f1)                      # U[k,p] = sum_q W~ f1(q)
    T1 = (U @ fNm1.reshape(M, -1)).reshape((M,) + fNm1.shape[1:])   # sum_p U[k,p] f_{N-1}[p,R]
    H = 2.0 * T1                                              # W~ symmetric in q<->p
    nR = N - 2
    if nR >= 1:
        Hs = contract_qp(Wt, fNm1)                             # Hs[k, R'] with R' an (N-3)-tuple
        for j in range(nR):
            # T3 term: f1(r_j) * Hs[k, R\\r_j]  -> insert axis j+1 for r_j
            shape = [1] * (nR + 1); shape[j + 1] = M
            H = H + f1.reshape(shape) * Hs.unsqueeze(j + 1)
    return H / N


def iterate(Wt, Om, N, closure, f=None, maxit=100000, tol=1e-11, liquid_floor=1e-7, log_every=0):
    """Fixed-point iteration from f ≡ 1 (or warm start).  Returns dict with f, status."""
    M = Om.shape[0]
    ndyn = N - 1 if closure == 'corr' else N
    Omn = [None] + [omega_n(Om, n) for n in range(1, ndyn + 1)]
    if f is None:
        f = [None] + [torch.ones((M,) * n, device=Om.device, dtype=Om.dtype) for n in range(1, ndyn + 1)]
    for it in range(1, maxit + 1):
        fnew = [None] * (ndyn + 1)
        Mlev = [None] * (ndyn + 1)
        # levels below the top: need f_{n+1}
        for n in range(1, ndyn):
            H = contract_qp(Wt, f[n + 1]) if n + 1 <= ndyn else None
            Mlev[n] = assemble_level(Om, Omn[n], H, n)
            fnew[n] = Mlev[n] / (1.0 + Mlev[n])
        if closure == 'corr':
            n = ndyn                                           # n = N-1
            H = top_level_corr(Wt, f[1], f[N - 1], N)
            Mlev[n] = assemble_level(Om, Omn[n], H, n)
            fnew[n] = Mlev[n] / (1.0 + Mlev[n])
        else:
            # memory closure at level N: X[K] = 1/(N-1) sum_i (Om_{N-1}(K\\i)/Om_N(K))^2 M_{N-1}[K\\i] f1(k_i)
            n = ndyn                                           # n = N
            if N == 2:
                # plain MCT for N = 2 (the published series uses standard MCT at N=2)
                H = top_level_corr(Wt, f[1], f[1], 2)
                Mlev[1] = assemble_level(Om, Omn[1], H, 1)
                fnew[1] = Mlev[1] / (1.0 + Mlev[1])
                fnew[2] = fnew[1][:, None] * fnew[1][None, :]
            else:
                X = torch.zeros_like(Omn[N])
                for i in range(N):
                    ratio = torch.movedim(Omn[N - 1].unsqueeze(0).expand((M,) + (M,) * (N - 1)), 0, i) / Omn[N]
                    shape = [1] * N; shape[i] = M
                    X += ratio ** 2 * torch.movedim(Mlev[N - 1].unsqueeze(0).expand((M,) + (M,) * (N - 1)), 0, i) \
                         * f[1].reshape(shape)
                X /= (N - 1)
                fnew[N] = X / (1.0 + X)
        # convergence bookkeeping on level 1
        d = float(torch.max(torch.abs(fnew[1] - f[1])))
        f = fnew
        fmax = float(f[1].max())
        if log_every and it % log_every == 0:
            print(f"    it={it:6d}  f1max={fmax:.6e}  dmax={d:.3e}", flush=True)
        if fmax < liquid_floor:
            return dict(f=f, status='liquid', iterations=it, dmax=d, f1max=fmax)
        if d < tol:
            return dict(f=f, status='glass' if fmax > 1e-3 else 'liquid', iterations=it, dmax=d, f1max=fmax)
    return dict(f=f, status='unresolved', iterations=maxit, dmax=d, f1max=fmax)


def solve(M, qmax, N, closure, phi, device, dtype, **kw):
    Wt, Om, k, S, c = build_vertex(M, qmax, phi, device, dtype)
    t0 = time.time()
    res = iterate(Wt, Om, N, closure, **kw)
    res['seconds'] = time.time() - t0
    res['phi'] = phi; res['M'] = M; res['N'] = N; res['closure'] = closure
    res['f1'] = res['f'][1].detach().cpu().numpy().tolist()
    res['k'] = k.tolist()
    del res['f']
    return res


def bisect(M, qmax, N, closure, lo, hi, tol, device, dtype, log, **kw):
    """Bisection on the glass/liquid status.  lo must be liquid, hi glass."""
    hist = []
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        r = solve(M, qmax, N, closure, mid, device, dtype, **kw)
        hist.append(dict(phi=mid, status=r['status'], iterations=r['iterations'], seconds=r['seconds'],
                         f1max=r['f1max']))
        print(f"  phi={mid:.7f}  {r['status']:10s} it={r['iterations']:6d} f1max={r['f1max']:.4e} "
              f"t={r['seconds']:.1f}s", flush=True)
        log.write(json.dumps(dict(M=M, qmax=qmax, N=N, closure=closure, **hist[-1])) + "\n"); log.flush()
        if r['status'] == 'glass':
            hi = mid
        elif r['status'] == 'liquid':
            lo = mid
        else:
            print("  unresolved (maxit reached): bracket NOT updated; stopping bisection", flush=True)
            hist[-1]['unresolved_stop'] = True
            break
    return lo, hi, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100)
    ap.add_argument('--qmax', type=float, default=40.0)
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--closure', choices=['corr', 'mem'], default='corr')
    ap.add_argument('--lo', type=float, required=True)
    ap.add_argument('--hi', type=float, required=True)
    ap.add_argument('--tol', type=float, default=2e-6)
    ap.add_argument('--dtype', choices=['float64', 'float32'], default='float64')
    ap.add_argument('--maxit', type=int, default=100000)
    ap.add_argument('--out', type=str, default='runs.jsonl')
    a = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dt = getattr(torch, a.dtype)
    print(f"device={dev} dtype={a.dtype} M={a.M} qmax={a.qmax} N={a.N} closure={a.closure}", flush=True)
    with open(a.out, 'a') as log:
        lo, hi, hist = bisect(a.M, a.qmax, a.N, a.closure, a.lo, a.hi, a.tol, dev, dt, log, maxit=a.maxit)
    print(f"RESULT M={a.M} N={a.N} closure={a.closure}: phi_c in ({lo:.7f}, {hi:.7f})", flush=True)
    with open(a.out, 'a') as log:
        log.write(json.dumps(dict(M=a.M, qmax=a.qmax, N=a.N, closure=a.closure, result=True, lo=lo, hi=hi)) + "\n")
