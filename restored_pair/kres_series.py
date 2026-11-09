#!/usr/bin/env python3
"""k-resolved exclusion-order series (SM Table S16). The k-resolved restored-pair kernel (brc_kres.py, admissible cage,
M = 100, q_max = 40) with the resummed weight 1/Z(c) replaced by its truncation at exclusion-series order n,
w_n = sum_{m<n} (1 - Z)^m.  n = 1 is exactly the diagonal N = 2 kernel (Parseval identity, SM Sec. S5 C);
n = inf is the k-resolved BRC (phi_c = 1.784).  Fold located by bisection from a tight Gaussian start.
Usage: python3 kres_series.py n lo hi nsteps   -> results/kres_series_n<n>.json"""
import sys, json, math, time, os
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import brc_kres as bk
n, lo, hi, nst = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
K = bk.BRCKernel()
orig = K.evaluate_adm
def evaluate_adm_n(f, rho):
    adm = K.admissible(f); rq = adm['rq']
    kk = np.concatenate([-K.q[:0:-1], K.q]); rr = np.concatenate([rq[:0:-1], rq])
    sr = bk.CubicSpline(kk, rr)
    R = lambda p: np.clip(np.where(np.abs(p) <= K.qmax, sr(np.minimum(np.abs(p), K.qmax)), 0.0), 0.0, 1.0)
    x = 1.0 - adm['Z']; W = sum(x ** m for m in range(n))
    Mb = np.empty(K.M); Md = np.empty(K.M); wc = K.dc * K.c ** 2
    for j, k in enumerate(K.k):
        p = np.sqrt(np.maximum(k * k + K.q[:, None] ** 2 - 2 * k * K.q[:, None] * K.x[None, :], 0.0))
        Ahat = (K.q * K.ft_q * rq)[:, None] * K.x[None, :] * R(p)
        Al_c = np.einsum('lcq,lq->lc', K.T, K.Pl @ Ahat.T)
        S = 4 * math.pi * np.einsum('lc,l->c', Al_c ** 2, K.inv2l1)
        Mb[j] = rho / k ** 2 * np.sum(wc * S * W); Md[j] = rho / k ** 2 * np.sum(wc * S)
    return Mb, Md, adm
K.evaluate_adm = evaluate_adm_n
res = []
def run(phi):
    rho = 6 * phi / math.pi
    r = K.iterate_adm(bk.gaussian_f(K.k, 0.008), rho, maxit=700, tol=1e-7, log_every=0)
    f = r['f']; st = r['status']
    rec = dict(phi=phi, status=st, iterations=r['iterations'], f7=float(np.interp(7, K.k, f)), f3=float(np.interp(3, K.k, f)),
               msd=float(-6 * math.log(max(f[0], 1e-300)) / K.k[0] ** 2), seconds=r['seconds'])
    res.append(rec)
    print(f"n={n} phi={phi:.4f}: {st} it={r['iterations']} f(3)={rec['f3']:.3f} f(7)={rec['f7']:.4f} [{r['seconds']:.0f}s]", flush=True)
    return st
for _ in range(nst):
    mid = 0.5 * (lo + hi)
    st = run(mid)
    if st == 'liquid': lo = mid
    elif st == 'glass': hi = mid        # keep resolved endpoints only; unresolved interior runs do not shrink bracket
print(f"n={n}: k-resolved fold in [{lo:.4f}, {hi:.4f}]", flush=True)
json.dump(dict(n=n, fold=[lo, hi], runs=res), open(os.path.join(HERE, 'results', f'kres_series_n{n}.json'), 'w'), indent=1)
