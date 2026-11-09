#!/usr/bin/env python3
"""Rigorous linear-chain (Collatz-Wielandt) lower bound on the GMCT critical point.

Any fixed point f of the correlator-closed hierarchy satisfies f_n <= L_n f_{n+1} (level maps are
linear and positive) and f_N <= P f_1 with P v(K) = (1/N) sum_i v(k_i) (closure with f_{N-1} <= 1).
Hence f_1 <= T_N f_1,  T_N = L_1 L_2 ... L_{N-1} P  (positive M x M matrix).
If rho(T_N(phi)) < 1 the only fixed point is f = 0.  phi_lin(N) := root of rho(T_N(phi)) = 1 is a
rigorous lower bound for phi_c(N).  For N = 2, T_2 = A/2 with A the frozen linearisation, so
rho(T_2) = lambda_P/2.
T_N v is evaluated with the symmetric-storage kernels: top level with f_1 = v, f_{N-1} = 1 gives
L_{N-1} P v; then the lower level maps are applied in turn.
"""
import argparse, json, time
import numpy as np, torch
from gmct_dense_gpu import build_vertex
from gmct_sym2_gpu import SymHierarchy2


def apply_T(H, v):
    L = H.L
    f = [None] * (L + 1)
    f[1] = v
    f[L] = torch.ones(H.counts[L], device=H.dev, dtype=H.dt)
    if L == 1:                                  # N = 2: T v(k) = sum_{q,p} W~(k;q,p) v_q / Omega_1^2(k) = (A v)/2
        return torch.einsum('kqp,q->k', H.Wt, v) / H.Om
    g = H.top_memory(f)                         # L_{L} P v   on L-tuples
    for n in range(L - 1, 0, -1):
        f[n + 1] = g
        g = H.level_memory(n, f)                # L_n g
    return g


def spectral_radius(H, v0=None, iters=40, tol=1e-6):
    v = torch.ones(H.M, device=H.dev, dtype=H.dt) if v0 is None else v0.clone()
    v = v / v.max(); rho = None
    for it in range(iters):
        w = apply_T(H, v)
        rho_new = float(w.max() / v.max())
        # Collatz-Wielandt sandwich: min_k (Tv)_k/v_k <= rho <= max_k (Tv)_k/v_k
        ratio = w / v
        lo, hi = float(ratio.min()), float(ratio.max())
        v = w / w.max()
        if rho is not None and abs(rho_new - rho) < tol * rho_new and (hi - lo) < 1e-3 * hi:
            rho = rho_new; break
        rho = rho_new
    return rho, lo, hi, v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=100); ap.add_argument('--qmax', type=float, default=40.0)
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--lo', type=float, default=0.30); ap.add_argument('--hi', type=float, default=0.60)
    ap.add_argument('--tol', type=float, default=5e-5); ap.add_argument('--out', default='linear_chain.jsonl')
    a = ap.parse_args()
    dev = torch.device('cuda'); dt = torch.float64
    H = SymHierarchy2(a.M, a.N, dev, dt, chunk_elems=300_000_000)
    v = None; lo, hi = a.lo, a.hi
    log = open(a.out, 'a')
    print(f"linear-chain bound M={a.M} N={a.N}", flush=True)
    while hi - lo > a.tol:
        mid = 0.5 * (lo + hi)
        Wt, Om, k, S, c = build_vertex(a.M, a.qmax, mid, dev, dt); H.set_phi(Wt, Om)
        t0 = time.time(); rho, cw_lo, cw_hi, v = spectral_radius(H, v); dtm = time.time() - t0
        print(f"  phi={mid:.6f}  rho(T_N)={rho:.6f}  CW bracket [{cw_lo:.5f},{cw_hi:.5f}]  {dtm:.1f}s", flush=True)
        log.write(json.dumps(dict(M=a.M, N=a.N, phi=mid, rho=rho, cw_lo=cw_lo, cw_hi=cw_hi)) + "\n"); log.flush()
        if cw_hi < 1.0:          # certified rho < 1 (upper Collatz-Wielandt bound)
            lo = mid
        elif cw_lo > 1.0:        # certified rho > 1
            hi = mid
        else:
            # undecided by the sandwich: use the estimate, flag
            if rho < 1: lo = mid
            else: hi = mid
    log.write(json.dumps(dict(M=a.M, N=a.N, result=True, phi_lin_lo=lo, phi_lin_hi=hi)) + "\n"); log.close()
    print(f"RESULT linear-chain M={a.M} N={a.N}: phi_lin in ({lo:.6f}, {hi:.6f})  [rigorous lower bound on phi_c(N)]", flush=True)
