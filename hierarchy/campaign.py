#!/usr/bin/env python3
"""Run a sequence of closure levels N on one grid; the glass bound of level N seeds level N+1."""
import argparse, json, torch
from gmct_dense_gpu import bisect
ap = argparse.ArgumentParser()
ap.add_argument('--M', type=int, required=True); ap.add_argument('--qmax', type=float, default=40.0)
ap.add_argument('--closure', default='corr'); ap.add_argument('--Ns', type=str, required=True)
ap.add_argument('--lo', type=float, required=True); ap.add_argument('--hi', type=float, required=True)
ap.add_argument('--tol', type=float, default=2e-6); ap.add_argument('--dtype', default='float64')
ap.add_argument('--maxit', type=int, default=200000); ap.add_argument('--out', required=True)
a = ap.parse_args()
dev = torch.device('cuda'); dt = getattr(torch, a.dtype)
lo, hi = a.lo, a.hi
with open(a.out, 'a') as log:
    for N in [int(x) for x in a.Ns.split(',')]:
        print(f"=== M={a.M} closure={a.closure} N={N}  bracket ({lo},{hi}) ===", flush=True)
        l, h, hist = bisect(a.M, a.qmax, N, a.closure, lo, hi, a.tol, dev, dt, log, maxit=a.maxit)
        print(f"RESULT M={a.M} N={N} closure={a.closure}: phi_c in ({l:.7f}, {h:.7f})", flush=True)
        log.write(json.dumps(dict(M=a.M, qmax=a.qmax, N=N, closure=a.closure, result=True, lo=l, hi=h)) + "\n"); log.flush()
        lo, hi = l, a.hi          # phi_c increases with N: next level's liquid bound = this level's
