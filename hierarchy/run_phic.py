#!/usr/bin/env python3
"""Bisection campaign phi_c(N) for the MK hierarchy (variants Bself/Bcoll), seeding each level with the
previous level's liquid bound (as campaign.py).  Dense solver (fp64) or sym2 solver (corr only).

Memory-closure labelling: the dense code's 'mem' branch with code order N evolves f_1..f_N and closes m_N;
in the schematic limit this is the bilinear closure at PUBLISHED label N+1 (SM Sec. S1 and Table S1).
With --mem_general_N2 the general branch is also used at code N=2 (= published label 3) instead of plain MCT.
Records in the JSONL carry both 'N' (code order) and 'label' (published label used for Lambda_c)."""
from __future__ import annotations
import argparse, json, time, sys, math, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # mk_vertex.py, gmct_*.py live here
import mk_vertex as V
from gmct_sym2_gpu import SymHierarchy2

ap = argparse.ArgumentParser()
ap.add_argument('--variant', required=True); ap.add_argument('--closure', default='corr', choices=['corr', 'mem'])
ap.add_argument('--Ns', required=True); ap.add_argument('--M', type=int, default=100); ap.add_argument('--qmax', type=float, default=40.0)
ap.add_argument('--solver', default='dense', choices=['dense', 'sym2'])
ap.add_argument('--lo', type=float, required=True); ap.add_argument('--hi', type=float, default=None)
ap.add_argument('--hi_factor', type=float, default=1.3)
ap.add_argument('--tol', type=float, default=2e-4); ap.add_argument('--maxit', type=int, default=100000)
ap.add_argument('--conv_tol', type=float, default=1e-11); ap.add_argument('--liquid_floor', type=float, default=1e-7)
ap.add_argument('--dtype', default='float64'); ap.add_argument('--gemm32', action='store_true')
ap.add_argument('--chunk', type=int, default=200_000_000); ap.add_argument('--warm', action='store_true')
ap.add_argument('--kcut', type=int, default=0); ap.add_argument('--mem_general_N2', action='store_true')
ap.add_argument('--out', required=True); ap.add_argument('--log_every', type=int, default=0)
a = ap.parse_args()
if a.solver == 'sym2' and a.closure != 'corr':
    ap.error('--solver sym2 only supports --closure corr')
dev = torch.device('cuda'); dt = getattr(torch, a.dtype)
log = open(a.out, 'a')
T0 = time.time()

def label_of(N):
    if a.closure == 'corr': return N
    return N + 1 if (N >= 3 or a.mem_general_N2) else N

def emit(rec):
    log.write(json.dumps(rec) + '\n'); log.flush()

state = dict(warm=None)

def solve(N, phi, H=None):
    torch.cuda.reset_peak_memory_stats()
    if a.solver == 'dense':
        Wt, Om, k = V.build_vertex_torch(a.M, a.qmax, phi, a.variant, dev, dt, a.kcut)
        t0 = time.time()
        if a.closure == 'mem' and N == 2 and a.mem_general_N2:
            r = V.iterate_dense(Wt, Om, 2, 'mem_general', maxit=a.maxit, tol=a.conv_tol, liquid_floor=a.liquid_floor, log_every=a.log_every)
        else:
            r = V.iterate_dense(Wt, Om, N, a.closure, maxit=a.maxit, tol=a.conv_tol, liquid_floor=a.liquid_floor, log_every=a.log_every)
        r['seconds'] = time.time() - t0; r.pop('f')
    else:
        Wt, Om, k = V.build_vertex_torch(a.M, a.qmax, phi, a.variant, dev, dt, a.kcut)
        H.set_phi(Wt, Om)
        t0 = time.time()
        r = H.iterate(f=state['warm'], maxit=a.maxit, tol=a.conv_tol, liquid_floor=a.liquid_floor, log_every=a.log_every)
        r['seconds'] = time.time() - t0
        fstate = r.pop('f')
        if a.warm and r['status'] == 'glass': state['warm'] = fstate
        del fstate
    rec = dict(type='solve', variant=a.variant, closure=a.closure, N=N, label=label_of(N), M=a.M, qmax=a.qmax, kcut=a.kcut,
               solver=a.solver, dtype=a.dtype, phi=phi, status=r['status'], iterations=r['iterations'], seconds=r['seconds'],
               f1max=r['f1max'], dmax=r['dmax'], E=r['E'], gpu_peak_GB=torch.cuda.max_memory_allocated() / 1e9,
               warm=bool(a.warm and a.solver == 'sym2'))
    print(f"  N={N} phi={phi:.7f} {r['status']:10s} it={r['iterations']:6d} f1max={r['f1max']:.4e} E={r['E']:.5f} t={r['seconds']:.1f}s gpu={rec['gpu_peak_GB']:.2f}GB", flush=True)
    emit(rec)
    return r['status']

lo, hi = a.lo, a.hi
for N in [int(x) for x in a.Ns.split(',')]:
    tN = time.time(); nsolve = 0
    H = None
    if a.solver == 'sym2':
        H = SymHierarchy2(a.M - a.kcut, N, dev, dt, chunk_elems=a.chunk, gemm32=a.gemm32)
        print(f"sym2 N={N}: top count {H.counts[N-1]:,}", flush=True)
    state['warm'] = None
    print(f"=== {a.variant} {a.closure} N={N} (label {label_of(N)}) M={a.M} qmax={a.qmax} solver={a.solver}: start lo={lo} hi={hi} ===", flush=True)
    # verify lo is liquid
    s = solve(N, lo, H); nsolve += 1
    if s != 'liquid':
        print(f"  !! lo={lo} is {s}, shrinking lo", flush=True)
        while s != 'liquid':
            lo = lo / a.hi_factor; s = solve(N, lo, H); nsolve += 1
    # find glass upper bound
    if hi is None: hi = lo * a.hi_factor
    s = solve(N, hi, H); nsolve += 1
    while s == 'liquid':
        lo = hi; hi = hi * a.hi_factor; s = solve(N, hi, H); nsolve += 1
    if s != 'glass':
        emit(dict(type='result', variant=a.variant, closure=a.closure, N=N, label=label_of(N), M=a.M, qmax=a.qmax, solver=a.solver,
                  result=True, lo=lo, hi=hi, phi_c=None, note=f'hi={hi} unresolved', seconds=time.time() - tN)); break
    hist = []
    while hi - lo > a.tol:
        mid = 0.5 * (lo + hi)
        s = solve(N, mid, H); nsolve += 1
        if s == 'glass': hi = mid
        elif s == 'liquid': lo = mid
        else:
            print("  unresolved (maxit reached): bracket NOT updated; stopping bisection", flush=True)
            hist.append(dict(phi=mid, unresolved=True)); break
    rec = dict(type='result', variant=a.variant, closure=a.closure, N=N, label=label_of(N), M=a.M, qmax=a.qmax, kcut=a.kcut,
               solver=a.solver, dtype=a.dtype, result=True, lo=lo, hi=hi, phi_c=0.5 * (lo + hi), half_width=0.5 * (hi - lo),
               n_solves=nsolve, seconds=time.time() - tN, unresolved=bool(hist))
    emit(rec)
    print(f"RESULT {a.variant} {a.closure} N={N} (label {label_of(N)}) M={a.M} qmax={a.qmax}: phi_c in ({lo:.6f}, {hi:.6f})  [{nsolve} solves, {time.time()-tN:.0f}s]", flush=True)
    if H is not None:
        del H; torch.cuda.empty_cache()
    # seed next level: liquid bound = this level's lo; hi re-probed from lo*hi_factor
    hi = None
print(f"DONE total {time.time()-T0:.0f}s", flush=True)
