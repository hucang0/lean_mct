#!/usr/bin/env python3
"""Quick scan of max_A Lambda(A;phi) for all variants (PY, VW) to bracket the folds; also checks
numerical h^(k) (node-averaged DST, the route used for VW) against the analytic PY h^(k) in the MZ closure."""
import math, time, json
import numpy as np
from gauss_py import Grid, Closures
import py_structure as PS
G = Grid()
def build(phi, kind):
    if kind == 'PY':
        st = PS.py_structure(phi, PS.Grid(G.N, G.dr))
        return Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'])
    if kind == 'PYnum':   # PY but with h^ from the numerical transform of g_node - 1 (VW route)
        st = PS.py_structure(phi, PS.Grid(G.N, G.dr))
        hk = PS.Grid(G.N, G.dr).k_from_r(st['g_node'] - 1)
        return Closures(G, st['rho'], st['y'], hk, st['ck'], 1 + st['rho'] * hk, y1=st['g1'])
    if kind.startswith('VW'):
        v = PS.vw_structure(phi, PS.Grid(G.N, G.dr), interior='continue' if kind == 'VW' else 'scale')
        return Closures(G, v['rho'], v['y'], v['hk'], v['hk'] / v['Sk'], v['Sk'], y1=v['g1'])
t0 = time.time()
for phi in (0.50, 0.52):
    a = build(phi, 'PY'); b = build(phi, 'PYnum')
    for A in (0.002, 0.005, 0.01):
        print(f"phi={phi} A={A}: K_MZ analytic h^ = {a.K('MZ', A):.6f}  numerical h^ = {b.K('MZ', A):.6f}")
print(f"[{time.time()-t0:.1f}s]")
for kind in ('PY', 'VW'):
    for phi in (0.40, 0.45, 0.50, 0.55, 0.60):
        C = build(phi, kind)
        row = []
        for v in ('a', 'a0', 'amix', 'MZ', 'MZ0', 'b0', 'GMCTV'):
            L, Am, _ = C.maxLam(v, n=24)
            row.append(f"{v}:{L:.3f}@A={Am:.4f}")
        print(f"{kind} phi={phi}: " + "  ".join(row), flush=True)
print(f"[{time.time()-t0:.1f}s]")
