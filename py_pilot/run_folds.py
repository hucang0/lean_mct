#!/usr/bin/env python3
"""phi_c (fold) and glass branch MSD(phi) for all Gaussian closures, PY and Verlet-Weis structure.
Writes results/folds.json (make_summary.py condenses it into results/summary_table.json)."""
import math, time, json, sys
import os
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import numpy as np
from scipy.optimize import brentq
from gauss_py import Grid, Closures
import py_structure as PS

G = Grid()
PG = PS.Grid(G.N, G.dr)


def build(phi, kind):
    if kind == 'PY':
        st = PS.py_structure(phi, PG)
        return Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'])
    if kind.startswith('VW'):
        interior = {'VW': 'scale', 'VWscale': 'scale', 'VWshift': 'shift'}[kind]
        v = PS.vw_structure(phi, PG, interior=interior)
        return Closures(G, v['rho'], v['y'], v['hk'], v['hk'] / v['Sk'], v['Sk'], y1=v['g1'])


def F(phi, kind, var):
    C = build(phi, kind)
    L, Am, (As, Ls) = C.maxLam(var, lo=1e-4, hi=0.3, n=28)
    # count interior local maxima of the coarse profile (should be one hump)
    nmax = int(np.sum((Ls[1:-1] > Ls[:-2]) & (Ls[1:-1] > Ls[2:])))
    return L - 1.0, Am, nmax, C


def fold(kind, var, lo=0.12, hi=0.70):
    flo = F(lo, kind, var)[0]; fhi = F(hi, kind, var)[0]
    if not (flo < 0 < fhi):
        return dict(error=f"no bracket: f({lo})={flo:.3g}, f({hi})={fhi:.3g}")
    phic = brentq(lambda p: F(p, kind, var)[0], lo, hi, xtol=2e-6)
    f, Am, nmax, C = F(phic, kind, var)
    return dict(phi_c=phic, A_c=Am, MSD_c=6 * Am, fs7_c=math.exp(-49 * Am), n_local_max=nmax)


def branch(kind, var, phic, phis):
    rows = []
    for phi in phis:
        if phi < phic:
            continue
        C = build(phi, kind)
        L, Am, _ = C.maxLam(var, lo=1e-4, hi=0.3, n=28)
        A = C.glass_A(var, Am, lo=1e-6)
        if A is None:
            rows.append(dict(phi=phi, A=None)); continue
        rows.append(dict(phi=phi, A=A, MSD=6 * A, fs7=math.exp(-49 * A), maxLam=L, A_fold=Am))
    return rows


if __name__ == '__main__':
    t0 = time.time()
    plan = [('PY', v) for v in ('a', 'a0', 'amix', 'MZ', 'MZ0', 'b0', 'GMCTV')] + \
           [('VWscale', v) for v in ('a', 'a0', 'amix', 'MZ', 'MZ0', 'b0', 'GMCTV')] + \
           [('VWshift', v) for v in ('a', 'a0')]
    phis = [0.30, 0.35, 0.40, 0.45, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
    out = {}
    for kind, var in plan:
        r = fold(kind, var)
        if 'phi_c' in r:
            r['branch'] = branch(kind, var, r['phi_c'], phis)
            br = ', '.join(f"{b['phi']:.2f}:{b['MSD']:.4f}" for b in r['branch'] if b.get('A'))
            print(f"[{time.time()-t0:6.1f}s] {kind:8s} {var:6s} phi_c={r['phi_c']:.4f} A_c={r['A_c']:.5f} MSD_c={r['MSD_c']:.4f} "
                  f"fs7_c={r['fs7_c']:.3f} humps={r['n_local_max']} | MSD branch {br}", flush=True)
        else:
            print(f"[{time.time()-t0:6.1f}s] {kind:8s} {var:6s} {r['error']}", flush=True)
        out[f"{kind}:{var}"] = r
        json.dump(out, open(os.path.join(HERE, 'results', 'folds.json'), 'w'), indent=1)
    print(f"done {time.time()-t0:.1f}s")
