#!/usr/bin/env python3
"""Grid convergence of the PY MZ (PMF / m->1 replica) and (a) folds; and an independent real-space evaluation of
q = gamma_s * g via the 1D radial kernel (MZ Eq. 26) at the fold, to cross-check the DST route."""
import math, json
import os
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import numpy as np
from scipy.optimize import brentq
from gauss_py import Grid, Closures
import py_structure as PS
out = {}
def fold(N, dr, var, cmax=30.0):
    G = Grid(N, dr); PG = PS.Grid(N, dr)
    def F(phi):
        st = PS.py_structure(phi, PG)
        C = Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'], cmax=cmax)
        return C.maxLam(var, lo=1e-3, hi=0.2, n=24)
    phic = brentq(lambda p: F(p)[0] - 1, 0.3, 0.6, xtol=1e-6)
    return phic, F(phic)[1]
for (N, dr, cmax) in ((2**16, 1e-3, 30), (2**17, 5e-4, 30), (2**18, 2.5e-4, 30), (2**17, 5e-4, 15), (2**18, 5e-4, 60)):
    for var in ('MZ', 'a'):
        p, A = fold(N, dr, var, cmax)
        print(f"N={N} dr={dr} cmax={cmax}: {var:3s} phi_c={p:.5f} A_c={A:.5f}", flush=True)
        out[f"{var}_N{N}_dr{dr}_cmax{cmax}"] = dict(phi_c=p, A_c=A)
# independent real-space check of K_MZ at phi=0.52, A=0.0085
phi, A = 0.52, 0.0085; s = 2 * A
G = Grid(); PG = PS.Grid(G.N, G.dr)
st = PS.py_structure(phi, PG)
C = Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'])
K_dst = C.K('MZ', A)
u = G.r; h = st['g_node'] - 1; du = G.dr
cs = np.arange(0.5, 12.0, 0.002)
q = np.empty_like(cs); qp = np.empty_like(cs)
for i, c in enumerate(cs):
    m = np.abs(u - c) < 12 * math.sqrt(s)
    uu = u[m]; hh = h[m]
    Em = np.exp(-(c - uu) ** 2 / (2 * s)); Ep = np.exp(-(c + uu) ** 2 / (2 * s))
    ker = uu / c / math.sqrt(2 * PI * s) * (Em - Ep) if False else uu / c / math.sqrt(2 * math.pi * s) * (Em - Ep)
    dker = uu / math.sqrt(2 * math.pi * s) * (-(Em - Ep) / c ** 2 + (-(c - uu) / s * Em + (c + uu) / s * Ep) / c)
    q[i] = 1 + np.sum(ker * hh) * du
    qp[i] = np.sum(dker * hh) * du
w = 4 * math.pi * cs ** 2 * 0.002 / 3
mm = q > 1e-11
K_real = np.sum(w[mm] * qp[mm] ** 2 / q[mm])
print(f"K_MZ(phi=0.52, A=0.0085): DST route {K_dst:.5f}  real-space kernel (c in [0.5,12]) {K_real:.5f}")
out['K_MZ_check'] = dict(dst=K_dst, real=K_real)
json.dump(out, open(os.path.join(HERE, 'results', 'conv_mz.json'), 'w'), indent=1)
