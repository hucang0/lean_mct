#!/usr/bin/env python3
"""Validate the generic DST machinery of gauss_py.py on the MK model (y == 1, g = Theta(r-1), c = h = Mayer f):
all routes must give the Letter's numbers: a = MZ = amix = 1.7762 (restored), a0 = MZ0 = b0 = 3.2369 (Z == 1)."""
import json, math
import os
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import numpy as np
from gauss_py import *
from scipy.optimize import minimize_scalar
G = Grid()
r = G.r
i1 = int(round(1 / G.dr)) - 1
# Mayer f in k space (exact), MK: h^ = c^ = f^ (shifted-neighbour Mayer function), S = 1
k = G.k
fk = -4 * PI / k ** 3 * (np.sin(k) - k * np.cos(k))
out = {}
C = Closures(G, rho=1.0, y_r=np.ones_like(r), hk=fk, ck=fk, Sk=np.ones_like(k), y1=1.0)
for v in ('a', 'MZ', 'amix', 'a0', 'MZ0', 'b0'):
    # rho-free: phi(A) = (pi/6)/(A K(A)); phi_c = min_A
    res = minimize_scalar(lambda la: (PI / 6) / (math.exp(la) * C.K(v, math.exp(la))),
                          bounds=(math.log(0.005), math.log(0.3)), method='bounded', options=dict(xatol=1e-8))
    out[v] = dict(phi_c=res.fun, A_c=math.exp(res.x))
    print(f"MK via {v:5s}: phi_c = {res.fun:.6f}, A_c = sigma2_c = {math.exp(res.x):.6f}, MSD_c = {6*math.exp(res.x):.4f}")
print("reference (brc_gauss_mk.py): restored 1.776241 (A=0.045310), Z=1 3.236939 (A=0.088393)")
json.dump(out, open(os.path.join(HERE, 'results', 'mk_paths_check.json'), 'w'), indent=1)
