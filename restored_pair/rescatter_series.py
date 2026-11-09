#!/usr/bin/env python3
"""The bond-rescattering series (SM Sec. S5 C, Table S16).  The resummed weight 1/Z = sum_{m>=0} (1-Z)^m: the term m is the
contribution of m+1 successive collisions of the same bond (single collision = m = 0 = diagonal N = 2).
Truncating after n collisions:  w_n(Z) = sum_{m<n} (1-Z)^m.
 (a) Gaussian cage, MSD level:  sigma^2 (rho/3) int d^3c J^2 w_n(Z) = 1   ->  phi_c(n), sigma^2_c(n)
 (b) VNG (non-Gaussian, k-resolved, static):  L_n(Z) = sum_{m=1}^n (1-Z)^m/m  (L_1 = 1-Z, L_inf = -ln Z)
Compared with the diagonal hierarchy order N (SM Table S1: phi_c(N) = 2.80, 3.26, 3.69, 4.10, 4.49 for N = 2..6)."""
import json, math, sys, os
import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import brc_gauss as bg
PI = math.pi

def m0n(s, n, ngrid=4000):
    c = bg.c_grid(s, ngrid); J = bg.J_gauss(c, s); Z = bg.Z_gauss(c, s)
    if n == 'inf':
        w = np.where(Z > 0, 1.0 / Z, 0.0)
    else:
        x = 1.0 - Z; w = sum(x ** m for m in range(n))
    return simpson(4 * PI * c ** 2 * J ** 2 * w, x=c) / 3.0

def fold(n):
    f = lambda ls: (PI / 6.0) / (math.exp(ls) * m0n(2 * math.exp(ls), n))
    r = minimize_scalar(f, bounds=(math.log(0.003), math.log(0.5)), method='bounded', options=dict(xatol=1e-9))
    return r.fun, math.exp(r.x)

out = dict(gauss=[], diag_hierarchy=dict(N=[2, 3, 4, 5, 6], phi_c=[2.80, 3.26, 3.69, 4.10, 4.49]))
for n in list(range(1, 13)) + [16, 24, 32, 'inf']:
    pc, s2 = fold(n)
    out['gauss'].append(dict(n=n, phi_c=pc, sigma2_c=s2, msd_c=6 * s2))
    print(f"Gaussian MSD level, collisions n={n}: phi_c={pc:.4f} sigma2_c={s2:.4f}", flush=True)
json.dump(out, open(os.path.join(HERE, 'results', 'rescatter_series_gauss.json'), 'w'), indent=1)
