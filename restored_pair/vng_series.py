#!/usr/bin/env python3
"""Fold of the VNG (non-Gaussian static cage) truncated after n bond collisions, L_n = sum_{m<=n} (1-Z)^m/m."""
import json, sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import vng
G = vng.Grid(0.004, 4.0)
gfold = {1: 3.2369, 2: 2.4320, 3: 2.1755, 4: 2.0528, 6: 1.9368, 8: 1.8831, 16: 1.8128}
out = []
for n in (1, 2, 3, 4, 6, 8, 16):
    lo, hi = gfold[n] - 0.06, gfold[n] + 0.10
    for _ in range(9):
        mid = 0.5 * (lo + hi)
        o = vng.solve(mid, f"n{n}", 0.02, G, 0.5, maxit=6000)
        if o['status'] == 'liquid': lo = mid
        else: hi = mid
    o = vng.solve(hi, f"n{n}", 0.02, G, 0.5, maxit=6000)
    out.append(dict(n=n, fold=[lo, hi], gauss_fold=gfold[n], msd_at_hi=o.get('msd'), alpha2_at_hi=o.get('alpha2')))
    print(f"VNG n={n}: fold in [{lo:.4f}, {hi:.4f}] (Gaussian {gfold[n]:.4f}); at hi: MSD={o.get('msd')}, a2={o.get('alpha2')}", flush=True)
json.dump(out, open(os.path.join(HERE, 'results', 'vng_series.json'), 'w'), indent=1)
