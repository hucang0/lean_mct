#!/usr/bin/env python3
"""Refine the VNG single-collision fold in (3.30, 3.40) (vng_scan.py's bracket (3.10, 3.30) did not contain it)."""
import json, sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import vng
G = vng.Grid(0.004, 4.0); lo, hi = 3.30, 3.40
for _ in range(10):
    mid = 0.5 * (lo + hi); o = vng.solve(mid, 'single', 0.02, G, 0.5, maxit=8000)
    if o['status'] == 'liquid': lo = mid
    else: hi = mid
    print(f"  single phi={mid:.5f} -> {o['status']} MSD={o.get('msd', float('nan')):.4f} it={o['iterations']}", flush=True)
print(f"single fold in [{lo:.5f}, {hi:.5f}]", flush=True)
json.dump(dict(fold=[lo, hi]), open(os.path.join(HERE, 'results', 'vng_single_fold.json'), 'w'))
