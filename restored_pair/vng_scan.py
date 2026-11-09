#!/usr/bin/env python3
"""VNG density scan and fold bisection for both variants (full = -ln Z, single = 1 - Z)."""
import json, sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import vng
G = vng.Grid(0.004, 4.0)
out = {}
for variant, phis, (lo, hi) in (('full', [1.78, 1.80, 1.85, 1.90, 2.00, 2.10, 2.20, 2.40, 2.60, 2.80, 3.20, 3.60], (1.70, 1.80)),
                                ('single', [3.30, 3.40, 3.60, 4.00, 4.50], (3.30, 3.40))):
    rows = []
    for phi in phis:
        o = vng.solve(phi, variant, 0.02, G, 0.5, maxit=6000)
        o.pop('r', None); o.pop('rho_tilde', None)
        rows.append(o)
        msg = (f"MSD={o['msd']:.4f} f(3,5,7,10)=" + " ".join(f"{o['f'][q]:.3f}" for q in ('k3','k5','k7','k10')) +
               f" fG7={o['fG_sameMSD']['k7']:.3f} a2={o['alpha2']:.3f}") if o['status'] != 'liquid' else ''
        print(f"{variant} phi={phi:.3f} {o['status']} it={o['iterations']} {msg}", flush=True)
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        o = vng.solve(mid, variant, 0.02, G, 0.5, maxit=6000)
        if o['status'] == 'liquid': lo = mid
        elif o['status'] == 'glass': hi = mid
        print(f"  bisect {variant}: phi={mid:.5f} -> {o['status']} (MSD={o.get('msd', float('nan')):.4f}, it={o['iterations']})", flush=True)
    out[variant] = dict(rows=rows, fold=[lo, hi])
    print(f"{variant}: fold in [{lo:.5f}, {hi:.5f}]", flush=True)
json.dump(out, open(os.path.join(HERE, 'results', 'vng_scan.json'), 'w'))
