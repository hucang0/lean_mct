#!/usr/bin/env python3
"""Matched ablation at the MSD level (Gaussian cage), hard core on (BRC-G = Szamel = replica m->1)
vs off (diagonal single-collision), as a function of phi.  Glass-branch sigma^2(phi), MSD plateau 6 sigma^2 (two independent draws from the cage) and
f_G(k) = exp(-k^2 sigma^2) at k = 3,5,7,10.  Uses brc_gauss.py (this folder). SM Table S15, SM Fig. S5."""
import json, math, sys, os
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import brc_gauss as bg

out = {}
for var in ('brc', 'diag'):
    phic, s2c = bg.critical(var)
    rows = []
    for phi in [1.78, 1.80, 1.85, 1.90, 2.00, 2.10, 2.20, 2.40, 2.60, 2.80, 3.00, 3.20, 3.24, 3.30, 3.40, 3.60, 4.00, 4.50]:
        if phi < phic:
            rows.append(dict(phi=phi, status='liquid'))
            continue
        s2 = bg.sigma2_glass(phi, var, sig2_c=s2c)
        rows.append(dict(phi=phi, status='glass', sigma2=s2, msd=6 * s2,
                         fG={k: math.exp(-k * k * s2) for k in (3, 5, 7, 10)}))
    out[var] = dict(phi_c=phic, sigma2_c=s2c, rows=rows)
    print(f"{var}: phi_c={phic:.6f} sigma2_c={s2c:.6f}")
    for r in rows:
        if r['status'] == 'glass':
            print(f"  phi={r['phi']:.2f} MSD={r['msd']:.4f} fG(3,5,7,10)=" + " ".join(f"{r['fG'][k]:.3f}" for k in (3,5,7,10)))
        else:
            print(f"  phi={r['phi']:.2f} liquid")
json.dump(out, open(os.path.join(HERE, 'results', 'gauss_curves.json'), 'w'), indent=1)

# dense curves for plotting
dense = {}
for var in ('brc', 'diag'):
    phic, s2c = out[var]['phi_c'], out[var]['sigma2_c']
    ph = np.concatenate([[phic], phic + np.geomspace(1e-4, 4.6 - phic, 70)])
    s2 = [s2c] + [bg.sigma2_glass(p, var, sig2_c=s2c) for p in ph[1:]]
    # unstable branch near the fold (for the C-shaped curve)
    dense[var] = dict(phi=ph.tolist(), msd=[6 * x for x in s2], f7=[math.exp(-49 * x) for x in s2])
json.dump(dense, open(os.path.join(HERE, 'results', 'gauss_curves_dense.json'), 'w'))
print('dense curves written')
