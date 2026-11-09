#!/usr/bin/env python3
"""MK sanity check: reproduce the Gaussian MSD-level folds of the Letter (1.776 restored, 3.237 Z=1),
using brc_gauss_mk.py, an identical copy of ../restored_pair/brc_gauss.py."""
import json, math, os
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import brc_gauss_mk as B
out = {}
for n in (4000, 8000):
    pb, sb = B.critical('brc', n); pd, sd = B.critical('diag', n)
    print(f"n={n}: restored phi_c={pb:.6f} sigma2_c={sb:.6f} MSD={6*sb:.4f} | Z=1: phi_c={pd:.6f} sigma2_c={sd:.6f} MSD={6*sd:.4f}")
    out[f'n{n}'] = dict(phi_c_restored=pb, sigma2_c_restored=sb, phi_c_Z1=pd, sigma2_c_Z1=sd)
s2 = B.sigma2_glass(2.0, 'brc', 8000)
print(f"phi=2.0 restored MSD={6*s2:.4f}  (SM Table: 0.119)")
out['MSD_phi2_restored'] = 6*s2
json.dump(out, open(os.path.join(HERE, 'results', 'mk_sanity.json'), 'w'), indent=1)
