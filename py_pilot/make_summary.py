#!/usr/bin/env python3
"""Condense results/folds.json (run_folds.py) into results/summary_table.json: per closure variant phi_c, MSD_c,
f_s(7) at the fold, and the glass-branch MSD and f_s(7) at phi = 0.52-0.64 (SM Table S17).
Usage: python3 make_summary.py [folds.json] [summary_table.json]"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'folds.json')
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'results', 'summary_table.json')
PHIS = (0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64)
out = {}
for key, r in json.load(open(src)).items():
    if 'phi_c' not in r:
        out[key] = dict(error=r.get('error')); continue
    br = {round(b['phi'], 2): b for b in r.get('branch', []) if b.get('A') is not None}
    out[key] = dict(phi_c=r['phi_c'], MSD_c=r['MSD_c'], fs7_c=r['fs7_c'],
                    MSD={f"{p:g}": (br[p]['MSD'] if p in br else None) for p in PHIS},     # None: liquid at p
                    fs7={f"{p:g}": (br[p]['fs7'] if p in br else None) for p in PHIS})
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
json.dump(out, open(dst, 'w'), indent=1)
print(f"wrote {dst} ({len(out)} variants)")
