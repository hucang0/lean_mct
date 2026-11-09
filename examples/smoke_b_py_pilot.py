#!/usr/bin/env python3
"""Smoke test (b): ordinary d = 3 hard spheres, Gaussian pair closures (SM Sec. S5 F, Table S17).
1. py_pilot/mk_sanity.py: the MK sanity check of the generic Gaussian machinery (1.776241 / 3.236939).
2. The PY potential-of-mean-force (Mangeat-Zamponi m -> 1 replica) fold, phi_c = 0.5184 (published 0.5216),
   on a coarse radial grid (N = 2^14, dr = 2e-3), and with the production code path of py_pilot/run_folds.py
   (N = 2^17, dr = 5e-4) for PY and Verlet-Weis structure (VW: 0.5345, published 0.5367).
CPU only, a few seconds."""
import json, math, runpy, sys
from _check import ROOT, check, finish
PP = ROOT / "py_pilot"
sys.path.insert(0, str(PP))

# 1. MK sanity check (runs the shipped script; writes py_pilot/results/mk_sanity.json)
sys.argv = [str(PP / "mk_sanity.py")]
runpy.run_path(str(PP / "mk_sanity.py"), run_name="__main__")
r = json.load(open(PP / "results" / "mk_sanity.json"))
check("MK restored phi_c (n=8000)", r["n8000"]["phi_c_restored"], 1.776241, 1e-5)
check("MK Z=1 phi_c (n=8000)", r["n8000"]["phi_c_Z1"], 3.236939, 1e-5)

# 2. PY MZ fold
from scipy.optimize import brentq
from gauss_py import Grid, Closures
import py_structure as PS


def fold(N, dr, var="MZ", cmax=30.0):
    G = Grid(N, dr); PG = PS.Grid(N, dr)
    def F(phi):
        st = PS.py_structure(phi, PG)
        C = Closures(G, st["rho"], st["y"], st["hk"], st["ck"], st["Sk"], y1=st["g1"], cmax=cmax)
        return C.maxLam(var, lo=1e-3, hi=0.2, n=24)
    phic = brentq(lambda p: F(p)[0] - 1.0, 0.3, 0.6, xtol=1e-6)
    return phic, F(phic)[1]


phic, A = fold(2 ** 14, 2e-3)
print(f"PY MZ fold, coarse grid (N = 2^14, dr = 0.002): phi_c = {phic:.5f}, MSD_c = 6A = {6 * A:.4f}")
check("PY MZ phi_c (coarse grid)", phic, 0.51845, 2e-4)

import run_folds as RF                       # production grid and fold solver of Table S17
for kind, ref in (("PY", 0.51845), ("VWscale", 0.53450)):
    r = RF.fold(kind, "MZ")
    print(f"{kind} MZ fold, production grid: phi_c = {r['phi_c']:.5f}, MSD_c = {r['MSD_c']:.4f}, f_s(7) = {r['fs7_c']:.3f}")
    check(f"{kind} MZ phi_c (production grid)", r["phi_c"], ref, 2e-4)
finish()
