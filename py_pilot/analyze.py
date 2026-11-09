#!/usr/bin/env python3
"""(1) MZ Eq. (30) literal form  F1 = -3A int r^2 dq/dA ln q dr  vs the IBP form (A/3) int d^3c q'^2/q.
(2) Region decomposition of K for MZ vs MZ0 and a vs a0 at the PY fold, and of the MK restored kernel:
    where is 1/q (1/Z) > 1 (amplifying) and where < 1 (q = gamma*g > 1 near contact: de-amplifying)?
(3) Small-s expansion of the difference MZ - a:  |grad q|^2/q ~ y J^2/Z + 2 y' J + Z y'^2/y  (y outside),
    i.e. the PMF closure = version (a) + cross term with the cavity (third-body mean) force + its square."""
import math, json
import os
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import numpy as np
from gauss_py import Grid, Closures, Z_gauss, J_gauss, kspace_K
import py_structure as PS
G = Grid(); PG = PS.Grid(G.N, G.dr)
out = {}
# (1)
for phi, A in ((0.5184, 0.00853), (0.52, 0.02)):
    st = PS.py_structure(phi, PG); s = 2 * A
    C = Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'])
    q, _ = G.r_from_k_with_deriv(st['hk'] * np.exp(-G.k ** 2 * A)); q = 1 + q
    dqdA, _ = G.r_from_k_with_deriv(-G.k ** 2 * st['hk'] * np.exp(-G.k ** 2 * A))
    m = (q > 1e-11) & (G.r < 30)
    F1 = -3 * A * np.sum(G.r[m] ** 2 * dqdA[m] * np.log(q[m])) * G.dr
    Lam_lit = (2 ** 3 * phi / 3) * F1
    Lam_ibp = C.Lam('MZ', A)
    print(f"phi={phi} A={A}: MZ Eq.(30) literal Lambda = {Lam_lit:.6f}   IBP form Lambda = {Lam_ibp:.6f}")
    out[f"eq30_vs_ibp_phi{phi}_A{A}"] = dict(literal=Lam_lit, ibp=Lam_ibp)
# (2)
def regions(c, w, f):
    reg = dict(inside=c < 1, shell=(c >= 1) & (c < 1.3), outer=c >= 1.3)
    return {k: float(np.sum(w[m] * f[m])) for k, m in reg.items()}
phi, A = 0.5184, 0.00853; s = 2 * A
st = PS.py_structure(phi, PG)
C = Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'])
q, qp = G.r_from_k_with_deriv(st['hk'] * np.exp(-G.k ** 2 * A)); q = 1 + q[C.ic]; qp = qp[C.ic]
mq = q > 1e-11
fMZ = np.where(mq, qp ** 2 / np.where(mq, q, 1), 0); fMZ0 = qp ** 2
rho = st['rho']
dMZ = {k: rho * A * v for k, v in regions(C.c, C.w, fMZ).items()}
dMZ0 = {k: rho * A * v for k, v in regions(C.c, C.w, fMZ0).items()}
print(f"\nPY phi={phi}, A={A} (MZ fold). Lambda contributions [inside c<1 | shell 1-1.3 | outer >1.3]:")
print("  MZ  (with 1/q): " + " | ".join(f"{k}={v:.3f}" for k, v in dMZ.items()) + f"  total={sum(dMZ.values()):.3f}")
print("  MZ0 (q->1)    : " + " | ".join(f"{k}={v:.3f}" for k, v in dMZ0.items()) + f"  total={sum(dMZ0.values()):.3f}")
print(f"  max q on grid = {q.max():.3f} at c = {C.c[np.argmax(q)]:.3f}  (q>1 => 1/q < 1 de-amplifies)")
out['regions_MZ_PY_fold'] = dict(MZ=dMZ, MZ0=dMZ0, qmax=float(q.max()))
phi, A = 0.3760, 0.04584; s = 2 * A
st = PS.py_structure(phi, PG)
C = Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'])
Z, J = Z_gauss(C.c, s), J_gauss(C.c, s); y = st['y'][C.ic]
fa = np.where(Z > 1e-280, y * J ** 2 / np.maximum(Z, 1e-300), 0); fa0 = y * J ** 2
da = {k: st['rho'] * A * v for k, v in regions(C.c, C.w, fa).items()}
da0 = {k: st['rho'] * A * v for k, v in regions(C.c, C.w, fa0).items()}
print(f"\nPY phi={phi}, A={A} ((a) fold):")
print("  a  : " + " | ".join(f"{k}={v:.3f}" for k, v in da.items()) + f"  total={sum(da.values()):.3f}")
print("  a0 : " + " | ".join(f"{k}={v:.3f}" for k, v in da0.items()) + f"  total={sum(da0.values()):.3f}")
out['regions_a_PY_fold'] = dict(a=da, a0=da0)
# MK for comparison at its fold
A = 0.045310; s = 2 * A; rhoMK = 6 * 1.776241 / math.pi
Z, J = Z_gauss(C.c, s), J_gauss(C.c, s)
dmk = {k: rhoMK * A * v for k, v in regions(C.c, C.w, np.where(Z > 1e-280, J ** 2 / np.maximum(Z, 1e-300), 0)).items()}
dmk0 = {k: rhoMK * A * v for k, v in regions(C.c, C.w, J ** 2).items()}
print(f"\nMK phi=1.7762, A={A}:")
print("  restored: " + " | ".join(f"{k}={v:.3f}" for k, v in dmk.items()) + f"  total={sum(dmk.values()):.3f}")
print("  Z=1     : " + " | ".join(f"{k}={v:.3f}" for k, v in dmk0.items()) + f"  total={sum(dmk0.values()):.3f}")
out['regions_MK_fold'] = dict(restored=dmk, Z1=dmk0)
# (3) small-s expansion of MZ - a
print("\nDecomposition  Lambda_MZ  vs  Lambda_a + cross(2 y' J) + grad(Z y'^2/y)   [rho A (1/3) int d^3c ...]")
for phi in (0.45, 0.5184):
    st = PS.py_structure(phi, PG)
    C = Closures(G, st['rho'], st['y'], st['hk'], st['ck'], st['Sk'], y1=st['g1'])
    y = st['y'][C.ic]; yp = np.gradient(y, C.c)
    for A in (0.002, 0.00853, 0.02, 0.045):
        s = 2 * A
        Z, J = Z_gauss(C.c, s), J_gauss(C.c, s)
        La = C.Lam('a', A); LMZ = C.Lam('MZ', A)
        cross = st['rho'] * A * np.sum(C.w * 2 * yp * J)
        grad = st['rho'] * A * np.sum(C.w * Z * yp ** 2 / y)
        print(f"  phi={phi} A={A}: Lam_a={La:.3f}  cross={cross:+.3f}  grad={grad:+.3f}  sum={La+cross+grad:.3f}  Lam_MZ={LMZ:.3f}")
        out[f"decomp_phi{phi}_A{A}"] = dict(a=La, cross=float(cross), grad=float(grad), MZ=LMZ)
json.dump(out, open(os.path.join(HERE, 'results', 'analysis.json'), 'w'), indent=1)
