#!/usr/bin/env python3
"""Validate PY input: g(1+) vs (1+phi/2)/(1-phi)^2, S(0) vs (1-phi)^4/(1+2phi)^2, y continuity at r=1,
analytic c^(k) vs numerical transform, h(r)=-1 inside core from the OZ inversion, VW contact = CS contact."""
import json, math
import os
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import numpy as np
from py_structure import *
out = {}
for (N, dr) in ((2 ** 16, 1e-3), (2 ** 17, 5e-4)):
    G = Grid(N, dr)
    for phi in (0.30, 0.45, 0.52, 0.58, 0.64):
        st = py_structure(phi, G)
        r = G.r
        i1 = i_contact(G)
        S0_exact = (1 - phi) ** 4 / (1 + 2 * phi) ** 2
        rho = st['rho']
        S0 = 1 / (1 - rho * c_py_k(np.array([0.0]), phi)[0])
        g1_exact = g_py_contact(phi)
        # y continuity: inside polynomial at r=1 vs 1+gamma at r=1
        yin1 = y_py_inside(1.0, phi)
        # h inside the core from OZ: gamma + c should be -1
        hin = st['gam'][r < 0.95] + st['c_r'][r < 0.95]
        # numerical c^(k) vs analytic (midpoint at jump)
        cr = st['c_r'].copy(); cr[i1] = 0.5 * (st['c_r'][i1 - 1] * 0 + (-(y_py_inside(1.0, phi))))
        cr[i1] = -0.5 * y_py_inside(1.0, phi)
        ck_num = G.k_from_r(cr)
        sel = G.k < 30
        err_ck = np.max(np.abs(ck_num[sel] - st['ck'][sel]))
        row = dict(g1=st['g1'], g1_exact=g1_exact, S0=S0, S0_exact=S0_exact, y_in_1=yin1,
                   max_dev_h_inside=float(np.max(np.abs(hin + 1))), err_ck_k_lt_30=float(err_ck),
                   Speak=float(np.max(st['Sk'])), kpeak=float(G.k[np.argmax(st['Sk'])]))
        print(f"N={N} dr={dr} phi={phi}: g(1+)={st['g1']:.6f} exact={g1_exact:.6f} | y_in(1)={yin1:.6f} | "
              f"S(0)={S0:.6e} exact={S0_exact:.6e} | max|h_in+1|={row['max_dev_h_inside']:.1e} | "
              f"max|ck_num-ck|={err_ck:.1e} | S_peak={row['Speak']:.3f} at k={row['kpeak']:.3f}")
        out[f"N{N}_phi{phi}"] = row
# Verlet-Weis
G = Grid()
for phi in (0.45, 0.52, 0.58):
    v = vw_structure(phi, G)
    print(f"VW phi={phi}: phi*={v['phis']:.4f} xi={v['xi']:.4f} dg1={v['dg1']:.4f} alpha={v['alpha']:.3f} "
          f"g(1+)={v['g1']:.5f} (CS {g_cs_contact(phi):.5f}, PY {g_py_contact(phi):.5f}) S(k->small)={v['Sk'][3]:.4f} "
          f"(CS S(0)={(1-phi)**4/(1+4*phi+4*phi**2-4*phi**3+phi**4):.4f})")
    out[f"VW_phi{phi}"] = dict(phis=v['phis'], xi=v['xi'], dg1=v['dg1'], alpha=v['alpha'], g1=float(v['g1']))
json.dump(out, open(os.path.join(HERE, 'results', 'structure_checks.json'), 'w'), indent=1)
