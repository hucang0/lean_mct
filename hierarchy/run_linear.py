#!/usr/bin/env python3
"""Linear-chain / Perron diagnostics for the MK hierarchy at phi = 1 (everything is exactly homogeneous in phi:
lamP ∝ phi, C ∝ phi, rho(T_N) ∝ phi^(N-1), so phi_lin(N) = rho(T_N; phi=1)^(-1/(N-1)) exactly).

Outputs results/linear_diag.json (next to this script; override with --out) with
  * per grid (M,qmax) and variant: lamP, C, w_max, omega_min, k_peak (Perron vector), k of w_max, Perron vector
  * rho(T_N), N=2..5 on M=100 (N=2,3 on M=200), with k-cutoffs kcut = 0,1,2,4 and "k >= 1", "k >= 2"
  * shape factors r_N = rho(T_N) (N-1)!/(lamP/2)^(N-1), phi_lin(N)
  * lamP(phi), C(phi) scans for PY (M=100) and MK (exactly linear) for figure F2
"""
import sys, os, json, time, math, argparse
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                     # mk_vertex.py, gmct_*.py live here
import mk_vertex as V
_ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
_ap.add_argument('--out', default=os.path.join(HERE, 'results', 'linear_diag.json'))
OUT = _ap.parse_args().out
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
dev = torch.device('cuda')
res = dict(perron_grids=[], rho_TN=[], scans={}, py_anchor=None)
T0 = time.time()

# ---- 1. Perron / C on the five grids at phi = 1 (+ k-cutoff variants on M=100, 200)
for (M, q) in [(100, 40.0), (200, 40.0), (400, 40.0), (100, 80.0), (200, 80.0), (800, 40.0)]:
    for var in ('Bself', 'Bcoll'):
        for kcut in ([0, 1, 2, 4, 5] if M in (100, 200) else [0]):
            Wt, Om, k = V.build_vertex_np(M, q, 1.0, var, kcut)
            p = V.perron(Wt, Om, k)
            rec = dict(M=M, qmax=q, h=q / M, kcut=kcut, kmin=float(k[0]), variant=var, phi=1.0,
                       lamP=p['lamP'], C=p['C'], w_max=p['w_max'], k_wmax=p['k_wmax'], omega_min=p['omega_min'],
                       k_peak=p['k_peak'], Mfrozen_max=p['Mfrozen_max'])
            if kcut == 0: rec['vP'] = p['vP']; rec['w'] = p['w']; rec['k'] = k.tolist()
            res['perron_grids'].append(rec)
            print(f"[perron] {var:5s} M={M:3d} qmax={q:4.0f} kcut={kcut} kmin={k[0]:.3f}: lamP={p['lamP']:.5f}  C={p['C']:.1f}  w_max={p['w_max']:.3f}@k={p['k_wmax']:.1f}  k_peak={p['k_peak']:.2f}", flush=True)
            del Wt
# PY anchor and PY on fine grids at its phi_c(2)
for (M, q) in [(100, 40.0), (200, 40.0), (400, 40.0)]:
    Wt, Om, k = V.build_vertex_np(M, q, V.PHI_C_PY_MCT, 'PY'); p = V.perron(Wt, Om, k)
    rec = dict(M=M, qmax=q, variant='PY', phi=V.PHI_C_PY_MCT, lamP=p['lamP'], C=p['C'], k_peak=p['k_peak'], w_max=p['w_max'], omega_min=p['omega_min'])
    if M == 100: rec['vP'] = p['vP']; rec['k'] = k.tolist(); res['py_anchor'] = rec
    res['perron_grids'].append(rec)
    print(f"[perron] PY M={M} phi_c(2): lamP={p['lamP']:.4f} C={p['C']:.1f} k_peak={p['k_peak']}", flush=True)

# ---- 2. scans for F2: PY M=100 phi in [0.30, 0.64]; MK at phi=1 (exact linear law), plus explicit check at several phi
phis = np.round(np.arange(0.30, 0.6401, 0.01), 4)
sc = []
for phi in phis:
    Wt, Om, k = V.build_vertex_np(100, 40.0, float(phi), 'PY'); p = V.perron(Wt, Om, k)
    sc.append(dict(phi=float(phi), lamP=p['lamP'], C=p['C'], k_peak=p['k_peak']))
res['scans']['PY_M100'] = sc
for var in ('Bself', 'Bcoll'):
    sc = []
    for phi in [0.1, 0.25, 0.5, 1.0, 1.5, 1.82, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        Wt, Om, k = V.build_vertex_np(100, 40.0, float(phi), var); p = V.perron(Wt, Om, k)
        sc.append(dict(phi=float(phi), lamP=p['lamP'], C=p['C'], k_peak=p['k_peak'], lamP_over_phi=p['lamP'] / phi, C_over_phi=p['C'] / phi))
    res['scans'][f'{var}_M100'] = sc
    print(f"[scan] {var}: lamP/phi = {[round(s['lamP_over_phi'],6) for s in sc]}", flush=True)
    print(f"[scan] {var}: C/phi    = {[round(s['C_over_phi'],3) for s in sc]}", flush=True)

# ---- 3. rho(T_N) with k-cutoffs
jobs = []
for var in ('Bself', 'Bcoll'):
    for N in (2, 3, 4, 5):
        for kcut in (0, 1, 2, 4, 5):          # M=100: kcut=2 <=> k>=1 ; kcut=5 <=> k>=2
            jobs.append((100, 40.0, var, N, kcut))
    for N in (2, 3):
        for kcut in (0, 1, 2, 4, 5, 10):      # M=200: kcut=5 <=> k>=1 ; kcut=10 <=> k>=2
            jobs.append((200, 40.0, var, N, kcut))
    for (M, q) in [(400, 40.0), (100, 80.0), (200, 80.0)]:
        for N in (2, 3):
            jobs.append((M, q, var, N, 0))
lam1 = {}
for (M, q, var, N, kcut) in jobs:
    r = V.rho_TN(M, q, 1.0, var, N, dev, kcut=kcut)
    Wt, Om, k = V.build_vertex_np(M, q, 1.0, var, kcut); p = V.perron(Wt, Om, k); del Wt
    r['lamP'] = p['lamP']; r['lamP_dense_eig_check_N2'] = (abs(r['rho'] - p['lamP'] / 2) / (p['lamP'] / 2)) if N == 2 else None
    r['schematic'] = (p['lamP'] / 2) ** (N - 1) / math.factorial(N - 1)
    r['r_N'] = r['rho'] / r['schematic']
    r['phi_lin'] = r['rho'] ** (-1.0 / (N - 1))
    res['rho_TN'].append(r)
    print(f"[rho] {var:5s} M={M} q={q:.0f} kcut={kcut} kmin={r['kmin']:.3f} N={N}: rho(T_N)={r['rho']:.6f} CW[{r['cw_lo']:.6f},{r['cw_hi']:.6f}] its={r['power_iters']} lamP/2={p['lamP']/2:.5f} r_N={r['r_N']:.4f} phi_lin={r['phi_lin']:.5f} vpeak_k={r['v_peak_k']:.2f} ({r['seconds']:.1f}s)", flush=True)
res['seconds'] = time.time() - T0
json.dump(res, open(OUT, 'w'))
print(f"DONE linear diagnostics in {time.time()-T0:.0f}s -> {OUT}", flush=True)
