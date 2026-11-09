import json, sys
import numpy as np
sys.path.insert(0, '.')
import n058_estimate as E
d = json.load(open('results/n058.json'))
L = {int(k): v for k, v in d['Lambda_c'].items()}
lp = {float(k): v for k, v in d['lamP'].items()}
phis = np.array(sorted(lp)); lps = np.array([lp[p] for p in phis])
l2 = np.interp(0.515912, phis, lps)
corr = {3: 0.531885, 4: 0.5441697, 5: 0.553815, 6: 0.5613, 7: 0.5671}
mem = {3: 0.533862, 4: 0.546851, 5: 0.556824, 6: 0.565}
print('N  Lc/4   corr_ratio dev%   mem_ratio dev%')
devc = {}
for N in range(3, 8):
    r = L[N] / 4
    rc = E.lamP(corr[N]) / E.lamP(0.515912); devc[N] = rc / r - 1
    rm = (E.lamP(mem[N]) / E.lamP(0.515914)) if N in mem else float('nan')
    print(N, round(r, 4), round(rc, 4), round(100 * (rc / r - 1), 2), round(rm, 4), round(100 * (rm / r - 1), 2))
# correlator: extrapolate the (linear-in-N) deviation and invert
Ns = np.array(sorted(devc)); dv = np.array([devc[n] for n in Ns]); a, b = np.polyfit(Ns, dv, 1)
print('corr deviation fit: dev = %.4f N + %.4f' % (a, b))
for N in range(7, 15):
    tgt = l2 * L[N] / 4 * (1 + a * N + b)
    print('corr-extrap N=%d phi=%.4f   law N=%d phi=%.4f' % (N, np.interp(tgt, lps, phis), N, np.interp(l2 * L[N] / 4, lps, phis)))
# dominant-mode check: peak k of Perron vector vs phi (fine grid)
h = E.QMAX / E.M; k = h * (np.arange(E.M) + 0.5)
def vec(phi):
    khat = np.arange(E.M) + 0.5
    rho = 6 * phi / np.pi; c = E.py_ck(k, phi); S = 1 / (1 - rho * c); Om = k ** 2 / S
    K = khat[:, None, None]; Q = khat[None, :, None]; P = khat[None, None, :]
    mask = (P >= np.abs(K - Q) + 0.5 - 1e-9) & (P <= K + Q - 0.5 + 1e-9)
    br = (Q ** 2 - P ** 2 + K ** 2) * c[None, :, None] + (P ** 2 - Q ** 2 + K ** 2) * c[None, None, :]
    Wt = np.where(mask, rho * h ** 5 / (32 * np.pi ** 2) * (Q * P / K ** 3) * br ** 2 * S[None, :, None] * S[None, None, :], 0)
    A = 2 * Wt.sum(2) / Om[:, None]
    w, V = np.linalg.eig(A); i = np.argmax(np.abs(w)); v = np.abs(V[:, i].real)
    return float(abs(w[i])), float(k[np.argmax(v)]), v / np.linalg.norm(v)
v0 = vec(0.5159)[2]
for p in [0.52, 0.55, 0.57, 0.58, 0.59, 0.595, 0.60, 0.605, 0.61, 0.62, 0.63, 0.64]:
    lam, kp, v = vec(p); print('phi %.3f lamP %.2f  peak k %.2f  overlap with phi_c(2) vector %.4f' % (p, lam, kp, float(v @ v0)))
