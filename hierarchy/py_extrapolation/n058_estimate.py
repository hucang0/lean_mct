#!/usr/bin/env python3
"""Estimate the GMCT order N at which PY hard-sphere phi_c(N) passes 0.58, via the
parameter-free single-mode 1/n dilution law of SM Sec. S1 E:
    lamP(phi_c(N)) / lamP(phi_c(2)) = Lambda_c(N) / 4,
lamP(phi) = spectral radius of A(k,q) = (2/Om1^2(k)) sum_p W~(k;q,p)  (SM Eq. perron),
on the Luo-Janssen grid (M=100, qmax=40), PY statics (same vertex as ../gmct_dense_gpu.py).
Lambda_c(N): schematic GMCT f_n/(1-f_n) = (Lam/n) f_{n+1}, closure f_N = f_1 f_{N-1}.
Note: analytic 3D Percus-Yevick S(k) is quantitative up to phi <= 0.60 (N <= 13 in SM Table S7)
and loses physical validity near random close packing (phi_rcp ~ 0.64); values tabulated above
phi = 0.60 are formal analytic continuations of the PY formula.
CPU only, no GPU."""
import json, sys
import numpy as np
from numpy.polynomial.legendre import leggauss

M, QMAX = 100, 40.0
XG, WG = leggauss(400); RR = 0.5 * (XG + 1.0); WW = 0.5 * WG


def py_ck(k, phi):
    l1 = (1 + 2 * phi) ** 2 / (1 - phi) ** 4
    l2 = -(1 + phi / 2) ** 2 / (1 - phi) ** 4
    cr = -l1 - 6 * phi * l2 * RR - 0.5 * phi * l1 * RR ** 3
    return np.array([4 * np.pi / kv * np.sum(WW * RR * cr * np.sin(kv * RR)) for kv in k])


def lamP(phi):
    h = QMAX / M; khat = np.arange(M) + 0.5; k = h * khat
    rho = 6 * phi / np.pi; c = py_ck(k, phi); S = 1.0 / (1.0 - rho * c); Om = k ** 2 / S
    K = khat[:, None, None]; Q = khat[None, :, None]; P = khat[None, None, :]
    mask = (P >= np.abs(K - Q) + 0.5 - 1e-9) & (P <= K + Q - 0.5 + 1e-9)
    br = (Q ** 2 - P ** 2 + K ** 2) * c[None, :, None] + (P ** 2 - Q ** 2 + K ** 2) * c[None, None, :]
    Wt = rho * h ** 5 / (32 * np.pi ** 2) * (Q * P / K ** 3) * br ** 2 * S[None, :, None] * S[None, None, :]
    Wt = np.where(mask, Wt, 0.0)
    A = 2.0 * Wt.sum(axis=2) / Om[:, None]
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def lam_c(N):
    """Minimal schematic coupling with a nontrivial plateau at closure order N."""
    if N == 2:
        return 4.0
    u = np.logspace(-12, 0, 20001)            # trial f_N
    def ok(L):
        f = u.copy(); fNm1 = None
        for n in range(N - 1, 0, -1):          # f_n = x/(1+x), x = (L/n) f_{n+1}
            x = (L / n) * f; f = x / (1 + x)
            if n == N - 1:
                fNm1 = f.copy()
        return np.max(f * fNm1 / u) >= 1.0     # closure u = f_1 f_{N-1} solvable
    lo, hi = 1.0, 4.0 * N
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


def main():
    Ls = {N: lam_c(N) for N in range(2, 41)}
    phis = np.round(np.arange(0.500, 0.6601, 0.0025), 4)
    lp = np.array([lamP(p) for p in phis])
    valid = phis <= 0.605
    phis_v, lp_v = phis[valid], lp[valid]
    out = {'grid': {'M': M, 'qmax': QMAX}, 'valid_phi_max': 0.605, 'Lambda_c': Ls,
           'lamP': dict(zip(map(float, phis), map(float, lp)))}
    for anchor_name, phi2 in [('corr', 0.515912), ('mem', 0.515914)]:
        l2 = float(np.interp(phi2, phis_v, lp_v))
        pred, pred_formal = {}, {}
        for N, L in Ls.items():
            target = l2 * L / 4.0
            pred[N] = float(np.interp(target, lp_v, phis_v)) if target <= lp_v[-1] else None
            pred_formal[N] = float(np.interp(target, lp, phis)) if target <= lp[-1] else None
        out['pred_' + anchor_name] = pred
        out['pred_formal_' + anchor_name] = pred_formal
    json.dump(out, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
