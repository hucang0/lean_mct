#!/usr/bin/env python3
"""Diagonal N = 2 hierarchy (hard core off), k-resolved, bipolar vertex W~_s of SM Sec. S2 D on the Franosch grid
(M = 100, q_max = 40; same discretisation as brc_kres.py BRCKernel.diag_bipolar). Plateau f(k) by
iteration from f = 1 at densities above phi_c(2) = 2.7966."""
import json, math, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
import numpy as np
PI = math.pi
M, qmax = 100, 40.0; h = qmax / M; k = h * (np.arange(M) + 0.5)
def mayer_ft(q):
    return -4.0 * PI / q ** 3 * (np.sin(q) - q * np.cos(q))
kh = np.arange(M) + 0.5
K_, Q_, P_ = kh[:, None, None], kh[None, :, None], kh[None, None, :]
mask = (P_ >= np.abs(K_ - Q_) + 0.5 - 1e-9) & (P_ <= K_ + Q_ - 0.5 + 1e-9)
W = h ** 5 / (16 * PI ** 2) * (Q_ * P_ / K_ ** 3) * (K_ ** 2 + Q_ ** 2 - P_ ** 2) ** 2 * mayer_ft(k)[None, :, None] ** 2
Wk2 = np.where(mask, W, 0.0) / k[:, None, None] ** 2
out = []
for phi in [2.70, 2.79, 2.80, 2.85, 3.00, 3.20, 3.60, 4.00, 4.50]:
    rho = 6 * phi / PI; f = np.ones(M)
    for it in range(200000):
        Mk = rho * np.einsum('kqp,q,p->k', Wk2, f, f); fn = Mk / (1 + Mk)
        d = np.max(np.abs(fn - f)); f = fn
        if f.max() < 1e-7 or d < 1e-12: break
    st = 'glass' if f.max() > 1e-3 else 'liquid'
    fa = {f"k{q}": float(np.interp(q, k, f)) for q in (1, 2, 3, 5, 7, 10)}
    msd2 = float(-6 * np.log(fa['k2']) / 4) if st == 'glass' else None
    out.append(dict(phi=phi, status=st, iterations=it, f=fa, msd_from_k2=msd2, k=k.tolist(), fk=f.tolist()))
    print(f"diag N=2 phi={phi:.2f}: {st} it={it} f(1,2,3,5,7,10)=" + " ".join(f"{fa[x]:.3f}" for x in fa) + (f"  MSD(k=2)={msd2:.4f}" if msd2 else ''), flush=True)
json.dump(out, open(os.path.join(HERE, 'results', 'diag_kres.json'), 'w'))
