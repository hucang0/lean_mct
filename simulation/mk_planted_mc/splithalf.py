#!/usr/bin/env python3
"""Split-half check of the cage decomposition: C_i from even frames, f_meas from odd frames (and vice versa).
Usage: python3 -I splithalf.py tmin tmax file.npz ..."""
import sys, numpy as np
ks = np.array([3.0, 5.0, 7.0, 10.0]); tmin, tmax = float(sys.argv[1]), float(sys.argv[2])
M = 590; i = np.arange(M) + 0.5; th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5 ** 0.5) * i
NV = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
for f in sys.argv[3:]:
    z = np.load(f); t = z['t_save']; tr = z['traj'].astype(np.float64)
    X = tr - tr.mean(axis=1, keepdims=True)
    sel = np.nonzero((t >= tmin) & (t <= min(tmax, t[-1])))[0]
    # split by time: first half vs second half of the window (stronger than even/odd: tests stationarity too)
    for name, (A_, B_) in (('evenodd', (sel[0::2], sel[1::2])), ('firstsecond', (sel[:len(sel)//2], sel[len(sel)//2:]))):
        Xa, Xb = X[A_], X[B_]
        C = np.einsum('tna,tnb->nab', Xa - Xa.mean(0), Xa - Xa.mean(0)) / (len(A_) - 1)
        rng = np.random.default_rng(1); T = len(B_)
        pa = rng.integers(0, T, 6000); pb = rng.integers(0, T, 6000); keep = pa != pb
        dn = np.linalg.norm(Xb[pa[keep]] - Xb[pb[keep]], axis=2); msd = np.mean(dn ** 2)
        fm = [np.mean(np.sinc(k * dn / np.pi)) for k in ks]; fg = [np.exp(-k * k * msd / 6) for k in ks]
        q = np.einsum('ma,nab,mb->nm', NV, C, NV); fa = [np.mean(np.exp(-k * k * q)) for k in ks]
        # rescale C so that its mean MSD matches the B half (removes drift of the overall cage size)
        sc = msd / (2 * np.trace(C, axis1=1, axis2=2).mean()); fa2 = [np.mean(np.exp(-k * k * q * sc)) for k in ks]
        exc = [(a - g) / (m - g) for m, g, a in zip(fm, fg, fa2)]
        print(f"{f.split('/')[-1]} {name}: MSD_B={msd:.4f} f_meas={np.round(fm,3)} gauss={np.round(fg,3)} aniso(C from A)={np.round(fa,3)} rescaled={np.round(fa2,3)} explained frac(k=5,7,10)={np.round(exc[1:],2)}")
