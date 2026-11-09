#!/usr/bin/env python3
"""Null test for the cage decomposition: per particle i, draw T iid isotropic Gaussian points with the measured
variance tr C_i / 3 (heterogeneous sizes, no anisotropy, no non-Gaussianity). Apply the same estimators.
f_aniso - f_meas on the surrogate = estimator bias from the noise in the sample covariance (T frames).
Usage: python3 -I bias_null.py tmin tmax file.npz ..."""
import sys, numpy as np
ks = np.array([3.0, 5.0, 7.0, 10.0]); tmin, tmax = float(sys.argv[1]), float(sys.argv[2])
M = 590; i = np.arange(M) + 0.5; th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5 ** 0.5) * i
NV = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
def est(Xw):
    T = len(Xw); dx = Xw - Xw.mean(0); C = np.einsum('tna,tnb->nab', dx, dx) / (T - 1)
    rng = np.random.default_rng(0); pa = rng.integers(0, T, 6000); pb = rng.integers(0, T, 6000); keep = pa != pb
    dn = np.linalg.norm(Xw[pa[keep]] - Xw[pb[keep]], axis=2); msd = np.mean(dn ** 2)
    fm = np.array([np.mean(np.sinc(k * dn / np.pi)) for k in ks]); fg = np.exp(-ks ** 2 * msd / 6)
    q = np.einsum('ma,nab,mb->nm', NV, C, NV); fa = np.array([np.mean(np.exp(-k * k * q)) for k in ks])
    return fm, fg, fa, C
for f in sys.argv[3:]:
    z = np.load(f); t = z['t_save']; tr = z['traj'].astype(np.float64); X = tr - tr.mean(axis=1, keepdims=True)
    sel = np.nonzero((t >= tmin) & (t <= min(tmax, t[-1])))[0]; Xw = X[sel]; T = len(sel)
    fm, fg, fa, C = est(Xw)
    s2 = np.trace(C, axis1=1, axis2=2) / 3
    rng = np.random.default_rng(7)
    Xs = rng.normal(size=Xw.shape) * np.sqrt(s2)[None, :, None]
    fms, fgs, fas, _ = est(Xs)
    corr = fa - (fas - fms)          # bias-corrected anisotropic-mixture prediction
    frac = (corr - fg) / (fm - fg)
    print(f"{f.split('/')[-1]} T={T}: f_meas(5,7,10)={np.round(fm[1:],3)} gauss={np.round(fg[1:],3)} aniso={np.round(fa[1:],3)} "
          f"null bias={np.round((fas-fms)[1:],3)} corrected={np.round(corr[1:],3)} explained={np.round(frac[1:],2)}")
