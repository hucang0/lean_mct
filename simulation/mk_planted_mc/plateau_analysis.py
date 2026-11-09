#!/usr/bin/env python3
"""Plateau observables and cage decomposition for MK planted-MC trajectories (read-only inputs; SM Sec. S4 D-E).
For each trajectory: centre-of-mass drift removed per frame (exact zero mode, as in analysis.py).
 (1) lag curves from the time origins: MSD(tau), F_s(k,tau) = <sinc(k|dx|)> (exact isotropic average), alpha2(tau);
 (2) plateau window [tmin, tmax] of saved frames: per-particle cage covariance C_i (3x3);
     f_meas(k) = <sinc(k|x(t)-x(t')|)> over random frame pairs in the window,
     f_gauss   = exp(-k^2 MSD/6) with MSD = 2 <tr C>,
     f_isomix  = E_i exp(-k^2 tr C_i/3)            (cage-to-cage heterogeneity of size only),
     f_anisomix= E_i <exp(-k^2 n.C_i.n)>_n          (individually Gaussian, anisotropic cages; n on a 590-pt sphere).
Usage: python3 -I plateau_analysis.py tmin tmax out.json file1.npz file2.npz ...
"""
import sys, json
import numpy as np
ks = np.array([3.0, 5.0, 7.0, 10.0])
tmin, tmax, outp = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
files = sys.argv[4:]
# quasi-uniform sphere directions (Fibonacci)
M = 590; i = np.arange(M) + 0.5
th = np.arccos(1 - 2 * i / M); ph = np.pi * (1 + 5 ** 0.5) * i
NV = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], 1)
res = {}
for f in files:
    z = np.load(f)
    t = z['t_save']; tr = z['traj'].astype(np.float64); origins = z['origins']
    X = tr - tr.mean(axis=1, keepdims=True)
    tdone = int(t[-1])
    # (1) lag curves
    lag = {}
    for o in origins:
        io = np.searchsorted(t, o)
        if io >= len(t) or t[io] != o: continue
        for j in range(io + 1, len(t)):
            tau = int(t[j] - o)
            d = X[j] - X[io]; r2 = (d ** 2).sum(1); r = np.sqrt(r2)
            rec = lag.setdefault(tau, [])
            rec.append([r2.mean(), (r2 ** 2).mean()] + [np.mean(np.sinc(k * r / np.pi)) for k in ks])
    taus = np.array(sorted(lag)); A = np.array([np.mean(lag[x], axis=0) for x in taus])
    msd = A[:, 0]; a2 = 3 * A[:, 1] / (5 * msd ** 2) - 1
    # (2) window statistics
    sel = np.nonzero((t >= tmin) & (t <= min(tmax, tdone)))[0]
    Xw = X[sel]; T = len(sel)
    xb = Xw.mean(0); dx = Xw - xb
    C = np.einsum('tna,tnb->nab', dx, dx) / (T - 1)
    trC = np.trace(C, axis1=1, axis2=2)
    rng = np.random.default_rng(0)
    pa = rng.integers(0, T, 6000); pb = rng.integers(0, T, 6000); keep = pa != pb
    dd = Xw[pa[keep]] - Xw[pb[keep]]; dn = np.linalg.norm(dd, axis=2)
    f_meas = [float(np.mean(np.sinc(k * dn / np.pi))) for k in ks]
    msd_w = float(np.mean(dn ** 2))
    f_gauss = [float(np.exp(-k * k * msd_w / 6)) for k in ks]
    f_iso = [float(np.mean(np.exp(-k * k * trC / 3))) for k in ks]
    q = np.einsum('ma,nab,mb->nm', NV, C, NV)            # (N, M) directional variances
    f_aniso = [float(np.mean(np.exp(-k * k * q))) for k in ks]
    ev = np.linalg.eigvalsh(C)                          # ascending
    res[f.split('/')[-1]] = dict(sweeps_done=tdone, window=[tmin, min(tmax, tdone)], frames=int(T),
        lag_tau=taus.tolist(), lag_msd=msd.tolist(), lag_alpha2=a2.tolist(), lag_fs={f"k{int(k)}": A[:, 2 + n].tolist() for n, k in enumerate(ks)},
        msd_window=msd_w, f_meas=f_meas, f_gauss=f_gauss, f_isomix=f_iso, f_anisomix=f_aniso,
        cv_trC=float(trC.std() / trC.mean()), aniso_ratio_mean=float(np.mean(ev[:, 2] / ev[:, 0])),
        alpha2_window=float(3 * np.mean(dn ** 4) / (5 * msd_w ** 2) - 1))
    r = res[f.split('/')[-1]]
    print(f"{f.split('/')[-1]} t={tdone} frames={T} MSDw={msd_w:.4f} f_meas={np.round(f_meas,3)} gauss={np.round(f_gauss,3)} iso={np.round(f_iso,3)} aniso={np.round(f_aniso,3)} cv={r['cv_trC']:.2f} lmax/lmin={r['aniso_ratio_mean']:.2f}", flush=True)
json.dump(res, open(outp, 'w'))
