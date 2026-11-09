#!/usr/bin/env python3
"""Summarise lag curves: MSD(tau), F_s(k=7,tau) at tau ~ 1e2..1e6 and log-slope dlnMSD/dlog10(tau) over the last decade."""
import json, sys, glob, numpy as np
rows = {}
for p in sys.argv[1:]:
    d = json.load(open(p))
    for name, r in d.items():
        tau = np.array(r['lag_tau']); m = np.array(r['lag_msd']); f7 = np.array(r['lag_fs']['k7'])
        def at(x): 
            j = np.argmin(np.abs(np.log(tau) - np.log(x))); return tau[j], m[j], f7[j]
        tmax = tau.max()
        out = {}
        for x in (1e2, 1e3, 1e4, 1e5, 1e6):
            if x <= tmax * 1.01: out[f"{x:.0e}"] = at(x)
        t1, m1, _ = at(tmax / 10); t2, m2, _ = at(tmax)
        slope = np.log10(m2 / m1) / np.log10(t2 / t1)
        phi = name.split('_')[1]
        rows.setdefault(phi, []).append(dict(name=name, msd_window=r['msd_window'], slope_last_decade=slope, **{k: v[1] for k, v in out.items()}))
        print(f"{name}: " + " ".join(f"MSD({k})={v[1]:.4f}" for k, v in out.items()) + f"  slope(last decade to {tmax:.0e})={slope:.3f}  MSDw={r['msd_window']:.4f}")
