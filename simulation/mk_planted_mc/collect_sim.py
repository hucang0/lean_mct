#!/usr/bin/env python3
"""Collect plateau observables per density (mean and s.d. over planted samples) into sim_summary.json
(SM Fig. S5, SM Table S13). Window [1e3, end of run]; CM drift removed; f_aniso bias-corrected with the
iid-Gaussian null of bias_null.py. Trajectories are grouped by the phi in their file name (run_phi<phi>_N1000_s*.npz).
The published file used phi = 1.80, 2.00 (seeds 1-3) and phi = 1.90, 2.10, ..., 3.60 (seeds 11-13); see README.md.
Usage: python3 collect_sim.py [--out FILE] run_phi*_N1000_s*.npz ...   (default --out ./results/sim_summary.json)"""
import argparse, glob, json, os, sys, numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from bias_null_lib import est
ap = argparse.ArgumentParser(description="collect plateau observables per density")
ap.add_argument("files", nargs="+", help="run_phi*_N1000_s*.npz trajectory files from mk_mc.py")
ap.add_argument("--out", default=os.path.join(HERE, "results", "sim_summary.json"))
args = ap.parse_args()
sets = {}
for f in sorted(args.files):
    if f.endswith('.tmp.npz'): continue
    sets.setdefault(float(os.path.basename(f).split('phi')[1][:4]), []).append(f)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
out = {}
for phi in sorted(sets):
    recs = []
    for f in sets[phi]:
        z = np.load(f); t = z['t_save']; tr = z['traj'].astype(np.float64); X = tr - tr.mean(axis=1, keepdims=True)
        sel = np.nonzero(t >= 1000)[0]; Xw = X[sel]
        fm, fg, fa, C, msd = est(Xw)
        s2 = np.trace(C, axis1=1, axis2=2) / 3
        Xs = np.random.default_rng(7).normal(size=Xw.shape) * np.sqrt(s2)[None, :, None]
        fms, _, fas, _, _ = est(Xs)
        recs.append(dict(file=f.split('/')[-1], sweeps=int(t[-1]), frames=len(sel), msd=msd, f_meas=fm.tolist(), f_gauss=fg.tolist(),
                         f_aniso=(fa - (fas - fms)).tolist(), f_iso=float(0)))  # placeholder: the isotropic cage mixture f_isomix is written by plateau_analysis.py (data/restore/plateau/)
    A = lambda key: np.array([r[key] for r in recs])
    out[f"{phi:.2f}"] = dict(n=len(recs), sweeps=[r['sweeps'] for r in recs], msd=[float(A('msd').mean()), float(A('msd').std(ddof=1))],
                             f_meas=[A('f_meas').mean(0).tolist(), A('f_meas').std(0, ddof=1).tolist()],
                             f_gauss=A('f_gauss').mean(0).tolist(), f_aniso=A('f_aniso').mean(0).tolist(), runs=recs)
    o = out[f"{phi:.2f}"]
    print(f"phi={phi:.2f} n={o['n']} sweeps={o['sweeps']} MSD={o['msd'][0]:.4f}±{o['msd'][1]:.4f} f_meas(3,5,7,10)={np.round(o['f_meas'][0],3)} gauss={np.round(o['f_gauss'],3)} aniso_corr={np.round(o['f_aniso'],3)}", flush=True)
json.dump(dict(k=[3, 5, 7, 10], window='t >= 1e3 sweeps to end of run', densities=out), open(args.out, 'w'), indent=1)
