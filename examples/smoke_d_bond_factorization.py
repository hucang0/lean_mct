#!/usr/bin/env python3
"""Smoke test (d): the bond-factorization analysis end to end, on tiny freshly simulated data (no trajectories
are shipped). CPU only, about 20 s (numba compilation included). Work files go to examples/_work/ (git-ignored).

MK planted MC (simulation/mk_planted_mc): two planted samples, N = 300, phi = 1.5, 3000 sweeps
  mk_mc.py -> analysis.py (tau_table.json, curves) -> conditional.py (R, R_marg, Re C along the bond)
  and the independent re-analysis code ../hard_spheres/mk_pearson.py on the same files.
  Reference (SM Tables S11-S12, N = 1000, 3 samples, phi = 1.5, t = tau_7): R_par = 2.54(3), Re C_par = 1.81(2).
Ordinary hard spheres (simulation/hard_spheres): two samples, N = 300, phi = 0.55
  hs_equil.py (2000 swap-MC sweeps) -> hs_dyn.py (3000 MC sweeps) -> hs_analysis.py -> hs_summary.py.
  Reference (SM Table S14, N = 1000, 6 samples, phi = 0.55): tau_7 = 2.7e3 sweeps, R_par(tau_7) = 4.07(5),
  Re C_par(tau_7) = 2.61(3), R_par(F_s = 0.45) = 3.22(5).
Tolerances are wide because the test systems are small and short; the point is that the pipeline runs and lands
on the published numbers."""
import json, os, shutil, subprocess, sys
from _check import ROOT, check, require, finish

W = ROOT / "examples" / "_work"
for sub in ("mk", "hs"):
    shutil.rmtree(W / sub, ignore_errors=True)
MK, HS = ROOT / "simulation" / "mk_planted_mc", ROOT / "simulation" / "hard_spheres"
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", NUMBA_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NPROC="2")


def run(*args):
    p = subprocess.run([sys.executable, *map(str, args)], env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:]); sys.exit(f"command failed: {args[0]}")
    return p.stdout


# ---- MK planted Monte Carlo
md = W / "mk" / "data"; md.mkdir(parents=True)
for s in (1, 2):
    run(MK / "mk_mc.py", "--N", 300, "--phi", 1.5, "--sweeps", 3000, "--norig", 4, "--seed", s, "--delta", 0.10,
        "--out", md / f"run_phi1.50_N300_s{s}.npz")
run(MK / "analysis.py", "--data", md, "--out", W / "mk" / "results", "--figures", W / "mk" / "figures", "--nproc", 2, "--no-plots")
run(MK / "conditional.py", "--data", md, "--tau-table", W / "mk" / "results" / "tau_table.json", "--out", W / "mk" / "reanalysis", "--nproc", 2)
run(HS / "mk_pearson.py", W / "mk" / "pearson.json", "--mk-data", md, "--tau-table", W / "mk" / "results" / "tau_table.json", "--N", 300)
C = json.load(open(W / "mk" / "reanalysis" / "conditional_summary.json"))["1.5_N300"]
Pn = json.load(open(W / "mk" / "pearson.json"))["1.5"]
v = C["values"]["tau7"]
print(f"MK phi=1.5 N=300: tau_7 = {C['tau7']:.0f} sweeps; R_par = {v['L-:R_glob']['mean']:.3f}, R_perp = {v['T-:R_glob']['mean']:.3f}, "
      f"R_iso = {v['Ipm:R_glob']['mean']:.3f}, R_marg_par = {v['L-:R_marg']['mean']:.3f}, Re C_par = {v['L-:ReC']['mean']:.3f}")
check("MK R_par(tau7)", v["L-:R_glob"]["mean"], 2.54, 0.4)
check("MK Re C_par(tau7)", v["L-:ReC"]["mean"], 1.81, 0.3)
require("MK R_perp < 1 < R_iso < R_par", v["T-:R_glob"]["mean"] < 1.0 < v["Ipm:R_glob"]["mean"] < v["L-:R_glob"]["mean"])
check("MK R_par: conditional.py vs mk_pearson.py", Pn["values"]["tau7"]["L-:R_glob"]["mean"], v["L-:R_glob"]["mean"], 1e-9)

# ---- ordinary polydisperse hard spheres
for s in (1, 2):
    eq = W / "hs" / "eq" / f"eq_phi0.550_N300_s{s}.npz"
    run(HS / "hs_equil.py", "--N", 300, "--phi", 0.55, "--seed", s, "--neq", 2000, "--out", eq)
    run(HS / "hs_dyn.py", "--init", eq, "--sweeps", 3000, "--delta", 0.05, "--norig", 4, "--seed", s,
        "--out", W / "hs" / "dyn" / f"run_phi0.550_N300_d0.050_s{s}.npz")
run(HS / "hs_analysis.py", W / "hs" / "curves", *sorted((W / "hs" / "dyn").glob("run_*.npz")))
run(HS / "hs_summary.py", W / "hs" / "curves", W / "hs" / "results")
T = json.load(open(W / "hs" / "results" / "summary_table.json"))["phi0.550_N300_d0.050"]
a7, a45 = T["at"]["tau7"], T["at"]["tF045"]
print(f"HS phi=0.55 N=300: tau_7 = {T['tau7']:.0f} sweeps; R_par(tau7) = {a7['L-:R_glob'][0]:.3f}, Re C_par(tau7) = {a7['L-:ReC'][0]:.3f}, "
      f"R_par(Fs=0.45) = {a45['L-:R_glob'][0]:.3f}, corr_par(tau7) = {a7['L:pearson'][0]:.3f}")
check("HS tau7 / 1e3", T["tau7"] / 1e3, 2.7, 0.8)
check("HS R_par(tau7)", a7["L-:R_glob"][0], 4.07, 0.6)
check("HS Re C_par(tau7)", a7["L-:ReC"][0], 2.61, 0.4)
check("HS R_par(Fs=0.45)", a45["L-:R_glob"][0], 3.22, 0.5)
finish()
