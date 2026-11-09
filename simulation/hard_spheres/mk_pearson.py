#!/usr/bin/env python3
"""Read-only re-analysis of the MK planted-MC trajectories (mk_mc.py output) with the HS bond
observables of hs_analysis.py: (1) validation -- must reproduce the published R_par, R_perp, Re C (SM Tables S11-S12);
(2) the model-free Pearson coefficient of bond-projected displacements, parallel / transverse, which is not in the SM.
Contacts, k sets, origins, binning from analysis.py (identical copy of ../mk_planted_mc/analysis.py; shifted metric
x_i - x_j - A_ij). Uses the N = 1000 runs at phi = 0.80, 1.50, 1.80, 2.00.
Usage: mk_pearson.py out.json --mk-data DIR --tau-table FILE [--N 1000]   (NPROC=6 worker processes by default)
  DIR: folder with the MK run_phi*_N1000_s*.npz; FILE: tau_table.json written by ../mk_planted_mc/analysis.py."""
import os, sys, glob, json
import numpy as np
from multiprocessing import Pool
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A
from hs_analysis import bond_obs, perp_of, iso_obs
from hs_summary import per_sample_ratio, at, cx


def process(path):
    d = np.load(path)
    traj = d["traj"].astype(np.float64); t_save = d["t_save"].astype(np.int64); Ash = d["A"].astype(np.float64)
    L, N, phi = float(d["L"]), int(d["N"]), float(d["phi"]); T = int(d["sweeps"])
    origins = d["origins"].astype(np.int64); origins = origins[origins < t_save[-1]]
    K7, k7 = A.lattice_kvecs(7.0, L)
    acc = A.Accum(int(np.ceil(A.PER_DECADE * np.log10(T))) + 2)
    rng = np.random.default_rng(12345)
    for t0 in origins:
        i0 = int(np.searchsorted(t_save, t0)); x0 = traj[i0]
        pairs, rhat, rmin, rvec = A.find_contacts(x0, Ash, L)
        ci, cj = pairs[:, 0], pairs[:, 1]
        ri = rng.integers(0, N, 4000); rj = (ri + rng.integers(1, N, 4000)) % N
        dr = x0[ri] - x0[rj] - Ash[ri, rj]; dr -= L * np.rint(dr / L); rrhat = dr / np.linalg.norm(dr, axis=1)[:, None]
        for f in range(i0 + 1, len(t_save)):
            lag = int(t_save[f] - t0); b = int(np.rint(A.PER_DECADE * np.log10(lag)))
            dx = traj[f] - x0; dx = dx - dx.mean(0)
            v = {"Fs7": np.cos(dx @ K7.T).mean()}
            bond_obs(v, "L", dx, ci, cj, k7 * rhat)
            bond_obs(v, "T", dx, ci, cj, k7 * perp_of(rhat))
            bond_obs(v, "rL", dx, ri, rj, k7 * rrhat)
            iso_obs(v, "I", dx, ci, cj, K7)
            acc.add(b, lag, **v)
    res = acc.result(); res["phi"] = phi; res["N"] = N
    return res


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MK re-analysis with the HS bond observables")
    ap.add_argument("out"); ap.add_argument("--mk-data", required=True); ap.add_argument("--tau-table", required=True)
    ap.add_argument("--N", type=int, default=1000, help="system size in the trajectory file names")
    a = ap.parse_args(); MKD, TAU = a.mk_data, a.tau_table
    files = []
    for p in ("0.80", "1.50", "1.80", "2.00"):
        files += sorted(glob.glob(f"{MKD}/run_phi{p}_N{a.N}_s*.npz"))
    with Pool(int(os.environ.get("NPROC", "6"))) as pool:
        runs = pool.map(process, files)
    taus = json.load(open(TAU))["tau"]
    out = {}
    for p in sorted(set(round(r["phi"], 3) for r in runs)):
        rs = [r for r in runs if round(r["phi"], 3) == p]
        obs = {"L-:R_glob": per_sample_ratio(rs, lambda r, j: r["L_pm_re"][j] / r["Fs7"][j] ** 2),
               "T-:R_glob": per_sample_ratio(rs, lambda r, j: r["T_pm_re"][j] / r["Fs7"][j] ** 2),
               "Ipm:R_glob": per_sample_ratio(rs, lambda r, j: r["I_pm_re"][j] / r["Fs7"][j] ** 2),
               "L-:ReC": per_sample_ratio(rs, lambda r, j: (cx(r, "L", "pm", j) / (cx(r, "L", "Fi", j) * np.conj(cx(r, "L", "Fj", j)))).real),
               "T-:ReC": per_sample_ratio(rs, lambda r, j: (cx(r, "T", "pm", j) / (cx(r, "T", "Fi", j) * np.conj(cx(r, "T", "Fj", j)))).real),
               "L-:Im_phi2": per_sample_ratio(rs, lambda r, j: r["L_pm_im"][j]),
               "Fs7": per_sample_ratio(rs, lambda r, j: r["Fs7"][j], den_key=None)}
        for lab in ("L", "T", "rL"):
            def pear(r, j, l=lab):
                c = r[f"{l}_uij"][j] - r[f"{l}_ui"][j] * r[f"{l}_uj"][j]
                return c / np.sqrt((r[f"{l}_uii"][j] - r[f"{l}_ui"][j] ** 2) * (r[f"{l}_ujj"][j] - r[f"{l}_uj"][j] ** 2))
            obs[f"{lab}:pearson"] = per_sample_ratio(rs, pear, den_key=None)
        for lab, num in (("L", "L_pm_re"), ("T", "T_pm_re"), ("I", "I_pm_re")):
            obs[f"{lab}:rhoG"] = per_sample_ratio(rs, lambda r, j, n=num: np.log(r[n][j] / r["Fs7"][j] ** 2) / (2 * np.log(1 / r["Fs7"][j])) if r[n][j] > 0 else None)
        key = f"{p:g}"
        tau7 = taus.get(key if a.N == 1000 else f"{key}_N{a.N}", {}).get("tau_k7")
        out[key] = {"tau7": tau7, "n_samples": len(rs), "values": {tn: {k: at(c, tv) for k, c in obs.items()}
                    for tn, tv in (("tau7", tau7), ("t1e4", 1e4), ("t100", 1e2)) if tv},
                    "curves": {k: [[c[0], c[1], c[2]] for c in v] for k, v in obs.items()}}
        for tn, vals in out[key]["values"].items():
            print(f"MK phi={key} {tn}: " + " ".join(f"{k}={vals[k]['mean']:.3f}±{(vals[k]['sem'] or 0):.3f}" for k in vals if vals[k]))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
