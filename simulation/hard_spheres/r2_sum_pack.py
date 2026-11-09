#!/usr/bin/env python3
"""Aggregate cache/packdyn_*.npz and cache/struct_*.npz -> results/packing.json.
For each group, packing measure and packing bin: contact pairs grouped by n_c (<=3, =4, >=5, >=6 as hs_summary);
per sample: R_par = Re phi2/Fs^2, R_marg_par = Re(F_i F_j*)/Fs^2, ReGamma_par = Re[phi2/(F_i F_j*)], Pearson_par, and R_perp,
then mean +- s.e.m. over samples at t* and tau7 (and tF045). Structural stats: Spearman(n_c, measure), mean measure per n_c
group, joint occupation of (n_c group, bin)."""
import os, sys, glob, json
sys.dont_write_bytecode = True
import numpy as np
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C
import r2_pack as PK

NCG = {"le3": (0, 3), "eq4": (4, 4), "ge5": (5, 10), "ge6": (6, 10), "all": (0, 10)}


def obs_from_sums(s, Fs):
    """s: summed fields (DF) over a set of bins -> dict."""
    I = {f: n for n, f in enumerate(PK.DF)}
    n = s[I["cnt"]]
    if n <= 0:
        return None
    m = lambda f: s[I[f]] / n
    out = {"n": n}
    for d in ("L", "T"):
        pm = m(f"{d}_pm_re") + 1j * m(f"{d}_pm_im")
        Fi = m(f"{d}_Fi_re") + 1j * m(f"{d}_Fi_im"); Fj = m(f"{d}_Fj_re") + 1j * m(f"{d}_Fj_im")
        prod = Fi * np.conj(Fj)
        out[f"R_{d}"] = pm.real / Fs ** 2
        out[f"Rm_{d}"] = prod.real / Fs ** 2
        out[f"ReG_{d}"] = (pm / prod).real
        ui, uj, uii, ujj, uij = m(f"{d}_ui"), m(f"{d}_uj"), m(f"{d}_uii"), m(f"{d}_ujj"), m(f"{d}_uij")
        out[f"P_{d}"] = (uij - ui * uj) / np.sqrt((uii - ui ** 2) * (ujj - uj ** 2))
    return out


def main():
    times = json.load(open(C.result_path("times.json")))
    th = json.load(open(C.result_path("pack_thresholds.json")))
    groups = {}
    for f in C.require_files(sorted(glob.glob(os.path.join(C.CACHE, "packdyn_*.npz"))), "R2 caches (run the corresponding trajectory analysis first)"):
        d = dict(np.load(f)); d = {k: (v.item() if v.shape == () else v) for k, v in d.items()}
        groups.setdefault(str(d["key"]), []).append(d)
    out = {}
    for key, rs in sorted(groups.items()):
        meas = list(rs[0]["measures"]); nbs = list(rs[0]["nbins"])
        offs = np.concatenate([[0], np.cumsum([11 * nb for nb in nbs])])
        tk = times[key]
        ent = {"n_samples": len(rs), "at": {}}
        for tn in ("tstar", "tau7", "tF045"):
            tv = tk.get(tn)
            if not tv:
                continue
            ent["at"][tn] = {}
            for im, mname in enumerate(meas):
                nb = nbs[im]
                res = {}
                for g, (lo, hi) in NCG.items():
                    for pb in list(range(nb)) + ["all"]:
                        vals = []
                        for r in rs:
                            j = C.value_at(r["lag"], None, tv)
                            if j is None:
                                continue
                            Fs = r["Fs7"][j] / r["nframes"][j]
                            S = r["S"][j]
                            rows = [offs[im] + nc * nb + p for nc in range(lo, hi + 1) for p in (range(nb) if pb == "all" else [pb])]
                            o = obs_from_sums(S[rows].sum(0), Fs)
                            if o is not None:
                                vals.append(o)
                        if not vals:
                            continue
                        res[f"{g}|{pb}"] = {q: list(C.sample_mean_sem([v[q] for v in vals])[:2]) for q in vals[0]}
                ent["at"][tn][mname] = res
        # structure
        sts = [np.load(f) for f in sorted(glob.glob(os.path.join(C.CACHE, "struct_*.npz"))) if str(np.load(f)["key"]) == key]
        coef = {k: np.array(v) for k, v in th[key]["coef"].items()}
        pm = [PK.pair_measures(s, coef) for s in sts]
        nc = np.concatenate([s["nc"] for s in sts])
        stru = {"mean_nc": float(nc.mean()), "nfix": int(sum(int(s["nfix"]) for s in sts)), "vsum": float(np.mean([s["vsum"].mean() for s in sts]))}
        for mname, var, nb in PK.MEASURES:
            if var is None:
                continue
            v = np.concatenate([p[var] for p in pm])
            rho = spearmanr(nc, v).correlation
            ed = np.array(th[key][mname])
            pb = np.searchsorted(ed, v, side="right")
            joint = {g: np.bincount(pb[(nc >= lo) & (nc <= hi)], minlength=nb).tolist() for g, (lo, hi) in NCG.items()}
            stru[mname] = {"spearman_nc": float(rho), "mean_by_ncg": {g: float(v[(nc >= lo) & (nc <= hi)].mean()) for g, (lo, hi) in NCG.items()},
                           "sd": float(v.std()), "joint_counts": joint, "edges": ed.tolist()}
        sij = np.concatenate([p["sij"] for p in pm])
        stru["spearman_nc_sij"] = float(spearmanr(nc, sij).correlation)
        stru["spearman_densrad_sij"] = float(spearmanr(np.concatenate([p["dens_rad"] for p in pm]), sij).correlation)
        stru["spearman_phirad_sij"] = float(spearmanr(np.concatenate([p["phi_rad"] for p in pm]), sij).correlation)
        ent["struct"] = stru
        out[key] = ent
        print(key, "spearman(nc, dens_rad)=%.3f (nc, phi_rad)=%.3f (nc, sij)=%.3f" % (stru["radc3"]["spearman_nc"], stru["rad3"]["spearman_nc"], stru["spearman_nc_sij"]), flush=True)
    C.jdump(out, os.path.join(C.RES, "packing.json"))


if __name__ == "__main__":
    main()
