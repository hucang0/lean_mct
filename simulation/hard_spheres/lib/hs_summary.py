#!/usr/bin/env python3
"""Aggregate hs_analysis.py caches over samples: per-sample ratios, mean +- s.e.m. over samples (as
orig_copies/conditional.py), values at tau7 [Fs(k7, tau7) = 1/e] and at the plateau time t* = minimum of the
beta-regime log-slope d ln MSD / d ln t (20 <= t <= tau7, sample-averaged MSD). The MK definition (flattest ln Fs,
orig_copies/analysis.plateau_estimate) is also stored, but in HS it selects the short-time end where ln Fs ~ 0.
Usage: hs_summary.py curves_dir results_dir
Writes results/hs_<group>.json (one per (phi, N, delta) group), results/summary_table.json, results/fig_curves.json."""
import os, sys, glob, json
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis as A

PD = A.PER_DECADE
NC_MAX = 10


def per_sample_ratio(rs, fn, den_min=0.15, den_key="Fs7"):
    bins = {}
    for r in rs:
        for j, lag in enumerate(r["lag"]):
            if den_key is not None and r[den_key][j] <= den_min:
                continue
            val = fn(r, j)
            if val is None or not np.isfinite(val):
                continue
            b = int(np.rint(PD * np.log10(lag)))
            bins.setdefault(b, []).append((lag, val))
    out = []
    for b in sorted(bins):
        arr = np.array(bins[b]); n = len(arr)
        out.append((float(np.exp(np.log(arr[:, 0]).mean())), float(arr[:, 1].mean()),
                    float(arr[:, 1].std(ddof=1) / np.sqrt(n)) if n > 1 else None, n))
    return out


def at(curve, t):
    if not curve or t is None:
        return None
    lags = np.array([c[0] for c in curve])
    if lags.max() < 0.9 * t or lags.min() > 1.1 * t:
        return None
    j = int(np.argmin(np.abs(np.log(lags) - np.log(t))))
    return {"t": curve[j][0], "mean": curve[j][1], "sem": curve[j][2], "n": curve[j][3]}


def cx(r, lab, part, j):
    return r[f"{lab}_{part}_re"][j] + 1j * r[f"{lab}_{part}_im"][j]


def nc_group(r, j, lo, hi):
    s = {q: 0j for q in ("pm", "Fi", "Fj")}; c = 0.0
    for n in range(lo, hi + 1):
        c += r[f"nc{n}_cnt"][j]
        for q in s:
            s[q] += r[f"nc{n}_{q}_sre"][j] + 1j * r[f"nc{n}_{q}_sim"][j]
    if c <= 0:
        return None
    return {q: v / c for q, v in s.items()}


def observables(rs):
    obs = {}
    F2 = lambda r, j, key="Fs7": r[key][j] ** 2
    for lab, fk in (("L", "Fs7"), ("T", "Fs7"), ("L5", "Fs5"), ("L10", "Fs10"), ("nL", "Fs7"), ("nT", "Fs7"), ("gL", "Fs7"), ("gT", "Fs7"),
                    ("sL", "Fs7"), ("sT", "Fs7")):
        obs[f"{lab}-:R_glob"] = per_sample_ratio(rs, lambda r, j, l=lab, k=fk: r[f"{l}_pm_re"][j] / F2(r, j, k), den_key=fk)
        obs[f"{lab}-:R_marg"] = per_sample_ratio(rs, lambda r, j, l=lab, k=fk: (cx(r, l, "Fi", j) * np.conj(cx(r, l, "Fj", j))).real / F2(r, j, k), den_key=fk)
        obs[f"{lab}-:ReC"] = per_sample_ratio(rs, lambda r, j, l=lab: (cx(r, l, "pm", j) / (cx(r, l, "Fi", j) * np.conj(cx(r, l, "Fj", j)))).real, den_key=fk)
        obs[f"{lab}-:Im_phi2"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_pm_im"][j], den_key=fk)
        obs[f"{lab}-:Im_prod"] = per_sample_ratio(rs, lambda r, j, l=lab: (cx(r, l, "Fi", j) * np.conj(cx(r, l, "Fj", j))).imag, den_key=fk)
        obs[f"{lab}:Fi_im"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_Fi_im"][j], den_key=fk)
        obs[f"{lab}+:R_glob"] = per_sample_ratio(rs, lambda r, j, l=lab, k=fk: r[f"{l}_pp_re"][j] / F2(r, j, k), den_key=fk)
        obs[f"{lab}+:ReC"] = per_sample_ratio(rs, lambda r, j, l=lab: (cx(r, l, "pp", j) / (cx(r, l, "Fi", j) * cx(r, l, "Fj", j))).real, den_key=fk)
    for lab in ("I", "nI", "gI", "sI"):
        for s in ("pm", "pp"):
            obs[f"{lab}{s}:R_glob"] = per_sample_ratio(rs, lambda r, j, l=lab, s=s: r[f"{l}_{s}_re"][j] / r["Fs7"][j] ** 2)
            obs[f"{lab}{s}:R_marg"] = per_sample_ratio(rs, lambda r, j, l=lab, s=s: r[f"{l}_prod_{s}_re"][j] / r["Fs7"][j] ** 2)
            obs[f"{lab}{s}:C"] = per_sample_ratio(rs, lambda r, j, l=lab, s=s: r[f"{l}_{s}_re"][j] / r[f"{l}_prod_{s}_re"][j])
    for lab in ("L", "T", "nL", "nT", "gL", "gT", "sL", "sT", "rL"):
        def pear(r, j, l=lab):
            ci = r[f"{l}_uij"][j] - r[f"{l}_ui"][j] * r[f"{l}_uj"][j]
            vi = r[f"{l}_uii"][j] - r[f"{l}_ui"][j] ** 2; vj = r[f"{l}_ujj"][j] - r[f"{l}_uj"][j] ** 2
            return ci / np.sqrt(vi * vj)
        obs[f"{lab}:pearson"] = per_sample_ratio(rs, pear, den_key=None)
        obs[f"{lab}:mean_rel"] = per_sample_ratio(rs, lambda r, j, l=lab: r[f"{l}_ui"][j] - r[f"{l}_uj"][j], den_key=None)
    # Gaussian-equivalent displacement correlation from R: rho_G = ln R / (2 ln(1/Fs))  (exact for Gaussian, iid-marginals)
    for lab, num in (("L", "L_pm_re"), ("T", "T_pm_re"), ("I", "I_pm_re")):
        obs[f"{lab}:rhoG"] = per_sample_ratio(rs, lambda r, j, n=num: np.log(r[n][j] / r["Fs7"][j] ** 2) / (2 * np.log(1 / r["Fs7"][j])) if r[n][j] > 0 and r["Fs7"][j] < 0.999 else None)
    obs["rand-:R"] = per_sample_ratio(rs, lambda r, j: r["r_pm"][j] / r["Fs7"][j] ** 2)
    obs["rand+:R"] = per_sample_ratio(rs, lambda r, j: r["r_pp"][j] / r["Fs7"][j] ** 2)
    obs["randL-:R"] = per_sample_ratio(rs, lambda r, j: r["rL_pm"][j] / r["Fs7"][j] ** 2)
    obs["anisoL/T"] = per_sample_ratio(rs, lambda r, j: r["L_pm_re"][j] / r["T_pm_re"][j])
    for q in ("end105", "end12"):
        obs[q] = per_sample_ratio(rs, lambda r, j, q=q: r[q][j], den_key=None)
    for q in ("Fs3", "Fs5", "Fs7", "Fs10", "msd"):
        obs[q] = per_sample_ratio(rs, lambda r, j, q=q: r[q][j], den_key=None)
    obs["alpha2"] = per_sample_ratio(rs, lambda r, j: 3 * r["r4"][j] / (5 * r["msd"][j] ** 2) - 1, den_key=None)
    # common-neighbour groups (parallel, '-'): n_c <= 3 ('few'), n_c = 4, n_c >= 5 ('many'), n_c >= 6
    h = np.sum([r["nc_hist"] for r in rs], 0)
    groups = {"few": (0, 3), "nc4": (4, 4), "many": (5, NC_MAX), "ge6": (6, NC_MAX)}
    for g, (a, b) in groups.items():
        def fR(r, j, a=a, b=b):
            q = nc_group(r, j, a, b); return None if q is None else q["pm"].real / r["Fs7"][j] ** 2
        def fC(r, j, a=a, b=b):
            q = nc_group(r, j, a, b); return None if q is None else (q["pm"] / (q["Fi"] * np.conj(q["Fj"]))).real
        def fI(r, j, a=a, b=b):
            q = nc_group(r, j, a, b); return None if q is None else q["pm"].imag
        def fP(r, j, a=a, b=b):
            c = sum(r[f"nc{n}_cnt"][j] for n in range(a, b + 1))
            if c <= 0:
                return None
            m = {q: sum(r[f"nc{n}_{q}_s"][j] for n in range(a, b + 1)) / c for q in ("ui", "uj", "uii", "ujj", "uij")}
            return (m["uij"] - m["ui"] * m["uj"]) / np.sqrt((m["uii"] - m["ui"] ** 2) * (m["ujj"] - m["uj"] ** 2))
        obs[f"nc_{g}:pearson"] = per_sample_ratio(rs, fP, den_key=None)
        obs[f"nc_{g}:R_glob"] = per_sample_ratio(rs, fR)
        obs[f"nc_{g}:ReC"] = per_sample_ratio(rs, fC)
        obs[f"nc_{g}:Im_phi2"] = per_sample_ratio(rs, fI)
    meta_nc = {"hist": h.tolist(), "mean": float(np.average(np.arange(len(h)), weights=h)),
               "groups": {g: list(v) for g, v in groups.items()},
               "frac": {g: float(h[a:b + 1].sum() / h.sum()) for g, (a, b) in groups.items()}}
    return obs, meta_nc


def main():
    cdir, rdir = sys.argv[1], sys.argv[2]
    os.makedirs(rdir, exist_ok=True)
    runs = [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(cdir, "curves_*.npz")))]
    groups = {}
    for r in runs:
        key = f"phi{float(r['phi']):.3f}_N{int(r['N'])}_d{float(r['delta']):.3f}"
        groups.setdefault(key, []).append(r)
    table, figc = {}, {}
    for key, rs in sorted(groups.items()):
        obs, meta_nc = observables(rs)
        lagF, mF, sF, _ = A.average_group(rs, "Fs7")
        tau_ps = [A.crossing_time(r["lag"], r["Fs7"]) for r in rs]
        tau_ok = [t for t in tau_ps if t is not None]
        tau7 = A.crossing_time(lagF, mF)
        pl = A.plateau_estimate(lagF, mF, tmin=20.0)          # MK definition (flattest ln Fs); in HS it picks the
        # short-time end (ln Fs ~ 0 there), so the HS plateau time is the beta-regime minimum of the MSD log-slope:
        lagM, mM, _, _ = A.average_group(rs, "msd")
        lt = np.log(lagM); lm = np.convolve(np.log(mM), np.ones(3) / 3, mode="same"); lm[0], lm[-1] = np.log(mM[0]), np.log(mM[-1])
        slope = np.gradient(lm, lt)
        win = (lagM >= 20) & (lagM <= (tau7 if tau7 else lagM.max())) & (np.arange(len(lagM)) > 0) & (np.arange(len(lagM)) < len(lagM) - 2)
        if win.sum() >= 3:
            iw = np.nonzero(win)[0][np.argmin(slope[win])]
            tstar, slope_star = float(lagM[iw]), float(slope[iw])
        else:
            tstar, slope_star = None, None
        # matched-F_s comparison point: F_s(k7, t) = 0.448, the MK phi = 2.0 plateau value at t = 1e4 sweeps
        times = {"tau7": tau7, "tstar": tstar, "tF045": A.crossing_time(lagF, mF, level=0.448)}
        ent = {"phi": float(rs[0]["phi"]), "N": int(rs[0]["N"]), "delta": float(rs[0]["delta"]), "n_samples": len(rs),
               "sweeps": [int(r["sweeps"]) for r in rs], "acc": float(np.mean([r["acc"] for r in rs])),
               "tau7": tau7, "tau7_per_sample": tau_ok,
               "tau7_sem": float(np.std(tau_ok, ddof=1) / np.sqrt(len(tau_ok))) if len(tau_ok) > 1 else None,
               "plateau_MKdef": {"f_s7": pl[0], "t": pl[1], "dlnF_dlog10t": pl[2]} if pl else None,
               "plateau": {"t_star": tstar, "dlnMSD_dlnt_min": slope_star,
                           "f_s7": float(np.interp(np.log(tstar), np.log(lagF), mF)) if tstar else None,
                           "msd": float(np.interp(np.log(tstar), lt, mM)) if tstar else None},
               "Z_mean": float(np.mean(np.concatenate([np.atleast_2d(r["Z"])[:, 1] for r in rs]))),
               "contacts_per_particle": float(np.mean([r["n_contacts"] for r in rs]) / rs[0]["N"]),
               "n_common": meta_nc, "kmag": rs[0]["kmag"].tolist(), "values": {}}
        # aging: Fs7 of first vs last origin near tau7
        ag = []
        for r in rs:
            f_, l_ = r["aging_first"], r["aging_last"]
            tt = tau7 if tau7 else lagF[-1] / 10
            if l_[:, 0].max() >= 0.9 * tt:
                i1 = np.argmin(np.abs(np.log(f_[:, 0] / tt))); i2 = np.argmin(np.abs(np.log(l_[:, 0] / tt)))
                ag.append([f_[i1, 1], l_[i2, 1]])
        ent["aging_Fs7_first_last_at_tau7"] = np.array(ag).tolist()
        for tn, tv in times.items():
            if tv is not None:
                ent["values"][tn] = {k: at(c, tv) for k, c in obs.items()}
        ent["curves"] = {k: [[c[0], c[1], c[2]] for c in v] for k, v in obs.items()}
        json.dump(ent, open(os.path.join(rdir, f"hs_{key}.json"), "w"), indent=1)
        g = lambda tn, k: (ent["values"].get(tn, {}).get(k) or {})
        row = {"at": {}}
        for tn in times:
            row["at"][tn] = {k: ([g(tn, k).get("mean"), g(tn, k).get("sem")] if g(tn, k) else None) for k in
                       ("L-:R_glob", "T-:R_glob", "Ipm:R_glob", "rand-:R", "randL-:R", "L-:R_marg", "L-:ReC", "T-:ReC",
                        "Ipm:C", "L-:Im_phi2", "L-:Im_prod", "end105", "Fs7", "msd", "alpha2", "gL-:R_glob", "gL-:ReC",
                        "gT-:R_glob", "gIpm:R_glob", "sL-:R_glob", "randL-:R", "sL-:ReC", "sIpm:R_glob", "nc_few:R_glob", "nc_many:R_glob",
                        "nc_few:ReC", "nc_many:ReC", "L5-:R_glob", "L5-:ReC", "L10-:R_glob", "L10-:ReC", "anisoL/T",
                        "L+:R_glob", "T+:R_glob", "Ipp:R_glob", "L:pearson", "T:pearson", "gL:pearson", "sL:pearson", "sT:pearson",
                        "rL:pearson", "nL:pearson", "nT:pearson", "nL-:R_glob", "nT-:R_glob", "nL-:ReC", "nIpm:R_glob", "gT:pearson",
                        "nc_few:pearson", "nc_many:pearson", "nc_nc4:pearson", "nc_ge6:pearson", "nc_nc4:R_glob", "nc_ge6:R_glob",
                        "nc_nc4:ReC", "nc_ge6:ReC", "nc_few:Im_phi2", "nc_many:Im_phi2", "L-:ImC" if False else "T-:Im_phi2", "L:rhoG", "T:rhoG", "I:rhoG", "L:mean_rel", "gL:mean_rel", "gL-:Im_phi2", "sT-:R_glob", "rL-:R_glob")}
            row["at"][tn]["t"] = times[tn]
        row.update({k: ent[k] for k in ("phi", "N", "delta", "n_samples", "acc", "tau7", "tau7_sem", "plateau", "Z_mean",
                                         "contacts_per_particle")})
        row["n_common_groups"] = meta_nc
        table[key] = row
        figc[key] = {k: ent["curves"][k] for k in ("L-:R_glob", "T-:R_glob", "Ipm:R_glob", "rand-:R", "L-:ReC", "L-:Im_phi2",
                                                   "L-:Im_prod", "Fs7", "msd", "end105", "gL-:R_glob", "sL-:R_glob", "L:pearson", "T:pearson", "sL:pearson", "T-:ReC")}
        figc[key]["tau7"] = tau7
        figc[key]["tstar"] = tstar
        def fmt(tn, k):
            v = g(tn, k)
            return f"{v['mean']:.2f}±{(v['sem'] or 0):.2f}" if v else "  -  "
        print(f"{key} n={len(rs)} acc={ent['acc']:.3f} tau7={tau7} plateau={ent['plateau']}")
        for tn in times:
            print(f"   {tn:5s}: R||={fmt(tn,'L-:R_glob')} Rperp={fmt(tn,'T-:R_glob')} Riso={fmt(tn,'Ipm:R_glob')} "
                  f"Rrand={fmt(tn,'rand-:R')} Rmarg={fmt(tn,'L-:R_marg')} ReC={fmt(tn,'L-:ReC')} ReCperp={fmt(tn,'T-:ReC')} "
                  f"Imphi2={fmt(tn,'L-:Im_phi2')} Improd={fmt(tn,'L-:Im_prod')} Pc={fmt(tn,'end105')} | gap R||={fmt(tn,'gL-:R_glob')} "
                  f"ReC={fmt(tn,'gL-:ReC')} 2nd R||={fmt(tn,'sL-:R_glob')} | nc few/many R||={fmt(tn,'nc_few:R_glob')}/{fmt(tn,'nc_many:R_glob')} "
                  f"ReC={fmt(tn,'nc_few:ReC')}/{fmt(tn,'nc_many:ReC')} | k5 {fmt(tn,'L5-:R_glob')} k10 {fmt(tn,'L10-:R_glob')}")
            print(f"          pearson ||={fmt(tn,'L:pearson')} perp={fmt(tn,'T:pearson')} gap||={fmt(tn,'gL:pearson')} 2nd||={fmt(tn,'sL:pearson')} "
                  f"2nd perp={fmt(tn,'sT:pearson')} n||={fmt(tn,'nL:pearson')} nperp={fmt(tn,'nT:pearson')} nc few/many={fmt(tn,'nc_few:pearson')}/{fmt(tn,'nc_many:pearson')} rand||={fmt(tn,'rL:pearson')} | rhoG ||={fmt(tn,'L:rhoG')} perp={fmt(tn,'T:rhoG')} iso={fmt(tn,'I:rhoG')} "
                  f"| mean rel ||={fmt(tn,'L:mean_rel')} gap={fmt(tn,'gL:mean_rel')} | 2nd perp R={fmt(tn,'sT-:R_glob')} Fs7={fmt(tn,'Fs7')} MSD={fmt(tn,'msd')}")
    json.dump(table, open(os.path.join(rdir, "summary_table.json"), "w"), indent=1)
    json.dump(figc, open(os.path.join(rdir, "fig_curves.json"), "w"))


if __name__ == "__main__":
    main()
