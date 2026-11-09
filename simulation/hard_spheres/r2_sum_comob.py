#!/usr/bin/env python3
"""Aggregate cache/comob_*.npz: time markers, R(r,t) curves (per-sample ratios, mean +- s.e.m. over samples), variance
decomposition, mobility correlations, random-pair controls, and decay-length fits.  Writes results/comob.json,
results/times.json, results/xi_fits.json.

Ratios per sample and lag bin (pairs x frames pooled inside a sample; Fs = frame-averaged global Fs7 of that sample):
  direction d in L (bond), T (average of the two transverse directions T1, T2), plus orientation-averaged I:
  R_d = (<cc>+<ss>)/Fs^2,  R_marg_d = (<c_i><c_j> + <s_i><s_j>)/Fs^2,  ReC_d = Re[phi2/(F_i F_j*)],
  Cov_cc,d/Fs^2 = (<cc> - <c_i><c_j>)/Fs^2, Cov_ss,d/Fs^2 likewise  [identity: R - R_marg = (Cov_cc + Cov_ss)/Fs^2],
  Cov_X/Fs^2: covariance of cos(k e1.dx_i) and cos(k e2.dx_j) for ORTHOGONAL transverse directions e1 _|_ e2,
  Cov_qq/Fs^2: covariance of the orientation-averaged self overlaps q_i, q_j,
  c_q = Pearson(q_i, q_j), c_m2 = Pearson(|dx_i|^2, |dx_j|^2)  (standard mobility correlation, 'g4'-type),
  R_pf: Re phi2 normalised by the PER-FRAME global Fs^2 before averaging (removes the origin-to-origin global
        mobility fluctuation, i.e. the k = 0 part of the mobility field, which in a finite box adds a constant to R(r)).
Points with Fs <= 0.15 dropped (as hs_summary).
"""
import os, sys, glob, json
sys.dont_write_bytecode = True
import numpy as np
from scipy.optimize import curve_fit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C

DEN_MIN = 0.15


def load_groups(prefix="comob"):
    groups = {}
    for f in sorted(glob.glob(os.path.join(C.CACHE, f"{prefix}_*.npz"))):
        d = dict(np.load(f))
        d = {k: (v.item() if v.shape == () else v) for k, v in d.items()}
        groups.setdefault(str(d["key"]), []).append(d)
    return groups


def time_markers(rs, key):
    """tau7, t*, tF045 from sample-averaged curves exactly as hs_summary (A.average_group on per-run curves)."""
    for r in rs:
        r["Fs7m"] = r["Fs7"] / r["nframes"]; r["msdm"] = r["msd"] / r["nframes"]
    lagF, mF, _, _ = C.A.average_group([{"lag": r["lag"], "Fs7": r["Fs7m"]} for r in rs], "Fs7")
    tau7 = C.A.crossing_time(lagF, mF)
    lagM, mM, _, _ = C.A.average_group([{"lag": r["lag"], "msd": r["msdm"]} for r in rs], "msd")
    lt = np.log(lagM); lm = np.convolve(np.log(mM), np.ones(3) / 3, mode="same"); lm[0], lm[-1] = np.log(mM[0]), np.log(mM[-1])
    slope = np.gradient(lm, lt)
    win = (lagM >= 20) & (lagM <= (tau7 if tau7 else lagM.max())) & (np.arange(len(lagM)) > 0) & (np.arange(len(lagM)) < len(lagM) - 2)
    tstar = float(lagM[np.nonzero(win)[0][np.argmin(slope[win])]]) if win.sum() >= 3 else None
    t = {"tau7": tau7, "tstar": tstar, "tF045": C.A.crossing_time(lagF, mF, level=0.448)}
    if key.startswith("mk"):
        t["tstar"] = None
        if key == "mk2.00":
            t["t1e4"] = 1e4
    t["Fs_curve"] = [lagF.tolist(), mF.tolist()]
    return t


def derived(r, j, which="S_s"):
    """All derived per-bin quantities for run r, lag bin j; returns dict name -> array over distance bins."""
    S = r[which][j]
    F = C.FI if False else None
    fields = list(r["fields"])
    I = {f: n for n, f in enumerate(fields)}
    cnt = S[:, I["cnt"]]
    ok = cnt > 0
    M = np.where(ok[:, None], S / np.maximum(cnt, 1)[:, None], np.nan)
    Fs = r["Fs7"][j] / r["nframes"][j]
    F2 = Fs * Fs
    g = lambda f: M[:, I[f]]
    out = {"cnt": cnt / r["nframes"][j], "Fs": np.full(len(cnt), Fs)}
    for d in ("L", "T1", "T2"):
        cc, ss, ci, si, cj, sj, im = g(f"{d}_cc"), g(f"{d}_ss"), g(f"{d}_ci"), g(f"{d}_si"), g(f"{d}_cj"), g(f"{d}_sj"), g(f"{d}_impm")
        re = cc + ss
        prod = (ci * cj + si * sj) + 1j * (si * cj - ci * sj)
        out[f"R_{d}"] = re / F2
        out[f"Rm_{d}"] = prod.real / F2
        out[f"ReC_{d}"] = ((re + 1j * im) / prod).real
        out[f"Ccc_{d}"] = (cc - ci * cj) / F2
        out[f"Css_{d}"] = (ss - si * sj) / F2
        out[f"Im_{d}"] = im
        ui, uj, uii, ujj, uij = g(f"{d}_ui"), g(f"{d}_uj"), g(f"{d}_uii"), g(f"{d}_ujj"), g(f"{d}_uij")
        out[f"P_{d}"] = (uij - ui * uj) / np.sqrt((uii - ui ** 2) * (ujj - uj ** 2))
    for q in ("R", "Rm", "ReC", "Ccc", "Css", "P", "Im"):
        out[f"{q}_T"] = 0.5 * (out[f"{q}_T1"] + out[f"{q}_T2"])
    out["Rc_T"] = out["R_T"] - out["Rm_T"]            # = (Ccc_T + Css_T): connected part of R_perp
    out["CX_T"] = (0.5 * (g("X12") + g("X21")) - 0.5 * (g("T1_ci") * g("T2_cj") + g("T2_ci") * g("T1_cj"))) / F2
    out["R_I"] = (g("Icc") + g("Iss")) / F2
    out["Rm_I"] = g("qi") * g("qj") / F2
    out["Cqq"] = (g("qij") - g("qi") * g("qj")) / F2
    out["Rc_I"] = out["R_I"] - out["Rm_I"]
    out["Isame"] = (g("Icc") - g("qij")) / F2          # same-direction excess of <cos cos> over the mobility product
    out["Iss"] = g("Iss") / F2
    out["c_q"] = (g("qij") - g("qi") * g("qj")) / np.sqrt((g("qii") - g("qi") ** 2) * (g("qjj") - g("qj") ** 2))
    out["c_m2"] = (g("mij") - g("mi") * g("mj")) / np.sqrt((g("mii") - g("mi") ** 2) * (g("mjj") - g("mj") ** 2))
    q2 = r["q2"][j] / r["nframes"][j]
    out["g4q"] = (g("qij") - F2) / (q2 - F2)           # global normalisation
    # detour: regression of transverse relative spread on longitudinal relative displacement
    Dp, Dt, DpDt, Dp2 = g("Dpar"), g("Dperp2"), g("DparDperp2"), g("Dpar2")
    out["detour_slope"] = (DpDt - Dp * Dt) / (Dp2 - Dp ** 2)          # d<Dperp^2>/dDpar
    out["detour_rel"] = out["detour_slope"] * np.sqrt(Dp2 - Dp ** 2) / Dt
    out["Dperp2"] = Dt; out["Dpar_sd"] = np.sqrt(Dp2 - Dp ** 2)
    # per-frame normalised
    nf = list(r["norm_fields"]); NI = {f: n for n, f in enumerate(nf)}
    Nw = {"S_s": "N_s", "S_r": "N_r", "S_rand": "N_rand"}[which]
    NS = r[Nw][j] / np.maximum(S[:, I["cnt"]], 1)[:, None]
    out["Rpf_L"] = NS[:, NI["L_cc"]] + NS[:, NI["L_ss"]]
    out["Rpf_T"] = 0.5 * (NS[:, NI["T1_cc"]] + NS[:, NI["T1_ss"]] + NS[:, NI["T2_cc"]] + NS[:, NI["T2_ss"]])
    out["Rpf_I"] = NS[:, NI["Icc"]] + NS[:, NI["Iss"]]
    return out


def aggregate(rs, which):
    """Per-sample derived quantities per lag bin -> mean, sem over samples. Returns lags, dict name -> (mean, sem) arrays
    of shape (nlag, ndist)."""
    per = {}
    for r in rs:
        for j, lag in enumerate(r["lag"]):
            Fs = r["Fs7"][j] / r["nframes"][j]
            if Fs <= DEN_MIN:
                continue
            b = int(np.rint(C.PD * np.log10(lag)))
            per.setdefault(b, []).append((lag, derived(r, j, which)))
    lags, out = [], {}
    bins = sorted(per)
    for b in bins:
        lags.append(float(np.exp(np.mean([np.log(a[0]) for a in per[b]]))))
    names = list(per[bins[0]][0][1].keys())
    for nm in names:
        mean, sem, nsam = [], [], []
        for b in bins:
            V = np.array([a[1][nm] for a in per[b]])
            mean.append(np.nanmean(V, 0)); n = len(V); nsam.append(n)
            sem.append(np.nanstd(V, 0, ddof=1) / np.sqrt(n) if n > 1 else np.full(V.shape[1], np.nan))
        out[nm] = (np.array(mean), np.array(sem))
    return np.array(lags), out, per, bins


# ------------------------------------------------------------------ fits
def f_exp_c(r, A, xi, c):
    return A * np.exp(-r / xi) + c


def f_exp(r, A, xi):
    return A * np.exp(-r / xi)


def f_oz_c(r, A, xi, c):
    return A * np.exp(-r / xi) / r + c


FORMS = {"exp+c": (f_exp_c, [1.0, 1.0, 0.0]), "exp": (f_exp, [1.0, 1.0]), "oz+c": (f_oz_c, [1.0, 1.0, 0.0])}


def fit_curve(x, y, s, form):
    f, p0 = FORMS[form]
    m = np.isfinite(y) & np.isfinite(s)
    if m.sum() < len(p0) + 2:
        return None
    p0 = list(p0); p0[0] = max(abs(y[m][0]), 1e-3) * (np.exp(x[m][0]) if form != "oz+c" else x[m][0] * np.exp(x[m][0]))
    try:
        p, cov = curve_fit(f, x[m], y[m], p0=p0, sigma=np.maximum(s[m], 1e-3), absolute_sigma=False, maxfev=20000,
                           bounds=([-np.inf, 0.05] + ([-np.inf] if len(p0) == 3 else []), [np.inf, 20.0] + ([np.inf] if len(p0) == 3 else [])))
    except Exception:
        return None
    res = (y[m] - f(x[m], *p)) / np.maximum(s[m], 1e-3)
    return {"p": p.tolist(), "xi": float(p[1]), "chi2_dof": float((res ** 2).sum() / max(m.sum() - len(p), 1))}


def jackknife_fit(samples_y, x, form, lo, hi):
    """samples_y: (nsam, ndist) per-sample curves. Fit mean curve (weights = sem), jackknife over samples."""
    Y = np.array(samples_y)
    n = len(Y)
    m = (x >= lo) & (x <= hi)
    xm = x[m]
    def one(Ys):
        y = np.nanmean(Ys, 0)[m]
        s = (np.nanstd(Ys, 0, ddof=1) / np.sqrt(len(Ys)))[m] if len(Ys) > 1 else np.full(m.sum(), 0.01)
        return fit_curve(xm, y, s, form)
    full = one(Y)
    if full is None or n < 3:
        return full
    jk = []
    for k in range(n):
        q = one(np.delete(Y, k, 0))
        if q is not None:
            jk.append(q["xi"])
    jk = np.array(jk)
    if len(jk) >= n - 1:
        full["xi_jk_se"] = float(np.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum()))
    full["window"] = [lo, hi]; full["form"] = form; full["n_bins"] = int(m.sum())
    return full


def main():
    groups = load_groups()
    C.require_files(groups, "R2 co-mobility caches (run r2_comob.py first)")
    times, out, fits = {}, {}, {}
    for key, rs in sorted(groups.items()):
        t = time_markers(rs, key)
        times[key] = {k: v for k, v in t.items() if k != "Fs_curve"}
        times[key]["Fs_curve"] = t["Fs_curve"]
        mk = key.startswith("mk")
        se, re_ = rs[0]["s_edges"], rs[0]["r_edges"]
        xs = 0.5 * (se[1:] + se[:-1]); xs[0] = 1.0
        xr = 0.5 * (re_[1:] + re_[:-1])
        ent = {"n_samples": len(rs), "times": times[key], "s_centres": xs, "r_centres": xr, "at": {}, "curves_contact": {}}
        aggs = {}
        for which in ("S_s", "S_r", "S_rand"):
            lags, agg, per, bins = aggregate(rs, which)
            aggs[which] = (lags, agg, per, bins)
        ent["lags"] = aggs["S_s"][0]
        tl = {k: v for k, v in times[key].items() if k in ("tau7", "tstar", "tF045", "t1e4") and v}
        keep = ["R_L", "R_T", "R_I", "Rm_L", "Rm_T", "Rm_I", "ReC_L", "ReC_T", "Ccc_T", "Css_T", "CX_T", "Cqq", "Isame", "Iss",
                "Ccc_L", "Css_L", "P_L", "P_T", "c_q", "c_m2", "g4q", "Rpf_L", "Rpf_T", "Rpf_I", "Im_L", "Im_T", "cnt",
                "detour_slope", "detour_rel", "Dperp2", "Dpar_sd", "Fs", "Rc_T", "Rc_I", "Rm_L", "ReC_L", "P_T1", "Ccc_T1", "Css_T1"]
        for tn, tv in tl.items():
            ent["at"][tn] = {}
            for which in ("S_s", "S_r", "S_rand"):
                lags, agg, per, bins = aggs[which]
                j = C.value_at(lags, None, tv)
                if j is None:
                    continue
                ent["at"][tn][which] = {"t": lags[j], **{nm: [agg[nm][0][j], agg[nm][1][j]] for nm in keep}}
        # time curves for contact bin (s-bin 0) and random pairs
        for which, ib in (("S_s", 0), ("S_rand", 0)):
            lags, agg, _, _ = aggs[which]
            ent["curves_contact"][which] = {"lag": lags, **{nm: [agg[nm][0][:, ib], agg[nm][1][:, ib]] for nm in
                                            ("R_L", "R_T", "R_I", "Rpf_T", "Rpf_I", "Rm_T", "Ccc_T", "Css_T", "CX_T", "Cqq", "c_q", "P_T", "P_L", "Fs")}}
        # time-resolved xi: per lag bin, absolute-r binning, window 1.95-4.95 (beyond the 2nd-shell peak)
        if not mk:
            lags, agg, per, bins = aggs["S_r"]
            xi_t = {}
            for qn in ("c_q", "CX_T"):
                rows = []
                for jb, b in enumerate(bins):
                    Ys = [a[1][qn] for a in per[b]]
                    for form in ("exp", "exp+c"):
                        fr = jackknife_fit(Ys, xr, form, 1.95, 4.95)
                        rows.append({"lag": lags[jb], "form": form, **(fr or {})})
                xi_t[qn] = rows
            ent["xi_vs_t"] = xi_t
        # fits at marker times: forms x windows x quantities x binnings
        if not mk:
            fits[key] = {}
            for tn, tv in tl.items():
                fits[key][tn] = []
                for which, xx, wins in (("S_s", xs, [(1.45, 4.05), (1.25, 4.05), (1.95, 4.05), (1.45, 3.45)]),
                                        ("S_r", xr, [(1.45, 4.95), (1.25, 4.95), (1.95, 4.95), (1.45, 3.95)])):
                    lags, agg, per, bins = aggs[which]
                    j = C.value_at(lags, None, tv)
                    if j is None:
                        continue
                    b = bins[j]
                    qlist = ("R_T", "R_I", "R_L", "Rpf_T", "Rpf_I", "c_q", "c_m2", "Cqq", "CX_T", "Ccc_T", "Rc_T", "Rc_I") if which == "S_r" else ("R_T", "R_I", "c_q")
                    for qn in qlist:
                        Ys = [a[1][qn] - (1 if qn in ("R_T", "R_I", "R_L", "Rpf_T", "Rpf_I") else 0) for a in per[b]]
                        for form in ("exp+c", "exp", "oz+c"):
                            for (lo, hi) in wins:
                                fr = jackknife_fit(Ys, xx, form, lo, hi)
                                if fr:
                                    fits[key][tn].append({"binning": which, "q": qn, **fr})
        out[key] = ent
        print(f"{key}: n={len(rs)} times={ {k: (round(v, 1) if v else None) for k, v in tl.items()} }", flush=True)
    C.jdump(times, os.path.join(C.RES, "times.json"))
    C.jdump(out, os.path.join(C.RES, "comob.json"))
    C.jdump(fits, os.path.join(C.RES, "xi_fits.json"))


if __name__ == "__main__":
    main()
