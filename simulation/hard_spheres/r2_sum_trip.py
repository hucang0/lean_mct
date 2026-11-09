#!/usr/bin/env python3
"""Aggregate cache/trip_*.npz -> results/triplets.json.
Per sample and lag bin, per class c (0 closed s_jk<1.05, 1 nb-closed 1.05-1.35, 2 open >=1.35; 'all'), per cos-theta bin a:
  b_ij = x1 - y1, b_ik = x2 - y2 (bond-projected relative displacements, x = centre i, y = outer particle):
     Pb   = Pearson(b_ij, b_ik)
     decomposition Cov(b_ij,b_ik) = S - P + J,  S = Cov(x1,x2) (shared centre, geometric), P = Cov(x1,y2)+Cov(y1,x2)
     (pair channel), J = Cov(y1,y2) (outer-outer: the only term that needs a j-k path other than through i).
     Tree (Gaussian-Markov) null: on a tree the outer particles correlate only through i,
        J_tree = Cov(x1,y1) Cov(x2,y2) cos(theta) / s_i^2,  s_i^2 = (Var x1 + Var x2)/2
     (exact for an isotropic Gaussian Markov field with bond-aligned pair covariance; reduces to
      Pb_tree = (1 - rho_par) cos(theta) / 2 when all bonds have the same rho_par). Pb_tree uses J_tree in place of J.
     Loop residual: dPb = Pb - Pb_tree;  dJ = (J - J_tree)/s_i^2.
  transverse (normal n to the triplet plane): Pt = Pearson(t_ij, t_ik), tree: J_t,tree = Cov(xt,y1t)Cov(xt,y2t)/Var(xt).
  co-breaking: K = P(both broken)/[P(ij broken) P(ik broken)], broken = r(t) > 1.35 sigma; also for 'still in contact'.
Mean +- s.e.m. over samples; MK angle-matched reference: MK per-angle-bin values reweighted with the HS class angle histogram.
"""
import os, sys, glob, json
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C

NANG = 10


def stats_from_sums(o):
    """o: array (..., NF) of sums -> dict of derived arrays (...)."""
    F = ["cnt", "cos", "x1", "x2", "y1", "y2", "x1x1", "x2x2", "y1y1", "y2y2", "x1x2", "x1y1", "x2y2", "x1y2", "x2y1", "y1y2",
         "cntt", "xt", "y1t", "y2t", "xtxt", "y1ty1t", "y2ty2t", "xty1t", "xty2t", "y1ty2t", "br1", "br2", "br12", "ok1", "ok2", "ok12"]
    I = {f: n for n, f in enumerate(F)}
    with np.errstate(invalid="ignore", divide="ignore"):
        n = o[..., I["cnt"]]
        m = lambda f: o[..., I[f]] / n
        cov = lambda a, b, ab: m(ab) - m(a) * m(b)
        vx1, vx2, vy1, vy2 = cov("x1", "x1", "x1x1"), cov("x2", "x2", "x2x2"), cov("y1", "y1", "y1y1"), cov("y2", "y2", "y2y2")
        Sx = cov("x1", "x2", "x1x2"); P = cov("x1", "y2", "x1y2") + cov("x2", "y1", "x2y1"); J = cov("y1", "y2", "y1y2")
        c11, c22 = cov("x1", "y1", "x1y1"), cov("x2", "y2", "x2y2")
        vb1 = vx1 + vy1 - 2 * c11; vb2 = vx2 + vy2 - 2 * c22
        cb = Sx - P + J
        s2 = 0.5 * (vx1 + vx2)
        cosm = m("cos")
        Jt = c11 * c22 * cosm / s2
        out = {"n": n, "cos": cosm, "Pb": cb / np.sqrt(vb1 * vb2), "Pb_tree": (Sx - P + Jt) / np.sqrt(vb1 * vb2),
               "J": J / s2, "J_tree": Jt / s2, "S": Sx / s2, "Pp": P / s2,
               "rho_pair": 0.5 * (c11 / np.sqrt(vx1 * vy1) + c22 / np.sqrt(vx2 * vy2))}
        out["dPb"] = out["Pb"] - out["Pb_tree"]; out["dJ"] = out["J"] - out["J_tree"]
        nt = o[..., I["cntt"]]
        mt = lambda f: o[..., I[f]] / nt
        covt = lambda a, b, ab: mt(ab) - mt(a) * mt(b)
        vxt, v1t, v2t = covt("xt", "xt", "xtxt"), covt("y1t", "y1t", "y1ty1t"), covt("y2t", "y2t", "y2ty2t")
        c1t, c2t, jt = covt("xt", "y1t", "xty1t"), covt("xt", "y2t", "xty2t"), covt("y1t", "y2t", "y1ty2t")
        vt1 = vxt + v1t - 2 * c1t; vt2 = vxt + v2t - 2 * c2t
        ct = vxt - c1t - c2t + jt
        jtt = c1t * c2t / vxt
        out["Pt"] = ct / np.sqrt(vt1 * vt2); out["Pt_tree"] = (vxt - c1t - c2t + jtt) / np.sqrt(vt1 * vt2)
        out["dPt"] = out["Pt"] - out["Pt_tree"]; out["Jt"] = jt / vxt; out["Jt_tree"] = jtt / vxt
        p1, p2, p12 = m("br1"), m("br2"), m("br12")
        out["Pbr"] = 0.5 * (p1 + p2); out["Kbr"] = p12 / (p1 * p2)
        k1, k2, k12 = m("ok1"), m("ok2"), m("ok12")
        out["Pok"] = 0.5 * (k1 + k2); out["Kok"] = k12 / (k1 * k2)
    return out


QN = ["n", "cos", "Pb", "Pb_tree", "dPb", "J", "J_tree", "dJ", "S", "Pp", "rho_pair", "Pt", "Pt_tree", "dPt", "Jt", "Jt_tree",
      "Pbr", "Kbr", "Pok", "Kok"]


def run_stats(r, j):
    S = r["S"][j]                                   # (3, NANG, NF)
    cls = stats_from_sums(S.sum(1))                 # (3,)
    allc = stats_from_sums(S.sum((0, 1)))           # scalar
    ang = stats_from_sums(S)                        # (3, NANG)
    ang_all = stats_from_sums(S.sum(0))             # (NANG,)
    return {"cls": cls, "all": allc, "ang": ang, "ang_all": ang_all}


def main():
    times = json.load(open(C.result_path("times.json")))
    groups = {}
    for f in C.require_files(sorted(glob.glob(os.path.join(C.CACHE, "trip_*.npz"))), "R2 caches (run the corresponding trajectory analysis first)"):
        d = dict(np.load(f)); d = {k: (v.item() if v.shape == () else v) for k, v in d.items()}
        groups.setdefault(str(d["key"]), []).append(d)
    out = {}
    for key, rs in sorted(groups.items()):
        per = {}
        for r in rs:
            for j, lag in enumerate(r["lag"]):
                b = int(np.rint(C.PD * np.log10(lag)))
                per.setdefault(b, []).append((lag, run_stats(r, j)))
        bins = sorted(per)
        lags = np.array([np.exp(np.mean([np.log(a[0]) for a in per[b]])) for b in bins])
        def agg(sel):
            res = {}
            for q in QN:
                mm, ss = [], []
                for b in bins:
                    V = np.array([sel(a[1])[q] for a in per[b]], float)
                    mm.append(np.nanmean(V, 0)); ss.append(np.nanstd(V, 0, ddof=1) / np.sqrt(len(V)) if len(V) > 1 else np.nan * V[0])
                res[q] = (np.array(mm), np.array(ss))
            return res
        A_cls, A_all, A_ang, A_angall = agg(lambda s: s["cls"]), agg(lambda s: s["all"]), agg(lambda s: s["ang"]), agg(lambda s: s["ang_all"])
        ncls = np.sum([r["ncls"] for r in rs], 0)
        cos_hist = np.sum([r["cos_hist"] for r in rs], 0)          # (3, 40)
        ent = {"n_samples": len(rs), "ntrip_per_config": float(np.mean([r["ntrip"] for r in rs])),
               "contacts_per_config": float(np.mean([r["ncontacts"] for r in rs])),
               "class_frac": (ncls / ncls.sum()).tolist(), "cos_hist": cos_hist.tolist(),
               "sjk_hist": np.sum([r["sjk_hist"] for r in rs], 0).tolist(), "sjk_edges": rs[0]["sjk_edges"].tolist(),
               "lags": lags.tolist(), "at": {},
               "curves": {q: {"cls_mean": A_cls[q][0].tolist(), "cls_sem": A_cls[q][1].tolist(),
                              "all_mean": A_all[q][0].tolist(), "all_sem": A_all[q][1].tolist()}
                          for q in ("Pb", "Pb_tree", "dPb", "dJ", "Kbr", "Pbr", "Pt", "dPt", "rho_pair", "J", "J_tree")}}
        tk = times.get(key, {})
        for tn in ("tau7", "tstar", "tF045", "t1e4"):
            tv = tk.get(tn)
            j = C.value_at(lags, None, tv) if tv else None
            if j is None:
                continue
            ent["at"][tn] = {"t": float(lags[j]),
                             "cls": {q: [A_cls[q][0][j].tolist(), A_cls[q][1][j].tolist()] for q in QN},
                             "all": {q: [float(A_all[q][0][j]), float(A_all[q][1][j])] for q in QN},
                             "ang": {q: [A_ang[q][0][j].tolist(), A_ang[q][1][j].tolist()] for q in QN},
                             "ang_all": {q: [A_angall[q][0][j].tolist(), A_angall[q][1][j].tolist()] for q in QN}}
        out[key] = ent
        print(key, "class frac", np.round(ncls / ncls.sum(), 3), flush=True)
    # MK angle-matched references for each HS class
    for key, ent in out.items():
        if key.startswith("mk"):
            continue
        ch = np.array(ent["cos_hist"])                 # (3, 40) -> 10 bins
        h10 = ch.reshape(3, NANG, 4).sum(2)
        ent["mk_ref"] = {}
        for mkkey in ("mk1.50", "mk1.80", "mk2.00"):
            if mkkey not in out:
                continue
            mk = out[mkkey]
            ent["mk_ref"][mkkey] = {}
            for tn in ("tau7", "tF045", "t1e4"):
                if tn not in mk["at"]:
                    continue
                ref = {}
                for q in ("Pb", "Pb_tree", "dPb", "Kbr", "Pt", "dJ", "J", "J_tree"):
                    v = np.array(mk["at"][tn]["ang_all"][q][0], float)
                    ref[q] = [float(np.nansum(h10[c] * v) / np.sum(h10[c][np.isfinite(v)])) for c in range(3)]
                ent["mk_ref"][mkkey][tn] = ref
    C.jdump(out, os.path.join(C.RES, "triplets.json"))


if __name__ == "__main__":
    main()
