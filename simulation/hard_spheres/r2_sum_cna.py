#!/usr/bin/env python3
"""Aggregate cache/cna_*.npz -> results/cna.json: 1551-bond fraction, icosahedral centres, 1551-cluster sizes,
endpoint persistence of bonds / 1551 bonds (still 1551; still bonded) / ico centres vs lag and its 1/e time,
and mobility (orientation-averaged self overlap at k7 and MSD) of particle groups defined by the t0 structure
(n_1551 = number of 1551 bonds of a particle: 0, 1-3, 4-7, >=8; ico centres; members of 1551 clusters >= 13)."""
import os, sys, glob, json
sys.dont_write_bytecode = True
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import r2_common as C


def main():
    times = json.load(open(C.result_path("times.json")))
    groups = {}
    for f in C.require_files(sorted(glob.glob(os.path.join(C.CACHE, "cna_*.npz"))), "R2 caches (run the corresponding trajectory analysis first)"):
        d = dict(np.load(f)); d = {k: (v.item() if v.shape == () else v) for k, v in d.items()}
        groups.setdefault(str(d["key"]), []).append(d)
    out = {}
    for key, rs in sorted(groups.items()):
        G = list(rs[0]["groups"])
        per = C.per_sample_curve(rs, lambda r, j: np.concatenate([r["pers"][j][:3] / r["nframes"][j],
                                  [r["pers"][j][3] / max(r["pers"][j][4], 1e-9)],
                                  r["q_sum"][j] / np.maximum(r["cnt"][j], 1), r["m2_sum"][j] / np.maximum(r["cnt"][j], 1),
                                  [r["Fs7"][j] / r["nframes"][j]]]))
        lags, mean, sem, n = C.curve_stats(per)
        nG = len(G)
        ent = {"n_samples": len(rs), "groups": G,
               "frac1551": C.sample_mean_sem([r["frac1551"] for r in rs])[:2],
               "ico_per_1000": C.sample_mean_sem([r["nico"] * 1000 / r["N"] for r in rs])[:2],
               "largest_1551_cluster": C.sample_mean_sem([r["largest"] for r in rs])[:2],
               "frac_in_1551_clusters_ge13": C.sample_mean_sem([r["frac_clus13"] for r in rs])[:2],
               "n1551_hist": np.sum([r["n1551_hist"] for r in rs], 0).tolist(),
               "lags": lags.tolist(),
               "pers_bond": [mean[:, 0].tolist(), sem[:, 0].tolist()], "pers_1551": [mean[:, 1].tolist(), sem[:, 1].tolist()],
               "pers_1551_bond": [mean[:, 2].tolist(), sem[:, 2].tolist()], "pers_ico": [mean[:, 3].tolist(), sem[:, 3].tolist()],
               "q_group": [mean[:, 4:4 + nG].tolist(), sem[:, 4:4 + nG].tolist()],
               "msd_group": [mean[:, 4 + nG:4 + 2 * nG].tolist(), sem[:, 4 + nG:4 + 2 * nG].tolist()]}
        ent["frac_groups"] = (np.sum([r["cnt"][0] / r["nframes"][0] for r in rs], 0) / np.sum([r["N"] for r in rs])).tolist()
        for nm, col in (("bond", 0), ("1551", 1), ("1551_bond", 2)):
            ent[f"t_1e_{nm}"] = C.A.crossing_time(lags, mean[:, col])
        tk = times[key]
        ent["at"] = {}
        for tn in ("tstar", "tau7"):
            j = C.value_at(lags, None, tk.get(tn))
            if j is None:
                continue
            qa = mean[j, 4 + G.index("all")]
            ent["at"][tn] = {"t": lags[j], "pers_bond": mean[j, 0], "pers_1551": mean[j, 1], "pers_1551_bond": mean[j, 2],
                             "pers_ico": mean[j, 3],
                             "q": {g: [mean[j, 4 + i], sem[j, 4 + i]] for i, g in enumerate(G)},
                             "q_over_all": {g: mean[j, 4 + i] / qa for i, g in enumerate(G)},
                             "msd_over_all": {g: mean[j, 4 + nG + i] / mean[j, 4 + nG + G.index("all")] for i, g in enumerate(G)}}
        ent["tau7"] = tk.get("tau7"); ent["tstar"] = tk.get("tstar")
        out[key] = ent
        a = ent["at"].get("tau7", {})
        print(f"{key}: frac1551={ent['frac1551'][0]:.3f} ico/1000={ent['ico_per_1000'][0]:.2f} largest={ent['largest_1551_cluster'][0]:.0f} "
              f"t1e(bond)={ent['t_1e_bond']} t1e(1551)={ent['t_1e_1551']} tau7={tk.get('tau7'):.4g}  "
              f"q/q_all@tau7: " + " ".join(f"{g}={v:.2f}" for g, v in a.get('q_over_all', {}).items()), flush=True)
    C.jdump(out, os.path.join(C.RES, "cna.json"))


if __name__ == "__main__":
    main()
