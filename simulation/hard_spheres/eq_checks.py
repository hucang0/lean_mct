#!/usr/bin/env python3
"""Equilibration / crystallization checks.
(a) swap-MC equilibration histories: Z (virtual-inflation pressure) averaged over the 2nd and 4th quarters of the
    run (stationarity), BMCSL reference, swap-dynamics Fs7 and MSD reached (decorrelation), Q6 and crystal-like fraction;
(b) dynamics runs: Z at the time origins (first half vs second half), Q6 / crystal fraction at the start and the end
    of the non-swap run, scaled g(r) contact peak, and local-size segregation (correlation of sigma_i with mean
    neighbour sigma) at start and end.
Usage: eq_checks.py results/eq_checks.json"""
import os, sys, glob, json
import numpy as np
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hs_struct import q6_stats, gr_scaled, neighbor_pairs


def size_corr(x, sig, L):
    P, d, r = neighbor_pairs(x, sig, L)
    N = len(sig); s = np.zeros(N); c = np.zeros(N)
    np.add.at(s, P[:, 0], sig[P[:, 1]]); np.add.at(s, P[:, 1], sig[P[:, 0]])
    np.add.at(c, P[:, 0], 1); np.add.at(c, P[:, 1], 1)
    m = c > 0
    return float(np.corrcoef(sig[m], s[m] / c[m])[0, 1])


def main():
    out = {"equil": {}, "dyn": {}}
    for f in sorted(glob.glob(os.path.join(HERE, "data/eq/eq_*.npz"))):
        z = np.load(f); h = z["hist"]; n = len(h)
        q2 = h[n // 4: n // 2, 1]; q4 = h[3 * n // 4:, 1]
        out["equil"][os.path.basename(f)] = dict(phi=float(z["phi"]), N=int(z["N"]), neq=int(z["neq"]),
            Z_q2=[float(q2.mean()), float(q2.std(ddof=1) / np.sqrt(len(q2)))], Z_q4=[float(q4.mean()), float(q4.std(ddof=1) / np.sqrt(len(q4)))],
            Z_bmcsl=float(z["Z_bmcsl"]), Fs7_swap_end=float(h[-1, 2]), MSD_swap_end=float(h[-1, 3]),
            Q6_max=float(h[:, 4].max()), xtal_max=float(h[:, 5].max()), acc_tr=float(z["acc_tr"]), acc_sw=float(z["acc_sw"]))
    for f in sorted(glob.glob(os.path.join(HERE, "data/dyn/run_*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        z = np.load(f)
        if bool(z["partial"]):
            continue
        sig, L = z["sig"], float(z["L"])
        Zs = np.atleast_2d(z["Z"])[:, 1]
        r = {"phi": float(z["phi"]), "N": int(z["N"]), "delta": float(z["delta"]), "acc": float(z["acc"]),
             "Z_first_half": float(Zs[: len(Zs) // 2].mean()), "Z_second_half": float(Zs[len(Zs) // 2:].mean())}
        for lab, x in (("start", z["x0"]), ("end", z["xfinal"])):
            Q6, xf = q6_stats(x, sig, L)
            rr, g = gr_scaled(x, sig, L)
            r[f"Q6_{lab}"] = Q6; r[f"xtal_{lab}"] = xf; r[f"g_peak_{lab}"] = float(g.max())
            r[f"size_corr_{lab}"] = size_corr(x, sig, L)
        out["dyn"][os.path.basename(f)] = r
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for k, v in out["equil"].items():
        print(k, {a: (np.round(b, 3).tolist() if isinstance(b, (list, float)) else b) for a, b in v.items()})
    for k, v in out["dyn"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})


if __name__ == "__main__":
    main()
