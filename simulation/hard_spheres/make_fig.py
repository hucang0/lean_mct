#!/usr/bin/env python3
"""Comparison figure: bond-resolved factorization in ordinary (polydisperse) HS vs the MK liquid.
(a) MK, phi = 2.0 (Letter Fig. 2(c) data, read-only): R(t) for contact-par, contact-perp, orientation-avg, random; Re C.
(b) ordinary HS, phi = PHI_HS (this study): the same five curves (same definitions).
(d) R_par and R_perp against the decorrelation Fs(k7, t) itself (matched-Fs comparison of the two systems).
(c) model-free Pearson coefficient of the bond-projected displacements of contact pairs, parallel (solid) and
    transverse (dashed), vs t / tau7: HS at all densities, MK at phi = 0.8, 1.5, 1.8 (re-analysis mk_pearson.py).
Writes figs/hs_vs_mk_bond.{pdf,png}, results/fig_panel_hs_phi<PHI>.csv and results/fig_panel_R_vs_Fs.csv (this folder).
Usage: make_fig.py [PHI_HS] [--hs-results DIR] [--mk-sim DIR]
  defaults: PHI_HS = 0.600; DIR = ../../data/simulation/hs_bond and ../../data/simulation (the shipped data)."""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "simulation")   # ../../data/simulation
_ap = argparse.ArgumentParser(description="HS vs MK bond-factorization comparison figure")
_ap.add_argument("phi_hs", nargs="?", default="0.600")
_ap.add_argument("--hs-results", default=os.path.join(REPO_DATA, "hs_bond"), help="hs_summary.py / mk_pearson.py outputs")
_ap.add_argument("--mk-sim", default=REPO_DATA, help="folder with results/curves_averaged.json and reanalysis/conditional_summary.json")
_a = _ap.parse_args()
SIM = _a.mk_sim
PHI_HS = _a.phi_hs
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
COL = {"L": "#2a78d6", "T": "#eb6834", "I": "#1baf7a", "C": "#6250d6", "r": "#7a7a75"}
INK2 = "#55554f"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
CV = json.load(open(f"{SIM}/results/curves_averaged.json"))
RC = json.load(open(f"{SIM}/reanalysis/conditional_summary.json"))
HS = json.load(open(os.path.join(_a.hs_results, "fig_curves.json")))
MKP = json.load(open(os.path.join(_a.hs_results, "mk_reanalysis_pearson.json")))
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9)); axs = axs.ravel()

# (a) MK phi = 2.0
ax = axs[0]; p = "2.0"
F = CV[p]["Fs7"]; tF = np.array(F["t"]); mF = np.array(F["mean"])
for key, c, ls, lab in (("c_Lpm", COL["L"], "-", r"contact $\parallel$"), ("c_Tpm", COL["T"], "-", r"contact $\perp$"),
                        ("c_pm", COL["I"], "-", "contact, orient.-avg."), ("r_pm", COL["r"], (0, (3, 1.5)), "random")):
    g = CV[p][key]; t = np.array(g["t"]); m = np.array(g["mean"]); Fi = np.interp(t, tF, mF); ok = Fi > 0.15
    ax.plot(t[ok], m[ok] / Fi[ok] ** 2, color=c, ls=ls, lw=1.4, label=lab)
cc = np.array(RC[p]["curves"]["L-:ReC"])
ax.plot(cc[:, 0], cc[:, 1], color=COL["C"], ls=(0, (3.5, 1.2, 1, 1.2)), lw=1.3, label=r"Re$\,C$ ($\parallel$)")
ax.set_xscale("log"); ax.set_xlim(1, 1.5e6); ax.set_ylim(0.5, 4.5)
ax.text(0.97, 0.93, r"MK, $\varphi=2.0$ (caged)", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
ax.set_xlabel("$t$ (MC sweeps)"); ax.set_ylabel(r"$R(t)=\mathrm{Re}\,\phi_2/F_s^2$")
ax.axhline(1, color="#bdbdb6", lw=0.6, zorder=0)
ax.legend(loc="upper left", fontsize=6.3, handlelength=2.2, labelspacing=0.2)
ax.text(-0.2, 1.03, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")

# (b) HS
ax = axs[1]; key = f"phi{PHI_HS}_N1000_d0.050"; h = HS[key]
for k, c, ls in (("L-:R_glob", COL["L"], "-"), ("T-:R_glob", COL["T"], "-"), ("Ipm:R_glob", COL["I"], "-"),
                 ("rand-:R", COL["r"], (0, (3, 1.5))), ("L-:ReC", COL["C"], (0, (3.5, 1.2, 1, 1.2)))):
    a = np.array(h[k], dtype=float)
    ax.plot(a[:, 0], a[:, 1], color=c, ls=ls, lw=1.4 if k != "L-:ReC" else 1.3)
for tv, lab in ((h["tstar"], r"$t^*$"), (h["tau7"], r"$\tau_7$")):
    if tv:
        ax.axvline(tv, color="#bdbdb6", lw=0.6, ls=":", zorder=0); ax.text(tv / 1.15, 3.6, lab, fontsize=6.5, color=INK2, ha="right")
ax.set_xscale("log"); ax.set_ylim(0.5, 4.5); ax.set_xlim(1, (h["tau7"] or 1e6) * 3)
ax.axhline(1, color="#bdbdb6", lw=0.6, zorder=0)
ax.text(0.03, 0.93, rf"ordinary HS (23% polydisp.), $\varphi={float(PHI_HS):.3g}$", transform=ax.transAxes, fontsize=7.5, color=INK2)
ax.set_xlabel("$t$ (MC sweeps)")
ax.text(-0.2, 1.03, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")

# (c) Pearson vs t/tau7
ax = axs[2]
hs_keys = [k for k in sorted(HS) if k.endswith("_N1000_d0.050")]
blues = plt.get_cmap("Blues")
for n, k in enumerate(hs_keys):
    h = HS[k]; t7 = h["tau7"]
    if not t7:
        continue
    col = blues(0.45 + 0.55 * n / max(len(hs_keys) - 1, 1))
    for q, ls in (("L:pearson", "-"), ("T:pearson", (0, (2.5, 1.5)))):
        a = np.array(h[q], dtype=float)
        ax.plot(a[:, 0] / t7, a[:, 1], color=col, ls=ls, lw=1.2, label=(rf"HS $\varphi={float(k[3:8]):.3g}$" if q == "L:pearson" else None))
greys = ["#c9a227", "#d97b29", "#a8431d"]
for n, p in enumerate(("0.8", "1.5", "1.8")):
    e = MKP[p]; t7 = e["tau7"]
    for q, ls in (("L:pearson", "-"), ("T:pearson", (0, (2.5, 1.5)))):
        a = np.array(e["curves"][q], dtype=float)
        ax.plot(a[:, 0] / t7, a[:, 1], color=greys[n], ls=ls, lw=1.1, label=(rf"MK $\varphi={p}$" if q == "L:pearson" else None))
ax.axhline(0, color="#bdbdb6", lw=0.6, zorder=0); ax.axvline(1, color="#bdbdb6", lw=0.6, ls=":", zorder=0)
ax.set_xscale("log"); ax.set_xlim(1e-3, 30); ax.set_ylim(-0.15, 1.0)
ax.set_xlabel(r"$t/\tau_7$"); ax.set_ylabel(r"corr$(\Delta x_i^{e},\Delta x_j^{e})$, contact pairs")
ax.text(0.03, 0.97, r"solid: $e=\hat{r}_{ij}$ ($\parallel$)" + "\n" + r"dashed: $e\perp\hat{r}_{ij}$", transform=ax.transAxes, fontsize=6.2, color=INK2, va="top")
ax.legend(loc="upper right", fontsize=6.0, handlelength=1.8, labelspacing=0.15, ncol=2, bbox_to_anchor=(1.0, 1.02))
ax.text(-0.2, 1.03, "(c)", transform=ax.transAxes, fontsize=9, fontweight="bold")
# (d) matched decorrelation: R_par (solid), R_perp (dashed) vs Fs(k7, t), both systems
ax = axs[3]
for n, k in enumerate(hs_keys):
    h = HS[k]; col = blues(0.45 + 0.55 * n / max(len(hs_keys) - 1, 1))
    F = np.array(h["Fs7"], dtype=float)
    for q, ls in (("L-:R_glob", "-"), ("T-:R_glob", (0, (2.5, 1.5)))):
        a = np.array(h[q], dtype=float); Fi = np.interp(np.log(a[:, 0]), np.log(F[:, 0]), F[:, 1])
        ax.plot(Fi, a[:, 1], color=col, ls=ls, lw=1.2, label=(rf"HS $\varphi={float(k[3:8]):.3g}$" if q == "L-:R_glob" else None))
for n, p in enumerate(("1.5", "1.8", "2.0")):
    Fm = CV[p]["Fs7"]; tF = np.array(Fm["t"]); mF = np.array(Fm["mean"])
    for q, ls in (("L-:R_glob", "-"), ("T-:R_glob", (0, (2.5, 1.5)))):
        a = np.array(RC[p]["curves"][q], dtype=float); Fi = np.interp(np.log(a[:, 0]), np.log(tF), mF)
        ax.plot(Fi, a[:, 1], color=greys[n], ls=ls, lw=1.1, label=(rf"MK $\varphi={p}$" if q == "L-:R_glob" else None))
ax.axhline(1, color="#bdbdb6", lw=0.6, zorder=0); ax.axvline(0.448, color="#bdbdb6", lw=0.6, ls=":", zorder=0)
ax.text(0.448, 4.55, r"MK $\varphi=2$ plateau $F_s$", fontsize=6.2, color=INK2, ha="center", va="bottom", clip_on=False)
ax.set_xlim(1.0, 0.15); ax.set_ylim(0.5, 4.5)
ax.set_xlabel(r"$F_s(k\simeq7,t)$  (decorrelation $\rightarrow$)"); ax.set_ylabel(r"$R_\parallel$ (solid), $R_\perp$ (dashed)")
ax.legend(loc="upper left", fontsize=6.0, handlelength=1.8, labelspacing=0.15, ncol=2)
ax.text(-0.2, 1.03, "(d)", transform=ax.transAxes, fontsize=9, fontweight="bold")
fig.tight_layout(w_pad=1.5, h_pad=1.2)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs/hs_vs_mk_bond.pdf")); fig.savefig(os.path.join(HERE, "figs/hs_vs_mk_bond.png"), dpi=200)

# data file for a Letter panel: HS curves at PHI_HS
h = HS[f"phi{PHI_HS}_N1000_d0.050"]
cols = ["L-:R_glob", "T-:R_glob", "Ipm:R_glob", "rand-:R", "L-:ReC", "L-:Im_phi2", "L-:Im_prod", "Fs7"]
grid = np.array(h["Fs7"], dtype=float)[:, 0]
with open(os.path.join(HERE, f"results/fig_panel_hs_phi{PHI_HS}.csv"), "w") as f:
    f.write(f"# ordinary polydisperse HS, phi={PHI_HS}, N=1000, delta=0.05; tau7={h['tau7']:.6g}; t_star={h['tstar']:.6g}; "
            "columns: t (MC sweeps), then mean and s.e.m. over samples of each quantity (NaN where Fs7<=0.15 for ratios)\n")
    f.write("t," + ",".join(f"{c}_mean,{c}_sem" for c in cols) + "\n")
    for t in grid:
        row = [t]
        for c in cols:
            a = np.array(h[c], dtype=float); j = np.argmin(np.abs(np.log(a[:, 0] / t)))
            row += [a[j, 1], a[j, 2]] if abs(np.log(a[j, 0] / t)) < 0.05 else [np.nan, np.nan]
        f.write(",".join(f"{x:.6g}" for x in row) + "\n")
print("wrote figs/hs_vs_mk_bond.{pdf,png} and results/fig_panel_hs_phi%s.csv" % PHI_HS)

# data file for the proposed Letter panel: R_par, R_perp, Re C_par, corr_par vs Fs(k7,t) for HS (all densities) and MK
rows = []
for k in hs_keys:
    h = HS[k]; F = np.array(h["Fs7"], dtype=float)
    A_ = {q: np.array(h[q], dtype=float) for q in ("L-:R_glob", "T-:R_glob", "L-:ReC", "L:pearson", "T:pearson")}
    for j in range(len(A_["L-:R_glob"])):
        t = A_["L-:R_glob"][j, 0]
        vals = []
        for q in ("L-:R_glob", "T-:R_glob", "L-:ReC", "L:pearson", "T:pearson"):
            a = A_[q]; i = np.argmin(np.abs(np.log(a[:, 0] / t)))
            vals += [a[i, 1], a[i, 2] if a[i, 2] == a[i, 2] else np.nan]
        rows.append(["HS", float(k[3:8]), t, np.interp(np.log(t), np.log(F[:, 0]), F[:, 1])] + vals)
for p in ("1.5", "1.8", "2.0"):
    Fm = CV[p]["Fs7"]; tF = np.array(Fm["t"]); mF = np.array(Fm["mean"])
    pk = {"2.0": "2"}.get(p, p)
    src = {"L-:R_glob": RC[p]["curves"]["L-:R_glob"], "T-:R_glob": RC[p]["curves"]["T-:R_glob"], "L-:ReC": RC[p]["curves"]["L-:ReC"],
           "L:pearson": MKP[pk]["curves"]["L:pearson"], "T:pearson": MKP[pk]["curves"]["T:pearson"]}
    A_ = {q: np.array([[c[0], c[1], c[2] if c[2] is not None else np.nan] for c in v], dtype=float) for q, v in src.items()}
    for j in range(len(A_["L-:R_glob"])):
        t = A_["L-:R_glob"][j, 0]
        vals = []
        for q in ("L-:R_glob", "T-:R_glob", "L-:ReC", "L:pearson", "T:pearson"):
            a = A_[q]; i = np.argmin(np.abs(np.log(a[:, 0] / t)))
            vals += [a[i, 1], a[i, 2]] if abs(np.log(a[i, 0] / t)) < 0.1 else [np.nan, np.nan]
        rows.append(["MK", float(p), t, np.interp(np.log(t), np.log(tF), mF)] + vals)
with open(os.path.join(HERE, "results/fig_panel_R_vs_Fs.csv"), "w") as f:
    f.write("# R_par, R_perp, Re C_par, corr_par, corr_perp (mean, sem over samples) vs Fs(k~7,t) for contact pairs; "
            "HS: this study (N=1000, MC delta=0.05); MK: Letter data (conditional_summary.json, curves_averaged.json) + mk_pearson.py\n")
    f.write("system,phi,t_sweeps,Fs7,R_par,R_par_sem,R_perp,R_perp_sem,ReC_par,ReC_par_sem,corr_par,corr_par_sem,corr_perp,corr_perp_sem\n")
    for r in rows:
        f.write(",".join([r[0]] + [f"{x:.6g}" for x in r[1:]]) + "\n")
print("wrote results/fig_panel_R_vs_Fs.csv", len(rows), "rows")
