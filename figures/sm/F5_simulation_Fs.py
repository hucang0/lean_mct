"""F5: planted Monte Carlo of MK at lambda = infinity. (a) F_s(k~7,t) for all densities (N=1000, mean +- s.e.m. over
3 samples; N=2000 dashed); (b) plateau heights (plateau_table.json: f_plateau where has_plateau, open symbols
F_s(k, t=10^4) elsewhere; N=2000 squares). Data: simulation/results/curves_averaged.json, plateau_table.json."""
import json, numpy as np, matplotlib.pyplot as plt
from _style import *

CV = json.load(open(f"{SIM}/results/curves_averaged.json")); PT = json.load(open(f"{SIM}/results/plateau_table.json"))
phis = ["0.8", "1.2", "1.5", "1.58", "1.65", "1.72", "1.8", "2.0"]
cmap = plt.get_cmap("Blues"); cols = [cmap(0.3 + 0.7*i/(len(phis)-1)) for i in range(len(phis))]

fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.25))
ax = axs[0]
for p, col in zip(phis, cols):
    g = CV[p]["Fs7"]; t = np.array(g["t"]); m = np.array(g["mean"]); s = np.array([x if x is not None else np.nan for x in g["sem"]], dtype=float)
    ax.plot(t, m, color=col, lw=1.3); ax.fill_between(t, m-s, m+s, color=col, alpha=0.25, lw=0)
    if float(p) >= 1.65: ax.text(t[-1]*1.25, m[-1], r"$%s$" % p, fontsize=6.3, color=col, va="center")
    elif p == "0.8":
        i = int(np.argmin(np.abs(m-0.32))); ax.text(t[i]*0.70, 0.32, r"$0.8$", fontsize=6.2, color=col, va="center", ha="right")
    elif p == "1.2":
        ax.text(580, 0.038, r"$1.2$", fontsize=6.0, color=col, va="center", ha="center", zorder=6)
    elif p == "1.5":
        ax.text(1.3e4, 0.038, r"$1.5$", fontsize=6.0, color=col, va="center", ha="center", zorder=6)
for p in ("1.5", "1.8", "2.0"):
    g = CV[p+"_N2000"]["Fs7"]; ax.plot(g["t"], g["mean"], color=C["ink"], lw=0.8, ls="--")
ax.plot([], [], color=C["ink"], lw=0.8, ls="--", label=r"$N_p=2000$")
ax.plot([], [], color=cols[4], lw=1.3, label=r"$N_p=1000$, $\varphi=0.8$–$2.0$")
ax.axhline(np.exp(-1), color=C["muted"], lw=0.6, ls=":")
ax.set_xscale("log"); ax.set_xlim(1, 2e7); ax.set_ylim(0, 1.0)
ax.set_xlabel(r"$t$ (MC sweeps)"); ax.set_ylabel(r"$F_s(k\simeq 7,t)$")
ax.legend(loc="upper right")
panel_label(ax, "(a)", x=-0.17)

ax = axs[1]
kcols = {"k3": C["blue"], "k5": C["orange"], "k7": C["aqua"], "k10": C["violet"]}
klab = {"k3": r"$k\simeq3$", "k5": r"$k\simeq5$", "k7": r"$k\simeq7$", "k10": r"$k\simeq10$"}
for kk, col in kcols.items():
    xs, ys, xq, yq = [], [], [], []
    for p in phis:
        r = PT[p][kk]
        if r["has_plateau"]: xs.append(float(p)); ys.append(r["f_plateau"])
        xq.append(float(p)); yq.append(r.get("Fs_at_t10000", np.nan))
    ax.plot(xq, yq, color=col, marker="o", mfc="white", ms=4, lw=0.8, ls=":")
    ax.plot(xs, ys, color=col, marker="o", ms=5.5, lw=1.6, label=klab[kk])
    for p in ("1.8", "2.0"):
        r = PT[p+"_N2000"][kk]
        if r["has_plateau"]: ax.plot([float(p)], [r["f_plateau"]], marker="s", mfc="none", mec=col, ms=7, ls="none")
ax.plot([], [], marker="o", mfc="white", color=C["ink2"], ms=4, lw=0.8, ls=":", label=r"$F_s(k,t=10^4)$, no plateau")
ax.plot([], [], marker="s", mfc="none", mec=C["ink2"], ls="none", ms=7, label=r"$N_p=2000$ plateau")
ax.axvspan(1.76, 1.84, color=C["grid"], lw=0, zorder=0); ax.axvline(1.82, color=COL_PHID, lw=0.9)
ax.text(1.815, 0.97, r"$\varphi_\times$", ha="right", va="top", fontsize=7.5, color=COL_PHID)
ax.set_xlim(0.7, 2.1); ax.set_ylim(0, 1.0)
ax.set_xlabel(r"$\varphi$"); ax.set_ylabel(r"plateau height $f_s(k)$")
ax.legend(loc="upper left", fontsize=6.5)
panel_label(ax, "(b)", x=-0.17)
fig.tight_layout(w_pad=1.5)
save(fig, "F5_simulation_Fs")
