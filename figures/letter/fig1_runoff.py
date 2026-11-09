"""Letter Fig. 2 (combined 4-panel horizontal):
(a) Perron test: lambda_P(phi_c(N))/lambda_P(phi_c(2)) vs N for PY and MK against Lambda_c(N)/4.
(b) MK critical packing fractions phi_c vs particle order N and intra-pair exclusion order ell.
(c) Bond-resolved factorization ratio R(t) = Re phi_2 / F_s^2 at phi = 2.0 in MK planted MC.
(d) Ordinary 3D hard spheres (phi = 0.50-0.60) vs MK (phi = 1.5-2.0): R_par (solid) and R_perp (dashed) vs F_s(k~7, t).
CPU only; no solver runs."""
import sys, json, glob as _g, numpy as np
from _style import *
import matplotlib.pyplot as plt
OUTL = OUT

H = json.load(open(f"{NUM}/results/headline.json"))
LAM = {int(k): v for k, v in H["Lambda_c_over_4"].items()}
PYR = {int(k): v for k, v in H["PY_paper_ratio"].items()}
rows = lambda var, clo: {r["label"]: r for r in H["phi_c"] if r["variant"] == var and r["closure"] == clo and "M" not in r and not r.get("grid_study")}
bs, bsm, bc, bcm = rows("Bself", "corr"), rows("Bself", "mem"), rows("Bcoll", "corr"), rows("Bcoll", "mem")
rat = lambda d, base, n: d[n]["phi_c"] / base[2]["phi_c"]

CV = json.load(open(f"{SIM}/results/curves_averaged.json"))
RC = json.load(open(f"{SIM}/reanalysis/conditional_summary.json"))
HS = json.load(open(f"{SIM}/hs_bond/fig_curves.json"))
RS = json.load(open(f"{RES}/rescatter_series_gauss.json"))

fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.85), gridspec_kw=dict(wspace=0.38))

# (a) Perron test
ax = axs[0]
Ns = np.arange(2, 7)
ax.plot(Ns, [LAM[n] for n in Ns], ls="--", color=COL_PERRON, lw=1.15, label=r"$\Lambda_c/4$")
ax.plot(Ns, [PYR[n] for n in Ns], marker="x", ms=5.0, mew=1.3, ls="none", color=COL_PY, label="PY")
ax.plot(Ns, [rat(bs, bs, n) for n in Ns], marker="o", ms=3.6, ls="-", color=COL_BSELF, label=r"$\widetilde W_s$")
ax.plot([3, 4, 5], [rat(bsm, bs, n) for n in (3, 4, 5)], marker="o", ms=3.6, mfc="white", ls="none", color=COL_BSELF)
ax.plot(Ns, [rat(bc, bc, n) for n in Ns], marker="s", ms=3.5, ls="-", color=COL_BCOLL, label=r"$\widetilde W_{\rm coll}$")
ax.set_xlabel(r"order $N$", fontsize=8.0)
ax.set_ylabel(r"$\lambda_{\rm P}(\varphi_c(N))/\lambda_{\rm P}(\varphi_c(2))$", fontsize=7.5)
ax.set_xticks(Ns); ax.set_xlim(1.75, 6.25); ax.set_ylim(0.95, 2.05)
ax.set_yticks([1.0, 1.5, 2.0])
ax.legend(loc="upper left", fontsize=6.6, labelspacing=0.14, handlelength=1.3, handletextpad=0.25, borderaxespad=0.12,
          frameon=False)
panel_label(ax, "(a)", x=-0.30, y=1.03)

# (b) MK critical packing fractions vs particle order N and exclusion order ell
ax = axs[1]
COL_ON, COL_OFF = C["green"], COL_BSELF
ax.axhspan(1.80, 1.93, color="#d8d6d0", zorder=0)
ax.axhline(2.40, color=COL_PHID, lw=0.85, ls=":")
ax.text(10.3, 2.45, r"$\varphi_p$", ha="right", va="bottom", fontsize=6.8, color=COL_PHID)
ax.axhline(1.776, color=COL_ON, lw=0.85, ls="-.")
ax.text(10.3, 1.70, r"$\varphi_G$", ha="right", va="top", fontsize=6.8, color=COL_ON)
y = [bs[n]["phi_c"] for n in Ns]; e = [bs[n]["half_width"] for n in Ns]
ax.errorbar(Ns, y, yerr=e, marker="s", mfc="white", ms=3.6, ls="-", color=COL_OFF, lw=1.2, capsize=1.3)
ax.plot(Ns, [bs[2]["phi_c"] * LAM[n] for n in Ns], ls="--", color=COL_PERRON, lw=0.95)
ax.text(5.3, 3.85, r"$N$", fontsize=7.2, color=COL_OFF, ha="left", va="center")
K = sorted((json.load(open(f)) for f in _g.glob(f"{RES}/kres_series_n*.json")), key=lambda d: d["n"])
kn = [d["n"] for d in K if d["n"] <= 10]
kp = [0.5 * (d["fold"][0] + d["fold"][1]) for d in K if d["n"] <= 10]
ke = [0.5 * (d["fold"][1] - d["fold"][0]) for d in K if d["n"] <= 10]
ax.errorbar(kn, kp, yerr=ke, fmt="o-", color=COL_ON, ms=3.5, lw=1.2, elinewidth=0.8, capsize=0, zorder=5)
g = [r for r in RS["gauss"] if r["n"] != "inf" and r["n"] <= 10]
ax.plot([r["n"] for r in g], [r["phi_c"] for r in g], "^", color=COL_ON, mfc="white", ms=3.2, mew=0.75, lw=0, zorder=4)
ax.text(3.8, 2.12, r"$\ell$", fontsize=7.2, color=COL_ON, ha="left", va="bottom")
ax.set_xlabel(r"order ($N$ or $\ell$)", fontsize=8.0)
ax.set_ylabel(r"$\varphi_c$", fontsize=7.8)
ax.set_xticks([1, 2, 4, 6, 8, 10]); ax.set_xlim(0.5, 10.6); ax.set_ylim(1.35, 5.20)
ax.set_yticks([2, 3, 4, 5])
panel_label(ax, "(b)", x=-0.26, y=1.03)

# (c) Bond-conditioned factorization ratio R(t) in MK
ax = axs[2]; p = "2.0"
F = CV[p]["Fs7"]; tF = np.array(F["t"]); mF = np.array(F["mean"])
series = (("c_Lpm", COL_BCOLL, "-", r"$\parallel$"),
          ("c_pm", COL_BSELF, "-", "avg."),
          ("r_pm", C["ink2"], (0, (3.2, 1.5)), "rand."),
          ("c_Tpm", COL_PY, "-", r"$\perp$"))
for key, col, ls, name in series:
    g = CV[p][key]; t = np.array(g["t"]); m = np.array(g["mean"]); Fi = np.interp(t, tF, mF); ok = Fi > 0.15
    ax.plot(t[ok], m[ok] / Fi[ok] ** 2, color=col, ls=ls, lw=1.25, label=name)
cc = np.array(RC[p]["curves"]["L-:ReC"])
ax.plot(cc[:, 0], cc[:, 1], color=COL_PERRON, ls=(0, (3.6, 1.3, 1.0, 1.3)), lw=1.15, label=r"$\mathrm{Re}\,\Gamma$")
ax.axhline(1, color=C["muted"], lw=0.6)
ax.set_xscale("log"); ax.set_xlim(1, 1.5e6); ax.set_ylim(0.55, 3.15)
ax.set_yticks([1.0, 2.0, 3.0])
ax.set_xticks([1e0, 1e3, 1e6])
ax.set_xlabel(r"$t$ (sweeps)", fontsize=8.0)
ax.set_ylabel(r"$R(t)=\mathrm{Re}\,\phi_2/F_s^2$", fontsize=7.6)
ax.legend(loc="upper left", ncol=2, fontsize=6.5, labelspacing=0.14, columnspacing=0.45,
          handlelength=1.25, handletextpad=0.25, borderaxespad=0.10, frameon=False)
panel_label(ax, "(c)", x=-0.26, y=1.03)

# (d) Ordinary 3D hard spheres vs MK: R_par (solid) and R_perp (dashed) vs F_s(k~7, t)
ax = axs[3]
cmap = plt.get_cmap("Blues")
hs_keys = [k for k in sorted(HS) if k.endswith("_N1000_d0.050")]
for n, k in enumerate(hs_keys):
    h = HS[k]; col = cmap(0.45 + 0.55 * n / max(len(hs_keys) - 1, 1))
    F = np.array(h["Fs7"], dtype=float)
    for q, ls in (("L-:R_glob", "-"), ("T-:R_glob", (0, (2.5, 1.5)))):
        a = np.array(h[q], dtype=float); Fi = np.interp(np.log(a[:, 0]), np.log(F[:, 0]), F[:, 1])
        ax.plot(Fi, a[:, 1], color=col, ls=ls, lw=1.15, label=("HS" if (q == "L-:R_glob" and n == len(hs_keys) - 1) else None))
oranges = ["#c9a227", "#d97b29", "#a8431d"]
for n, p in enumerate(("1.5", "1.8", "2.0")):
    Fm = CV[p]["Fs7"]; tF = np.array(Fm["t"]); mF = np.array(Fm["mean"])
    for q, ls in (("L-:R_glob", "-"), ("T-:R_glob", (0, (2.5, 1.5)))):
        a = np.array(RC[p]["curves"][q], dtype=float); Fi = np.interp(np.log(a[:, 0]), np.log(tF), mF)
        ax.plot(Fi, a[:, 1], color=oranges[n], ls=ls, lw=1.1, label=("MK" if (q == "L-:R_glob" and n == 2) else None))
ax.plot([], [], color="#444444", ls="-", lw=1.1, label=r"$R_\parallel$")
ax.plot([], [], color="#444444", ls="--", lw=0.8, label=r"$R_\perp$")
ax.axhline(1, color=C["muted"], lw=0.6); ax.axvline(0.45, color=C["muted"], lw=0.6, ls=":")
ax.set_xlim(1.0, 0.18); ax.set_ylim(0.55, 4.2)
ax.set_yticks([1, 2, 3, 4]); ax.set_xticks([1.0, 0.6, 0.2])
ax.set_xlabel(r"$F_s(k\simeq7,t)$", fontsize=8.0)
ax.set_ylabel(r"$R_\parallel, R_\perp$", fontsize=7.6)
ax.legend(loc="upper left", fontsize=6.4, ncol=2, columnspacing=0.5, handlelength=1.2, handletextpad=0.25, labelspacing=0.15, frameon=False)
panel_label(ax, "(d)", x=-0.24, y=1.03)

fig.subplots_adjust(left=0.06, right=0.99, bottom=0.23, top=0.88)
fig.savefig(f"{OUTL}/fig1_runoff.pdf"); fig.savefig(f"{OUTL}/fig1_runoff.png"); print("wrote 4-panel fig1_runoff")
