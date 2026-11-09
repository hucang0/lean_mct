"""SM Fig. S6: Bond-resolved pair factorization in ordinary (polydisperse) 3D hard spheres vs. the MK liquid.
(a) MK, phi = 2.0: R(t) for contact-par, contact-perp, orientation-avg, random, and Re Gamma.
(b) Ordinary 3D hard spheres (23% polydispersity), phi = 0.585: the same five curves.
(c) Model-free Pearson correlation of bond-projected displacements of contact pairs, parallel (solid) and
    transverse (dashed), vs. t / tau_7 for HS (phi = 0.50-0.60) and MK (phi = 0.8, 1.5, 1.8).
(d) R_par (solid) and R_perp (dashed) vs. decorrelation F_s(k ~ 7, t) for both systems."""
import json
import numpy as np
from _style import *
import matplotlib.pyplot as plt

PHI_HS = "0.585"
COL = {"L": C["blue"], "T": C["orange"], "I": C["aqua"], "C": C["violet"], "r": C["ink2"]}
INK2 = C["ink2"]

CV = json.load(open(f"{SIM}/results/curves_averaged.json"))
RC = json.load(open(f"{SIM}/reanalysis/conditional_summary.json"))
HS = json.load(open(f"{SIM}/hs_bond/fig_curves.json"))
MKP = json.load(open(f"{SIM}/hs_bond/mk_reanalysis_pearson.json"))

fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.8)); axs = axs.ravel()

# (a) MK phi = 2.0
ax = axs[0]; p = "2.0"
F = CV[p]["Fs7"]; tF = np.array(F["t"]); mF = np.array(F["mean"])
for key, c, ls, lab in (("c_Lpm", COL["L"], "-", r"contact $\parallel$"),
                        ("c_Tpm", COL["T"], "-", r"contact $\perp$"),
                        ("c_pm", COL["I"], "-", "contact, orient.-avg."),
                        ("r_pm", COL["r"], (0, (3, 1.5)), "random")):
    g = CV[p][key]; t = np.array(g["t"]); m = np.array(g["mean"]); Fi = np.interp(t, tF, mF); ok = Fi > 0.15
    ax.plot(t[ok], m[ok] / Fi[ok] ** 2, color=c, ls=ls, lw=1.4, label=lab)
cc = np.array(RC[p]["curves"]["L-:ReC"])
ax.plot(cc[:, 0], cc[:, 1], color=COL["C"], ls=(0, (3.5, 1.2, 1, 1.2)), lw=1.3, label=r"$\mathrm{Re}\,\Gamma$ ($\parallel$)")
ax.set_xscale("log"); ax.set_xlim(1, 1.5e6); ax.set_ylim(0.5, 4.5)
ax.text(0.97, 0.92, r"MK, $\varphi=2.0$ (caged)", transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
ax.set_xlabel(r"$t$ (MC sweeps)"); ax.set_ylabel(r"$R(t)=\mathrm{Re}\,\phi_2/F_s^2$")
ax.axhline(1, color=C["muted"], lw=0.6, zorder=0)
ax.legend(loc="upper left", fontsize=6.5, handlelength=2.2, labelspacing=0.18)
panel_label(ax, "(a)", x=-0.18, y=1.02)

# (b) HS phi = 0.585
ax = axs[1]; key = f"phi{PHI_HS}_N1000_d0.050"; h = HS[key]
for k, c, ls in (("L-:R_glob", COL["L"], "-"), ("T-:R_glob", COL["T"], "-"), ("Ipm:R_glob", COL["I"], "-"),
                 ("rand-:R", COL["r"], (0, (3, 1.5))), ("L-:ReC", COL["C"], (0, (3.5, 1.2, 1, 1.2)))):
    a = np.array(h[k], dtype=float)
    ax.plot(a[:, 0], a[:, 1], color=c, ls=ls, lw=1.4 if k != "L-:ReC" else 1.3)
for tv, lab in ((h["tstar"], r"$t^*$"), (h["tau7"], r"$\tau_7$")):
    if tv:
        ax.axvline(tv, color=C["muted"], lw=0.6, ls=":", zorder=0)
        ax.text(tv / 1.15, 3.6, lab, fontsize=6.8, color=INK2, ha="right")
ax.set_xscale("log"); ax.set_ylim(0.5, 4.5); ax.set_xlim(1, (h["tau7"] or 1e6) * 3)
ax.axhline(1, color=C["muted"], lw=0.6, zorder=0)
ax.text(0.03, 0.92, rf"ordinary HS ($23\%$ polydisp.), $\varphi={float(PHI_HS):.3g}$", transform=ax.transAxes, fontsize=7.5, color=INK2)
ax.set_xlabel(r"$t$ (MC sweeps)"); ax.set_ylabel(r"$R(t)=\mathrm{Re}\,\phi_2/F_s^2$")
panel_label(ax, "(b)", x=-0.18, y=1.02)

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
ax.axhline(0, color=C["muted"], lw=0.6, zorder=0); ax.axvline(1, color=C["muted"], lw=0.6, ls=":", zorder=0)
ax.set_xscale("log"); ax.set_xlim(1e-3, 30); ax.set_ylim(-0.15, 1.0)
ax.set_xlabel(r"$t/\tau_7$"); ax.set_ylabel(r"$\mathrm{corr}(\Delta x_i^{e},\Delta x_j^{e})$, contact pairs")
ax.text(0.03, 0.96, r"solid: $\mathbf{e}=\hat{\mathbf{r}}_{ij}$ ($\parallel$)" + "\n" + r"dashed: $\mathbf{e}\perp\hat{\mathbf{r}}_{ij}$",
        transform=ax.transAxes, fontsize=6.5, color=INK2, va="top")
ax.legend(loc="upper right", fontsize=6.2, handlelength=1.8, labelspacing=0.15, ncol=2, bbox_to_anchor=(1.0, 1.01))
panel_label(ax, "(c)", x=-0.18, y=1.02)

# (d) matched decorrelation: R_par (solid), R_perp (dashed) vs Fs(k7, t)
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
ax.axhline(1, color=C["muted"], lw=0.6, zorder=0); ax.axvline(0.448, color=C["muted"], lw=0.6, ls=":", zorder=0)
ax.text(0.448, 4.55, r"MK $\varphi=2$ plateau $F_s$", fontsize=6.5, color=INK2, ha="center", va="bottom", clip_on=False)
ax.set_xlim(1.0, 0.15); ax.set_ylim(0.5, 4.5)
ax.set_xlabel(r"$F_s(k\simeq7,t)$  (decorrelation $\rightarrow$)"); ax.set_ylabel(r"$R_\parallel$ (solid), $R_\perp$ (dashed)")
ax.legend(loc="upper left", fontsize=6.2, handlelength=1.8, labelspacing=0.15, ncol=2)
panel_label(ax, "(d)", x=-0.18, y=1.02)

fig.tight_layout(w_pad=1.5, h_pad=1.2)
save(fig, "F7_hs_vs_mk_bond")
